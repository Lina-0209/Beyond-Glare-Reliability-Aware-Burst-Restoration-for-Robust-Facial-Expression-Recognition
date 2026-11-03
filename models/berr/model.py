"""End-to-end 4-to-1 burst eyeglass reflection removal model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .alignment import GeometryAligner, OffsetHead, align_feature_pyramid, warp_with_flow
from .backbone import SharedNAFEncoder
from .composite import soft_paste_roi
from .degradation import DegradationHead, ReliabilityFusion
from .restoration import RestorationDecoder


class BurstEyeglassReflectionRemoval(nn.Module):
    """Shared-backbone, degradation-aware burst restoration.

    Frame 0 fixes the output pose. All frames share one encoder; degradation and
    optional offset prediction are lightweight heads on shared shallow features.
    """

    def __init__(
        self,
        roi_size: int = 128,
        width: int = 32,
        encoder_blocks: tuple[int, int, int] = (2, 2, 4),
        decoder_blocks: tuple[int, int] = (2, 2),
        reliability_temperature: float = 0.15,
        use_offset_head: bool = False,
        max_residual_flow: float = 4.0,
        feather_pixels: int = 3,
    ) -> None:
        super().__init__()
        self.roi_size = int(roi_size)
        self.feather_pixels = int(feather_pixels)
        self.use_offset_head = bool(use_offset_head)
        self.aligner = GeometryAligner(self.roi_size)
        self.encoder = SharedNAFEncoder(4, width, tuple(encoder_blocks))
        self.degradation_head = DegradationHead(width)
        self.offset_head = OffsetHead(width, max_residual_flow) if self.use_offset_head else None
        self.fusion = ReliabilityFusion(reliability_temperature)
        self.decoder = RestorationDecoder(width, tuple(decoder_blocks))

    @staticmethod
    def _paste_alpha(alpha_roi: torch.Tensor, roi_bbox: torch.Tensor, height: int, width: int) -> torch.Tensor:
        full = alpha_roi.new_zeros(alpha_roi.size(0), 1, height, width)
        for index, bbox in enumerate(roi_bbox):
            x0, y0, x1, y1 = [int(value) for value in bbox.tolist()]
            if x1 <= x0 or y1 <= y0:
                continue
            resized = F.interpolate(alpha_roi[index : index + 1], (y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
            full[index, :, y0:y1, x0:x1] = resized[0]
        return full.clamp(0, 1)

    def _encode_burst(self, rois: torch.Tensor, geometry: torch.Tensor):
        batch_size, frames = rois.shape[:2]
        geometry_stack = geometry[:, None].expand(-1, frames, -1, -1, -1)
        encoded = self.encoder(torch.cat((rois, geometry_stack), dim=2).flatten(0, 1))
        return [level.view(batch_size, frames, *level.shape[1:]) for level in encoded]

    def _predict_and_apply_offsets(self, pyramid, rois):
        batch_size, frames = rois.shape[:2]
        if self.offset_head is None:
            flow = rois.new_zeros(batch_size, frames, 2, self.roi_size, self.roi_size)
            return pyramid, rois, flow

        shallow = pyramid[0]
        anchor = shallow[:, :1].expand_as(shallow)
        flow = self.offset_head(shallow.flatten(0, 1), anchor.flatten(0, 1)).view(
            batch_size, frames, 2, self.roi_size, self.roi_size
        )
        # Reference coordinates must remain unchanged.
        reference_mask = flow.new_ones(batch_size, frames, 1, 1, 1)
        reference_mask[:, 0] = 0
        flow = flow * reference_mask
        aligned_pyramid = []
        for level in pyramid:
            flat = align_feature_pyramid([level.flatten(0, 1)], flow.flatten(0, 1), self.roi_size)[0]
            aligned_pyramid.append(flat.view_as(level))
        aligned_rois = warp_with_flow(rois.flatten(0, 1), flow.flatten(0, 1)).view_as(rois)
        return aligned_pyramid, aligned_rois, flow

    def forward(self, burst, landmarks, roi_bbox, geometry_heatmap_roi) -> dict[str, torch.Tensor]:
        if burst.ndim != 5 or burst.size(2) != 3:
            raise ValueError(f"Expected burst (B,T,3,H,W), got {burst.shape}")
        geometry = geometry_heatmap_roi if geometry_heatmap_roi.ndim == 4 else geometry_heatmap_roi.unsqueeze(1)
        if geometry.shape[-2:] != (self.roi_size, self.roi_size):
            geometry = F.interpolate(geometry, (self.roi_size, self.roi_size), mode="bilinear", align_corners=False)

        rois, transforms = self.aligner(burst, landmarks, roi_bbox)
        pyramid = self._encode_burst(rois, geometry)
        pyramid, aligned_rois, residual_flow = self._predict_and_apply_offsets(pyramid, rois)

        batch_size, frames = burst.shape[:2]
        logits = self.degradation_head(pyramid[0].flatten(0, 1)).view(
            batch_size, frames, 1, self.roi_size, self.roi_size
        )
        degradation = torch.sigmoid(logits)
        fused_features, weights_per_scale = self.fusion(pyramid, logits)
        anchor_features = [level[:, 0] for level in pyramid]
        anchor_roi = aligned_rois[:, 0]
        visible_roi, prior_roi = self.decoder(anchor_features, fused_features, anchor_roi)

        reliability = weights_per_scale[0]
        evidence_roi = (reliability * aligned_rois).sum(dim=1)
        all_degraded = degradation.prod(dim=1)
        anchor_mask = degradation[:, 0]
        source = visible_roi * (1.0 - all_degraded) + prior_roi * all_degraded
        restored_roi = anchor_roi * (1.0 - anchor_mask) + source * anchor_mask

        anchor_full = burst[:, 0]
        final = soft_paste_roi(
            anchor_full, restored_roi, roi_bbox,
            alpha_mask=anchor_mask, feather_pixels=self.feather_pixels,
        )
        alpha_full = self._paste_alpha(anchor_mask, roi_bbox, burst.size(-2), burst.size(-1))
        return {
            "final": final,
            "restored_roi": restored_roi,
            "visible_roi": visible_roi,
            "prior_roi": prior_roi,
            "anchor_full": anchor_full,
            "anchor_roi": anchor_roi,
            "aligned_rois": aligned_rois,
            "transforms": transforms,
            "residual_flows": residual_flow,
            "degradation_logits": logits,
            "degradation": degradation,
            "reliability": reliability,
            "reliability_pyramid": weights_per_scale,
            "evidence_roi": evidence_roi,
            "all_degraded_roi": all_degraded,
            "temporal_variance": aligned_rois.var(dim=1, unbiased=False),
            "geometry_heatmap_roi": geometry,
            "alpha_full": alpha_full,
        }

    def align_supervision_masks(self, masks, outputs, roi_bbox):
        return self.aligner.align_masks(
            masks, outputs["transforms"], roi_bbox, outputs["residual_flows"]
        )
