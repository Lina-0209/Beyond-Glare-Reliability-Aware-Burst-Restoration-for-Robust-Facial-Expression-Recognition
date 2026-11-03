"""Landmark alignment and optional feature-offset correction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.common.roi_utils import crop_and_resize

from .geometry import align_burst_to_anchor_batch, warp_frames


def warp_with_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``image`` with full-resolution pixel-space flow."""
    n, _, height, width = image.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
        torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    base = torch.stack((xs, ys), dim=-1).unsqueeze(0).expand(n, -1, -1, -1)
    normalized = torch.stack(
        (flow[:, 0] * 2.0 / max(width - 1, 1), flow[:, 1] * 2.0 / max(height - 1, 1)), dim=-1
    )
    return F.grid_sample(image, base + normalized, mode="bilinear", padding_mode="border", align_corners=True)


class GeometryAligner(nn.Module):
    """Parameter-free face alignment; frame 0 defines output coordinates."""

    def __init__(self, roi_size: int) -> None:
        super().__init__()
        self.roi_size = int(roi_size)

    def forward(self, burst, landmarks, roi_bbox):
        batch_size, frames = burst.shape[:2]
        anchor = torch.zeros(batch_size, dtype=torch.long, device=burst.device)
        aligned, transforms = align_burst_to_anchor_batch(burst, landmarks, anchor)
        rois = torch.stack(
            [crop_and_resize(aligned[:, frame], roi_bbox, self.roi_size) for frame in range(frames)], dim=1
        )
        return rois, transforms

    def align_masks(self, masks, transforms, roi_bbox, residual_flow):
        batch_size, frames, _, height, width = masks.shape
        aligned = warp_frames(
            masks.flatten(0, 1), transforms.flatten(0, 1), output_size=(height, width)
        ).view_as(masks)
        rois = torch.stack(
            [crop_and_resize(aligned[:, frame], roi_bbox, self.roi_size) for frame in range(frames)], dim=1
        )
        return warp_with_flow(rois.flatten(0, 1), residual_flow.flatten(0, 1)).view_as(rois).clamp(0, 1)


class OffsetHead(nn.Module):
    """Optional lightweight residual offset head for future real-video data."""

    def __init__(self, channels: int, max_flow: float = 4.0) -> None:
        super().__init__()
        self.max_flow = float(max_flow)
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1),
        )

    def forward(self, features: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat((features, anchor), dim=1))) * self.max_flow


def align_feature_pyramid(
    pyramid: list[torch.Tensor], flow: torch.Tensor, roi_size: int
) -> list[torch.Tensor]:
    aligned = []
    for features in pyramid:
        scale = features.size(-1) / float(roi_size)
        scaled_flow = F.interpolate(flow, features.shape[-2:], mode="bilinear", align_corners=False) * scale
        aligned.append(warp_with_flow(features, scaled_flow))
    return aligned
