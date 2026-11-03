"""Differentiable ROI compositing utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _feather_alpha(
    mask: torch.Tensor,
    feather_pixels: int = 5,
) -> torch.Tensor:
    """Smooth the edges of a binary/soft mask for blending.

    Args:
        mask: (B, 1, H, W) alpha mask in [0, 1].
        feather_pixels: Gaussian blur kernel half-size for feathering.

    Returns:
        Feathered mask of the same shape.
    """
    if feather_pixels <= 0:
        return mask
    k = 2 * feather_pixels + 1
    sigma = feather_pixels / 2.0
    x = torch.arange(k, dtype=mask.dtype, device=mask.device) - feather_pixels
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.view(1, 1, k, k)

    return F.conv2d(mask, kernel_2d, padding=feather_pixels)


def soft_paste_roi(
    anchor_full: torch.Tensor,
    refined_roi: torch.Tensor,
    roi_bbox: torch.Tensor,
    alpha_mask: torch.Tensor | None = None,
    feather_pixels: int = 5,
) -> torch.Tensor:
    """Alpha-blended paste of refined ROI into the full image.

    Args:
        anchor_full: (B, 3, H, W) full anchor image.
        refined_roi: (B, 3, roi_H, roi_W) refined ROI.
        roi_bbox: (B, 4) [x0, y0, x1, y1].
        alpha_mask: optional (B, 1, roi_H, roi_W) soft mask; if None uses a feathered box.
        feather_pixels: edge feathering radius.

    Returns:
        (B, 3, H, W) pasted image.
    """
    B, C, H, W = anchor_full.shape
    result = anchor_full.clone()

    for i in range(B):
        x0, y0, x1, y1 = [int(v) for v in roi_bbox[i].tolist()]
        rh, rw = y1 - y0, x1 - x0
        if rh <= 0 or rw <= 0:
            continue

        roi_resized = F.interpolate(
            refined_roi[i : i + 1], size=(rh, rw), mode="bilinear", align_corners=False
        ).squeeze(0)

        if alpha_mask is not None:
            a = F.interpolate(
                alpha_mask[i : i + 1], size=(rh, rw), mode="bilinear", align_corners=False
            )
            a = _feather_alpha(a, feather_pixels).squeeze(0).clamp(0, 1)
        else:
            a = torch.ones(1, rh, rw, dtype=anchor_full.dtype, device=anchor_full.device)
            a = _feather_alpha(a.unsqueeze(0), feather_pixels).squeeze(0).clamp(0, 1)

        result[i, :, y0:y1, x0:x1] = a * roi_resized + (1 - a) * anchor_full[i, :, y0:y1, x0:x1]

    return result
