"""Parameter-free four-frame alignment and RGB averaging benchmark baseline."""
from __future__ import annotations

import numpy as np
import torch

from models.berr.geometry import align_burst_to_anchor_batch
from .preprocessing import _as_rgb_array, _resize_rgb


@torch.inference_mode()
def aligned_mean_rgb(frames, landmarks, *, image_size=256, device="cpu"):
    """Use BERR's similarity-warp primitive, then an equal RGB mean.

    No masking, learned reliability, ROI paste, or new network weights are used.
    Coordinates are provided in each original frame's pixel coordinate system.
    Border extension and bilinear sampling follow the existing BERR geometry.
    """
    if len(frames) != 4:
        raise ValueError("aligned mean requires exactly four frames")
    images = [_as_rgb_array(frame) for frame in frames]
    points = np.asarray(landmarks, dtype=np.float32).copy()
    if points.ndim != 3 or points.shape[0] != 4 or points.shape[2] != 2 or points.shape[1] < 3:
        raise ValueError("landmarks must have shape [4,K,2], K >= 3")
    if not np.isfinite(points).all():
        raise ValueError("non-finite landmarks")
    for index, frame in enumerate(images):
        points[index, :, 0] *= image_size / frame.shape[1]
        points[index, :, 1] *= image_size / frame.shape[0]
    resized = np.stack([_resize_rgb(frame, image_size) for frame in images])
    burst = torch.from_numpy(resized).permute(0, 3, 1, 2).float().div(255).unsqueeze(0).to(device)
    landmarks_t = torch.from_numpy(points).unsqueeze(0).to(device)
    aligned, _ = align_burst_to_anchor_batch(
        burst, landmarks_t, torch.zeros(1, dtype=torch.long, device=device))
    mean = aligned.mean(dim=1)[0].permute(1, 2, 0).clamp(0, 1)
    return (mean.cpu().numpy() * 255).round().astype(np.uint8)
