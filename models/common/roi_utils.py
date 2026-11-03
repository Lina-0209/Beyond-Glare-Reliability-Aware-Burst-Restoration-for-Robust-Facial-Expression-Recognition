"""ROI helpers for heuristic eye-region extraction and crop handling."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


def clamp_bbox(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def heuristic_eye_roi_bbox(height: int, width: int) -> Tuple[int, int, int, int]:
    """A dependency-free upper-face ROI covering both eyes and eyeglasses."""
    roi_width = int(width * 0.56)
    roi_height = int(height * 0.24)
    center_x = width // 2
    center_y = int(height * 0.37)
    x0 = center_x - roi_width // 2
    y0 = center_y - roi_height // 2
    x1 = x0 + roi_width
    y1 = y0 + roi_height
    return clamp_bbox(x0, y0, x1, y1, width, height)


def make_batch_heuristic_bboxes(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    bbox = torch.tensor(heuristic_eye_roi_bbox(height, width), dtype=torch.long, device=device)
    return bbox.unsqueeze(0).repeat(batch_size, 1)


def crop_and_resize(images: torch.Tensor, bboxes: torch.Tensor, output_size: int | Sequence[int]) -> torch.Tensor:
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    crops = []
    for image, bbox in zip(images, bboxes):
        x0, y0, x1, y1 = [int(value) for value in bbox.tolist()]
        crop = image[:, y0:y1, x0:x1]
        crop = F.interpolate(crop.unsqueeze(0), size=tuple(output_size), mode="bilinear", align_corners=False)
        crops.append(crop.squeeze(0))
    return torch.stack(crops, dim=0)


def paste_crop(base_image: torch.Tensor, roi_crop: torch.Tensor, bbox: Iterable[int]) -> torch.Tensor:
    x0, y0, x1, y1 = [int(value) for value in bbox]
    resized = F.interpolate(roi_crop.unsqueeze(0), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
    output = base_image.clone()
    output[:, y0:y1, x0:x1] = resized.squeeze(0)
    return output
