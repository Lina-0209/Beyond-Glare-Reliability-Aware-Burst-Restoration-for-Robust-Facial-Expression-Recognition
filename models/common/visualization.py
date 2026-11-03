"""Small visualization helpers for training and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid


def draw_bbox_overlay(image: torch.Tensor, bbox: torch.Tensor | list[int] | tuple[int, int, int, int], color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> torch.Tensor:
    overlay = image.detach().cpu().float().clone()
    if overlay.dim() == 4:
        overlay = overlay[0]
    if overlay.size(0) == 1:
        overlay = overlay.repeat(3, 1, 1)
    h, w = overlay.shape[-2:]
    if torch.is_tensor(bbox):
        if bbox.dim() > 1:
            bbox = bbox[0]
        x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    else:
        x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    for channel, value in enumerate(color):
        overlay[channel, y0:y1, x0] = value
        overlay[channel, y0:y1, x1 - 1] = value
        overlay[channel, y0, x0:x1] = value
        overlay[channel, y1 - 1, x0:x1] = value
    return overlay.clamp(0.0, 1.0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().float()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def _prepare_grid_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu().float()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor for visualization, got shape {tuple(tensor.shape)}")
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor.clamp(0.0, 1.0)


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(tensor).save(path)


def save_image_grid(tensors: list[torch.Tensor], path: str | Path, nrow: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = nrow or len(tensors)
    prepared = [_prepare_grid_tensor(tensor) for tensor in tensors]
    target_h, target_w = prepared[0].shape[-2:]
    normalized = []
    for tensor in prepared:
        if tensor.shape[-2:] != (target_h, target_w):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        normalized.append(tensor)
    batch = torch.stack(normalized, dim=0)
    grid = make_grid(batch, nrow=nrow)
    tensor_to_image(grid).save(path)
