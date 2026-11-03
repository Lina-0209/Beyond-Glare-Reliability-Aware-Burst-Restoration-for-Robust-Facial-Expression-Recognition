"""Shared image loading and tensor conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_files(root: str | Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()])


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def resize_and_center_crop(image: Image.Image, size: int) -> Image.Image:
    return ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return TF.to_tensor(image)


def load_image_tensor(path: str | Path, image_size: int, grayscale: bool = False) -> torch.Tensor:
    image = load_mask_image(path) if grayscale else load_rgb_image(path)
    image = resize_and_center_crop(image, image_size)
    tensor = pil_to_tensor(image)
    if grayscale and tensor.dim() == 3:
        tensor = tensor[:1]
    return tensor


def save_tensor_as_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    image = TF.to_pil_image(tensor)
    image.save(path)
