"""Utilities for glare-mask post-processing."""

from __future__ import annotations

import torch

from .losses import total_variation_loss


def sigmoid_mask(mask_logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(mask_logits)


def threshold_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (mask >= threshold).float()


def mask_smoothness(mask: torch.Tensor) -> torch.Tensor:
    return total_variation_loss(mask)
