"""Shared loss helpers for stage-wise glare restoration."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target)


def l2_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def total_variation_loss(mask: torch.Tensor) -> torch.Tensor:
    """Encourage smooth masks while keeping the implementation lightweight."""
    loss_h = torch.mean(torch.abs(mask[..., 1:, :] - mask[..., :-1, :]))
    loss_w = torch.mean(torch.abs(mask[..., :, 1:] - mask[..., :, :-1]))
    return loss_h + loss_w


def mask_sparsity_loss(mask: torch.Tensor) -> torch.Tensor:
    return mask.mean()


def composition_consistency_loss(
    coarse_face: torch.Tensor,
    glare_layer: torch.Tensor,
    glare_mask: torch.Tensor,
    anchor_input: torch.Tensor,
) -> torch.Tensor:
    composed = torch.clamp(coarse_face + glare_layer * glare_mask, 0.0, 1.0)
    return F.l1_loss(composed, anchor_input)
