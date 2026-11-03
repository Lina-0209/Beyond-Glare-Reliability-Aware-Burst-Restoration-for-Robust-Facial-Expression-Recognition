"""Lightweight degradation head and multi-scale reliability fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DegradationHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, max(channels // 2, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // 2, 8), 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class ReliabilityFusion(nn.Module):
    def __init__(self, temperature: float = 0.15) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(
        self, pyramid: list[torch.Tensor], logits: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Fuse BxTxCxHxW features at each scale with dense reliability."""
        fused, weights_per_scale = [], []
        for features in pyramid:
            batch_size, frames = features.shape[:2]
            scaled_logits = F.interpolate(
                logits.flatten(0, 1), features.shape[-2:], mode="bilinear", align_corners=False
            ).view(batch_size, frames, 1, *features.shape[-2:])
            weights = torch.softmax(-scaled_logits / self.temperature, dim=1)
            fused.append((weights * features).sum(dim=1))
            weights_per_scale.append(weights)
        return fused, weights_per_scale
