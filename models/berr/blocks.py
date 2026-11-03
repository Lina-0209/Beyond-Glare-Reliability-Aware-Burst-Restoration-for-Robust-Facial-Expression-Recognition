"""NAFNet-style blocks for low-level image restoration."""

from __future__ import annotations

import torch
from torch import nn


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class LayerNorm2d(nn.Module):
    """Per-pixel channel normalization used by NAFNet."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class NAFBlock(nn.Module):
    """Compact non-linear-activation-free residual block."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.gate1 = SimpleGate()
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden // 2, hidden // 2, 1))
        self.conv2 = nn.Conv2d(hidden // 2, channels, 1)

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, hidden, 1)
        self.gate2 = SimpleGate()
        self.ffn2 = nn.Conv2d(hidden // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(self.conv1(self.norm1(x)))
        residual = self.gate1(residual)
        residual = self.conv2(residual * self.attention(residual))
        y = x + residual * self.beta
        feed_forward = self.ffn2(self.gate2(self.ffn1(self.norm2(y))))
        return y + feed_forward * self.gamma


def naf_stack(channels: int, depth: int) -> nn.Sequential:
    return nn.Sequential(*(NAFBlock(channels) for _ in range(depth)))
