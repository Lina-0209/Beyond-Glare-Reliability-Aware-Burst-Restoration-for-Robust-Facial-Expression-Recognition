"""Shared multi-scale feature encoder."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import naf_stack


class SharedNAFEncoder(nn.Module):
    """One weight-shared encoder applied to every burst frame."""

    def __init__(self, in_channels: int = 4, width: int = 32, blocks: tuple[int, int, int] = (2, 2, 4)) -> None:
        super().__init__()
        if len(blocks) != 3:
            raise ValueError("encoder blocks must contain exactly three levels")
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)
        self.level1 = naf_stack(width, blocks[0])
        self.down1 = nn.Conv2d(width, 2 * width, 2, stride=2)
        self.level2 = naf_stack(2 * width, blocks[1])
        self.down2 = nn.Conv2d(2 * width, 4 * width, 2, stride=2)
        self.level3 = naf_stack(4 * width, blocks[2])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        level1 = self.level1(self.stem(x))
        level2 = self.level2(self.down1(level1))
        level3 = self.level3(self.down2(level2))
        return [level1, level2, level3]
