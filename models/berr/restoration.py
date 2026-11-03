"""Single multi-scale NAFNet-style restoration decoder."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import naf_stack


class RestorationDecoder(nn.Module):
    """Decode anchor and reliability-fused features into visible/prior estimates."""

    def __init__(self, width: int = 32, blocks: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        if len(blocks) != 2:
            raise ValueError("decoder blocks must contain exactly two levels")
        self.deep_fuse = nn.Conv2d(8 * width, 4 * width, 1)
        self.deep_blocks = naf_stack(4 * width, blocks[1])
        self.up2 = nn.ConvTranspose2d(4 * width, 2 * width, 2, stride=2)
        self.mid_fuse = nn.Conv2d(6 * width, 2 * width, 1)
        self.mid_blocks = naf_stack(2 * width, blocks[0])
        self.up1 = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.shallow_fuse = nn.Conv2d(3 * width, width, 1)
        self.shallow_blocks = naf_stack(width, blocks[0])
        self.head = nn.Conv2d(width, 6, 3, padding=1)

    def forward(
        self, anchor_features: list[torch.Tensor], fused_features: list[torch.Tensor], anchor_rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.deep_blocks(self.deep_fuse(torch.cat((anchor_features[2], fused_features[2]), dim=1)))
        x = self.mid_blocks(self.mid_fuse(torch.cat((self.up2(x), anchor_features[1], fused_features[1]), dim=1)))
        x = self.shallow_blocks(self.shallow_fuse(torch.cat((self.up1(x), anchor_features[0], fused_features[0]), dim=1)))
        visible_delta, prior_delta = self.head(x).chunk(2, dim=1)
        visible = (anchor_rgb + 0.5 * torch.tanh(visible_delta)).clamp(0, 1)
        prior = (anchor_rgb + 0.5 * torch.tanh(prior_delta)).clamp(0, 1)
        return visible, prior
