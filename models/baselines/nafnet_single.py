"""Fixed lightweight NAFNet-style single-frame restoration baseline."""

from __future__ import annotations

import torch
from torch import nn
from models.berr.blocks import naf_stack


class NAFNetSingle(nn.Module):
    def __init__(self, width=24, blocks=(1, 1, 2, 2, 1, 1, 1)):
        super().__init__(); b=list(blocks); self.intro=nn.Conv2d(3,width,3,padding=1)
        self.enc1=naf_stack(width,b[0]); self.down1=nn.Conv2d(width,width*2,2,2)
        self.enc2=naf_stack(width*2,b[1]); self.down2=nn.Conv2d(width*2,width*4,2,2)
        self.enc3=naf_stack(width*4,b[2]); self.down3=nn.Conv2d(width*4,width*8,2,2)
        self.mid=naf_stack(width*8,b[3])
        self.up3=nn.Sequential(nn.Conv2d(width*8,width*8,1),nn.PixelShuffle(2)); self.dec3=naf_stack(width*2,b[4])
        self.up2=nn.Sequential(nn.Conv2d(width*2,width*4,1),nn.PixelShuffle(2)); self.dec2=naf_stack(width,b[5])
        self.up1=nn.Sequential(nn.Conv2d(width,width*2,1),nn.PixelShuffle(2)); self.dec1=naf_stack(width//2,b[6])
        self.skip3=nn.Conv2d(width*4,width*2,1); self.skip2=nn.Conv2d(width*2,width,1); self.skip1=nn.Conv2d(width,width//2,1)
        self.outro=nn.Conv2d(width//2,3,3,padding=1)

    def forward(self,x):
        e1=self.enc1(self.intro(x)); e2=self.enc2(self.down1(e1)); e3=self.enc3(self.down2(e2)); m=self.mid(self.down3(e3))
        d3=self.dec3(self.up3(m)+self.skip3(e3)); d2=self.dec2(self.up2(d3)+self.skip2(e2)); d1=self.dec1(self.up1(d2)+self.skip1(e1))
        return (x+self.outro(d1)).clamp(0,1)

