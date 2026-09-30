"""Shared building blocks for UNet-style architectures.

Ported from the reference ``spectrai/networks/layers.py``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .utils import get_normalization, get_pooling

__all__ = ["BasicConv", "UNetConv", "Down", "Up", "Upsampler"]


class BasicConv(nn.Module):
    """Conv + optional activation + optional normalization."""

    def __init__(self, dims, channels_in, channels_out, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 normalization="BatchNorm", activation=nn.ReLU(True), act_first=True):
        super().__init__()
        self.dims = dims

        conv = {
            1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d,
        }[dims](
            channels_in, channels_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
        )

        layers: list[nn.Module] = [conv]
        if activation and act_first:
            layers.append(activation)
        norm = get_normalization(channels_out, normalization, dims) if normalization and normalization != "None" else None
        if norm:
            layers.append(norm)
        if activation and not act_first:
            layers.append(activation)
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class UNetConv(nn.Module):
    """UNet convolution block (with optional residual connection)."""

    def __init__(self, dims, in_channels, out_channels, mid_channels=None,
                 normalization="BatchNorm", activation=None, res=False):
        super().__init__()
        self.res = res
        if not mid_channels:
            mid_channels = out_channels

        self.initial_conv = BasicConv(dims, in_channels, mid_channels, kernel_size=3, padding=1,
                                      normalization=normalization, activation=activation)
        res_conv = []
        for _ in range(2):
            res_conv.append(BasicConv(dims, mid_channels, mid_channels, kernel_size=3, padding=1,
                                      normalization=normalization, activation=activation))
        self.res_conv = nn.Sequential(*res_conv)
        self.final_conv = BasicConv(dims, mid_channels, out_channels, kernel_size=3, padding=1,
                                    normalization=normalization, activation=activation)

    def forward(self, x):
        x = self.initial_conv(x)
        if self.res:
            res = self.res_conv(x)
            res = res + x
            out = self.final_conv(res)
        else:
            out = self.final_conv(x)
        return out


class Down(nn.Module):
    """UNet contracting layer (maxpool + conv)."""

    def __init__(self, dims, in_channels, out_channels, normalization, activation, res=False):
        super().__init__()
        maxpool_conv = [
            get_pooling("MaxPool", size=2, dims=dims),
            UNetConv(dims, in_channels, out_channels, None, normalization, activation, res),
        ]
        self.maxpool_conv = nn.Sequential(*maxpool_conv)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """UNet expanding layer (upsample + concat + conv)."""

    def __init__(self, dims, in_channels, out_channels, normalization, activation,
                 bilinear=True, res=False):
        super().__init__()
        self.dims = dims
        if dims in (1, 3):
            self.up = nn.Upsample(scale_factor=2)
            self.conv = UNetConv(dims, in_channels, out_channels, in_channels // 2, normalization, activation, res)
        elif dims == 2:
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                self.conv = UNetConv(dims, in_channels, out_channels, in_channels // 2, normalization, activation, res)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = UNetConv(dims, in_channels, out_channels, None, normalization, activation, res)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.dims == 1:
            diffY = x2.size(2) - x1.size(2)
            x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        elif self.dims == 2:
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        elif self.dims == 3:
            diffZ = x2.size(2) - x1.size(2)
            diffY = x2.size(3) - x1.size(3)
            diffX = x2.size(4) - x1.size(4)
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2,
                            diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Upsampler(nn.Sequential):
    """Pixel-shuffle upsampler for hyperspectral super-resolution."""

    def __init__(self, scale, channels, kernel_size, normalization="BatchNorm",
                 activation=None, bias=True):
        m: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                conv = nn.Conv2d(channels, 4 * channels, kernel_size, padding=kernel_size // 2, bias=bias)
                m.append(conv)
                m.append(nn.PixelShuffle(2))
                norm = get_normalization(channels, normalization, dims=2)
                if norm:
                    m.append(norm)
                if activation:
                    m.append(activation)
        elif scale == 3:
            m.append(nn.Conv2d(channels, 9 * channels, kernel_size, padding=kernel_size // 2, bias=bias))
            m.append(nn.PixelShuffle(3))
            norm = get_normalization(channels, normalization, dims=2)
            if norm:
                m.append(norm)
            if activation:
                m.append(activation)
        else:
            raise NotImplementedError(f"Unsupported upsampling scale {scale}")

        super().__init__(*m)
