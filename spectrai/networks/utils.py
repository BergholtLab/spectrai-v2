"""Shared network helpers: normalization and pooling factories.

Ported from the reference ``spectrai/networks/layer_utils.py``.
"""

from __future__ import annotations

from torch import nn

__all__ = ["get_normalization", "get_pooling"]


def get_normalization(channels: int, normalization: str, dims: int):
    """Return the appropriate normalization layer for ``channels``/``dims``."""
    if normalization == "BatchNorm":
        if dims == 1:
            return nn.BatchNorm1d(channels)
        if dims == 2:
            return nn.BatchNorm2d(channels)
        if dims == 3:
            return nn.BatchNorm3d(channels)
    elif normalization == "InstanceNorm":
        if dims == 1:
            return nn.InstanceNorm1d(channels)
        if dims == 2:
            return nn.InstanceNorm2d(channels)
        if dims == 3:
            return nn.InstanceNorm3d(channels)
    elif normalization == "LayerNorm":
        return nn.LayerNorm(channels)
    elif normalization == "GroupNorm":
        return nn.GroupNorm(channels // 4, channels)
    return None


def get_pooling(pooling: str, size, stride=None, padding=0, dilation=1, dims: int = 2):
    """Return the appropriate pooling layer for ``dims``."""
    if pooling == "MaxPool":
        if dims == 1:
            return nn.MaxPool1d(size, stride=stride, padding=padding, dilation=dilation)
        if dims == 2:
            return nn.MaxPool2d(size, stride=stride, padding=padding, dilation=dilation)
        if dims == 3:
            return nn.MaxPool3d(size, stride=stride, padding=padding, dilation=dilation)
    elif pooling == "AvgPool":
        if dims == 1:
            return nn.AvgPool1d(size, stride=stride, padding=padding)
        if dims == 2:
            return nn.AvgPool2d(size, stride=stride, padding=padding)
        if dims == 3:
            return nn.AvgPool3d(size, stride=stride, padding=padding)
    elif pooling == "AdaptiveAvgPool":
        if dims == 1:
            return nn.AdaptiveAvgPool1d(size)
        if dims == 2:
            return nn.AdaptiveAvgPool2d(size)
        if dims == 3:
            return nn.AdaptiveAvgPool3d(size)
    return None
