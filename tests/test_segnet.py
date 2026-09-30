"""Forward-shape tests for the SegNet architecture (Phase 2 port)."""

from __future__ import annotations

import pytest
import torch

from spectrai.networks.nets.segnet import SegNet


def test_segnet_2d_segmentation_shape():
    net = SegNet(dims=2, num_classes=3, channels=32)
    net.eval()
    x = torch.randn(2, 32, 32, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 3, 32, 32)


def test_segnet_3d_segmentation_shape():
    # 3D convention from the reference registry:
    #   SegNet(dims=3, num_classes=num_classes, channels=1, img_channels=channels)
    # with channels=1 (input channel dim) and img_channels=spectrum_length.
    # The 3D encoder/decoder uses Conv3d over the spatial volume; the
    # classifier flattens (C, D) into a 2D feature map and projects to
    # num_classes.  Input shape: (batch, 1, 32, 32, 32).
    # Output shape: (batch, num_classes, H, W) = (batch, 3, 32, 32).
    net = SegNet(dims=3, num_classes=3, channels=1, img_channels=32)
    net.eval()
    x = torch.randn(2, 1, 32, 32, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 3, 32, 32)


def test_segnet_1d_not_implemented():
    # The reference registry rejects 1D (spectral) data for SegNet;
    # the ported class is only wired for dims 2 and 3, so a dims=1
    # construction/forward must fail (NotImplementedError or a conv
    # channel mismatch).
    with pytest.raises((NotImplementedError, RuntimeError)):
        net = SegNet(dims=1, num_classes=3, channels=32)
        net.eval()
        with torch.no_grad():
            net(torch.randn(2, 1, 32))
