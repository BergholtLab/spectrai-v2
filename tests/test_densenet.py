"""Forward-shape tests for the DenseNet architectures (Phase 2 port)."""

from __future__ import annotations

import pytest
import torch

from spectrai.networks.nets.densenet import densenet121, densenet161, densenet169, densenet201


@pytest.mark.parametrize("factory", [densenet121, densenet161, densenet169, densenet201])
def test_densenet_1d_classification_shape(factory):
    """1D classification on spectra (Spectra data format)."""
    net = factory(dims=1, channels=1, num_classes=3)
    x = torch.randn(2, 1, 32)
    y = net(x)
    assert y.shape == (2, 3)


@pytest.mark.parametrize("factory", [densenet121, densenet161, densenet169, densenet201])
def test_densenet_2d_classification_shape(factory):
    """2D classification on hyperspectral image data (32x32 image, 4 spectral channels)."""
    net = factory(dims=2, channels=4, num_classes=3)
    x = torch.randn(2, 4, 32, 32)
    y = net(x)
    assert y.shape == (2, 3)
