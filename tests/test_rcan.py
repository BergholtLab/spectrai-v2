"""Forward-shape tests for the RCAN architecture (Phase 2 port)."""

from __future__ import annotations

import pytest
import torch

from spectrai.networks.nets.rcan import Hyperspectral_RCAN
from spectrai.networks.registry import setup_network
from torch.nn import PReLU


def test_rcan_2d_super_resolution_shape():
    """2D super-resolution: channels-first image in -> scale*H x scale*W out."""
    spectrum_length = 32
    scale = 2
    net = Hyperspectral_RCAN(
        spectrum_length=spectrum_length,
        scale=scale,
        activation=PReLU(),
        n_resblocks=2,
        n_resgroups=2,
    ).float()

    # channels-first 2D image with 32 spectral channels
    x = torch.randn(2, spectrum_length, 16, 16)
    y = net(x)
    assert y.shape == (2, spectrum_length, 32, 32)


def test_rcan_2d_super_resolution_scale3_shape():
    spectrum_length = 32
    scale = 3
    net = Hyperspectral_RCAN(
        spectrum_length=spectrum_length,
        scale=scale,
        activation=PReLU(),
        n_resblocks=2,
        n_resgroups=2,
    ).float()

    x = torch.randn(1, spectrum_length, 16, 16)
    y = net(x)
    assert y.shape == (1, spectrum_length, 48, 48)


def test_rcan_1d_not_implemented():
    """The reference registry rejects RCAN for 1D (spectral) data.

    RCAN is Conv2d/PixelShuffle-based and only implemented for 2D image
    data, so the registry must raise ``NotImplementedError`` when asked to
    build it with 1D spectral data.
    """
    Task_Options = {"task": "Super-Resolution", "classes": 0}
    NH = {"network": "RCAN", "dimension": "2D", "activation": "ReLU",
          "normalization": "None"}
    TH = {"spectrum_length": 32, "input_image_size": 16, "target_image_size": 32}
    DMO = {"data_format": "Spectra"}

    with pytest.raises(NotImplementedError):
        setup_network(Task_Options, NH, TH, DMO)


def test_rcan_wrong_task_not_implemented():
    """The reference registry rejects RCAN for any non-Super-Resolution task."""
    Task_Options = {"task": "Denoising", "classes": 0}
    NH = {"network": "RCAN", "dimension": "2D", "activation": "ReLU",
          "normalization": "None"}
    TH = {"spectrum_length": 32, "input_image_size": 16, "target_image_size": 16}
    DMO = {"data_format": "Image: H, W, C"}

    with pytest.raises(NotImplementedError):
        setup_network(Task_Options, NH, TH, DMO)
