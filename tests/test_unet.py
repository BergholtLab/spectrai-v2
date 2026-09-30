"""Forward-shape tests for the UNet / ResUNet architecture (Phase 1 port)."""

from __future__ import annotations

import pytest
import torch

from spectrai.networks.registry import setup_network


def test_unet_1d_denoising_shape():
    Task_Options = {"task": "Denoising", "classes": 0}
    NH = {"network": "ResUNet", "dimension": "2D", "activation": "PReLU", "normalization": "None"}
    TH = {"spectrum_length": 500, "input_image_size": 16, "target_image_size": 16}
    DMO = {"data_format": "Spectra"}

    net = setup_network(Task_Options, NH, TH, DMO)
    x = torch.randn(4, 1, 500)
    y = net(x)
    assert y.shape == x.shape


def test_unet_1d_calibration_shape():
    Task_Options = {"task": "Calibration", "classes": 0}
    NH = {"network": "UNet", "dimension": "2D", "activation": "ReLU", "normalization": "None"}
    TH = {"spectrum_length": 500, "input_image_size": 16, "target_image_size": 16}
    DMO = {"data_format": "Spectra"}

    net = setup_network(Task_Options, NH, TH, DMO)
    x = torch.randn(2, 1, 500)
    y = net(x)
    assert y.shape == x.shape


def test_unet_2d_denoising_shape():
    Task_Options = {"task": "Denoising", "classes": 0}
    NH = {"network": "UNet", "dimension": "2D", "activation": "ReLU", "normalization": "None"}
    TH = {"spectrum_length": 32, "input_image_size": 16, "target_image_size": 16}
    DMO = {"data_format": "Image: H, W, C"}

    net = setup_network(Task_Options, NH, TH, DMO)
    # channels-first 2D image with 32 spectral channels
    x = torch.randn(2, 32, 16, 16)
    y = net(x)
    assert y.shape == x.shape


def test_unet_1d_classification_shape():
    Task_Options = {"task": "Classification", "classes": 3}
    NH = {"network": "UNet", "dimension": "2D", "activation": "ReLU", "normalization": "None"}
    TH = {"spectrum_length": 500, "input_image_size": 16, "target_image_size": 16}
    DMO = {"data_format": "Spectra"}

    net = setup_network(Task_Options, NH, TH, DMO)
    x = torch.randn(4, 1, 500)
    y = net(x)
    assert y.shape == (4, 3)


def test_unet_not_implemented_3d_superres():
    Task_Options = {"task": "Super-Resolution", "classes": 0}
    NH = {"network": "UNet", "dimension": "3D", "activation": "ReLU", "normalization": "None"}
    TH = {"spectrum_length": 32, "input_image_size": 8, "target_image_size": 64}
    DMO = {"data_format": "Image: H, W, C"}

    with pytest.raises(NotImplementedError):
        setup_network(Task_Options, NH, TH, DMO)
