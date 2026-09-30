"""Registry dispatch tests: setup_network + edit_network glue for all architectures.

These exercise the registry branches (the subagent net tests import the nets
directly; these verify the reference's per-(task, dimension, data_format)
dispatch and the head-swap in ``edit_network``).
"""

from __future__ import annotations

import pytest
import torch

from spectrai.networks.registry import edit_network, setup_network


def _cfg(task, network, dimension="2D", data_format="Spectra", classes=3,
         spectrum_length=500, input_image_size=16, target_image_size=16,
         activation="ReLU", normalization="None"):
    return dict(
        Task_Options={"task": task, "classes": classes},
        Network_Hyperparameters={"network": network, "dimension": dimension,
                                 "activation": activation, "normalization": normalization},
        Training_Hyperparameters={"spectrum_length": spectrum_length,
                                  "input_image_size": input_image_size,
                                  "target_image_size": target_image_size},
        DataManager_Options={"data_format": data_format},
    )


def test_registry_unet_denoising_1d():
    cfg = _cfg("Denoising", "UNet")
    net = setup_network(**cfg)
    assert net(torch.randn(2, 1, 500)).shape == (2, 1, 500)


def test_registry_resnet_classification_1d():
    cfg = _cfg("Classification", "ResNet", spectrum_length=32)
    net = setup_network(**cfg)
    # Reference passes img_channels=channels, so 1D input is (B, spectrum_length, L).
    assert net(torch.randn(2, 32, 32)).shape == (2, 3)


def test_registry_resnet_rejects_denoising():
    cfg = _cfg("Denoising", "ResNet")
    with pytest.raises(NotImplementedError):
        setup_network(**cfg)


def test_registry_rcan_superres_2d():
    cfg = _cfg("Super-Resolution", "RCAN", data_format="Image: H, W, C",
               spectrum_length=32, input_image_size=16, target_image_size=32)
    net = setup_network(**cfg)
    assert net(torch.randn(2, 32, 16, 16)).shape == (2, 32, 32, 32)


def test_registry_rcan_rejects_1d():
    cfg = _cfg("Super-Resolution", "RCAN", data_format="Spectra")
    with pytest.raises(NotImplementedError):
        setup_network(**cfg)


def test_registry_efficientnet_classification_2d():
    cfg = _cfg("Classification", "EfficientNet", data_format="Image: H, W, C",
               spectrum_length=32, target_image_size=32)
    net = setup_network(**cfg)
    assert net(torch.randn(2, 32, 32, 32)).shape == (2, 3)


def test_registry_efficientnet_rejects_1d():
    cfg = _cfg("Classification", "EfficientNet", data_format="Spectra")
    with pytest.raises(NotImplementedError):
        setup_network(**cfg)


def test_registry_segnet_segmentation_2d():
    cfg = _cfg("Segmentation", "SegNet", data_format="Image: H, W, C", spectrum_length=32)
    net = setup_network(**cfg)
    assert net(torch.randn(2, 32, 32, 32)).shape == (2, 3, 32, 32)


def test_registry_segnet_rejects_denoising():
    cfg = _cfg("Denoising", "SegNet", data_format="Image: H, W, C")
    with pytest.raises(NotImplementedError):
        setup_network(**cfg)


def test_registry_densenet_classification_1d():
    # The reference's 1D DenseNet path requires a real normalization layer;
    # normalization='None' is a latent reference limitation (not supported by the reference).
    cfg = _cfg("Classification", "DenseNet", spectrum_length=32, normalization="BatchNorm")
    net = setup_network(**cfg)
    assert net(torch.randn(2, 1, 32)).shape == (2, 3)


def test_registry_densenet_rejects_denoising():
    cfg = _cfg("Denoising", "DenseNet")
    with pytest.raises(NotImplementedError):
        setup_network(**cfg)


def test_registry_invalid_network_raises_valueerror():
    cfg = _cfg("Denoising", "NotANet")
    with pytest.raises(ValueError):
        setup_network(**cfg)


def test_edit_network_resnet_head_swap():
    # Build for 3 classes, then edit the head to 10 classes; verify the swap.
    cfg_build = _cfg("Classification", "ResNet", classes=3, spectrum_length=32)
    cfg_edit = _cfg("Classification", "ResNet", classes=10, spectrum_length=32)
    net = setup_network(**cfg_build)
    x = torch.randn(2, 32, 32)
    out_before = net(x).shape
    net = edit_network(net, **cfg_edit)
    assert net(x).shape == (2, 10)
    assert out_before == (2, 3)


def test_edit_network_unet_head_swap_classification_2d():
    cfg = _cfg("Classification", "UNet", data_format="Image: H, W, C",
               classes=10, spectrum_length=32)
    net = setup_network(**cfg)
    net = edit_network(net, **cfg)
    assert net(torch.randn(2, 32, 16, 16)).shape == (2, 10)
