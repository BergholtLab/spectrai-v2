"""Tests for spectral + image transforms and the transform pipeline builders."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectrai.transforms import prepare_transforms
from spectrai.transforms import spectral as T


def _sample(data, target_type="spectrum", name="x"):
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    return {"input": data, "target": {"data": data, "type": target_type}, "name": name}


def test_crop_spectrum(spectrum):
    out = T.CropSpectrum(10, 100)(_sample(torch.from_numpy(spectrum)))
    assert out["input"].shape[-1] == 90


def test_pad_spectrum_longer(spectrum):
    out = T.PadSpectrum(400)(_sample(torch.from_numpy(spectrum)))
    assert out["input"].shape[-1] == 400


def test_pad_spectrum_shorter(spectrum):
    short = spectrum[:100]
    out = T.PadSpectrum(200)(_sample(torch.from_numpy(short)))
    assert out["input"].shape[-1] == 200


def test_max_normalize_spectrum(spectrum):
    out = T.MaxNormalizeSpectrum()(_sample(torch.from_numpy(spectrum)))
    assert np.isclose(float(torch.amax(out["input"])), 1.0, atol=1e-5)


def test_auc_normalize_spectrum(spectrum):
    out = T.AUCNormalizeSpectrum()(_sample(torch.from_numpy(spectrum)))
    assert np.isclose(float(torch.sum(out["input"])), 1.0, atol=1e-5)


def test_min_background_spectrum(spectrum):
    out = T.MinBackgroundSpectrum()(_sample(torch.from_numpy(spectrum)))
    assert np.isclose(float(torch.amin(out["input"])), 0.0, atol=1e-6)


def test_poly_background_spectrum(spectrum):
    out = T.PolyBackgroundSpectrum(3)(_sample(torch.from_numpy(spectrum)))
    assert tuple(out["input"].shape) == (1, 500)


def test_als_background_spectrum(spectrum):
    out = T.ALSBackgroundSpectrum()(_sample(torch.from_numpy(spectrum)))
    assert tuple(out["input"].shape) == (1, 500)


def test_flip_axis(spectrum):
    out = T.FlipAxis(-1)(_sample(torch.from_numpy(spectrum)))
    assert out["input"].shape == torch.from_numpy(spectrum).shape


def test_make_channels_first_last():
    x = {"input": torch.randn(1, 500), "target": {"data": torch.randn(1, 500), "type": "spectrum"}, "name": "x"}
    cf = T.MakeChannelsFirst()(x)
    assert cf["input"].shape == (500, 1)  # channels-first: last dim -> first
    cl = T.MakeChannelsLast()(cf)
    assert cl["input"].shape == (1, 500)  # round-trip


def test_add_dim_first_last():
    x = {"input": torch.randn(1, 500), "target": {"data": torch.randn(1, 500), "type": "spectrum"}, "name": "x"}
    f = T.AddDimFirst()(x)
    assert f["input"].dim() == 3
    l = T.AddDimLast()(x)
    assert l["input"].dim() == 3


def test_to_tensor_to_numpy_roundtrip(spectrum):
    t = T.ToTensor()(_sample(spectrum))
    assert isinstance(t["input"], torch.Tensor)
    back = T.ToNumpy()(_sample(torch.from_numpy(spectrum)))
    assert isinstance(back["input"], np.ndarray)


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


def test_crop_image_spectrum(hyperspectral_image):
    img = hyperspectral_image[:, :, 5:30]
    out = T.CropImageSpectrum(2, 25)(_sample(torch.from_numpy(img), "hyperspectral_image"))
    assert out["input"].shape[-1] == 23


def test_pad_crop_image_center(hyperspectral_image):
    out = T.PadCropImage(16, False)(_sample(torch.from_numpy(hyperspectral_image), "hyperspectral_image"))
    assert out["input"].shape[0] == 16 and out["input"].shape[1] == 16


def test_pad_crop_image_random(hyperspectral_image):
    out = T.PadCropImage(16, True)(_sample(torch.from_numpy(hyperspectral_image), "hyperspectral_image"))
    assert out["input"].shape[0] == 16 and out["input"].shape[1] == 16


def test_rotate_image(hyperspectral_image):
    out = T.RotateImage()(_sample(torch.from_numpy(hyperspectral_image), "hyperspectral_image"))
    assert out["input"].shape == torch.from_numpy(hyperspectral_image).shape


def test_max_normalize_image(hyperspectral_image):
    out = T.MaxNormalizeImage()(_sample(torch.from_numpy(hyperspectral_image), "hyperspectral_image"))
    assert np.isclose(float(torch.amax(out["input"])), 1.0, atol=1e-5)


def test_auc_normalize_image(hyperspectral_image):
    out = T.AUCNormalizeImage()(_sample(torch.from_numpy(hyperspectral_image), "hyperspectral_image"))
    per_pixel = torch.sum(out["input"], -1)
    assert np.isclose(float(per_pixel.min()), 1.0, atol=1e-3)


def test_min_background_image(hyperspectral_image):
    out = T.MinBackgroundImage()(_sample(torch.from_numpy(hyperspectral_image), "hyperspectral_image"))
    assert out["input"].shape == torch.from_numpy(hyperspectral_image).shape


def test_poly_background_image(hyperspectral_image):
    out = T.PolyBackgroundImage(3)(_sample(torch.from_numpy(hyperspectral_image), "hyperspectral_image"))
    assert out["input"].shape == torch.from_numpy(hyperspectral_image).shape


def test_bicubic_downsample_image(hyperspectral_image):
    out = T.BicubicDownsampleImage(8)(_sample(torch.from_numpy(hyperspectral_image), "hyperspectral_image"))
    assert out["input"].shape[0] == 8 and out["input"].shape[1] == 8


def test_skip_downsample_image(hyperspectral_image):
    out = T.SkipDownsampleImage(2)(_sample(torch.from_numpy(hyperspectral_image), "hyperspectral_image"))
    assert out["input"].shape[0] < hyperspectral_image.shape[0]


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


SPECTRAL_CFG = dict(
    Task_Options={"task": "Denoising"},
    Network_Hyperparameters={"network": "ResUNet", "dimension": "2D"},
    Training_Hyperparameters={"spectrum_length": 500},
    Preprocessing={"spectral_crop_start": 0, "spectral_crop_end": 500,
                   "background_subtraction": "None", "data_normalization": "Max Value"},
    Data_Augmentation={"horizontal_flip": 0, "vertical_flip": 0, "rotation": 0, "random_crop": 0,
                       "spectral_shift": 0.1, "spectral_flip": 1, "spectral_background": 0, "mixup": 1},
    DataManager_Options={"data_format": "Spectra", "data_directory": "False", "shuffle": "False",
                         "seed": "None", "train_split": "None", "val_split": "None", "test_split": "None"},
)


def test_prepare_spectral_pipeline(spectrum):
    pipe = prepare_transforms(**SPECTRAL_CFG)
    out = pipe(_sample(spectrum))
    assert out["input"].shape[-1] == 500
    assert out["input"].dim() >= 2


IMAGE_CFG = dict(
    Task_Options={"task": "Denoising"},
    Network_Hyperparameters={"network": "UNet", "dimension": "2D"},
    Training_Hyperparameters={"spectrum_length": 32, "target_image_size": 16, "input_image_size": 16},
    Preprocessing={"spectral_crop_start": 0, "spectral_crop_end": 32,
                   "background_subtraction": "None", "data_normalization": "Max Value"},
    Data_Augmentation={"horizontal_flip": 1, "vertical_flip": 1, "rotation": 1, "random_crop": 1,
                       "spectral_shift": 0.05, "spectral_flip": 1, "spectral_background": 0, "mixup": 0},
    DataManager_Options={"data_format": "Image: H, W, C", "data_directory": "False", "shuffle": "False",
                         "seed": "None", "train_split": "None", "val_split": "None", "test_split": "None"},
)


def test_prepare_image_pipeline(hyperspectral_image):
    pipe = prepare_transforms(**IMAGE_CFG)
    out = pipe(_sample(hyperspectral_image, "hyperspectral_image"))
    # channels-first after pipeline
    assert out["input"].shape[1] == 16 and out["input"].shape[2] == 16
