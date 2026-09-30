"""Tests for the dataset + dataloader pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectrai.data import (
    prepare_data,
    prepare_dataloader,
    split_data,
)
from spectrai.data.dataset import SpectralDataset
from spectrai.transforms import prepare_transforms

SPECTRAL_CFG = dict(
    Task_Options={"task": "Denoising"},
    Network_Hyperparameters={"network": "ResUNet", "dimension": "2D"},
    Training_Hyperparameters={"spectrum_length": 500, "batch_size": 8},
    Preprocessing={"spectral_crop_start": 0, "spectral_crop_end": 500,
                   "background_subtraction": "None", "data_normalization": "Max Value"},
    Data_Augmentation={"horizontal_flip": 0, "vertical_flip": 0, "rotation": 0, "random_crop": 0,
                       "spectral_shift": 0.0, "spectral_flip": 0, "spectral_background": 0, "mixup": 0},
    DataManager_Options={"data_format": "Spectra", "data_directory": "False", "shuffle": "False",
                         "seed": "None", "train_split": "None", "val_split": "None", "test_split": "None"},
)


def test_spectral_dataset_getitem(spectra):
    n = len(spectra)
    input_df = pd.DataFrame({"Data": [spectra[i] for i in range(n)]})
    target_df = pd.DataFrame({"Data": [spectra[i] for i in range(n)]})
    pipe = prepare_transforms(**SPECTRAL_CFG)
    ds = SpectralDataset(input_df, target_df, "spectrum", "spectrum", False, pipe, 0)
    assert len(ds) == n
    item = ds[0]
    assert item["input"].shape[-1] == 500
    assert item["name"] == "0"


def test_prepare_data_non_directory(spectra):
    dmo = dict(SPECTRAL_CFG["DataManager_Options"])
    dmo["train_input_data"] = spectra  # non-directory: pass array directly
    dmo["train_target_data"] = spectra
    inp, tgt, it, tt, directory = prepare_data(
        dmo, SPECTRAL_CFG["Task_Options"], "train", 0,
    )
    # prepare_dataset_from_input expects a path for .mat; here we pass an array,
    # which load_sample returns as-is.
    assert it == "spectrum"
    assert tt == "spectrum"
    assert directory is False


def test_split_data_denoising(spectra):
    n = len(spectra)
    input_df = pd.DataFrame({"Data": [spectra[i] for i in range(n)]})
    target_df = pd.DataFrame({"Data": [spectra[i] for i in range(n)]})
    dmo = dict(SPECTRAL_CFG["DataManager_Options"])
    dmo["train_split"] = 70
    dmo["val_split"] = 20
    dmo["test_split"] = 10

    data, it, tt, directory = split_data(
        SPECTRAL_CFG["Task_Options"], dmo, input_df, target_df, "spectrum", "spectrum", False,
    )
    assert len(data["train_input"]) + len(data["val_input"]) + len(data["test_input"]) == n
    assert len(data["train_target"]) == len(data["train_input"])
    assert len(data["val_target"]) == len(data["val_input"])
    assert len(data["test_target"]) == len(data["test_input"])


def test_prepare_dataloader(spectra):
    n = len(spectra)
    input_df = pd.DataFrame({"Data": [spectra[i] for i in range(n)]})
    target_df = pd.DataFrame({"Data": [spectra[i] for i in range(n)]})
    pipe = prepare_transforms(**SPECTRAL_CFG)
    loader = prepare_dataloader(
        SPECTRAL_CFG["Task_Options"], SPECTRAL_CFG["Training_Hyperparameters"],
        SPECTRAL_CFG["DataManager_Options"], input_df, target_df, "spectrum", "spectrum",
        False, pipe, 0,
    )
    batch = next(iter(loader))
    assert batch["input"].shape[0] == 8  # batch_size
    assert batch["input"].shape[-1] == 500


def test_classification_dataset(spectra):
    n = len(spectra)
    labels = np.repeat([0, 1], n // 2)
    df = pd.DataFrame({"Data": [spectra[i] for i in range(n)], "Encoded_Labels": labels})
    from spectrai.data.dataset import ClassificationDataset

    pipe = prepare_transforms(**SPECTRAL_CFG)
    ds = ClassificationDataset(df, None, "spectrum", "class", False, pipe, 0)
    item = ds[0]
    assert item["target"]["data"] in (0, 1)


def test_superres_dataset_length(spectra):
    from spectrai.data.dataset import ImageSuperResDataset

    n = len(spectra)
    target_df = pd.DataFrame({"Data": [spectra[i] for i in range(n)]})
    input_df = pd.DataFrame({"Data": [spectra[i] for i in range(n)]})
    ds = ImageSuperResDataset(target_df, target_df, "hyperspectral_image", "hyperspectral_image",
                              False, None, 0)
    assert len(ds) == n
    ds_apply = ImageSuperResDataset(input_df, None, "hyperspectral_image", "hyperspectral_image",
                                    False, None, 1)
    assert len(ds_apply) == n
