"""Data preparation pipeline: load, split, transform, and build DataLoaders.

Ported from the reference ``spectrai/dataloader/dataloaders.py`` and
``spectrai/utils/utilities.py`` (data-prep parts).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from .dataset import (
    ClassificationDataset,
    ImageSegmentationDataset,
    ImageSuperResDataset,
    SpectralDataset,
    load_mask,
    load_sample,
)
from ..transforms import prepare_transforms

__all__ = [
    "prepare_data",
    "get_data",
    "load_data_from_directory",
    "load_data_from_input",
    "split_data",
    "prepare_dataloader",
    "prepare_dataset_directory",
    "prepare_dataset_from_input",
    "prepare_classification_dataset",
]


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------


def prepare_dataset_directory(dataset_path: str) -> pd.DataFrame:
    """Build a one-column (``Data``) DataFrame of file paths in a directory."""
    data = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    return pd.DataFrame({"Data": data})


def prepare_dataset_from_input(sample, sample_type: str) -> pd.DataFrame:
    """Build a one-column DataFrame from a single input file."""
    if sample_type == "spectrum":
        spectra = load_sample(sample)
        return pd.DataFrame(((x,) for x in spectra), columns=["Data"])
    if sample_type == "hyperspectral_image":
        return pd.DataFrame({"Data": [load_sample(sample)]})
    if sample_type == "mask":
        return pd.DataFrame({"Data": [load_mask(sample)]})
    raise ValueError(f"unknown sample_type {sample_type!r}")


def prepare_classification_dataset(dataset_path: str, sample_type: str) -> pd.DataFrame:
    """Build a classification DataFrame (``Data`` + ``Labels`` + ``Encoded_Labels``).

    Expects ``dataset_path`` to contain one sub-folder per class.
    """
    data: list = []
    labels: list = []
    lb = LabelEncoder()

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        entries = os.listdir(folder_path)
        if len(entries) > 1:
            for sample_name in entries:
                data.append(os.path.join(folder_path, sample_name))
                labels.append(folder)
        else:
            single = os.path.join(folder_path, entries[0])
            sub_df = prepare_dataset_from_input(single, sample_type)
            for i in range(len(sub_df)):
                data.append(sub_df["Data"][i])
                labels.append(folder)

    dataset = pd.DataFrame({"Data": data, "Labels": labels})
    dataset["Encoded_Labels"] = lb.fit_transform(dataset["Labels"])
    return dataset


# ---------------------------------------------------------------------------
# prepare_data + helpers
# ---------------------------------------------------------------------------


def get_data(DataManager_Options, Task_Options, dataset_type, apply=0):
    if dataset_type == "train":
        return (
            DataManager_Options["train_input_data"],
            DataManager_Options["train_target_data"],
        )
    if dataset_type == "val":
        return (
            DataManager_Options["val_input_data"],
            DataManager_Options["val_target_data"],
        )
    if dataset_type == "test" and Task_Options["task"] == "Super-Resolution":
        if apply:
            return DataManager_Options["test_target_data"], None
        return None, DataManager_Options["test_target_data"]
    # test
    if apply:
        return DataManager_Options["test_input_data"], None
    return DataManager_Options["test_input_data"], DataManager_Options["test_target_data"]


def load_data_from_directory(DataManager_Options, Task_Options, input_data, target_data, input_type, apply=0):
    if Task_Options["task"] == "Classification":
        if apply:
            input_data = prepare_dataset_directory(input_data)
        else:
            input_data = prepare_classification_dataset(input_data, input_type)
            target_data = "None"
        target_type = "class"
    elif Task_Options["task"] == "Super-Resolution":
        if apply:
            input_data = prepare_dataset_directory(input_data)
        else:
            input_data = "None"
            target_data = prepare_dataset_directory(target_data)
        target_type = "hyperspectral_image"
    elif Task_Options["task"] == "Segmentation":
        input_data = prepare_dataset_directory(input_data)
        if not apply:
            target_data = prepare_dataset_directory(target_data)
        target_type = "mask"
    else:  # Calibration / Denoising
        input_data = prepare_dataset_directory(input_data)
        if not apply:
            target_data = prepare_dataset_directory(target_data)
        target_type = "spectrum" if DataManager_Options["data_format"] == "Spectra" else "hyperspectral_image"
    return input_data, target_data, input_type, target_type


def load_data_from_input(DataManager_Options, Task_Options, input_data, target_data, input_type, apply=0):
    if Task_Options["task"] == "Classification":
        target_type = "class"
    elif Task_Options["task"] == "Super-Resolution":
        target_type = "hyperspectral_image"
    elif Task_Options["task"] == "Segmentation":
        target_type = "mask"
    else:
        target_type = "spectrum" if DataManager_Options["data_format"] == "Spectra" else "hyperspectral_image"

    input_data = prepare_dataset_from_input(input_data, input_type)
    if not apply:
        target_data = prepare_dataset_from_input(target_data, target_type)
    return input_data, target_data, input_type, target_type


def prepare_data(DataManager_Options, Task_Options, dataset_type, apply=0):
    input_data, target_data = get_data(DataManager_Options, Task_Options, dataset_type, apply)

    input_type = "spectrum" if DataManager_Options["data_format"] == "Spectra" else "hyperspectral_image"

    if DataManager_Options["data_directory"] == "True":
        directory = True
        input_data, target_data, input_type, target_type = load_data_from_directory(
            DataManager_Options, Task_Options, input_data, target_data, input_type, apply
        )
    else:
        directory = False
        input_data, target_data, input_type, target_type = load_data_from_input(
            DataManager_Options, Task_Options, input_data, target_data, input_type, apply
        )
    return input_data, target_data, input_type, target_type, directory


def _as_split(value):
    if value is None or (isinstance(value, str) and value == "None"):
        return 0.0
    return float(value)


def split_data(Task_Options, DataManager_Options, input_data, target_data, input_type, target_type, directory):
    val_test_split = (_as_split(DataManager_Options["val_split"]) + _as_split(DataManager_Options["test_split"])) * 0.01
    test_split = _as_split(DataManager_Options["test_split"]) * 0.01

    data: dict = {}
    if Task_Options["task"] == "Super-Resolution" and isinstance(target_data, pd.DataFrame):
        y_train, y_val_test = train_test_split(target_data, test_size=val_test_split)
        data["train_input"] = None
        data["val_input"] = None
        data["test_input"] = None
        data["train_target"] = y_train.reset_index(drop=True)
    else:
        if isinstance(target_data, pd.DataFrame):
            x_train, x_val_test, y_train, y_val_test = train_test_split(
                input_data, target_data, test_size=val_test_split
            )
            data["train_target"] = y_train.reset_index(drop=True)
        else:
            x_train, x_val_test = train_test_split(input_data, test_size=val_test_split)
            data["train_target"] = None
        data["train_input"] = x_train.reset_index(drop=True)

    if test_split > 0.0:
        if Task_Options["task"] == "Super-Resolution" and isinstance(target_data, pd.DataFrame):
            y_val, y_test = train_test_split(y_val_test, test_size=test_split / val_test_split)
            data["val_target"] = y_val.reset_index(drop=True)
            data["test_target"] = y_test.reset_index(drop=True)
        else:
            if isinstance(target_data, pd.DataFrame):
                x_val, x_test, y_val, y_test = train_test_split(
                    x_val_test, y_val_test, test_size=test_split / val_test_split
                )
                data["val_target"] = y_val.reset_index(drop=True)
                data["test_target"] = y_test.reset_index(drop=True)
            else:
                x_val, x_test = train_test_split(x_val_test, test_size=test_split / val_test_split)
                data["val_target"] = None
                data["test_target"] = None
            data["val_input"] = x_val.reset_index(drop=True)
            data["test_input"] = x_test.reset_index(drop=True)
    else:
        if Task_Options["task"] == "Super-Resolution" and isinstance(target_data, pd.DataFrame):
            data["val_target"] = y_val_test.reset_index(drop=True)
            data["test_target"] = None
        else:
            if isinstance(target_data, pd.DataFrame):
                data["val_target"] = y_val_test.reset_index(drop=True)
            else:
                data["val_target"] = None
            data["val_input"] = x_val_test.reset_index(drop=True)
            data["test_input"] = None
            data["test_target"] = None
    return data, input_type, target_type, directory


# ---------------------------------------------------------------------------
# Dataloader construction
# ---------------------------------------------------------------------------


def _dataset_class(task: str):
    return {
        "Calibration": SpectralDataset,
        "Denoising": SpectralDataset,
        "Segmentation": ImageSegmentationDataset,
        "Super-Resolution": ImageSuperResDataset,
        "Classification": ClassificationDataset,
    }[task]


def prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options,
                       input_data, target_data, input_type, target_type, directory,
                       transform_list, apply):
    cls = _dataset_class(Task_Options["task"])
    user_dataset = cls(input_data, target_data, input_type, target_type, directory, transform_list, apply)

    shuffle = DataManager_Options.get("shuffle") == "True"
    return DataLoader(
        user_dataset,
        batch_size=int(Training_Hyperparameters["batch_size"]),
        shuffle=shuffle,
        num_workers=0,
    )
