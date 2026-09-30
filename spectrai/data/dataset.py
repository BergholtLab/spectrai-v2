"""Spectral dataset classes (PyTorch ``Dataset``).

Ported from the reference ``spectrai/dataloader/spectral_dataset.py``.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset


def load_sample(sample_path):
    """Load a ``.mat`` or ``.npy`` data sample (or return a preloaded array)."""
    if isinstance(sample_path, str):
        _, ext = os.path.splitext(sample_path)
        if ext == ".mat":
            sample_data = scipy.io.loadmat(sample_path)
            sample_values = list(sample_data.values())
            return sample_values[3]
        if ext == ".npy":
            return np.load(sample_path)
        raise ValueError(f"Input file with extension {ext} is not a valid file type")
    return sample_path


def load_mask(path: str) -> np.ndarray:
    """Load a mask image and convert to a 0-indexed float array."""
    mask_image = Image.open(path)
    mask = np.asarray(mask_image)
    mask = mask.astype(np.float64)
    mask = mask - 1
    return mask


class SpectralDataset(Dataset):
    """Base spectral dataset.

    Arguments:
        input_data: DataFrame (or list) of input samples / paths.
        target_data: DataFrame (or list) of target samples / paths.
        input_type: 'spectrum' or 'hyperspectral_image'.
        target_type: 'spectrum', 'hyperspectral_image', 'mask', or 'class'.
        directory: whether data is a directory of files or a single input file.
        apply: flag indicating target data is unavailable (inference).
        transform: composed transform pipeline (or None).
    """

    def __init__(self, input_data, target_data, input_type, target_type, directory, transform, apply):
        self.input_data = input_data
        self.target_data = target_data
        self.input_type = input_type
        self.target_type = target_type
        self.directory = directory
        self.transform = transform
        self.apply = apply
        self.on_epoch_end()

    def get_name(self, sample_path, idx):
        if isinstance(sample_path, str):
            return os.path.splitext(os.path.basename(sample_path))[0]
        return str(idx)

    def __getitem__(self, idx):
        if self.directory:
            name = self.get_name(self.input_data["Data"][idx], idx)
            image = load_sample(self.input_data["Data"][idx])
            if not self.apply:
                target = load_sample(self.target_data["Data"][idx])
        else:
            name = str(idx)
            image = self.input_data["Data"][idx]
            if not self.apply:
                target = self.target_data["Data"][idx]

        if self.apply:
            sample = {"input": image, "target": {"data": image, "type": self.input_type}, "name": name}
        else:
            sample = {"input": image, "target": {"data": target, "type": self.target_type}, "name": name}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def on_epoch_end(self):
        pass

    def __len__(self):
        return len(self.input_data)


class ClassificationDataset(SpectralDataset):
    """Spectral / image classification dataset (labels live in ``input_data``)."""

    def __getitem__(self, idx):
        name = self.get_name(self.input_data["Data"][idx], idx)
        spectrum = load_sample(self.input_data["Data"][idx])
        if not self.apply:
            target = self.input_data["Encoded_Labels"][idx]

        if self.apply:
            sample = {"input": spectrum, "target": {"data": spectrum, "type": self.input_type}, "name": name}
        else:
            sample = {"input": spectrum, "target": {"data": target, "type": self.target_type}, "name": name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ImageSuperResDataset(SpectralDataset):
    """Spectral image super-resolution dataset.

    For training the input is derived from the (downsampled) target; for
    application only the input is available.
    """

    def __getitem__(self, idx):
        if self.apply:
            if self.directory:
                name = self.get_name(self.input_data["Data"][idx], idx)
                image = load_sample(self.input_data["Data"][idx])
            else:
                name = str(idx)
                image = self.input_data["Data"][idx]
        else:
            if self.directory:
                name = self.get_name(self.target_data["Data"][idx], idx)
                image = load_sample(self.target_data["Data"][idx])
            else:
                name = str(idx)
                image = self.target_data["Data"][idx]
            target = image

        if self.apply:
            sample = {"input": image, "target": {"data": image, "type": self.input_type}, "name": name}
        else:
            sample = {"input": image, "target": {"data": target, "type": self.target_type}, "name": name}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        if self.apply:
            return len(self.input_data)
        return len(self.target_data)


class ImageSegmentationDataset(SpectralDataset):
    """Spectral image segmentation dataset (image + mask target)."""

    def __getitem__(self, idx):
        if self.directory:
            name = self.get_name(self.input_data["Data"][idx], idx)
            image = load_sample(self.input_data["Data"][idx])
            if not self.apply:
                target = load_mask(self.target_data["Data"][idx])
        else:
            name = str(idx)
            image = self.input_data["Data"][idx]
            if not self.apply:
                target = self.target_data["Data"][idx]

        if self.apply:
            sample = {"input": image, "target": {"data": image, "type": self.input_type}, "name": name}
        else:
            sample = {"input": image, "target": {"data": target, "type": self.target_type}, "name": name}

        if self.transform:
            sample = self.transform(sample)

        return sample


__all__ = [
    "load_sample",
    "load_mask",
    "SpectralDataset",
    "ClassificationDataset",
    "ImageSuperResDataset",
    "ImageSegmentationDataset",
]
