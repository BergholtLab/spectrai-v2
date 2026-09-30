"""Transform construction helpers + spectral transform classes.

The ported :mod:`spectrai.transforms.spectral` module holds the transform
classes. This module holds :func:`prepare_transforms` (and the spectral /
image pipeline builders) — the functions the reference's
``dataloaders.py`` used to build the composed pipeline.
"""

from __future__ import annotations

from torch import nn
from torchvision import transforms

from . import spectral as T

__all__ = ["prepare_transforms", "prepare_spectral_transforms", "prepare_image_transforms"]


def prepare_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                       Preprocessing, Data_Augmentation, DataManager_Options):
    """Compose the transform pipeline for a task (training or val/test)."""
    if DataManager_Options["data_format"] == "Spectra":
        lst = prepare_spectral_transforms(
            Network_Hyperparameters, Training_Hyperparameters, Preprocessing, Data_Augmentation
        )
    else:
        lst = prepare_image_transforms(
            Task_Options, Network_Hyperparameters, Training_Hyperparameters,
            Preprocessing, Data_Augmentation, DataManager_Options,
        )
    return transforms.Compose(lst)


def prepare_spectral_transforms(Network_Hyperparameters, Training_Hyperparameters,
                                Preprocessing, Data_Augmentation):
    spectrum_length = int(Training_Hyperparameters["spectrum_length"])
    crop_start = int(Preprocessing["spectral_crop_start"])
    crop_end = int(Preprocessing["spectral_crop_end"])
    background = Preprocessing["background_subtraction"]
    normalization = Preprocessing["data_normalization"]
    shift = float(Data_Augmentation["spectral_shift"])
    flip = Data_Augmentation["spectral_flip"]

    lst: list[nn.Module | T.Transform] = [
        T.ToTensor(),
        T.Squeeze(),
        T.CropSpectrum(crop_start, crop_end),
        T.PadSpectrum(spectrum_length),
        T.ShiftSpectrum(shift),
    ]
    if flip:
        lst.append(T.FlipAxis(-1))

    if background != "None":
        if background == "3rd Order Polynomial":
            lst.append(T.PolyBackgroundSpectrum(3))
        elif background == "5th Order Polynomial":
            lst.append(T.PolyBackgroundSpectrum(5))
        elif background == "Minimum Value Offset":
            lst.append(T.MinBackgroundSpectrum())

    if normalization != "None":
        if normalization == "Max Value":
            lst.append(T.MaxNormalizeSpectrum())
        elif normalization == "Area Under The Curve":
            lst.append(T.AUCNormalizeSpectrum())

    lst.append(T.AddDimLast())
    if Network_Hyperparameters["network"] != "ResNet":
        lst.append(T.MakeChannelsLast())
    return lst


def prepare_image_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                             Preprocessing, Data_Augmentation, DataManager_Options):
    target_size = int(Training_Hyperparameters["target_image_size"])
    input_size = int(Training_Hyperparameters["input_image_size"])
    crop_start = int(Preprocessing["spectral_crop_start"])
    crop_end = int(Preprocessing["spectral_crop_end"])
    background = Preprocessing["background_subtraction"]
    normalization = Preprocessing["data_normalization"]
    shift = float(Data_Augmentation["spectral_shift"])
    spectral_flip = Data_Augmentation["spectral_flip"]
    hflip = Data_Augmentation["horizontal_flip"]
    vflip = Data_Augmentation["vertical_flip"]
    rotation = Data_Augmentation["rotation"]
    random_crop = Data_Augmentation["random_crop"]

    lst: list[nn.Module | T.Transform] = [T.ToTensor()]
    if DataManager_Options["data_format"] == "Image: C, H, W":
        lst.append(T.MakeChannelsLast())

    lst += [
        T.PadCropImage(target_size, bool(random_crop)),
        T.CropImageSpectrum(crop_start, crop_end),
        T.ShiftImageSpectrum(shift),
    ]
    if vflip:
        lst.append(T.FlipAxis(0))
    if hflip:
        lst.append(T.FlipAxis(1))
    if spectral_flip:
        lst.append(T.FlipAxis(2))
    if rotation:
        lst.append(T.RotateImage())

    if background != "None":
        if background == "3rd Order Polynomial":
            lst.append(T.PolyBackgroundImage(3))
        elif background == "5th Order Polynomial":
            lst.append(T.PolyBackgroundImage(5))
        elif background == "Minimum Value Offset":
            lst.append(T.MinBackgroundImage())

    if normalization != "None":
        if normalization == "Max Value":
            lst.append(T.MaxNormalizeImage())
        elif normalization == "Area Under The Curve":
            lst.append(T.AUCNormalizeImage())

    if Task_Options["task"] == "Super-Resolution":
        lst.append(T.BicubicDownsampleImage(input_size))

    lst.append(T.MakeChannelsFirst())
    if Network_Hyperparameters["dimension"] == "3D":
        lst.append(T.AddDimFirst())
    return lst
