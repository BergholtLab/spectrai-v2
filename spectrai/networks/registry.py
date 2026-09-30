"""Network registry: build and adapt networks from config sections.

Ported from the reference ``spectrai/networks/networks.py`` — all seven
architectures (UNet, ResUNet, ResNet, RCAN, EfficientNet, SegNet, DenseNet)
with the reference's exact per-``(task, dimension, data_format)`` dispatch and
its ``NotImplementedError``/``ValueError`` gaps preserved 1:1.
"""

from __future__ import annotations

from torch import nn

from .layers import BasicConv
from .nets.densenet import densenet121
from .nets.efficientnet import EfficientNet
from .nets.rcan import Hyperspectral_RCAN
from .nets.resnet import ResNet18
from .nets.segnet import SegNet
from .nets.unet import UNet

__all__ = ["setup_network", "edit_network"]


def _make_activation(name: str) -> nn.Module:
    if name == "PReLU":
        return nn.PReLU()
    if name == "LeakyReLU":
        return nn.LeakyReLU()
    return nn.ReLU()


def setup_network(Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options):
    """Initialise a neural network given user input parameters."""
    task = Task_Options["task"]
    channels = int(Training_Hyperparameters["spectrum_length"])
    image_size = int(Training_Hyperparameters["target_image_size"])
    normalization = Network_Hyperparameters["normalization"]

    if task in ("Classification", "Segmentation"):
        num_classes = int(Task_Options["classes"])
    else:
        num_classes = 0

    if task == "Super-Resolution":
        scale = int(Training_Hyperparameters["target_image_size"]) // int(Training_Hyperparameters["input_image_size"])
    else:
        scale = None

    activation = _make_activation(Network_Hyperparameters["activation"])
    dims = 2 if Network_Hyperparameters["dimension"] == "2D" else 3

    # ------------ UNet / ResUNet ------------
    if Network_Hyperparameters["network"] in ("UNet", "ResUNet"):
        res = Network_Hyperparameters["network"] == "ResUNet"
        if DataManager_Options["data_format"] == "Spectra":
            if task not in ("Calibration", "Classification", "Denoising"):
                raise NotImplementedError(
                    "1D UNet/ResUNet architecture (for spectra) only implemented for "
                    "calibration, classification, and denoising"
                )
            net = UNet(dims=1, channels=1, num_classes=num_classes, n_blocks=3,
                       normalization=normalization, activation=activation, task=task, res=res).float()
        else:  # Hyperspectral image data
            if dims == 2:
                net = UNet(dims=dims, channels=channels, num_classes=num_classes, n_blocks=3,
                           normalization=normalization, activation=activation, task=task,
                           scale=scale, res=res).float()
            elif dims == 3:
                if task == "Super-Resolution":
                    raise NotImplementedError("UNet/ResUNet architecture not implemented for 3D super-resolution")
                net = UNet(dims=dims, channels=1, img_channels=channels, num_classes=num_classes, n_blocks=3,
                           normalization=normalization, activation=activation, task=task,
                           scale=scale, res=res).float()

    # -------------- ResNet --------------
    elif Network_Hyperparameters["network"] == "ResNet":
        if task == "Classification":
            if DataManager_Options["data_format"] == "Spectra":
                net = ResNet18(dims=1, img_channels=channels, res_channels=channels,
                               num_classes=num_classes, normalization=normalization,
                               activation=activation).float()
            else:  # Hyperspectral image data
                if dims == 2:
                    net = ResNet18(dims=dims, img_channels=channels, res_channels=channels,
                                   num_classes=num_classes, normalization=normalization,
                                   activation=activation).float()
                elif dims == 3:
                    net = ResNet18(dims=dims, img_channels=1, res_channels=64,
                                   num_classes=num_classes, normalization=normalization,
                                   activation=activation).float()
        else:
            raise NotImplementedError("ResNet architecture has only been implemented for classification")

    # --------------- RCAN ---------------
    elif Network_Hyperparameters["network"] == "RCAN":
        if DataManager_Options["data_format"] == "Spectra":
            raise NotImplementedError("RCAN architecture not implemented for 1D (spectral) data")
        else:  # Hyperspectral image data
            if task == "Super-Resolution" and dims == 2:
                net = Hyperspectral_RCAN(spectrum_length=channels, scale=scale, activation=activation).float()
            else:
                raise NotImplementedError("RCAN architecture only implemented for 2D super-resolution")

    # ----------- EfficientNet -----------
    elif Network_Hyperparameters["network"] == "EfficientNet":
        if DataManager_Options["data_format"] == "Spectra":
            raise NotImplementedError("EfficientNet architecture not implemented for 1D (spectral) data")
        else:  # Hyperspectral image data
            if task == "Classification" and dims == 2:
                override_params = {"image_size": image_size, "num_classes": num_classes}
                net = EfficientNet.from_name("efficientnet-b0", channels, **override_params).float()
            else:
                raise NotImplementedError("EfficientNet architecture only implemented for 2D classification")

    # ------------- SegNet -------------
    elif Network_Hyperparameters["network"] == "SegNet":
        if DataManager_Options["data_format"] == "Spectra":
            raise NotImplementedError("SegNet architecture not implemented for 1D (spectral) data")
        else:  # Hyperspectral image data
            if task == "Segmentation":
                if dims == 2:
                    net = SegNet(dims=dims, num_classes=num_classes, channels=channels).float()
                elif dims == 3:
                    net = SegNet(dims=dims, num_classes=num_classes, channels=1,
                                 img_channels=channels).float()
            else:
                raise NotImplementedError("SegNet architecture only implemented for segmentation")

    # ------------- DenseNet -------------
    elif Network_Hyperparameters["network"] == "DenseNet":
        if task == "Classification":
            if DataManager_Options["data_format"] == "Spectra":
                net = densenet121(dims=1, channels=1, num_classes=num_classes,
                                  normalization=normalization, activation=activation).float()
            else:  # Hyperspectral image data
                if dims == 2:
                    net = densenet121(dims=dims, channels=channels, num_classes=num_classes,
                                      normalization=normalization, activation=activation).float()
                elif dims == 3:
                    net = densenet121(dims=dims, channels=1, num_classes=num_classes,
                                      normalization=normalization, activation=activation).float()
        else:
            raise NotImplementedError("ResNet architecture has only been implemented for classification")

    else:
        raise ValueError(f"{Network_Hyperparameters['network']} is not a valid network")

    return net


def edit_network(net, Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options):
    """Edit a network to match the desired number of output classes."""
    classes = int(Task_Options["classes"])
    channels = int(Training_Hyperparameters["spectrum_length"])

    if Network_Hyperparameters["network"] in ("UNet", "ResUNet"):
        if DataManager_Options["data_format"] == "Spectra":
            net.tail = nn.Linear(1, classes)
        else:
            if Task_Options["task"] == "Classification":
                net.tail = nn.Linear(channels, classes)
            elif Task_Options["task"] == "Segmentation":
                if Network_Hyperparameters["dimension"] == "2D":
                    net.tail = BasicConv(2, net.filter_config[0], classes, kernel_size=1,
                                         normalization=None, activation=None)
                else:
                    net.tail = BasicConv(2, net.img_channels, classes, kernel_size=1,
                                         normalization=None, activation=None)
    elif Network_Hyperparameters["network"] == "ResNet":
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, classes)
    elif Network_Hyperparameters["network"] == "EfficientNet":
        in_features = net._fc.in_features
        net._fc = nn.Linear(in_features, classes)
    elif Network_Hyperparameters["network"] == "SegNet":
        if Network_Hyperparameters["dimension"] == "2D":
            net.classifier = nn.Conv2d(channels, classes, kernel_size=3, stride=1, padding=1)
        else:
            net.classifier[1] = nn.Conv2d(net.img_channels, classes, kernel_size=3, stride=1, padding=1)
    return net
