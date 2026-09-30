"""Utility helpers: metrics, mixup, and config validation.

Ported from the reference ``spectrai/utils/utilities.py`` (the metric/mixup
helpers and ``check_inputs``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from skimage.metrics import structural_similarity as sk_ssim

from .config.schema import (
    ACTIVATIONS,
    BACKGROUND_SUBTRACTION,
    CRITERIA,
    DATA_FORMATS,
    DATA_NORMALIZATION,
    DIMENSIONS,
    NETWORKS,
    NORMALIZATIONS,
    OPTIMIZERS,
    SCHEDULERS,
    TASKS,
    TRAINING_OPTIONS,
)

__all__ = [
    "check_inputs",
    "AverageMeter",
    "calc_psnr",
    "calc_ssim",
    "mixup_data",
    "mixup_criterion",
]


def check_inputs(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                 Preprocessing, Data_Augmentation, DataManager_Options):
    """Validate user-supplied parameter dictionaries against the allowed sets."""
    if Task_Options["task"] not in TASKS:
        raise ValueError(f"{Task_Options['task']} is not a valid task")
    if Training_Options["training_option"] not in TRAINING_OPTIONS:
        raise ValueError(f"{Training_Options['training_option']} is not a valid training option")
    if Network_Hyperparameters["network"] not in NETWORKS:
        raise ValueError(f"{Network_Hyperparameters['network']} is not a valid network")
    if Network_Hyperparameters["dimension"] not in DIMENSIONS:
        raise ValueError(f"{Network_Hyperparameters['dimension']} is not a valid dimension")
    if Network_Hyperparameters["activation"] not in ACTIVATIONS:
        raise ValueError(f"{Network_Hyperparameters['activation']} is not a valid activation")
    if Network_Hyperparameters["normalization"] not in NORMALIZATIONS:
        raise ValueError(f"{Network_Hyperparameters['normalization']} is not a valid normalization")
    if Training_Hyperparameters["optimizer"] not in OPTIMIZERS:
        raise ValueError(f"{Training_Hyperparameters['optimizer']} is not a valid optimizer")
    if Training_Hyperparameters["scheduler"] not in SCHEDULERS:
        raise ValueError(f"{Training_Hyperparameters['scheduler']} is not a valid scheduler")
    if Training_Hyperparameters["criterion"] not in CRITERIA:
        raise ValueError(f"{Training_Hyperparameters['criterion']} is not a valid criterion")
    if Preprocessing["background_subtraction"] not in BACKGROUND_SUBTRACTION:
        raise ValueError(f"{Preprocessing['background_subtraction']} is not a valid background subtraction")
    if Preprocessing["data_normalization"] not in DATA_NORMALIZATION:
        raise ValueError(f"{Preprocessing['data_normalization']} is not a valid data normalization")
    if DataManager_Options["data_format"] not in DATA_FORMATS:
        raise ValueError(f"{DataManager_Options['data_format']} is not a valid data format")


class AverageMeter:
    """Record mini-batch metric values during training."""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def calc_psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio between output and target."""
    mse = nn.MSELoss()(output, target)
    return 10 * math.log10(1 / max(mse, 1e-12))


def calc_ssim(output: torch.Tensor, target: torch.Tensor) -> float:
    """Structural similarity between output and target (per-batch average)."""
    out = output.detach().cpu().numpy()
    tgt = target.detach().cpu().numpy()

    if out.ndim == 4:
        total = 0.0
        batch = out.shape[0]
        for i in range(batch):
            o = np.squeeze(out[i]).transpose(1, 2, 0)
            t = np.squeeze(tgt[i]).transpose(1, 2, 0)
            total += sk_ssim(o, t, data_range=o.max() - t.max(), channel_axis=-1)
        return total / batch

    o = np.squeeze(out).transpose(1, 2, 0)
    t = np.squeeze(tgt).transpose(1, 2, 0)
    return sk_ssim(o, t, data_range=o.max() - t.max(), channel_axis=-1)


def mixup_data(x, y, alpha=1.0):
    """Mixup: combine a sample with a random in-batch partner."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: weighted combination of the two partners' losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
