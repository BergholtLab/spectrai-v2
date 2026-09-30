"""Preview helpers: return a preprocessed sample (and model output) for the GUI.

Ported from the reference ``spectrai/previewer.py``.
"""

from __future__ import annotations

import numpy as np

from .trainer import initialise_dataset

__all__ = ["preview_preprocessing", "preview_progress"]


def preview_preprocessing(Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                          Preprocessing, Data_Augmentation, DataManager_Options):
    """Return preprocessed and augmented input and target data (first batch)."""
    preview_loader, _, _ = initialise_dataset(Task_Options, Network_Hyperparameters,
                                              Training_Hyperparameters, Preprocessing,
                                              Data_Augmentation, DataManager_Options, False)
    batch = next(iter(preview_loader))
    x = batch["input"]
    y = batch["target"]["data"]
    x = np.squeeze(x[0].numpy())
    if Task_Options["task"] != "Classification":
        y = np.squeeze(y[0].numpy())
    else:
        y = np.squeeze(y[0].numpy())

    x = np.ascontiguousarray(np.moveaxis(x, 0, -1))
    if Task_Options["task"] == "Segmentation":
        y = np.ascontiguousarray(y)
    elif Task_Options["task"] != "Classification":
        y = np.ascontiguousarray(np.moveaxis(y, 0, -1))

    return {"input": x, "target": y}


def preview_progress(Task_Options, net, device, preview_loader):
    """Return input, target, and model output for the first batch of the loader."""
    batch = next(iter(preview_loader))
    inputs = batch["input"].float().to(device)
    target = batch["target"]["data"]
    if Task_Options["task"] in ("Segmentation", "Classification"):
        target = target.long()
    else:
        target = target.float()

    output = net(inputs)

    x = inputs.cpu().detach().numpy()
    y = output.cpu().detach().numpy()
    target = target.numpy()

    x = np.ascontiguousarray(np.moveaxis(np.squeeze(x[0]), 0, -1))

    if Task_Options["task"] == "Classification":
        y = np.squeeze(y[0])
        y = np.argmax(y)
        target = np.squeeze(target[0])
    elif Task_Options["task"] == "Segmentation":
        y = np.ascontiguousarray(np.argmax(np.squeeze(y[0]), axis=0))
        target = np.ascontiguousarray(np.squeeze(target[0]))
    else:
        y = np.ascontiguousarray(np.moveaxis(np.squeeze(y[0]), 0, -1))
        target = np.ascontiguousarray(np.moveaxis(np.squeeze(target[0]), 0, -1))

    return {"input": x, "output": y, "target": target}
