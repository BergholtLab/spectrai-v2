#!/usr/bin/env python3
"""Example 2 — train a 2D hyperspectral image classifier (DenseNet) for one epoch.

Uses the bundled ``image_classification.yml`` config and a small synthetic
image dataset generated on the fly (so it runs anywhere, no data download).
Run from the repo root:

    python examples/train_image_classification.py
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from spectrai.config import load_config, apply_cli_overrides
from spectrai.trainer import train_epoch


def make_classified_images(root: str, side: int = 32, channels: int = 16, n_per_class: int = 6) -> str:
    """Create a tiny 2-class image dataset: one subfolder per class."""
    rng = np.random.default_rng(0)
    for cls in range(2):
        cls_dir = os.path.join(root, str(cls))
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            img = rng.random((side, side, channels), dtype=np.float32) + cls * 0.3
            np.save(os.path.join(cls_dir, f"img_{i}.npy"), img)
    return root


def main() -> int:
    cfg = apply_cli_overrides(load_config("image_classification.yml"), {
        "classes": 2,
        "network": "DenseNet",
        "dimension": "2D",
        "spectrum_length": 16,
        "input_image_size": 32,
        "target_image_size": 32,
        "spectral_crop_start": 0,
        "spectral_crop_end": 16,
        "epochs": 1,
        "batch_size": 8,
    })
    d = cfg.as_legacy_dicts()
    dmo = d["DataManager_Options"]
    dmo.update({
        "data_format": "Image: H, W, C",
        "data_directory": "True",
        "val_input_data": "None", "val_target_data": "None",
        "test_input_data": "None", "test_target_data": "None",
        "train_split": "None", "val_split": "None", "test_split": "None",
        "shuffle": "False", "seed": "42",
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = make_classified_images(tmpdir)
        dmo["train_input_data"] = data_dir
        dmo["train_target_data"] = "None"

        out = train_epoch(
            d["Training_Options"], d["Task_Options"], d["Network_Hyperparameters"],
            d["Training_Hyperparameters"], d["Preprocessing"], d["Data_Augmentation"],
            dmo, d["State_Dicts"].get("net_state_dict"),
            d["State_Dicts"].get("optimizer_state_dict"), d["State_Dicts"].get("scheduler_state_dict"),
            epochs=1, verbose=True,
        )
        print("train metrics:", out.get("train_metrics"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
