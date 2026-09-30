#!/usr/bin/env python3
"""Example 1 — train a 1D spectral denoising model (UNet) for one epoch.

Uses the bundled ``spectral_denoising.yml`` config and the small bundled
spectra dataset. Run from the repo root:

    python examples/train_denoising.py

This is the smallest end-to-end pipeline: config -> data -> transforms ->
network -> one training epoch -> reported loss.
"""
import os
import sys

# Make the repo root importable regardless of where the script is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from spectrai.config import load_config
from spectrai.trainer import train_epoch


def main() -> int:
    cfg = load_config("spectral_denoising.yml").as_legacy_dicts()

    # Point at the bundled data (resolved relative to the repo root).
    dmo = cfg["DataManager_Options"]
    for key in ("train_input_data", "train_target_data"):
        value = dmo.get(key)
        if value and not os.path.isabs(value):
            dmo[key] = os.path.join(ROOT, "data", value)

    out = train_epoch(
        cfg["Training_Options"],
        cfg["Task_Options"],
        cfg["Network_Hyperparameters"],
        cfg["Training_Hyperparameters"],
        cfg["Preprocessing"],
        cfg["Data_Augmentation"],
        cfg["DataManager_Options"],
        cfg["State_Dicts"].get("net_state_dict"),
        cfg["State_Dicts"].get("optimizer_state_dict"),
        cfg["State_Dicts"].get("scheduler_state_dict"),
        epochs=1,
        verbose=True,
    )
    print("train metrics:", out.get("train_metrics"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
