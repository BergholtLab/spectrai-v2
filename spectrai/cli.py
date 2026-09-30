"""Command-line interface.

Replaces the reference ``spectrai/{train,evaluate,apply,preview}.py`` scripts
with a single ``spectrai`` entry point and subcommands:

    spectrai train     --config spectral_denoising.yml [--epochs 2 ...]
    spectrai evaluate  --config image_segmentation.yml
    spectrai apply     --config spectral_calibration.yml
    spectrai preview   --config spectral_denoising.yml

Every flag from the reference CLI is preserved.
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import apply_cli_overrides, load_config
from .trainer import (
    apply_pretrained,
    evaluate_pretrained,
    train_epoch,
)
from .previewer import preview_preprocessing

# Flags accepted by the reference CLI (train.py / evaluate.py / apply.py / preview.py).
_OVERRIDABLE_FLAGS = [
    # Task
    "task", "classes",
    # Training options
    "training_option", "pretrained_network", "pretrained_classes",
    # Network hyperparameters
    "network", "dimension", "activation", "normalization",
    # Training hyperparameters
    "epochs", "batch_size", "learning_rate",
    "input_image_size", "target_image_size", "spectrum_length",
    "optimizer", "scheduler", "criterion",
    # Preprocessing
    "spectral_crop_start", "spectral_crop_end",
    "background_subtraction", "data_normalization",
    # Data augmentation
    "horizontal_flip", "vertical_flip", "rotation", "random_crop",
    "spectral_shift", "spectral_flip", "spectral_background", "mixup",
    # Data manager
    "data_format", "data_directory",
    "train_input_data", "val_input_data", "test_input_data",
    "train_target_data", "val_target_data", "test_target_data",
    "shuffle", "seed", "train_split", "val_split", "test_split",
]

_FLAG_TYPES = {
    "task": str, "classes": int,
    "training_option": str, "pretrained_network": str, "pretrained_classes": int,
    "network": str, "dimension": str, "activation": str, "normalization": str,
    "epochs": int, "batch_size": int, "learning_rate": float,
    "input_image_size": int, "target_image_size": int, "spectrum_length": int,
    "optimizer": str, "scheduler": str, "criterion": str,
    "spectral_crop_start": int, "spectral_crop_end": int,
    "background_subtraction": str, "data_normalization": str,
    "horizontal_flip": int, "vertical_flip": int, "rotation": int, "random_crop": int,
    "spectral_shift": float, "spectral_flip": int, "spectral_background": int, "mixup": int,
    "data_format": str, "data_directory": str,
    "train_input_data": str, "val_input_data": str, "test_input_data": str,
    "train_target_data": str, "val_target_data": str, "test_target_data": str,
    "shuffle": str, "seed": str,
    "train_split": float, "val_split": float, "test_split": float,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectrai",
        description="spectrai: a deep learning framework for spectral data",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd, desc in [
        ("train", "Train a model from scratch (or via transfer learning)"),
        ("evaluate", "Evaluate a pre-trained model on test data"),
        ("apply", "Apply a pre-trained model to new data (no targets)"),
        ("preview", "Preview preprocessed sample data"),
    ]:
        p = sub.add_parser(cmd, help=desc)
        p.add_argument("--config", default="image_superresolution.yml", type=str,
                       help="Config file name or path (default: image_superresolution.yml)")
        for flag in _OVERRIDABLE_FLAGS:
            p.add_argument(f"--{flag}", type=_FLAG_TYPES[flag], default=None)
        p.add_argument("--verbose", action="store_true", help="Print per-epoch metrics")
        p.add_argument("--save-frequency", type=int, default=0,
                       help="Save model every N epochs (0 = only at end)")
    return parser


def _resolve_data_paths(config) -> None:
    """Resolve relative data paths against the repo-local ``data/`` directory.

    Mirrors the reference behaviour: paths that aren't absolute are joined to
    ``<repo>/data`` so that ``spectrai train --config spectral_denoising.yml``
    works out of the box with the bundled test data.
    """
    dmo = config.DataManager_Options
    repo_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    for key in ("train_input_data", "val_input_data", "test_input_data",
                "train_target_data", "val_target_data", "test_target_data"):
        value = getattr(dmo, key)
        if value and not os.path.isabs(value):
            setattr(dmo, key, os.path.join(repo_data, value))


def _legacy_dicts(config):
    return config.as_legacy_dicts()


def _run_train(args, config):
    d = _legacy_dicts(config)
    _resolve_data_paths(config)
    d = _legacy_dicts(config)  # re-dump after path resolution
    epochs = d["Training_Hyperparameters"]["epochs"]
    out = train_epoch(
        d["Training_Options"], d["Task_Options"], d["Network_Hyperparameters"],
        d["Training_Hyperparameters"], d["Preprocessing"], d["Data_Augmentation"],
        d["DataManager_Options"], d["State_Dicts"].get("net_state_dict"),
        d["State_Dicts"].get("optimizer_state_dict"), d["State_Dicts"].get("scheduler_state_dict"),
        epochs, results_preview=0, current_epoch=1, max_epochs=epochs,
        save_frequency=args.save_frequency, verbose=args.verbose,
    )
    _print_metrics("train", out)
    return out


def _run_evaluate(args, config):
    d = _legacy_dicts(config)
    _resolve_data_paths(config)
    d = _legacy_dicts(config)
    out = evaluate_pretrained(
        d["Training_Options"], d["Task_Options"], d["Network_Hyperparameters"],
        d["Training_Hyperparameters"], d["Preprocessing"], d["Data_Augmentation"],
        d["DataManager_Options"], d["State_Dicts"].get("net_state_dict"), results_preview=0,
    )
    _print_metrics("evaluate", out)
    return out


def _run_apply(args, config):
    d = _legacy_dicts(config)
    _resolve_data_paths(config)
    d = _legacy_dicts(config)
    out = apply_pretrained(
        d["Training_Options"], d["Task_Options"], d["Network_Hyperparameters"],
        d["Training_Hyperparameters"], d["Preprocessing"], d["Data_Augmentation"],
        d["DataManager_Options"], d["State_Dicts"].get("net_state_dict"),
        results_preview=0, apply=1,
    )
    print("apply:", out)
    return out


def _run_preview(args, config):
    d = _legacy_dicts(config)
    _resolve_data_paths(config)
    d = _legacy_dicts(config)
    out = preview_preprocessing(
        d["Task_Options"], d["Network_Hyperparameters"], d["Training_Hyperparameters"],
        d["Preprocessing"], d["Data_Augmentation"], d["DataManager_Options"],
    )
    print("preview input shape:", out["input"].shape,
          "| target shape:", out["target"].shape if hasattr(out["target"], "shape") else type(out["target"]))
    return out


def _print_metrics(label, out):
    import numpy as np

    def r(m):
        if not m:
            return None
        return {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                for k, v in m.items()}

    for key in ("train_metrics", "val_metrics", "test_metrics"):
        if key in out and out[key] is not None:
            print(f"{label} {key}: {r(out[key])}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load + validate the base config.
    config = load_config(args.config)

    # Apply CLI overrides (only non-None values), then re-validate.
    overrides = {flag: getattr(args, flag, None) for flag in _OVERRIDABLE_FLAGS}
    config = apply_cli_overrides(config, overrides)

    if args.command == "train":
        _run_train(args, config)
    elif args.command == "evaluate":
        _run_evaluate(args, config)
    elif args.command == "apply":
        _run_apply(args, config)
    elif args.command == "preview":
        _run_preview(args, config)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
