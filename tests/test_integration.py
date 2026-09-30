"""Phase 3 integration tests: full pipeline (config → data → net → train → validate).

These run end-to-end on small synthetic datasets to confirm the rewritten
spectrai pipeline works for representative (task, network, dimension) combos.
They exercise the real DataLoader, transforms, trainer, and registry together —
not the nets in isolation.

Data layouts mirror the reference pipeline's actual contracts:
- Spectral (Denoising/Calibration): flat ``.npy`` of spectra, one per row.
- Classification: a *directory* with one subfolder per class (the pipeline
  derives labels from folder names via ``prepare_classification_dataset``).
- Super-Resolution: a *directory* of high-res ``.npy`` images; the input is
  derived by downsampling the target.
- Segmentation: a *directory* of images + a *directory* of masks (``.png``,
  1-indexed classes, loaded via PIL).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from spectrai.config.loader import load_config
from spectrai.trainer import train_epoch


def _cfg_dict(config_name: str = "spectral_denoising.yml") -> dict:
    return load_config(config_name).model_dump()


def _common_overrides(cfg: dict) -> dict:
    """Deterministic, fast, augmentation-free settings."""
    cfg["Data_Augmentation"]["spectral_shift"] = 0
    cfg["Data_Augmentation"]["spectral_flip"] = 0
    cfg["Data_Augmentation"]["spectral_background"] = 0
    cfg["Data_Augmentation"]["mixup"] = 0
    cfg["Data_Augmentation"]["horizontal_flip"] = 0
    cfg["Data_Augmentation"]["vertical_flip"] = 0
    cfg["Data_Augmentation"]["rotation"] = 0
    cfg["Data_Augmentation"]["random_crop"] = 0
    cfg["DataManager_Options"]["seed"] = "42"
    cfg["Training_Hyperparameters"]["epochs"] = 1
    cfg["Training_Hyperparameters"]["batch_size"] = 8
    cfg["Training_Hyperparameters"]["learning_rate"] = 0.001
    return cfg


def _clear_splits(cfg: dict) -> dict:
    for k in ("val_input_data", "val_target_data", "test_input_data", "test_target_data"):
        cfg["DataManager_Options"][k] = "None"
    for k in ("train_split", "val_split", "test_split"):
        cfg["DataManager_Options"][k] = "None"
    return cfg


def _run_train(cfg: dict) -> dict:
    return train_epoch(
        cfg["Training_Options"],
        cfg["Task_Options"],
        cfg["Network_Hyperparameters"],
        cfg["Training_Hyperparameters"],
        cfg["Preprocessing"],
        cfg["Data_Augmentation"],
        cfg["DataManager_Options"],
        None, None, None,
        epochs=1,
        max_epochs=1,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# 1D spectral
# ---------------------------------------------------------------------------


def test_pipeline_spectral_denoising_unet():
    """1D spectral denoising with UNet (flat .npy input/target)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(0)
        clean = rng.random((48, 50), dtype=np.float32)
        noisy = clean + 0.1 * rng.random((48, 50), dtype=np.float32)
        np.save(os.path.join(tmpdir, "input.npy"), noisy)
        np.save(os.path.join(tmpdir, "target.npy"), clean)

        cfg = _common_overrides(_cfg_dict("spectral_denoising.yml"))
        cfg["Training_Hyperparameters"]["spectrum_length"] = 50
        cfg["Preprocessing"]["spectral_crop_start"] = 0
        cfg["Preprocessing"]["spectral_crop_end"] = 50
        cfg["DataManager_Options"]["train_input_data"] = os.path.join(tmpdir, "input.npy")
        cfg["DataManager_Options"]["train_target_data"] = os.path.join(tmpdir, "target.npy")
        cfg["DataManager_Options"]["data_directory"] = "False"
        cfg = _clear_splits(cfg)

        out = _run_train(cfg)
        assert "network" in out
        assert np.isfinite(out["train_metrics"]["loss"])


def test_pipeline_spectral_denoising_resunet():
    """1D spectral denoising with ResUNet (residual variant)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(5)
        clean = rng.random((32, 60), dtype=np.float32)
        noisy = clean + 0.1 * rng.random((32, 60), dtype=np.float32)
        np.save(os.path.join(tmpdir, "input.npy"), noisy)
        np.save(os.path.join(tmpdir, "target.npy"), clean)

        cfg = _common_overrides(_cfg_dict("spectral_denoising.yml"))
        cfg["Training_Hyperparameters"]["spectrum_length"] = 60
        cfg["Preprocessing"]["spectral_crop_start"] = 0
        cfg["Preprocessing"]["spectral_crop_end"] = 60
        cfg["Network_Hyperparameters"]["network"] = "ResUNet"
        cfg["DataManager_Options"]["train_input_data"] = os.path.join(tmpdir, "input.npy")
        cfg["DataManager_Options"]["train_target_data"] = os.path.join(tmpdir, "target.npy")
        cfg["DataManager_Options"]["data_directory"] = "False"
        cfg = _clear_splits(cfg)

        out = _run_train(cfg)
        assert "network" in out
        assert np.isfinite(out["train_metrics"]["loss"])


# ---------------------------------------------------------------------------
# 1D spectral classification (directory of per-class folders)
# ---------------------------------------------------------------------------


def _make_classification_dir(tmpdir: str, n_per_class: int, spectrum_len: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    for cls in range(3):
        cls_dir = os.path.join(tmpdir, str(cls))
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            spec = rng.random((spectrum_len,), dtype=np.float32)
            # Give each class a distinct mean so the net can learn *something*.
            spec = spec + cls * 0.3
            np.save(os.path.join(cls_dir, f"spec_{i}.npy"), spec)
    return tmpdir


def test_pipeline_spectral_classification_resnet():
    """1D spectral classification with ResNet (3 classes, directory layout)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = _make_classification_dir(tmpdir, n_per_class=10, spectrum_len=40, seed=1)

        cfg = _common_overrides(_cfg_dict("spectral_classification.yml"))
        cfg["Task_Options"]["classes"] = 3
        cfg["Training_Hyperparameters"]["spectrum_length"] = 40
        cfg["Preprocessing"]["spectral_crop_start"] = 0
        cfg["Preprocessing"]["spectral_crop_end"] = 40
        cfg["Network_Hyperparameters"]["network"] = "ResNet"
        cfg["Network_Hyperparameters"]["normalization"] = "BatchNorm"
        cfg["DataManager_Options"]["data_format"] = "Spectra"
        cfg["DataManager_Options"]["data_directory"] = "True"
        cfg["DataManager_Options"]["train_input_data"] = data_dir
        cfg["DataManager_Options"]["train_target_data"] = "None"
        cfg = _clear_splits(cfg)

        out = _run_train(cfg)
        assert "network" in out
        assert "accuracy" in out["train_metrics"]
        assert np.isfinite(out["train_metrics"]["loss"])


# ---------------------------------------------------------------------------
# 2D image classification (directory of per-class folders)
# ---------------------------------------------------------------------------


def _make_image_classification_dir(tmpdir: str, n_per_class: int, channels: int, side: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    for cls in range(2):
        cls_dir = os.path.join(tmpdir, str(cls))
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            img = rng.random((side, side, channels), dtype=np.float32) + cls * 0.3
            np.save(os.path.join(cls_dir, f"img_{i}.npy"), img)
    return tmpdir


def test_pipeline_image_classification_densenet_2d():
    """2D hyperspectral image classification with DenseNet (2 classes).

    densenet121 has 4 dense blocks + 3 transition pools (halving spatial size
    three times), so it needs a >=32x32 spatial size to survive — identical to
    the reference architecture.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = _make_image_classification_dir(tmpdir, n_per_class=6, channels=8, side=32, seed=2)

        cfg = _common_overrides(_cfg_dict("image_classification.yml"))
        cfg["Task_Options"]["classes"] = 2
        cfg["Training_Hyperparameters"]["spectrum_length"] = 8
        cfg["Training_Hyperparameters"]["input_image_size"] = 32
        cfg["Training_Hyperparameters"]["target_image_size"] = 32
        cfg["Preprocessing"]["spectral_crop_start"] = 0
        cfg["Preprocessing"]["spectral_crop_end"] = 8
        cfg["Network_Hyperparameters"]["network"] = "DenseNet"
        cfg["Network_Hyperparameters"]["dimension"] = "2D"
        cfg["Network_Hyperparameters"]["normalization"] = "BatchNorm"
        cfg["DataManager_Options"]["data_format"] = "Image: H, W, C"
        cfg["DataManager_Options"]["data_directory"] = "True"
        cfg["DataManager_Options"]["train_input_data"] = data_dir
        cfg["DataManager_Options"]["train_target_data"] = "None"
        cfg = _clear_splits(cfg)

        out = _run_train(cfg)
        assert "network" in out
        assert "accuracy" in out["train_metrics"]
        assert np.isfinite(out["train_metrics"]["loss"])


# ---------------------------------------------------------------------------
# 2D image segmentation (image dir + mask dir)
# ---------------------------------------------------------------------------


def _make_segmentation_dirs(tmpdir: str, n: int, channels: int, side: int, seed: int):
    rng = np.random.default_rng(seed)
    in_dir = os.path.join(tmpdir, "images")
    mask_dir = os.path.join(tmpdir, "masks")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for i in range(n):
        img = rng.random((side, side, channels), dtype=np.float32)
        np.save(os.path.join(in_dir, f"img_{i}.npy"), img)
        # Mask: 2-class per-pixel, 1-indexed for PIL (load_mask subtracts 1).
        mask = rng.integers(1, 3, size=(side, side), dtype=np.uint8)
        Image.fromarray(mask, mode="L").save(os.path.join(mask_dir, f"mask_{i}.png"))
    return in_dir, mask_dir


def test_pipeline_image_segmentation_segnet_2d():
    """2D hyperspectral image segmentation with SegNet (2 classes).

    SegNet has 5 encoder pooling stages (halving spatial size five times), so
    it needs a >=32x32 spatial size to survive — identical to the reference,
    whose segmentation config uses 64x64 images.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir, mask_dir = _make_segmentation_dirs(tmpdir, n=6, channels=8, side=32, seed=3)

        cfg = _common_overrides(_cfg_dict("image_segmentation.yml"))
        cfg["Task_Options"]["classes"] = 2
        cfg["Training_Hyperparameters"]["spectrum_length"] = 8
        cfg["Training_Hyperparameters"]["input_image_size"] = 32
        cfg["Training_Hyperparameters"]["target_image_size"] = 32
        cfg["Preprocessing"]["spectral_crop_start"] = 0
        cfg["Preprocessing"]["spectral_crop_end"] = 8
        cfg["Network_Hyperparameters"]["network"] = "SegNet"
        cfg["Network_Hyperparameters"]["dimension"] = "2D"
        cfg["DataManager_Options"]["data_format"] = "Image: H, W, C"
        cfg["DataManager_Options"]["data_directory"] = "True"
        cfg["DataManager_Options"]["train_input_data"] = in_dir
        cfg["DataManager_Options"]["train_target_data"] = mask_dir
        cfg = _clear_splits(cfg)

        out = _run_train(cfg)
        assert "network" in out
        assert np.isfinite(out["train_metrics"]["loss"])


# ---------------------------------------------------------------------------
# 2D super-resolution (high-res image directory)
# ---------------------------------------------------------------------------


def test_pipeline_image_superres_rcan_2d():
    """2D super-resolution with RCAN (high-res 16x16 → low-res 8x8 input).

    RCAN builds its body from ``int(spectrum_length/2)`` channels with a
    reduction-16 channel-attention block, so ``spectrum_length`` must be at
    least 32 (body >= 16 channels) — identical to the reference, whose
    super-resolution config uses ``spectrum_length: 500``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(4)
        in_dir = os.path.join(tmpdir, "highres")
        os.makedirs(in_dir, exist_ok=True)
        for i in range(6):
            hr = rng.random((16, 16, 32), dtype=np.float32)
            np.save(os.path.join(in_dir, f"hr_{i}.npy"), hr)

        cfg = _common_overrides(_cfg_dict("image_superresolution.yml"))
        cfg["Task_Options"]["classes"] = 0
        cfg["Training_Hyperparameters"]["spectrum_length"] = 32
        cfg["Training_Hyperparameters"]["input_image_size"] = 8
        cfg["Training_Hyperparameters"]["target_image_size"] = 16
        cfg["Preprocessing"]["spectral_crop_start"] = 0
        cfg["Preprocessing"]["spectral_crop_end"] = 32
        cfg["Network_Hyperparameters"]["network"] = "RCAN"
        cfg["Network_Hyperparameters"]["dimension"] = "2D"
        cfg["DataManager_Options"]["data_format"] = "Image: H, W, C"
        cfg["DataManager_Options"]["data_directory"] = "True"
        cfg["DataManager_Options"]["train_target_data"] = in_dir
        cfg["DataManager_Options"]["train_input_data"] = "None"
        cfg = _clear_splits(cfg)

        out = _run_train(cfg)
        assert "network" in out
        assert "psnr" in out["train_metrics"]
        assert np.isfinite(out["train_metrics"]["loss"])


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_help_runs():
    """The spectrai CLI entry point should respond to --help."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-m", "spectrai.cli", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower() or "train" in result.stdout
