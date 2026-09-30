"""Tests for the config layer: schema validation, YAML loading, CLI overrides."""

from __future__ import annotations

import os

import pytest

from spectrai.config import apply_cli_overrides, load_config

CONFIGS = sorted(os.listdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")))


@pytest.mark.parametrize("name", CONFIGS)
def test_all_legacy_configs_load(name):
    cfg = load_config(name)
    assert cfg.Task_Options.task in (
        "Calibration", "Classification", "Denoising", "Segmentation", "Super-Resolution",
    )
    d = cfg.as_legacy_dicts()
    assert "Task_Options" in d and "Network_Hyperparameters" in d
    # Normalization must preserve the 'None' sentinel (legacy code checks == 'None').
    assert d["Network_Hyperparameters"]["normalization"] in (
        "BatchNorm", "GroupNorm", "InstanceNorm", "LayerNorm", "None",
    )


def test_denoising_config_values():
    cfg = load_config("spectral_denoising.yml")
    assert cfg.Task_Options.task == "Denoising"
    assert cfg.Network_Hyperparameters.network in ("UNet", "ResUNet")
    assert cfg.DataManager_Options.data_format == "Spectra"
    assert cfg.Training_Hyperparameters.criterion in ("L1", "L2 / MSE")


def test_invalid_task_rejected():
    from pydantic import ValidationError

    from spectrai.config.schema import Config

    base = load_config("spectral_denoising.yml").model_dump()
    base["Task_Options"]["task"] = "NotATask"
    with pytest.raises(ValidationError):
        Config.model_validate(base)


def test_invalid_optimizer_rejected():
    from pydantic import ValidationError

    from spectrai.config.schema import Config

    base = load_config("spectral_denoising.yml").model_dump()
    base["Training_Hyperparameters"]["optimizer"] = "NOTANOPT"
    with pytest.raises(ValidationError):
        Config.model_validate(base)


def test_cli_override_batch_size():
    cfg = load_config("spectral_denoising.yml")
    cfg2 = apply_cli_overrides(cfg, {"batch_size": 16})
    assert cfg2.Training_Hyperparameters.batch_size == 16
    # Other fields unchanged.
    assert cfg2.Network_Hyperparameters.network == cfg.Network_Hyperparameters.network


def test_cli_override_activation():
    cfg = load_config("spectral_denoising.yml")
    cfg2 = apply_cli_overrides(cfg, {"activation": "PReLU"})
    assert cfg2.Network_Hyperparameters.activation == "PReLU"


def test_cli_override_invalid_value_rejected():
    from pydantic import ValidationError

    cfg = load_config("spectral_denoising.yml")
    with pytest.raises(ValidationError):
        apply_cli_overrides(cfg, {"optimizer": "NOTANOPT"})


def test_none_override_is_noop():
    cfg = load_config("spectral_denoising.yml")
    cfg2 = apply_cli_overrides(cfg, {"batch_size": None})
    assert cfg2.Training_Hyperparameters.batch_size == cfg.Training_Hyperparameters.batch_size
