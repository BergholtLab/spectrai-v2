"""Config layer: validated schema + legacy-YAML loader + CLI overrides."""

from .schema import (  # noqa: F401
    Config,
    TaskOptions,
    TrainingOptions,
    NetworkHyperparameters,
    TrainingHyperparameters,
    Preprocessing,
    DataAugmentation,
    DataManagerOptions,
    StateDicts,
)
from .loader import load_config, default_config_path, apply_cli_overrides  # noqa: F401

__all__ = [
    "Config",
    "TaskOptions",
    "TrainingOptions",
    "NetworkHyperparameters",
    "TrainingHyperparameters",
    "Preprocessing",
    "DataAugmentation",
    "DataManagerOptions",
    "StateDicts",
    "load_config",
    "default_config_path",
    "apply_cli_overrides",
]
