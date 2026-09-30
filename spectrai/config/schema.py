"""Pydantic schema for a full spectrai run configuration.

Replaces the reference project's stringly-typed ``.yml`` dictionaries with a
validated, typed model. All eight legacy config files in ``configs/`` are
accepted and round-trip through :func:`spectrai.config.loader.load_config`.
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enumeration helpers
# ---------------------------------------------------------------------------

TASKS = ("Calibration", "Classification", "Denoising", "Segmentation", "Super-Resolution")
TRAINING_OPTIONS = ("Apply Pre-Trained Network", "Train From Scratch", "Transfer Learning")
NETWORKS = ("UNet", "ResUNet", "ResNet", "RCAN", "EfficientNet", "SegNet", "DenseNet")
DIMENSIONS = ("2D", "3D")
ACTIVATIONS = ("ReLU", "PReLU", "LeakyReLU")
NORMALIZATIONS = ("BatchNorm", "GroupNorm", "InstanceNorm", "LayerNorm", "None")
OPTIMIZERS = ("Adam", "Adagrad", "SGD", "RMSprop")
SCHEDULERS = ("Constant", "Step", "Multiplicative", "Cyclic", "OneCycle", "ReduceOnPlateau")
CRITERIA = ("L1", "L2 / MSE", "Cross Entropy", "Binary Cross Entropy")
BACKGROUND_SUBTRACTION = (
    "None",
    "Automatic Least Squares",
    "3rd Order Polynomial",
    "5th Order Polynomial",
    "Minimum Value Offset",
)
DATA_NORMALIZATION = ("None", "Max Value", "Area Under The Curve")
DATA_FORMATS = ("Image: H, W, C", "Image: C, H, W", "Spectra")

# 'None' as a string is used by the legacy configs; accept it everywhere.
NoneStr = Optional[str]


def _none_str(value: object) -> Optional[str]:
    """Coerce the legacy ``'None'`` sentinel to real ``None``."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    # Legacy configs also store 0/None as the string '0' / 'None' for
    # optional numeric fields; preserve them as their string form so that
    # ``as_legacy_dicts`` can round-trip to the legacy stringly-typed dict.
    return str(value)


def _optional_int(value: object) -> Optional[int]:
    """Coerce the legacy ``'None'`` / ``0`` sentinels to real ``None`` / ``0``."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return 0
    return int(value)


class _ConfigSection(BaseModel):
    """Common section settings: forbid unknown keys so typos surface early."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class TaskOptions(_ConfigSection):
    task: str = Field(description="Deep learning task.")
    classes: int = Field(default=0, ge=0)

    @field_validator("task")
    @classmethod
    def _check_task(cls, v: str) -> str:
        if v not in TASKS:
            raise ValueError(f"{v!r} is not a valid task; must be one of {TASKS}")
        return v


class TrainingOptions(_ConfigSection):
    training_option: str = "Train From Scratch"
    pretrained_network: Optional[str] = None
    pretrained_classes: Optional[int] = 0

    @field_validator("training_option")
    @classmethod
    def _check_option(cls, v: str) -> str:
        if v not in TRAINING_OPTIONS:
            raise ValueError(f"{v!r} is not a valid training option; must be one of {TRAINING_OPTIONS}")
        return v

    @field_validator("pretrained_network")
    @classmethod
    def _none(cls, v: object) -> Optional[str]:
        return _none_str(v)

    @field_validator("pretrained_classes")
    @classmethod
    def _int(cls, v: object) -> Optional[int]:
        return _optional_int(v)


class NetworkHyperparameters(_ConfigSection):
    network: str = "UNet"
    dimension: str = "2D"
    activation: str = "ReLU"
    normalization: str = "None"

    @field_validator("network")
    @classmethod
    def _check_network(cls, v: str) -> str:
        if v not in NETWORKS:
            raise ValueError(f"{v!r} is not a valid network; must be one of {NETWORKS}")
        return v

    @field_validator("dimension")
    @classmethod
    def _check_dim(cls, v: str) -> str:
        if v not in DIMENSIONS:
            raise ValueError(f"{v!r} is not a valid dimension; must be one of {DIMENSIONS}")
        return v

    @field_validator("activation")
    @classmethod
    def _check_act(cls, v: str) -> str:
        if v not in ACTIVATIONS:
            raise ValueError(f"{v!r} is not a valid activation; must be one of {ACTIVATIONS}")
        return v

    @field_validator("normalization")
    @classmethod
    def _check_norm(cls, v: str) -> str:
        if v not in NORMALIZATIONS:
            raise ValueError(f"{v!r} is not a valid normalization; must be one of {NORMALIZATIONS}")
        return v


class TrainingHyperparameters(_ConfigSection):
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = 1e-4
    input_image_size: int = 16
    target_image_size: int = 16
    spectrum_length: int = 500
    optimizer: str = "Adam"
    scheduler: str = "Constant"
    criterion: str = "L1"

    @field_validator("optimizer")
    @classmethod
    def _check_opt(cls, v: str) -> str:
        if v not in OPTIMIZERS:
            raise ValueError(f"{v!r} is not a valid optimizer; must be one of {OPTIMIZERS}")
        return v

    @field_validator("scheduler")
    @classmethod
    def _check_sched(cls, v: str) -> str:
        if v not in SCHEDULERS:
            raise ValueError(f"{v!r} is not a valid scheduler; must be one of {SCHEDULERS}")
        return v

    @field_validator("criterion")
    @classmethod
    def _check_crit(cls, v: str) -> str:
        if v not in CRITERIA:
            raise ValueError(f"{v!r} is not a valid criterion; must be one of {CRITERIA}")
        return v


class Preprocessing(_ConfigSection):
    spectral_crop_start: int = 0
    spectral_crop_end: int = 500
    background_subtraction: str = "None"
    data_normalization: str = "Max Value"

    @field_validator("background_subtraction")
    @classmethod
    def _check_bg(cls, v: str) -> str:
        if v not in BACKGROUND_SUBTRACTION:
            raise ValueError(
                f"{v!r} is not a valid background subtraction; must be one of {BACKGROUND_SUBTRACTION}"
            )
        return v

    @field_validator("data_normalization")
    @classmethod
    def _check_norm(cls, v: str) -> str:
        if v not in DATA_NORMALIZATION:
            raise ValueError(f"{v!r} is not a valid normalization; must be one of {DATA_NORMALIZATION}")
        return v


class DataAugmentation(_ConfigSection):
    horizontal_flip: int = 0
    vertical_flip: int = 0
    rotation: int = 0
    random_crop: int = 0
    spectral_shift: float = 0.0
    spectral_flip: int = 0
    spectral_background: int = 0
    mixup: int = 0


class DataManagerOptions(_ConfigSection):
    data_format: str = "Spectra"
    data_directory: str = "False"
    train_input_data: Optional[str] = None
    val_input_data: Optional[str] = None
    test_input_data: Optional[str] = None
    train_target_data: Optional[str] = None
    val_target_data: Optional[str] = None
    test_target_data: Optional[str] = None
    shuffle: str = "False"
    seed: Optional[str] = None
    # Legacy configs use the string 'None' for these; accept it pre-validation.
    train_split: Optional[Union[float, str]] = None
    val_split: Optional[Union[float, str]] = None
    test_split: Optional[Union[float, str]] = None

    @field_validator("data_format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if v not in DATA_FORMATS:
            raise ValueError(f"{v!r} is not a valid data format; must be one of {DATA_FORMATS}")
        return v

    _none_fields = (
        "train_input_data",
        "val_input_data",
        "test_input_data",
        "train_target_data",
        "val_target_data",
        "test_target_data",
        "seed",
    )

    @field_validator(*_none_fields)  # type: ignore[misc]
    @classmethod
    def _none(cls, v: object) -> Optional[str]:
        return _none_str(v)

    @field_validator("train_split", "val_split", "test_split")
    @classmethod
    def _split(cls, v: object) -> Optional[float]:
        if v is None or (isinstance(v, str) and v.strip().lower() == "none"):
            return None
        f = float(v)
        if not 0.0 <= f <= 100.0:
            raise ValueError(f"split {f} must be in [0, 100]")
        return f


class StateDicts(_ConfigSection):
    net_state_dict: Optional[str] = None
    optimizer_state_dict: Optional[str] = None
    scheduler_state_dict: Optional[str] = None

    @field_validator("net_state_dict", "optimizer_state_dict", "scheduler_state_dict")
    @classmethod
    def _none(cls, v: object) -> Optional[str]:
        return _none_str(v)


class Config(BaseModel):
    """Top-level run configuration, mirroring the legacy ``.yml`` sections."""

    model_config = ConfigDict(extra="forbid")

    Task_Options: TaskOptions
    Training_Options: TrainingOptions
    Network_Hyperparameters: NetworkHyperparameters
    Training_Hyperparameters: TrainingHyperparameters
    Preprocessing: Preprocessing
    Data_Augmentation: DataAugmentation
    DataManager_Options: DataManagerOptions
    State_Dicts: StateDicts = Field(default_factory=StateDicts)

    # ------------------------------------------------------------------
    def as_legacy_dicts(self) -> dict[str, dict]:
        """Return the legacy stringly-typed section dicts.

        This is the bridge that lets the ported (legacy-shaped) core modules
        keep working unchanged while the new code has access to a validated
        config. Values are plain ``str``/``int``/``float`` exactly as the
        original configs used them (``'None'`` for nulls, ``'True'``/``'False'``
        for booleans).
        """
        return {
            "Task_Options": _section_dict(self.Task_Options),
            "Training_Options": _section_dict(self.Training_Options),
            "Network_Hyperparameters": _section_dict(self.Network_Hyperparameters),
            "Training_Hyperparameters": _section_dict(self.Training_Hyperparameters),
            "Preprocessing": _section_dict(self.Preprocessing),
            "Data_Augmentation": _section_dict(self.Data_Augmentation),
            "DataManager_Options": _section_dict(self.DataManager_Options),
            "State_Dicts": _section_dict(self.State_Dicts),
        }


def _section_dict(section: BaseModel) -> dict:
    out: dict = {}
    for key, value in section.model_dump().items():
        # Legacy code checks both `is None` and `== 'None'`. Preserve the
        # 'None' sentinel for option fields (normalization, background,
        # data_normalization, shuffle, data_directory, splits, seed, paths,
        # pretrained_network) so both forms work.
        out[key] = "None" if value is None else value
    return out
