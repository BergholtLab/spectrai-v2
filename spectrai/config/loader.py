"""Load legacy ``.yml`` configs into a validated :class:`~spectrai.config.schema.Config`."""

from __future__ import annotations

import os
from typing import Any, Optional

import yaml

from .schema import Config


def load_config(config: Optional[str | os.PathLike] = None, *, path: Optional[str] = None) -> Config:
    """Load a spectrai config.

    Arguments:
        config: Path to a ``.yml`` config file, or the *name* of a bundled
            config (e.g. ``'spectral_denoising.yml'``) resolved against the
            package ``configs/`` directory.
        path: Explicit directory to resolve a bare config name against
            (defaults to the bundled ``configs/`` folder).

    Returns:
        A validated :class:`Config`.
    """
    raw = _read_yaml(config, path)
    return Config.model_validate(raw)


def default_config_path(name: str, *, path: Optional[str] = None) -> str:
    """Resolve a config name to an absolute path."""
    if name is None:
        raise ValueError("config name is required")
    if os.path.isabs(name) and os.path.exists(name):
        return name
    # Search order: explicit `path`, the package-bundled configs, then the
    # repo-top-level `configs/` directory (the canonical published location).
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "configs"),
        os.path.join(os.path.dirname(os.path.dirname(here)), "configs"),
        "configs",
    ]
    if path:
        candidates.insert(0, path)
    for base in candidates:
        candidate = name if os.path.isabs(name) else os.path.join(base, name)
        if os.path.exists(candidate):
            return candidate
    searched = ", ".join(candidates)
    raise FileNotFoundError(f"config not found: {name} (looked in: {searched})")


def _read_yaml(config: Optional[str | os.PathLike], path: Optional[str]) -> dict[str, Any]:
    if config is None:
        raise ValueError("no config specified")
    cfg_path = default_config_path(str(config), path=path)
    with open(cfg_path, "r") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"config {cfg_path} did not parse to a mapping")
    return data


def apply_cli_overrides(config: Config, overrides: dict[str, Any]) -> Config:
    """Apply CLI overrides to a validated config and re-validate.

    ``overrides`` maps *field names* (as they appear in any section, e.g.
    ``'batch_size'``, ``'activation'``) to new values. Only non-``None``
    values are applied, mirroring the reference CLI behaviour.
    """
    merged: dict[str, Any] = config.model_dump()
    for key, value in overrides.items():
        if value is None:
            continue
        for section_name, section in merged.items():
            if isinstance(section, dict) and key in section:
                section[key] = value
                break
    return Config.model_validate(merged)
