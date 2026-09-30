# Changelog

All notable changes to `spectrai-v2` are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-09-30

### Added
- Modern rewrite of the `spectrai` framework as a clean, tested PyTorch package.
- Typed, validated config system (pydantic) with backward-compatible legacy `.yml`
  loading and CLI overrides.
- All six architecture families: UNet/ResUNet (1D/2D/3D), ResNet
  (18/34/50/101/152), DenseNet (121/161/169/201), SegNet (2D/3D),
  EfficientNet-b0 (2D), RCAN (2D).
- Single `spectrai` CLI (`train` / `evaluate` / `apply` / `preview`) preserving the
  original flag contract.
- Full test suite (141 tests) including a **parity suite** that verifies the
  rewrite produces numerically identical outputs to the reference for every
  architecture.
- `LICENSE` (Apache-2.0), `README`, `examples/`, and GitHub Actions CI
  (ruff + pytest on Python 3.10/3.11/3.12, wheel build).

### Changed
- Dropped the Python `<3.9` cap and hard-pinned dependencies; now
  `Python >= 3.10` with flexible, modern dependency ranges.
- Split the monolithic reference `trainer.py` into a testable pipeline
  (config / data / transforms / networks / trainer).
- Replaced MATLAB-GUI coupling with a pure Python package.
- `BasicConv` defaults `activation=nn.ReLU(True)` to exactly match the
  reference, keeping `state_dict` keys compatible.

### Removed
- MATLAB GUI (`spectrai.mlapp`).
- Redundant bundled data (kept a small sample set so the CLI runs out of the box).
