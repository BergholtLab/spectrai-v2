# spectrai-v2

A deep learning framework for **spectral data** — 1D spectra and 2D/3D
hyperspectral images — built on PyTorch. `spectrai-v2` is a modern, fully-tested
re-implementation of the original [spectrai](https://github.com/conor-horgan/spectrai)
framework: a typed, validated config system, a clean network registry, a testable
training loop, and a single `spectrai` CLI that preserves the original command
contract.

**Status:** full 1:1 port of all five tasks and six architecture families,
verified end-to-end and numerically parity-checked against the reference
(141 passing tests, including a rewrite-vs-reference parity suite).

> **Note on naming:** the *package* (what you `import`) is `spectrai` and the
> *command* is `spectrai`. The *distribution* you install from source or a wheel
> is **`spectrai-v2`** — this is the project/repo name to avoid colliding with
> the legacy `spectrai` project.

---

## Contents

- [What it does](#what-it-does)
- [Installation](#installation)
- [Quick start (5 minutes)](#quick-start-5-minutes)
- [Training](#training)
- [Evaluation](#evaluation)
- [Applying a model to new data](#applying-a-model-to-new-data)
- [Previewing preprocessing](#previewing-preprocessing)
- [Configuration reference](#configuration-reference)
- [Python API](#python-api)
- [Using your own data](#using-your-own-data)
- [Running the tests](#running-the-tests)
- [Project layout](#project-layout)
- [License](#license)

---

## What it does

### Tasks
| Task | Description |
|---|---|
| **Denoising** | Learn a clean-signal mapping from noisy spectra / images |
| **Calibration** | Map uncalibrated spectra to a reference |
| **Classification** | Per-sample classification (spectra or images) |
| **Segmentation** | Per-pixel image segmentation |
| **Super-Resolution** | Upsample low-res hyperspectral images (2D) |

### Architectures
| Network | Dimensions | Tasks (as implemented) |
|---|---|---|
| **UNet / ResUNet** | 1D, 2D, 3D | denoising, calibration, classification, segmentation, super-resolution (2D) |
| **ResNet** (18/34/50/101/152) | 1D, 2D, 3D | classification |
| **DenseNet** (121/161/169/201) | 1D, 2D, 3D | classification |
| **SegNet** | 2D, 3D | segmentation |
| **EfficientNet-b0** | 2D | classification |
| **RCAN** | 2D | super-resolution |

> `spectrai-v2` **preserves the reference's gaps**: an unsupported
> `(network, task, dimension)` combination raises the same `NotImplementedError`
> the original did, rather than silently inventing new behaviour.

---

## Installation

Requires **Python ≥ 3.10** and PyTorch ≥ 2.0.

### Option A — install from this repository (recommended)
```bash
git clone https://github.com/BergholtLab/spectrai-v2.git
cd spectrai-v2

# Create and activate a virtual environment (strongly recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install the package + dependencies (editable)
pip install -e .
```

### Option B — with dev tooling (pytest, ruff, coverage)
```bash
pip install -e ".[dev]"
```

> **macOS / Linux PEP 668:** if your system Python is "externally managed",
> create the virtual environment above first (it resolves the issue), or pass
> `pip install -e . --break-system-packages` at your own risk.
>
> **No GPU? No problem.** PyTorch installs a CPU build by default on
> Linux/macOS. For an NVIDIA GPU, install the CUDA-enabled torch before
> `pip install -e .` (see <https://pytorch.org/get-started/locally/>).

### Verify the install
```bash
spectrai --help
python -c "import spectrai; print('spectrai', spectrai.__version__, 'OK')"
pytest -q        # expect: 141 passed
```

---

## Quick start (5 minutes)

A small `data/` set and `configs/` ship with the package, so these commands work
**out of the box** — no dataset download needed.

```bash
# 1. Train 1D spectral denoising for one epoch (UNet).
spectrai train --config spectral_denoising.yml --epochs 1 --verbose

# 2. Or run the bundled Python example (same pipeline, scripted):
python examples/train_denoising.py

# 3. Or a 2D hyperspectral image classifier (DenseNet), synthetic data:
python examples/train_image_classification.py
```

That's the whole loop — config → data → transforms → network → training →
reported loss.

---

## Training

`spectrai train` loads a config, applies any CLI overrides, and runs the
training loop. The most useful flags:

```bash
spectrai train \
  --config spectral_denoising.yml \
  --network UNet \                # UNet | ResUNet | ResNet | DenseNet | ...
  --dimension 2D \                # 1D | 2D | 3D
  --activation ReLU \             # ReLU | PReLU | LeakyReLU | ...
  --normalization BatchNorm \     # BatchNorm | GroupNorm | LayerNorm | InstanceNorm | None
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --criterion "L1" \
  --data_format "Spectra" \
  --seed 42 \
  --verbose \                     # print per-epoch metrics
  --save-frequency 0              # save every N epochs (0 = only at the end)
```

The full flag list is available with `spectrai train --help`. Config files live
in `configs/` (one per task):

| Config | Task |
|---|---|
| `spectral_denoising.yml` | 1D denoising |
| `spectral_calibration.yml` | 1D calibration |
| `spectral_classification.yml` | 1D classification |
| `image_classification.yml` | 2D image classification |
| `image_classification3D.yml` | 3D image classification |
| `image_segmentation.yml` | 2D segmentation |
| `image_segmentation3D.yml` | 3D segmentation |
| `image_superresolution.yml` | 2D super-resolution |

The trained model is written to `saved_models/` as a `.pt` checkpoint containing
the network state dict, optimizer state, and scheduler state (so training can be
resumed).

---

## Evaluation

Evaluate a trained model on validation/test data:

```bash
spectrai evaluate --config image_segmentation.yml --verbose
```

To load a specific saved checkpoint, add `--pretrained_network` with the path to
the `.pt` file (see `spectrai evaluate --help` for the full flag set).

---

## Applying a model to new data

Run inference on new data (no targets required):

```bash
spectrai apply --config spectral_denoising.yml
```

Outputs are written to `apply_outputs/`.

---

## Previewing preprocessing

See exactly what shape/normalization the model receives before you train:

```bash
spectrai preview --config spectral_denoising.yml
```

---

## Configuration reference

Configs are validated `pydantic` models loaded from the `.yml` files in
`configs/`. CLI flags override individual fields; only non-`None` overrides are
applied. Sections:

| Section | Key fields |
|---|---|
| `Task_Options` | `task`, `classes` |
| `Training_Options` | training mode, pretrained checkpoint |
| `Network_Hyperparameters` | `network`, `dimension`, `activation`, `normalization` |
| `Training_Hyperparameters` | `epochs`, `batch_size`, `learning_rate`, image sizes, `spectrum_length`, `optimizer`, `scheduler`, `criterion` |
| `Preprocessing` | spectral crop window, background subtraction, data normalization |
| `Data_Augmentation` | spatial & spectral augmentation, mixup |
| `DataManager_Options` | `data_format`, paths, splits, `shuffle`, `seed` |

Open any file in `configs/` to see the full set of options for that task.

---

## Python API

### Train programmatically
```python
from spectrai.config import load_config
from spectrai.trainer import train_epoch

cfg = load_config("spectral_denoising.yml").as_legacy_dicts()
out = train_epoch(
    cfg["Training_Options"], cfg["Task_Options"], cfg["Network_Hyperparameters"],
    cfg["Training_Hyperparameters"], cfg["Preprocessing"], cfg["Data_Augmentation"],
    cfg["DataManager_Options"],
    cfg["State_Dicts"]["net_state_dict"],
    cfg["State_Dicts"]["optimizer_state_dict"],
    cfg["State_Dicts"]["scheduler_state_dict"],
    epochs=cfg["Training_Hyperparameters"]["epochs"],
)
```

### Use a network directly
```python
from spectrai.networks.nets.unet import UNet
from spectrai.networks.nets.resnet import ResNet18, ResNet50
from spectrai.networks.nets.densenet import densenet121
from spectrai.networks.nets.segnet import SegNet
from spectrai.networks.nets.rcan import Hyperspectral_RCAN
from spectrai.networks.nets.efficientnet import EfficientNet
```

---

## Using your own data

- **Spectra (1D):** a directory of `.mat` files (each a single spectrum) or a
  flat `.mat` of stacked spectra. Point `DataManager_Options.train_input_data`
  (and the target equivalent for supervised tasks) at your files.
- **Images (2D/3D):** a directory with one subfolder per class (classification),
  or separate input/target folders (segmentation, super-resolution). Use
  `data_format: "Image: H, W, C"` (or `D, H, W, C` for 3D).

Set `train_split` / `val_split` / `test_split` (e.g. `0.7 0.15 0.15`) or provide
explicit file lists. See a config in `configs/` for the exact field names for
your task.

---

## Running the tests

```bash
pytest                       # full suite (141 tests)
pytest tests/test_parity.py  # rewrite vs. reference numerical parity
pytest -k integration        # full-pipeline end-to-end runs
```

The **parity suite** builds each network from the reference and the rewrite with
identical constructor args, loads the reference's weights into the rewrite, and
asserts the forward outputs are numerically identical — a per-architecture
guarantee that the rewrite is behaviourally faithful.

> The parity tests compare against a read-only reference checkout at
> `/tmp/spectrai`; if that path isn't present they are skipped.

---

## Project layout

```
spectrai/
├── config/         # pydantic schema + legacy-yml loader + CLI overrides
├── data/           # dataset classes, file/dir loading, splitting, dataloaders
├── transforms/     # spectral + image transform pipelines
├── networks/
│   ├── layers.py   # BasicConv, UNetConv, Down/Up, Upsampler
│   ├── utils.py    # get_normalization, get_pooling
│   ├── registry.py # setup_network / edit_network
│   └── nets/       # unet, resnet, rcan, efficientnet, segnet, densenet
├── trainer.py      # train / validate / evaluate / apply
├── previewer.py    # preprocessing preview
└── cli.py          # spectrai {train, evaluate, apply, preview}
```

---

## Contributing

1. Fork and clone.
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -e ".[dev]"`
4. `pytest` to confirm a green baseline, then make your change.
5. Open a PR. CI (`.github/workflows/ci.yml`) runs `ruff` + `pytest` on
   Python 3.10/3.11/3.12 and builds the wheel on every push/PR.

## License

[Apache License 2.0](LICENSE).
