"""Phase 3 parity tests: rewrite vs. reference spectrai, net by net.

For each architecture we build the reference network and the rewritten network
with *identical* constructor args, load the reference's ``state_dict`` into the
rewrite, and assert the forward outputs are numerically identical (within a
tight tolerance). This proves the rewrite is behaviourally faithful at the
network level — independent of data-loading or training, which are exercised
separately by ``test_integration.py``.

Both packages are named ``spectrai``. The reference uses *absolute* imports
(``from spectrai.networks.layer_utils import ...``), so it must be imported
while ``spectrai`` still resolves to the reference. We therefore:

1. Load the reference first (``/tmp/spectrai`` on ``sys.path``), caching the
   net classes.
2. Then load the rewrite under a distinct top-level name (``rw_spectrai``) so
   it never collides with the cached reference.
"""

from __future__ import annotations

import importlib
import os
import sys

import torch

REF_DIR = "/tmp/spectrai"
REPO_DIR = "/home/bergholtlab/.openclaw/workspace/spectrai"
_NETS = ("unet", "resnet", "rcan", "densenet", "segnet", "efficientnet")

# ---------------------------------------------------------------------------
# Phase 1: load the reference (absolute imports resolve to /tmp/spectrai).
# ---------------------------------------------------------------------------
if REF_DIR not in sys.path:
    sys.path.insert(0, REF_DIR)

_REF: dict[str, object] = {}
for _net in _NETS:
    _REF[_net] = importlib.import_module(f"spectrai.networks.nets.{_net}")

# ---------------------------------------------------------------------------
# Phase 2: load the rewrite under a distinct top-level name.
# ---------------------------------------------------------------------------
import importlib.util
import types


def _install_rewrite() -> None:
    if "rw_spectrai" in sys.modules:
        return
    top = types.ModuleType("rw_spectrai")
    top.__path__ = [os.path.join(REPO_DIR, "spectrai")]
    sys.modules["rw_spectrai"] = top
    for sub in ("networks", "networks.nets"):
        dotted = f"rw_spectrai.{sub}"
        path = os.path.join(REPO_DIR, "spectrai", *sub.split("."))
        pkg = importlib.import_module(dotted)
        sys.modules[dotted] = pkg
        setattr(sys.modules[dotted.rsplit(".", 1)[0]], sub.split(".")[-1], pkg)
    for net in _NETS:
        importlib.import_module(f"rw_spectrai.networks.nets.{net}")


_install_rewrite()


def _pair(net: str):
    return _REF[net], importlib.import_module(f"rw_spectrai.networks.nets.{net}")


def _assert_same(a: torch.Tensor, b: torch.Tensor, msg: str, rtol=1e-5, atol=1e-6):
    assert a.shape == b.shape, f"{msg}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    a = a.float()
    b = b.float()
    max_diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, rtol=rtol, atol=atol), f"{msg}: max abs diff {max_diff}"


def _parity(ref_cls, mine_cls, ctor_kwargs, input_tensor, msg):
    ref = ref_cls(**ctor_kwargs)
    mine = mine_cls(**ctor_kwargs)
    mine.load_state_dict(ref.state_dict())
    ref.eval()
    mine.eval()
    with torch.no_grad():
        _assert_same(mine(input_tensor), ref(input_tensor), msg)


def test_parity_unet_denoising_1d():
    ref, mine = _pair("unet")
    _parity(
        ref.UNet, mine.UNet,
        dict(dims=1, channels=1, num_classes=0, n_blocks=3,
             normalization="BatchNorm", activation=torch.nn.ReLU(), task="Denoising", res=False),
        torch.randn(4, 1, 128),
        "UNet 1D denoising",
    )


def test_parity_unet_classification_2d():
    ref, mine = _pair("unet")
    _parity(
        ref.UNet, mine.UNet,
        dict(dims=2, channels=8, num_classes=3, n_blocks=3,
             normalization="BatchNorm", activation=torch.nn.ReLU(), task="Classification", res=True),
        torch.randn(4, 8, 16, 16),
        "ResUNet 2D classification",
    )


def test_parity_resnet18_classification_2d():
    ref, mine = _pair("resnet")
    _parity(
        ref.ResNet18, mine.ResNet18,
        dict(dims=2, img_channels=8, res_channels=8, num_classes=3,
             normalization="BatchNorm", activation=torch.nn.ReLU()),
        torch.randn(4, 8, 16, 16),
        "ResNet18 2D classification",
    )


def test_parity_rcan_superres_2d():
    ref, mine = _pair("rcan")
    _parity(
        ref.Hyperspectral_RCAN, mine.Hyperspectral_RCAN,
        dict(spectrum_length=32, scale=2, activation=torch.nn.ReLU()),
        torch.randn(2, 32, 16, 16),
        "RCAN 2D super-resolution",
    )


def test_parity_densenet121_classification_2d():
    ref, mine = _pair("densenet")
    _parity(
        ref.densenet121, mine.densenet121,
        dict(dims=2, channels=8, num_classes=3,
             normalization="BatchNorm", activation=torch.nn.ReLU(inplace=True)),
        torch.randn(2, 8, 32, 32),
        "DenseNet121 2D classification",
    )


def test_parity_segnet_segmentation_2d():
    ref, mine = _pair("segnet")
    _parity(
        ref.SegNet, mine.SegNet,
        dict(dims=2, num_classes=3, channels=8),
        torch.randn(2, 8, 32, 32),
        "SegNet 2D segmentation",
    )


def test_parity_efficientnet_classification_2d():
    ref, mine = _pair("efficientnet")
    r = ref.EfficientNet.from_name("efficientnet-b0", 8, image_size=32, num_classes=3)
    m = mine.EfficientNet.from_name("efficientnet-b0", 8, image_size=32, num_classes=3)
    m.load_state_dict(r.state_dict())
    r.eval()
    m.eval()
    x = torch.randn(2, 8, 32, 32)
    with torch.no_grad():
        _assert_same(m(x), r(x), "EfficientNet-b0 2D classification")
