"""Forward-shape tests for the ResNet18/34/50/101/152 port (Phase 2).

The reference registry (``spectrai/networks/networks.py``) supports only the
``Classification`` task for ResNet and raises ``NotImplementedError`` for every
other task.  Supported dimension / data-format combos:

- Spectra (1D):      ``dims=1, img_channels=channels, res_channels=channels``
- Image 2D:          ``dims=2, img_channels=channels, res_channels=channels``
- Image 3D:          ``dims=3, img_channels=1,       res_channels=64``

Forward output is a per-sample classification logit of shape ``(B, num_classes)``.
"""

from __future__ import annotations

import torch
from torch import nn

import pytest

from spectrai.networks.nets.resnet import (
    ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
    BasicBlock, Bottleneck,
)


# ---------------------------------------------------------------------------
# 1D / Spectra classification (dims=1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [ResNet18, ResNet34, ResNet50])
def test_resnet_1d_classification_shape(factory):
    channels = 8
    num_classes = 3
    net = factory(dims=1, img_channels=channels, res_channels=channels,
                  num_classes=num_classes, normalization="None",
                  activation=nn.ReLU()).float()
    net.eval()
    x = torch.randn(2, channels, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, num_classes)


# ---------------------------------------------------------------------------
# 2D image classification (dims=2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152])
def test_resnet_2d_classification_shape(factory):
    channels = 4
    num_classes = 3
    net = factory(dims=2, img_channels=channels, res_channels=channels,
                  num_classes=num_classes, normalization="None",
                  activation=nn.ReLU()).float()
    net.eval()
    x = torch.randn(2, channels, 32, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, num_classes)


# ---------------------------------------------------------------------------
# 3D image classification (dims=3, img_channels=1, res_channels=64)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [ResNet18, ResNet50])
def test_resnet_3d_classification_shape(factory):
    num_classes = 3
    net = factory(dims=3, img_channels=1, res_channels=64,
                  num_classes=num_classes, normalization="None",
                  activation=nn.ReLU()).float()
    net.eval()
    x = torch.randn(2, 1, 32, 32, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, num_classes)


# ---------------------------------------------------------------------------
# Helper / block classes exist and are usable
# ---------------------------------------------------------------------------
def test_blocks_expansion():
    assert BasicBlock.expansion == 1
    assert Bottleneck.expansion == 4


def test_basicblock_rejects_bad_dilation():
    with pytest.raises(NotImplementedError):
        BasicBlock(1, 4, 4, dilation=2, normalization="None")


def test_basicblock_rejects_bad_groups():
    with pytest.raises(ValueError):
        BasicBlock(1, 4, 4, groups=2, normalization="None")


def test_resnet_rejects_bad_dilation_tuple():
    with pytest.raises(ValueError):
        ResNet(1, BasicBlock, [2, 2, 2, 2],
               replace_stride_with_dilation=[False, False])


def test_resnet_dims_preserved():
    net = ResNet(2, BasicBlock, [1, 1, 1, 1],
                 img_channels=2, res_channels=4, num_classes=3,
                 normalization="None", activation=nn.ReLU())
    assert net.dims == 2
    assert isinstance(net.fc, nn.Linear)
    assert net.fc.in_features == 4 * 8 * 1  # res_channels*8 * expansion
    assert net.fc.out_features == 3


# ---------------------------------------------------------------------------
# NotImplementedError / ValueError paths that live inside the reference
# ``resnet.py`` (not in the registry).  The reference registry also raises
# NotImplementedError for non-Classification tasks, but that gate is in the
# registry (handled separately in Phase 2), so we test only the in-class
# restrictions here.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152])
@pytest.mark.parametrize("dilate", [[True, False, False], [False, True, False], [False, False, True]])
def test_resnet_dilate_replaced_stride_not_supported_by_basicblock(factory, dilate):
    """BasicBlock raises NotImplementedError when dilation > 1.
    The reference ResNet passes ``replace_stride_with_dilation`` to dilate
    the corresponding layer, which then sets dilation>1 for the BasicBlock
    -> raises.  Bottleneck (ResNet50/101/152) allows dilation>1, so it does
    not raise; BasicBlock (ResNet18/34) does."""
    is_basic = factory is ResNet18 or factory is ResNet34
    kwargs = dict(dims=1, img_channels=4, res_channels=4, num_classes=3,
                  normalization="None", activation=nn.ReLU(),
                  replace_stride_with_dilation=dilate)
    if is_basic:
        with pytest.raises(NotImplementedError):
            factory(**kwargs)
    else:
        # Bottleneck handles dilation fine; forward still works.
        net = factory(**kwargs)
        net.eval()
        x = torch.randn(2, 4, 32)
        with torch.no_grad():
            y = net(x)
        assert y.shape == (2, 3)
