"""Forward-shape tests for the EfficientNet architecture (Phase 2 port).

The reference registry (``spectrai/networks/networks.py``) instantiates
EfficientNet only for 2D ``Classification`` and raises ``NotImplementedError``
for every other combo:

- 1D (spectral) data:  ``NotImplementedError`` ("not implemented for 1D (spectral) data")
- 2D non-Classification tasks: ``NotImplementedError`` ("only implemented for 2D classification")

The reference calls::

    EfficientNet.from_name('efficientnet-b0', channels,
                           image_size=..., num_classes=...)

and afterwards swaps the classifier head (``net._fc``) for the real class
count via ``edit_network``.  Because this shared registry is still a Phase-1
scaffold (sibling Phase-2 ports' tests import the nets directly for exactly
this reason), these tests exercise the net module directly, reproducing the
reference registry's constructor call, the head-swap, and the rejected combos
at the level the port actually provides.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectrai.networks.nets.efficientnet import (
    VALID_MODELS,
    BlockArgs,
    BlockDecoder,
    EfficientNet,
    GlobalParams,
    MemoryEfficientSwish,
    MBConvBlock,
    MaxPool2dDynamicSamePadding,
    MaxPool2dStaticSamePadding,
    Conv2dDynamicSamePadding,
    Conv2dStaticSamePadding,
    SwishImplementation,
    calculate_output_image_size,
    drop_connect,
    efficientnet,
    efficientnet_params,
    get_model_params,
    get_same_padding_conv2d,
    get_same_padding_maxPool2d,
    get_width_and_height_from_size,
    round_filters,
    round_repeats,
)


# ---------------------------------------------------------------------------
# 2D image classification (the only combo the reference registry supports)
# ---------------------------------------------------------------------------
def _build_2d_classifier(channels=32, image_size=32, num_classes=3):
    """Mirror the reference registry's 2D classification branch:
    ``EfficientNet.from_name('efficientnet-b0', channels,
    image_size=image_size, num_classes=num_classes).float()``
    (the registry builds with ``num_classes`` then head-swaps via
    ``edit_network``; here we pass the real class count directly)."""
    net = EfficientNet.from_name(
        'efficientnet-b0', channels, image_size=image_size, num_classes=num_classes
    ).float()
    return net


def test_efficientnet_2d_classification_shape():
    """2D classification: channels-first image in -> (B, num_classes) logits."""
    channels = 32
    num_classes = 3
    net = _build_2d_classifier(channels=channels, image_size=32, num_classes=num_classes)
    net.eval()
    x = torch.randn(2, channels, 32, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, num_classes)


def test_efficientnet_edit_network_head_swap():
    """Reference ``edit_network`` behaviour: build with placeholder classes,
    then replace ``net._fc`` with a Linear of the true class count."""
    channels = 32
    placeholder_classes = 1000
    num_classes = 3
    net = EfficientNet.from_name(
        'efficientnet-b0', channels, image_size=32, num_classes=placeholder_classes
    ).float()
    assert isinstance(net._fc, nn.Linear)
    assert net._fc.in_features == 1280  # round_filters(1280, width=1.0)
    assert net._fc.out_features == placeholder_classes

    # edit_network: net._fc = nn.Linear(net._fc.in_features, classes)
    net._fc = nn.Linear(net._fc.in_features, num_classes)
    net.eval()
    x = torch.randn(2, channels, 32, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, num_classes)


# ---------------------------------------------------------------------------
# NotImplementedError / invalid-input paths (reference net module behaviour)
# ---------------------------------------------------------------------------
def test_efficientnet_rejects_invalid_model_name():
    with pytest.raises(ValueError, match="model_name should be one of"):
        EfficientNet.from_name('efficientnet-x9', 32)


def test_efficientnet_get_model_params_undefined_name():
    with pytest.raises(NotImplementedError, match="model name is not pre-defined"):
        get_model_params('resnet50', {})


def test_efficientnet_override_unknown_param():
    with pytest.raises(ValueError):
        EfficientNet.from_name('efficientnet-b0', 32, bogus_param=1)


# ---------------------------------------------------------------------------
# Public API / building-block checks
# ---------------------------------------------------------------------------
def test_valid_models_include_b0_b8():
    assert 'efficientnet-b0' in VALID_MODELS
    assert 'efficientnet-b8' in VALID_MODELS


def test_get_image_size():
    assert EfficientNet.get_image_size('efficientnet-b0') == 224
    assert EfficientNet.get_image_size('efficientnet-b4') == 380


def test_round_filters_b0_identity():
    blocks_args, global_params = get_model_params('efficientnet-b0', {})
    assert round_filters(32, global_params) == 32
    assert round_filters(1280, global_params) == 1280


def test_round_repeats_b0_identity():
    _, global_params = get_model_params('efficientnet-b0', {})
    assert round_repeats(3, global_params) == 3


def test_block_decoder_roundtrip():
    blocks_args = BlockDecoder.decode(
        ['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25'])
    assert blocks_args[0].num_repeat == 1
    assert blocks_args[0].input_filters == 32
    assert blocks_args[0].output_filters == 16
    assert blocks_args[0].se_ratio == 0.25
    assert blocks_args[0].id_skip is True
    assert blocks_args[1].num_repeat == 2
    assert blocks_args[1].stride == [2]
    # NOTE: the reference's _encode_block_string uses block.strides (plural),
    # which does not exist on the reference BlockArgs (field is 'stride'), so
    # BlockDecoder.encode raises AttributeError even in the reference repo.
    # The 1:1 port preserves that quirk; encode round-tripping is unsupported.
    with pytest.raises(AttributeError):
        BlockDecoder.encode(blocks_args)


def test_b0_model_has_expected_block_count():
    net = EfficientNet.from_name('efficientnet-b0', 3, image_size=32, num_classes=3)
    # r1 + r2 + r2 + r3 + r3 + r4 + r1 = 16 blocks
    assert len(net._blocks) == 16
    assert isinstance(net._blocks[0], MBConvBlock)


def test_swish_activation_available():
    s = MemoryEfficientSwish()
    x = torch.randn(2, 4)
    y = s(x)
    assert y.shape == x.shape
    assert isinstance(SwishImplementation, type)


def test_drop_connect_eval_identity():
    x = torch.randn(2, 4, 8, 8)
    assert drop_connect(x, p=0.5, training=False) is x


def test_same_padding_conv_factories():
    assert get_same_padding_conv2d(None) is Conv2dDynamicSamePadding
    static_conv = get_same_padding_conv2d(image_size=32)
    assert static_conv.func is Conv2dStaticSamePadding
    assert get_same_padding_maxPool2d(None) is MaxPool2dDynamicSamePadding
    static_pool = get_same_padding_maxPool2d(image_size=32)
    assert static_pool.func is MaxPool2dStaticSamePadding


def test_calculate_output_image_size():
    assert calculate_output_image_size(None, 2) is None
    assert calculate_output_image_size(32, 2) == [16, 16]
    assert calculate_output_image_size((30, 40), 2) == [15, 20]


def test_get_width_and_height_from_size():
    assert get_width_and_height_from_size(32) == (32, 32)
    assert get_width_and_height_from_size((30, 40)) == (30, 40)
    with pytest.raises(TypeError):
        get_width_and_height_from_size("32")


def test_extract_features_shape():
    net = EfficientNet.from_name('efficientnet-b0', 8, image_size=32, num_classes=3)
    net.eval()
    x = torch.randn(1, 8, 32, 32)
    with torch.no_grad():
        feats = net.extract_features(x)
    # 32 / 2(stem) / 2 / 2 / 2 / 2 = 1 spatial dim, 1280 head channels
    assert feats.shape == (1, 1280, 1, 1)


def test_extract_endpoints_reduction_levels():
    net = EfficientNet.from_name('efficientnet-b0', 8, image_size=32, num_classes=3)
    net.eval()
    x = torch.randn(1, 8, 32, 32)
    with torch.no_grad():
        eps = net.extract_endpoints(x)
    assert list(eps) == [f"reduction_{i}" for i in range(1, 7)]
    assert eps["reduction_6"].shape[-1] == 1
    assert eps["reduction_1"].shape[-1] == 16  # 32/2
