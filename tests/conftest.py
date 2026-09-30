"""Shared fixtures for the spectrai test suite.

Generates small synthetic spectra and hyperspectral images so the suite is
self-contained and does not depend on the bundled ``.mat`` test data.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def spectrum(rng) -> np.ndarray:
    """A single synthetic 1D spectrum, length 500 (float64)."""
    x = np.linspace(0, 1, 500)
    base = x
    peaks = np.exp(-((x - 0.3) ** 2) / 0.01) * 0.5
    return (base + peaks + rng.normal(0, 0.02, 500)).astype(np.float64)


@pytest.fixture
def spectra(rng, spectrum) -> np.ndarray:
    """A batch of synthetic spectra, shape (N, 500)."""
    n = 64
    out = np.stack([spectrum + rng.normal(0, 0.05, 500) for _ in range(n)])
    return out.astype(np.float64)


@pytest.fixture
def hyperspectral_image(rng) -> np.ndarray:
    """A synthetic hyperspectral image, shape (H, W, C) with C=32 channels."""
    h, w, c = 32, 32, 32
    x = np.linspace(0, 1, c)
    image = np.zeros((h, w, c), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            image[i, j] = x * (i + j) / (h + w)
    image += rng.normal(0, 0.02, (h, w, c))
    return image


@pytest.fixture
def mask() -> np.ndarray:
    """A synthetic 0/1 segmentation mask, shape (H, W)."""
    h, w = 32, 32
    m = np.zeros((h, w), dtype=np.uint8)
    m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1
    return m
