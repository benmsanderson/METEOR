"""
Unit tests for the AgMERRA baseline loader.
============================================

The AgMERRA 1980-2010 climatological means are bundled with METEOR and
loaded by ``meteor.impacts.ggcm.baseline.load_agmerra_baseline``.

These tests verify:
- Arrays have the expected 0.5-degree global grid shape (360 × 720).
- Temperature values are physically plausible (global land range ≈ −60 … 45 °C).
- Precipitation values are all ≥ 1.0 mm/yr (floor applied to avoid division
  by zero in the W-ratio calculation of the GGCMI polynomial).
- Both arrays are of dtype float64.
"""

import numpy as np
import pytest

from meteor.impacts.ggcm.baseline import load_agmerra_baseline


@pytest.fixture(scope="module")
def agmerra():
    """Load baseline once per test module."""
    return load_agmerra_baseline()


#  Testing agmerra array shapes are correct


def test_temperature_shape(agmerra):
    T_agmerra, _ = agmerra
    assert T_agmerra.shape == (
        360,
        720,
    ), "Expected global 0.5-degree grid (360 lat × 720 lon)"


def test_precipitation_shape(agmerra):
    _, W_agmerra = agmerra
    assert W_agmerra.shape == (360, 720)


#  Testing agmerra data types are correct
def test_temperature_dtype(agmerra):
    T_agmerra, _ = agmerra
    assert T_agmerra.dtype == np.float64


def test_precipitation_dtype(agmerra):
    _, W_agmerra = agmerra
    assert W_agmerra.dtype == np.float64


def test_temperature_is_ndarray(agmerra):
    T_agmerra, _ = agmerra
    assert isinstance(T_agmerra, np.ndarray)
    # Must NOT be a masked array – ocean cells were already filled
    assert not isinstance(T_agmerra, np.ma.MaskedArray)


def test_precipitation_is_ndarray(agmerra):
    _, W_agmerra = agmerra
    assert isinstance(W_agmerra, np.ndarray)
    assert not isinstance(W_agmerra, np.ma.MaskedArray)


#  Testing agmerra physical plausibility and spatial variation
def test_temperature_range(agmerra):
    """Global mean temperatures should lie in a plausible range."""
    T_agmerra, _ = agmerra
    assert T_agmerra.min() >= -80.0, "Some cells below −80 °C is implausible"
    assert T_agmerra.max() <= 60.0, "Some cells above 60 °C is implausible"


def test_precipitation_floor(agmerra):
    """All W_agmerra values must be >= 1.0 mm/yr (floor applied in loader)."""
    _, W_agmerra = agmerra
    assert np.all(W_agmerra >= 1.0), (
        "Precipitation floor of 1 mm/yr not applied; "
        "division-by-zero risk in W-ratio computation"
    )


def test_precipitation_no_negative(agmerra):
    _, W_agmerra = agmerra
    assert np.all(W_agmerra >= 0.0)


def test_temperature_has_spatial_variation(agmerra):
    """Non-constant: pole-to-equator temperature gradient should exist."""
    T_agmerra, _ = agmerra
    assert T_agmerra.std() > 1.0, "Suspiciously uniform temperature grid"


def test_precipitation_has_spatial_variation(agmerra):
    _, W_agmerra = agmerra
    assert W_agmerra.std() > 1.0
