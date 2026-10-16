"""
Unit tests for the AgMERRA baseline loader.
============================================

The AgMERRA 1980-2010 climatological means are bundled with METEOR and
loaded by ``meteor.impacts.ggcm.baseline.load_agmerra_baseline``.

These tests verify:
- Arrays have the expected 0.5-degree global grid shape (360 × 720).
- Temperature values are physically plausible (global land range ≈ −60 … 45 °C).
- Precipitation values are all ≥ 1.0 mm/yr (floor applied to avoid division
  by zero in the precip-ratio calculation of the GGCMI polynomial).
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
    temp_baseline, _ = agmerra
    assert temp_baseline.shape == (
        360,
        720,
    ), "Expected global 0.5-degree grid (360 lat × 720 lon)"


def test_precipitation_shape(agmerra):
    _, precip_baseline = agmerra
    assert precip_baseline.shape == (360, 720)


#  Testing agmerra data types are correct
def test_temperature_dtype(agmerra):
    temp_baseline, _ = agmerra
    assert temp_baseline.dtype == np.float64


def test_precipitation_dtype(agmerra):
    _, precip_baseline = agmerra
    assert precip_baseline.dtype == np.float64


def test_temperature_is_ndarray(agmerra):
    temp_baseline, _ = agmerra
    assert isinstance(temp_baseline, np.ndarray)
    # Must NOT be a masked array – ocean cells were already filled
    assert not isinstance(temp_baseline, np.ma.MaskedArray)


def test_precipitation_is_ndarray(agmerra):
    _, precip_baseline = agmerra
    assert isinstance(precip_baseline, np.ndarray)
    assert not isinstance(precip_baseline, np.ma.MaskedArray)


#  Testing agmerra physical plausibility and spatial variation
def test_temperature_range(agmerra):
    """Global mean temperatures should lie in a plausible range."""
    temp_baseline, _ = agmerra
    assert temp_baseline.min() >= -80.0, "Some cells below −80 °C is implausible"
    assert temp_baseline.max() <= 60.0, "Some cells above 60 °C is implausible"


def test_precipitation_floor(agmerra):
    """All precip_baseline values must be >= 1.0 mm/yr (floor applied in loader)."""
    _, precip_baseline = agmerra
    assert np.all(precip_baseline >= 1.0), (
        "Precipitation floor of 1 mm/yr not applied; "
        "division-by-zero risk in precip-ratio computation"
    )


def test_precipitation_no_negative(agmerra):
    _, precip_baseline = agmerra
    assert np.all(precip_baseline >= 0.0)


def test_temperature_has_spatial_variation(agmerra):
    """Non-constant: pole-to-equator temperature gradient should exist."""
    temp_baseline, _ = agmerra
    assert temp_baseline.std() > 1.0, "Suspiciously uniform temperature grid"


def test_precipitation_has_spatial_variation(agmerra):
    _, precip_baseline = agmerra
    assert precip_baseline.std() > 1.0
