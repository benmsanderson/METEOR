"""
Unit tests for impacts utilities
===============================

Tests the utility functions in impacts.utils module.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

from meteor.impacts.utils import (
    check_monthly_dimension,
    convert_temperature_units,
    create_monthly_time_axis,
    ensure_spatial_coordinates,
    group_by_season,
    validate_temperature_data,
)


# Temperature validation tests
def test_validate_temperature_data_valid_celsius():
    """Test validation with valid Celsius data."""
    data = xr.DataArray([10, 20, 30], dims=["x"])
    # Should not raise
    validate_temperature_data(data, "celsius")


def test_validate_temperature_data_valid_kelvin():
    """Test validation with valid Kelvin data."""
    data = xr.DataArray([283, 293, 303], dims=["x"])
    # Should not raise
    validate_temperature_data(data, "kelvin")


def test_validate_temperature_data_invalid_type():
    """Test validation with invalid input type."""
    with pytest.raises(
        ValueError, match="Temperature data must be an xarray.DataArray"
    ):
        validate_temperature_data([10, 20, 30], "celsius")


def test_validate_temperature_data_celsius_out_of_range():
    """Test Celsius data outside reasonable range."""
    data = xr.DataArray([-100, 0, 100], dims=["x"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_temperature_data(data, "celsius")
        assert len(w) > 0
        assert "outside expected range" in str(w[0].message)


def test_validate_temperature_data_kelvin_out_of_range():
    """Test Kelvin data outside reasonable range."""
    data = xr.DataArray([100, 200, 500], dims=["x"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_temperature_data(data, "kelvin")
        assert len(w) > 0
        assert "outside expected range" in str(w[0].message)


def test_validate_temperature_data_custom_range():
    """Test validation with custom temperature range."""
    data = xr.DataArray([10, 20, 30], dims=["x"])

    # Should warn with narrow custom range
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_temperature_data(data, "celsius", temp_range=(15, 25))
        assert len(w) > 0


def test_validate_temperature_data_unknown_units():
    """Test validation with unknown units."""
    data = xr.DataArray([10, 20, 30], dims=["x"])

    # Should raise ValueError for unknown units
    with pytest.raises(ValueError, match="Unknown temperature units: fahrenheit"):
        validate_temperature_data(data, "fahrenheit")


# Temperature unit conversion tests
def test_convert_celsius_to_kelvin():
    """Test Celsius to Kelvin conversion."""
    data = xr.DataArray([0, 20, 100], dims=["x"], attrs={"units": "celsius"})
    result = convert_temperature_units(data, "celsius", "kelvin")

    expected = [273.15, 293.15, 373.15]
    np.testing.assert_array_almost_equal(result.values, expected)
    assert result.attrs["units"] == "kelvin"


def test_convert_kelvin_to_celsius():
    """Test Kelvin to Celsius conversion."""
    data = xr.DataArray([273.15, 293.15, 373.15], dims=["x"])
    result = convert_temperature_units(data, "kelvin", "celsius")

    expected = [0, 20, 100]
    np.testing.assert_array_almost_equal(result.values, expected)


def test_convert_celsius_to_fahrenheit():
    """Test Celsius to Fahrenheit conversion."""
    data = xr.DataArray([0, 20, 100], dims=["x"])
    result = convert_temperature_units(data, "celsius", "fahrenheit")

    expected = [32, 68, 212]
    np.testing.assert_array_almost_equal(result.values, expected)


def test_convert_fahrenheit_to_celsius():
    """Test Fahrenheit to Celsius conversion."""
    data = xr.DataArray([32, 68, 212], dims=["x"])
    result = convert_temperature_units(data, "fahrenheit", "celsius")

    expected = [0, 20, 100]
    np.testing.assert_array_almost_equal(result.values, expected)


def test_convert_temperature_unknown_units():
    """Test conversion with unknown units."""
    data = xr.DataArray([10, 20, 30], dims=["x"])

    with pytest.raises(ValueError, match="Unknown source units"):
        convert_temperature_units(data, "unknown", "celsius")

    with pytest.raises(ValueError, match="Unknown target units"):
        convert_temperature_units(data, "celsius", "unknown")


# Monthly dimension checking tests
def test_check_monthly_dimension_valid():
    """Test with valid monthly data."""
    data = xr.DataArray(np.random.rand(24), dims=["month"])
    # Should not raise
    check_monthly_dimension(data)


def test_check_monthly_dimension_missing():
    """Test with missing month dimension."""
    data = xr.DataArray(np.random.rand(24), dims=["time"])

    with pytest.raises(ValueError, match="must have a 'month' dimension"):
        check_monthly_dimension(data)


def test_check_monthly_dimension_short():
    """Test with less than 12 months."""
    data = xr.DataArray(np.random.rand(6), dims=["month"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_monthly_dimension(data)
        assert len(w) > 0
        assert "only 6 months" in str(w[0].message)


def test_check_monthly_dimension_custom_name():
    """Test with custom dimension name."""
    data = xr.DataArray(np.random.rand(24), dims=["time"])
    # Should not raise when specifying correct dimension name
    check_monthly_dimension(data, dim_name="time")


# Spatial coordinate detection tests
def test_ensure_spatial_coordinates_standard():
    """Test with standard lat/lon coordinates."""
    data = xr.DataArray(
        np.random.rand(10, 10),
        dims=["lat", "lon"],
        coords={"lat": np.arange(10), "lon": np.arange(10)},
    )

    lat_name, lon_name = ensure_spatial_coordinates(data)
    assert lat_name == "lat"
    assert lon_name == "lon"


def test_ensure_spatial_coordinates_alternative():
    """Test with alternative coordinate names."""
    data = xr.DataArray(
        np.random.rand(10, 10),
        dims=["latitude", "longitude"],
        coords={"latitude": np.arange(10), "longitude": np.arange(10)},
    )

    lat_name, lon_name = ensure_spatial_coordinates(data)
    assert lat_name == "latitude"
    assert lon_name == "longitude"


def test_ensure_spatial_coordinates_missing():
    """Test with missing spatial coordinates."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["dimension1", "dimension2"])

    with pytest.raises(ValueError, match="Could not find latitude coordinate"):
        ensure_spatial_coordinates(data)


# Monthly time axis creation tests
def test_create_monthly_time_axis_basic():
    """Test basic time axis creation."""
    time_axis = create_monthly_time_axis(1850, 24)  # 2 years

    assert len(time_axis) == 24
    assert time_axis[0] == 1850.0  # January 1850
    assert time_axis[11] == 1850 + 11 / 12  # December 1850
    assert time_axis[12] == 1851.0  # January 1851

    assert "start_year" in time_axis.attrs
    assert time_axis.attrs["start_year"] == 1850


def test_create_monthly_time_axis_custom_dimension():
    """Test with custom dimension name."""
    time_axis = create_monthly_time_axis(2000, 12, dim_name="time")

    assert time_axis.dims == ("time",)
    assert len(time_axis) == 12


# Seasonal grouping tests
def test_group_by_season_standard():
    """Test with standard meteorological seasons."""
    # Create 24 months of data (2 years)
    monthly_values = np.arange(24)  # 0, 1, 2, ..., 23

    monthly_data = xr.DataArray(
        monthly_values, dims=["month"], coords={"month": np.arange(24)}
    )

    seasonal_data = group_by_season(monthly_data)

    assert isinstance(seasonal_data, xr.Dataset)
    assert "DJF" in seasonal_data
    assert "MAM" in seasonal_data
    assert "JJA" in seasonal_data
    assert "SON" in seasonal_data

    # Check DJF includes December (11), January (0), February (1), etc.
    # First year: months 11, 0, 1 -> values 11, 0, 1 -> mean = 4
    # Second year: months 23, 12, 13 -> values 23, 12, 13 -> mean = 16
    # Overall DJF mean should be (4 + 16) / 2 = 10
    djf_mean = seasonal_data["DJF"].values
    expected_djf = (11 + 0 + 1 + 23 + 12 + 13) / 6  # All DJF months
    np.testing.assert_almost_equal(djf_mean, expected_djf)


def test_group_by_season_custom():
    """Test with custom season definitions."""
    # Create 24 months of data (2 years)
    monthly_values = np.arange(24)  # 0, 1, 2, ..., 23

    monthly_data = xr.DataArray(
        monthly_values, dims=["month"], coords={"month": np.arange(24)}
    )

    custom_seasons = {
        "WINTER": [11, 0, 1, 2],  # Extended winter
        "SUMMER": [5, 6, 7, 8],  # Extended summer
    }

    seasonal_data = group_by_season(monthly_data, seasons=custom_seasons)

    assert "WINTER" in seasonal_data
    assert "SUMMER" in seasonal_data
    assert "DJF" not in seasonal_data  # Standard seasons not included


def test_group_by_season_missing_dimension():
    """Test with missing month dimension."""
    bad_data = xr.DataArray([1, 2, 3], dims=["time"])

    with pytest.raises(ValueError, match="must have a 'month' dimension"):
        group_by_season(bad_data)
