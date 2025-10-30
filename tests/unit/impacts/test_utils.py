"""
Unit tests for impacts utilities
===============================

Tests the utility functions in impacts.utils module.
"""

import pytest
import numpy as np
import xarray as xr
import warnings

from meteor.impacts.utils import (
    validate_temperature_data,
    convert_temperature_units,
    check_monthly_dimension,
    ensure_spatial_coordinates,
    calculate_global_mean,
    create_monthly_time_axis,
    group_by_season,
)


class TestValidateTemperatureData:
    """Test temperature validation function."""

    def test_valid_celsius(self):
        """Test validation with valid Celsius data."""
        data = xr.DataArray([10, 20, 30], dims=["x"])
        # Should not raise
        validate_temperature_data(data, "celsius")

    def test_valid_kelvin(self):
        """Test validation with valid Kelvin data."""
        data = xr.DataArray([283, 293, 303], dims=["x"])
        # Should not raise
        validate_temperature_data(data, "kelvin")

    def test_invalid_type(self):
        """Test validation with invalid input type."""
        with pytest.raises(
            ValueError, match="Temperature data must be an xarray.DataArray"
        ):
            validate_temperature_data([10, 20, 30], "celsius")

    def test_celsius_out_of_range(self):
        """Test Celsius data outside reasonable range."""
        data = xr.DataArray([-100, 0, 100], dims=["x"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_temperature_data(data, "celsius")
            assert len(w) > 0
            assert "outside expected range" in str(w[0].message)

    def test_kelvin_out_of_range(self):
        """Test Kelvin data outside reasonable range."""
        data = xr.DataArray([100, 200, 500], dims=["x"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_temperature_data(data, "kelvin")
            assert len(w) > 0
            assert "outside expected range" in str(w[0].message)

    def test_custom_range(self):
        """Test validation with custom temperature range."""
        data = xr.DataArray([10, 20, 30], dims=["x"])

        # Should warn with narrow custom range
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_temperature_data(data, "celsius", temp_range=(15, 25))
            assert len(w) > 0

    def test_unknown_units(self):
        """Test validation with unknown units."""
        data = xr.DataArray([10, 20, 30], dims=["x"])

        # Should raise ValueError for unknown units
        with pytest.raises(ValueError, match="Unknown temperature units: fahrenheit"):
            validate_temperature_data(data, "fahrenheit")

    def test_coordinate_detection_errors(self):
        """Test error handling in coordinate detection."""
        # Test data with coordinates that definitely won't match lat/lon patterns
        data_no_coords = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=["dimension_1", "dimension_2"],
            coords={"dimension_1": [0, 1], "dimension_2": [0, 1]},
        )

        # Should raise error when lat/lon not found (hits lines 182 or 185)
        with pytest.raises(ValueError, match="Could not find .* coordinate"):
            ensure_spatial_coordinates(data_no_coords)


class TestConvertTemperatureUnits:
    """Test temperature unit conversion."""

    def test_celsius_to_kelvin(self):
        """Test Celsius to Kelvin conversion."""
        data = xr.DataArray([0, 20, 100], dims=["x"], attrs={"units": "celsius"})
        result = convert_temperature_units(data, "celsius", "kelvin")

        expected = [273.15, 293.15, 373.15]
        np.testing.assert_array_almost_equal(result.values, expected)
        assert result.attrs["units"] == "kelvin"

    def test_kelvin_to_celsius(self):
        """Test Kelvin to Celsius conversion."""
        data = xr.DataArray([273.15, 293.15, 373.15], dims=["x"])
        result = convert_temperature_units(data, "kelvin", "celsius")

        expected = [0, 20, 100]
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_celsius_to_fahrenheit(self):
        """Test Celsius to Fahrenheit conversion."""
        data = xr.DataArray([0, 20, 100], dims=["x"])
        result = convert_temperature_units(data, "celsius", "fahrenheit")

        expected = [32, 68, 212]
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_fahrenheit_to_celsius(self):
        """Test Fahrenheit to Celsius conversion."""
        data = xr.DataArray([32, 68, 212], dims=["x"])
        result = convert_temperature_units(data, "fahrenheit", "celsius")

        expected = [0, 20, 100]
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_same_units(self):
        """Test conversion with same source and target units."""
        data = xr.DataArray([10, 20, 30], dims=["x"])
        result = convert_temperature_units(data, "celsius", "celsius")

        np.testing.assert_array_equal(result.values, data.values)
        assert result is not data  # Should be a copy

    def test_unknown_units(self):
        """Test conversion with unknown units."""
        data = xr.DataArray([10, 20, 30], dims=["x"])

        with pytest.raises(ValueError, match="Unknown source units"):
            convert_temperature_units(data, "unknown", "celsius")

        with pytest.raises(ValueError, match="Unknown target units"):
            convert_temperature_units(data, "celsius", "unknown")


class TestCheckMonthlyDimension:
    """Test monthly dimension checking."""

    def test_valid_monthly_data(self):
        """Test with valid monthly data."""
        data = xr.DataArray(np.random.rand(24), dims=["month"])
        # Should not raise
        check_monthly_dimension(data)

    def test_missing_month_dimension(self):
        """Test with missing month dimension."""
        data = xr.DataArray(np.random.rand(24), dims=["time"])

        with pytest.raises(ValueError, match="must have a 'month' dimension"):
            check_monthly_dimension(data)

    def test_short_monthly_data(self):
        """Test with less than 12 months."""
        data = xr.DataArray(np.random.rand(6), dims=["month"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_monthly_dimension(data)
            assert len(w) > 0
            assert "only 6 months" in str(w[0].message)

    def test_custom_dimension_name(self):
        """Test with custom dimension name."""
        data = xr.DataArray(np.random.rand(24), dims=["time"])
        # Should not raise when specifying correct dimension name
        check_monthly_dimension(data, dim_name="time")


class TestEnsureSpatialCoordinates:
    """Test spatial coordinate detection."""

    def test_standard_coordinates(self):
        """Test with standard lat/lon coordinates."""
        data = xr.DataArray(
            np.random.rand(10, 10),
            dims=["lat", "lon"],
            coords={"lat": np.arange(10), "lon": np.arange(10)},
        )

        lat_name, lon_name = ensure_spatial_coordinates(data)
        assert lat_name == "lat"
        assert lon_name == "lon"

    def test_alternative_coordinates(self):
        """Test with alternative coordinate names."""
        data = xr.DataArray(
            np.random.rand(10, 10),
            dims=["latitude", "longitude"],
            coords={"latitude": np.arange(10), "longitude": np.arange(10)},
        )

        lat_name, lon_name = ensure_spatial_coordinates(data)
        assert lat_name == "latitude"
        assert lon_name == "longitude"

    def test_missing_coordinates(self):
        """Test with missing spatial coordinates."""
        data = xr.DataArray(np.random.rand(10, 10), dims=["dimension1", "dimension2"])

        with pytest.raises(ValueError, match="Could not find latitude coordinate"):
            ensure_spatial_coordinates(data)


class TestCalculateGlobalMean:
    """Test global mean calculation."""

    def setup_method(self):
        """Set up test data."""
        # Create simple test data
        lats = np.array([0, 30, 60])  # Different latitudes for weighting test
        lons = np.array([0, 180])

        # Data that's latitude-dependent (higher values at equator)
        data_values = np.array([[3, 3], [2, 2], [1, 1]])  # shape: (lat, lon)

        self.test_data = xr.DataArray(
            data_values, dims=["lat", "lon"], coords={"lat": lats, "lon": lons}
        )

    def test_basic_global_mean(self):
        """Test basic global mean calculation."""
        result = calculate_global_mean(self.test_data)

        # Should be a scalar (all spatial dims removed)
        assert result.ndim == 0
        assert "operation" in result.attrs
        assert result.attrs["operation"] == "area_weighted_global_mean"

    def test_weighting_effect(self):
        """Test that latitude weighting affects results."""
        # With latitude weighting, equatorial values should have more influence
        weighted_mean = calculate_global_mean(self.test_data)

        # Simple unweighted mean
        unweighted_mean = self.test_data.mean()

        # Weighted mean should be higher due to higher equatorial values
        assert float(weighted_mean) > float(unweighted_mean)

    def test_custom_weights(self):
        """Test with custom weights."""
        # Equal weights (should give simple average)
        equal_weights = xr.ones_like(self.test_data)
        result = calculate_global_mean(self.test_data, weights=equal_weights)

        expected = self.test_data.mean()
        np.testing.assert_almost_equal(float(result), float(expected))

    def test_auto_coordinate_detection(self):
        """Test automatic coordinate detection."""
        # Rename coordinates
        renamed_data = self.test_data.rename({"lat": "latitude", "lon": "longitude"})

        result = calculate_global_mean(renamed_data)
        assert result.ndim == 0  # Spatial dims should be removed


class TestCreateMonthlyTimeAxis:
    """Test monthly time axis creation."""

    def test_basic_time_axis(self):
        """Test basic time axis creation."""
        time_axis = create_monthly_time_axis(1850, 24)  # 2 years

        assert len(time_axis) == 24
        assert time_axis[0] == 1850.0  # January 1850
        assert time_axis[11] == 1850 + 11 / 12  # December 1850
        assert time_axis[12] == 1851.0  # January 1851

        assert "start_year" in time_axis.attrs
        assert time_axis.attrs["start_year"] == 1850

    def test_custom_dimension_name(self):
        """Test with custom dimension name."""
        time_axis = create_monthly_time_axis(2000, 12, dim_name="time")

        assert time_axis.dims == ("time",)
        assert len(time_axis) == 12


class TestGroupBySeason:
    """Test seasonal grouping function."""

    def setup_method(self):
        """Set up test data."""
        # Create 24 months of data (2 years)
        monthly_values = np.arange(24)  # 0, 1, 2, ..., 23

        self.monthly_data = xr.DataArray(
            monthly_values, dims=["month"], coords={"month": np.arange(24)}
        )

    def test_standard_seasons(self):
        """Test with standard meteorological seasons."""
        seasonal_data = group_by_season(self.monthly_data)

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

    def test_custom_seasons(self):
        """Test with custom season definitions."""
        custom_seasons = {
            "WINTER": [11, 0, 1, 2],  # Extended winter
            "SUMMER": [5, 6, 7, 8],  # Extended summer
        }

        seasonal_data = group_by_season(self.monthly_data, seasons=custom_seasons)

        assert "WINTER" in seasonal_data
        assert "SUMMER" in seasonal_data
        assert "DJF" not in seasonal_data  # Standard seasons not included

    def test_missing_month_dimension(self):
        """Test with missing month dimension."""
        bad_data = xr.DataArray([1, 2, 3], dims=["time"])

        with pytest.raises(ValueError, match="must have a 'month' dimension"):
            group_by_season(bad_data)
