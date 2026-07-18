"""
Unit tests for degree days calculator
=====================================

Tests the DegreeDaysCalculator implementation.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

from meteor.impacts.calculators.degree_days import (
    DegreeDaysCalculator,
    validate_temperature_input_and_convert,
)
from meteor.impacts.impacts_core import ImpactResult


def test_degree_days_calculator_init():
    """Test calculator initialization."""
    calc = DegreeDaysCalculator(base_temperature=20.0, sigma_m_c1=1.5, name="TestDD")

    assert calc.base_temperature == 20.0
    assert calc.sigma_m_c1 == 1.5
    assert calc.name == "TestDD"


def test_validate_temperature_input_and_convert():
    """Test validation with suspicious temperature values."""
    # Very hot temperatures that would be unusual even for Kelvin
    hot_data = xr.DataArray(
        [450, 460, 470], dims=["month"]
    )  # ~177-197°C, unusual even for Kelvin

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        converted = validate_temperature_input_and_convert(hot_data)
        assert len(w) > 0, "Should warn about high temperature values"
        assert converted.equals(
            hot_data - 273.15
        ), "Should convert from Kelvin to Celsius"

    # Mixed range that suggests unit confusion
    mixed_data = xr.DataArray(
        [50, 150, 250], dims=["month"]
    )  # Mixed range triggering mixed warning

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        converted = validate_temperature_input_and_convert(mixed_data)
        assert len(w) > 0, "Should warn about mixed/unusual temperature values"
        assert "unusual" in str(w[0].message)
        assert converted.equals(mixed_data), "Should not convert mixed data"

    # Very cold temperatures
    cold_data = xr.DataArray([-200, -150, -100], dims=["month"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        converted = validate_temperature_input_and_convert(cold_data)
        assert len(w) > 0, "Should warn about very cold temperature values"
        assert "unusual" in str(w[0].message)
        assert converted.equals(cold_data), "Should not convert cold data"

    float_data = 300.0  # Single float value
    assert np.allclose(
        validate_temperature_input_and_convert(float_data), 26.85
    ), "Should convert float Kelvin to Celsius"

    float_data = -50.0  # Single float value in Celsius range
    assert (
        validate_temperature_input_and_convert(float_data) == -50.0
    ), "Should not convert float Celsius"

    with pytest.raises(ValueError, match="Input must be an xarray.DataArray or float"):
        validate_temperature_input_and_convert("not a data array or float")


def test_degree_days_calculate_batches_realizations():
    """Passing a 2D ``(realization, month)`` DataArray must produce the same
    per-realization output as looping ``.calculate()`` over each row.

    The MeteorInterface impact path relies on this: it avoids the expensive
    per-realization Python loop by batching the whole ensemble into one call.
    """
    calc = DegreeDaysCalculator(base_temperature=18.0)

    rng = np.random.default_rng(0)
    n_real, n_month = 8, 12 * 30  # 30 years
    t = np.arange(n_month)
    temps = (
        15.0
        + 8.0 * np.sin(2 * np.pi * t / 12.0)[None, :]
        + 0.4 * rng.standard_normal((n_real, n_month))
    )
    ds = xr.DataArray(
        temps, dims=["realization", "month"], coords={"month": t}
    )

    # Per-realization loop (reference)
    serial_hdd = np.stack(
        [calc.calculate(ds[i]).data["annual_hdd"].values for i in range(n_real)]
    )
    serial_cdd = np.stack(
        [calc.calculate(ds[i]).data["annual_cdd"].values for i in range(n_real)]
    )

    # Batched
    result = calc.calculate(ds)
    batched_hdd = result.data["annual_hdd"].values
    batched_cdd = result.data["annual_cdd"].values

    assert batched_hdd.shape == serial_hdd.shape
    assert batched_cdd.shape == serial_cdd.shape
    np.testing.assert_allclose(batched_hdd, serial_hdd, rtol=0, atol=1e-9)
    np.testing.assert_allclose(batched_cdd, serial_cdd, rtol=0, atol=1e-9)


class TestDegreeDaysCalculator:
    """Test DegreeDaysCalculator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = DegreeDaysCalculator(base_temperature=18.0)

        # Create test data: 24 months of temperature data (2 years)
        # Monthly pattern: cold winter, warm summer
        monthly_temps = np.array(
            [
                # Year 1
                -5,
                -3,
                2,
                8,
                15,
                22,
                25,
                24,
                18,
                12,
                5,
                0,
                # Year 2
                -4,
                -2,
                3,
                9,
                16,
                23,
                26,
                25,
                19,
                13,
                6,
                1,
            ]
        )

        self.test_data = xr.DataArray(
            monthly_temps,
            dims=["month"],
            coords={"month": np.arange(24)},
            attrs={"units": "celsius"},
        )

        # Create spatial test data (simple 2x2 grid)
        self.spatial_data = xr.DataArray(
            np.random.normal(10, 5, (24, 2, 2)),  # 24 months, 2x2 grid
            dims=["month", "lat", "lon"],
            coords={"month": np.arange(24), "lat": [50, 60], "lon": [0, 10]},
        )

    def test_validate_input_valid(self):
        """Test validation with valid input."""
        # Should not raise any exceptions
        self.calculator.validate_input(self.test_data)

    def test_validate_input_invalid_type(self):
        """Test validation with invalid input type."""
        with pytest.raises(ValueError, match="Input must be an xarray.DataArray"):
            self.calculator.validate_input([1, 2, 3])

    def test_validate_input_missing_month_dim(self):
        """Test validation with missing month dimension."""
        bad_data = xr.DataArray([1, 2, 3], dims=["time"])

        with pytest.raises(ValueError, match="must have a dimension named 'month'"):
            self.calculator.validate_input(bad_data)

    def test_calculate_basic(self):
        """Test basic degree days calculation."""
        result = self.calculator.calculate(self.test_data)

        assert isinstance(result, ImpactResult)
        assert result.calculator_name == "DegreeDays"

        # Check all expected variables are present
        expected_vars = ["monthly_hdd", "monthly_cdd", "annual_hdd", "annual_cdd"]
        for var in expected_vars:
            assert var in result, f"Missing variable: {var}"

        # Check metadata
        assert result.metadata["base_temperature"] == 18.0
        assert "method" in result.metadata
        assert "reference" in result.metadata

    def test_monthly_results_shape(self):
        """Test that monthly results have correct shape."""
        result = self.calculator.calculate(self.test_data)

        monthly_hdd = result["monthly_hdd"]
        monthly_cdd = result["monthly_cdd"]

        # Should have same shape as input
        assert monthly_hdd.shape == self.test_data.shape
        assert monthly_cdd.shape == self.test_data.shape

        # Should have month dimension
        assert "month" in monthly_hdd.dims
        assert "month" in monthly_cdd.dims

    def test_annual_results_shape(self):
        """Test that annual results have correct shape."""
        result = self.calculator.calculate(self.test_data)

        annual_hdd = result["annual_hdd"]
        annual_cdd = result["annual_cdd"]

        # Should have 2 years of data
        expected_years = self.test_data.sizes["month"] // 12
        assert annual_hdd.shape == (expected_years,)
        assert annual_cdd.shape == (expected_years,)
        assert "year" in annual_hdd.dims  # Should have year dimension

    def test_degree_days_logic(self):
        """Test that degree days logic is correct."""
        result = self.calculator.calculate(self.test_data)

        monthly_hdd = result["monthly_hdd"]
        monthly_cdd = result["monthly_cdd"]

        # When temperature < base_temperature, should have HDD but not CDD
        cold_months = self.test_data < self.calculator.base_temperature
        hot_months = self.test_data > self.calculator.base_temperature

        # Check that cold months have HDD > 0 and CDD = 0
        cold_hdd = monthly_hdd.where(cold_months, drop=False)
        cold_cdd = monthly_cdd.where(cold_months, drop=False)

        # Most cold months should have some HDD
        assert (cold_hdd > 0).sum() > 0
        # All cold months should have CDD = 0 (where cold_months is True)
        cold_cdd_values = cold_cdd.where(cold_months).values
        # Remove NaN values (where cold_months is False)
        cold_cdd_actual = cold_cdd_values[~np.isnan(cold_cdd_values)]
        np.testing.assert_array_equal(cold_cdd_actual, 0)

        # Check that hot months have CDD > 0 and HDD = 0
        hot_hdd = monthly_hdd.where(hot_months, drop=False)
        hot_cdd = monthly_cdd.where(hot_months, drop=False)

        # All hot months should have HDD = 0 (where hot_months is True)
        hot_hdd_values = hot_hdd.where(hot_months).values
        # Remove NaN values (where hot_months is False)
        hot_hdd_actual = hot_hdd_values[~np.isnan(hot_hdd_values)]
        np.testing.assert_array_equal(hot_hdd_actual, 0)
        # Most hot months should have some CDD
        hot_cdd_actual = hot_cdd.where(hot_months).values
        hot_cdd_actual = hot_cdd_actual[~np.isnan(hot_cdd_actual)]
        assert (hot_cdd_actual > 0).sum() > 0

    def test_spatial_data(self):
        """Test calculation with spatial data."""
        result = self.calculator.calculate(self.spatial_data)

        monthly_hdd = result["monthly_hdd"]
        _ = result["monthly_cdd"]  # noqa: F841

        # Should preserve spatial dimensions
        assert "lat" in monthly_hdd.dims
        assert "lon" in monthly_hdd.dims
        assert monthly_hdd.shape == self.spatial_data.shape

        # Annual data should have spatial dimensions but not month
        annual_hdd = result["annual_hdd"]
        assert "lat" in annual_hdd.dims
        assert "lon" in annual_hdd.dims
        assert "month" not in annual_hdd.dims
        assert "year" in annual_hdd.dims  # Should have year dimension instead

    def test_attributes(self):
        """Test that output variables have proper attributes."""
        result = self.calculator.calculate(self.test_data)

        for var_name in ["monthly_hdd", "monthly_cdd", "annual_hdd", "annual_cdd"]:
            var = result[var_name]

            assert hasattr(var, "attrs")
            assert "long_name" in var.attrs
            assert "units" in var.attrs
            assert "base_temperature" in var.attrs
            assert var.attrs["units"] == "degree-days"
            assert "18.0°C" in var.attrs["base_temperature"]

    def test_different_base_temperature(self):
        """Test calculation with different base temperature."""
        calc_15 = DegreeDaysCalculator(base_temperature=15.0)
        calc_21 = DegreeDaysCalculator(base_temperature=21.0)

        result_15 = calc_15.calculate(self.test_data)
        result_21 = calc_21.calculate(self.test_data)

        # Lower base temperature should generally result in more CDD, less HDD
        hdd_15 = result_15["annual_hdd"].sum()
        hdd_21 = result_21["annual_hdd"].sum()
        cdd_15 = result_15["annual_cdd"].sum()
        cdd_21 = result_21["annual_cdd"].sum()

        assert hdd_15 < hdd_21, "Lower base temp should have less HDD"
        assert cdd_15 > cdd_21, "Lower base temp should have more CDD"

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with temperature exactly at base temperature
        exact_temp = xr.DataArray([18.0] * 12, dims=["month"])
        result = self.calculator.calculate(exact_temp)

        # Should not crash and should return valid structure
        assert "monthly_hdd" in result
        assert "monthly_cdd" in result
        assert "annual_hdd" in result
        assert "annual_cdd" in result

        # All values should be non-negative
        assert (result["monthly_hdd"] >= 0).all()
        assert (result["monthly_cdd"] >= 0).all()
        assert (result["annual_hdd"] >= 0).all()
        assert (result["annual_cdd"] >= 0).all()

        # Test with more realistic temperature variation that would clearly
        # produce degree days (well above and below base temperature)
        varied_temp = xr.DataArray(
            [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 32.0, 30.0, 25.0, 20.0, 15.0, 10.0],
            dims=["month"],
        )
        result2 = self.calculator.calculate(varied_temp)

        # Should have both heating and cooling degree days
        total_hdd = result2["annual_hdd"].sum()
        total_cdd = result2["annual_cdd"].sum()

        assert total_hdd > 0, "Should have heating degree days in winter months"
        assert (
            total_cdd > 0
        ), "Should have cooling degree days in summer months"  # Test with very short time series
        short_data = xr.DataArray([10.0, 20.0, 15.0], dims=["month"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.calculator.calculate(short_data)
            # Should warn about incomplete year
            assert any(
                "less than a full 12-month cycle" in str(warning.message)
                for warning in w
            )

    def test_parameter_effects(self):
        """Test that different parameters affect results."""
        default_calc = DegreeDaysCalculator()
        modified_calc = DegreeDaysCalculator(
            sigma_m_c1=2.0,  # Different from default 1.45
            sigma_m_c2=0.5,  # Different from default 0.29
        )

        result_default = default_calc.calculate(self.test_data)
        result_modified = modified_calc.calculate(self.test_data)

        # Results should be different (not exactly equal)
        hdd_default = result_default["annual_hdd"].sum()
        hdd_modified = result_modified["annual_hdd"].sum()

        assert not np.isclose(
            hdd_default, hdd_modified
        ), "Different parameters should give different results"

    def test_temperature_validation_edge_cases(self):
        """Test temperature validation with edge cases to hit lines 102, 262-268."""

        # Test very high temperature values that trigger warning (line 102)
        # Need all 12 months to avoid the incomplete year warning
        # Use simple high values - all 150 to trigger max > 100 warning quickly
        high_temp_data = xr.DataArray(
            [150] * 12,  # Simple high values that will trigger max > 100 warning
            dims=["month"],
            coords={"month": range(1, 13)},
        )

        calculator = DegreeDaysCalculator(base_temperature=18.0)

        # Should trigger warning for unusual temperature values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculator.calculate(high_temp_data)

            # Check that warning was raised about unusual temperatures
            warning_messages = [str(warning.message).lower() for warning in w]
            assert any(
                "unusual" in msg for msg in warning_messages
            ), f"Expected 'unusual' warning, got: {warning_messages}"

        # Should still produce valid results
        assert "annual_hdd" in result.data
        assert "annual_cdd" in result.data
