"""
Integration tests for METEOR-impacts workflow
============================================

Tests the complete workflow from METEOR outputs through impact calculations.
"""

import pytest
import numpy as np
import xarray as xr

from meteor.impacts import DegreeDaysCalculator, ImpactEnsemble
from meteor.impacts.ensemble import apply_impact_calculator, ensemble_statistics


class TestMeteorImpactsWorkflow:
    """Test complete METEOR-impacts workflow."""

    def setup_method(self):
        """Set up test data mimicking METEOR outputs."""
        # Create synthetic METEOR-like output
        # Simulate 10 years of monthly temperature data on a small spatial grid
        n_months = 120  # 10 years
        n_lat = 3
        n_lon = 4

        # Create realistic temperature patterns
        lats = np.array([40, 50, 60])  # Northern hemisphere
        lons = np.array([0, 10, 20, 30])
        months = np.arange(n_months)

        # Base temperature varies with latitude (colder at higher latitudes)
        base_temp = 15 - 0.5 * (lats - 50)  # 17.5°C at 40N, 12.5°C at 60N

        # Seasonal cycle (amplitude decreases with latitude)
        seasonal_amplitude = 15 - 0.2 * (lats - 40)  # Larger cycles at higher latitudes

        # Create temperature field
        temperature_data = np.zeros((n_months, n_lat, n_lon))

        for i, month in enumerate(months):
            # Seasonal cycle (peaks in July, month 6)
            seasonal_cycle = seasonal_amplitude * np.cos(
                2 * np.pi * (month % 12 - 6) / 12
            )

            # Add some warming trend (0.5°C per decade)
            warming_trend = 0.05 * (month / 12)

            # Broadcast across spatial dimensions
            monthly_temp = (
                base_temp[:, np.newaxis] + seasonal_cycle[:, np.newaxis] + warming_trend
            )

            # Add small random variations
            monthly_temp = monthly_temp + np.random.normal(0, 1, (n_lat, n_lon))
            temperature_data[i, :, :] = monthly_temp

        # Convert to Celsius (METEOR typically outputs in Kelvin)
        self.temperature_kelvin = xr.DataArray(
            temperature_data + 273.15,
            dims=["month", "lat", "lon"],
            coords={"month": months, "lat": lats, "lon": lons},
            attrs={"units": "K", "long_name": "Temperature"},
        )

        self.temperature_celsius = self.temperature_kelvin - 273.15
        self.temperature_celsius.attrs = {"units": "C", "long_name": "Temperature"}

        # Create ensemble by adding different noise realizations
        self.ensemble_size = 5
        self.temperature_ensemble = []

        for i in range(self.ensemble_size):
            # Add different random noise to each ensemble member
            noise = np.random.normal(0, 0.5, temperature_data.shape)
            ensemble_member = self.temperature_celsius + noise
            ensemble_member.attrs = self.temperature_celsius.attrs
            self.temperature_ensemble.append(ensemble_member)

    def test_single_realization_workflow(self):
        """Test impact calculation for single realization."""
        # Initialize calculator
        calculator = DegreeDaysCalculator(base_temperature=18.0)

        # Calculate impacts
        result = calculator.calculate(self.temperature_celsius)

        # Verify structure of results
        assert "monthly_hdd" in result
        assert "monthly_cdd" in result
        assert "annual_hdd" in result
        assert "annual_cdd" in result

        # Check spatial and temporal dimensions
        monthly_hdd = result["monthly_hdd"]
        assert monthly_hdd.shape == self.temperature_celsius.shape
        assert "lat" in monthly_hdd.dims
        assert "lon" in monthly_hdd.dims
        assert "month" in monthly_hdd.dims

        annual_hdd = result["annual_hdd"]
        expected_years = self.temperature_celsius.sizes["month"] // 12
        assert annual_hdd.shape == (
            expected_years,
            len(self.temperature_celsius.lat),
            len(self.temperature_celsius.lon),
        )

        # Verify physical reasonableness
        # At higher latitudes (colder), should have more HDD
        lat_means = annual_hdd.mean(
            dim=["year", "lon"]
        )  # Average over years and longitude
        assert lat_means.isel(lat=2).values > lat_means.isel(lat=0).values  # 60N > 40N

    def test_ensemble_workflow(self):
        """Test impact calculation for ensemble."""
        calculator = DegreeDaysCalculator(base_temperature=18.0)

        # Apply calculator to ensemble
        ensemble_results = apply_impact_calculator(
            calculator, self.temperature_ensemble
        )

        assert len(ensemble_results) == self.ensemble_size
        assert all(
            isinstance(r.data["monthly_hdd"], xr.DataArray) for r in ensemble_results
        )

        # Check ensemble metadata
        for i, result in enumerate(ensemble_results):
            assert result.metadata["ensemble_member"] == i
            assert result.metadata["ensemble_size"] == self.ensemble_size

    def test_impact_ensemble_class(self):
        """Test using ImpactEnsemble class."""
        calculator = DegreeDaysCalculator(base_temperature=18.0)
        ensemble = ImpactEnsemble(calculator)

        # Calculate ensemble
        results = ensemble.calculate_ensemble(self.temperature_ensemble)

        assert len(results) == self.ensemble_size

        # Test ensemble statistics
        mean_hdd = ensemble.ensemble_mean("annual_hdd")
        std_hdd = ensemble.ensemble_std("annual_hdd")

        # Check shapes
        expected_shape = results[0]["annual_hdd"].shape
        assert mean_hdd.shape == expected_shape
        assert std_hdd.shape == expected_shape

        # Standard deviation should be positive
        assert (std_hdd > 0).all()

        # Test percentiles
        percentiles = ensemble.ensemble_percentiles("annual_hdd", [25, 50, 75])
        assert isinstance(percentiles, dict)
        assert 25 in percentiles
        assert 50 in percentiles
        assert 75 in percentiles

        # P25 <= P50 <= P75
        assert (percentiles[25] <= percentiles[50]).all()
        assert (percentiles[50] <= percentiles[75]).all()

    def test_ensemble_statistics_function(self):
        """Test standalone ensemble_statistics function."""
        calculator = DegreeDaysCalculator(base_temperature=18.0)
        ensemble_results = apply_impact_calculator(
            calculator, self.temperature_ensemble
        )

        # Calculate statistics
        stats = ensemble_statistics(
            ensemble_results,
            "annual_hdd",
            statistics=["mean", "std", "quantile_10", "quantile_90"],
        )

        assert isinstance(stats, xr.Dataset)
        assert "ensemble_mean" in stats
        assert "ensemble_std" in stats
        assert "ensemble_p10" in stats
        assert "ensemble_p90" in stats

        # Check attributes
        assert stats.attrs["variable"] == "annual_hdd"
        assert stats.attrs["ensemble_size"] == self.ensemble_size

    def test_convert_to_dataset(self):
        """Test converting ensemble results to unified dataset."""
        calculator = DegreeDaysCalculator(base_temperature=18.0)
        ensemble = ImpactEnsemble(calculator)
        ensemble.calculate_ensemble(self.temperature_ensemble)

        # Convert to dataset
        dataset = ensemble.to_dataset(["monthly_hdd", "monthly_cdd"])

        assert isinstance(dataset, xr.Dataset)
        assert "monthly_hdd" in dataset
        assert "monthly_cdd" in dataset
        assert "ensemble_member" in dataset.dims
        assert dataset.sizes["ensemble_member"] == self.ensemble_size

        # All spatial and temporal dimensions should be preserved
        assert "lat" in dataset.dims
        assert "lon" in dataset.dims
        assert "month" in dataset.dims

    def test_different_base_temperatures(self):
        """Test sensitivity to base temperature parameter."""
        base_temps = [15.0, 18.0, 21.0]
        results = {}

        for base_temp in base_temps:
            calc = DegreeDaysCalculator(base_temperature=base_temp)
            result = calc.calculate(self.temperature_celsius)
            results[base_temp] = result

        # Lower base temperature should generally result in more CDD, less HDD
        total_hdd_15 = results[15.0]["annual_hdd"].sum()
        total_hdd_21 = results[21.0]["annual_hdd"].sum()
        total_cdd_15 = results[15.0]["annual_cdd"].sum()
        total_cdd_21 = results[21.0]["annual_cdd"].sum()

        assert total_hdd_15 < total_hdd_21, "Lower base temp should have less HDD"
        assert total_cdd_15 > total_cdd_21, "Lower base temp should have more CDD"

    def test_temperature_unit_conversion(self):
        """Test that calculator works with both Kelvin and Celsius."""
        # Test with Celsius
        calc_celsius = DegreeDaysCalculator(base_temperature=18.0)
        result_celsius = calc_celsius.calculate(self.temperature_celsius)

        # Test with Kelvin (adjust base temperature accordingly)
        calc_kelvin = DegreeDaysCalculator(base_temperature=18.0 + 273.15)
        result_kelvin = calc_kelvin.calculate(self.temperature_kelvin)

        # Results should be very similar (allowing for small numerical differences)
        hdd_celsius = result_celsius["annual_hdd"]
        hdd_kelvin = result_kelvin["annual_hdd"]
        cdd_celsius = result_celsius["annual_cdd"]
        cdd_kelvin = result_kelvin["annual_cdd"]

        # Use relative tolerance due to different numerical paths
        # Only compare where both have non-zero values to avoid issues with zeros
        mask_hdd = (hdd_celsius > 1) & (hdd_kelvin > 1)
        mask_cdd = (cdd_celsius > 1) & (cdd_kelvin > 1)

        if mask_hdd.any():
            np.testing.assert_allclose(
                hdd_celsius.values[mask_hdd],
                hdd_kelvin.values[mask_hdd],
                rtol=1e-2,
                err_msg="HDD results should be similar regardless of input temperature units",
            )

        if mask_cdd.any():
            np.testing.assert_allclose(
                cdd_celsius.values[mask_cdd],
                cdd_kelvin.values[mask_cdd],
                rtol=1e-2,
                err_msg="CDD results should be similar regardless of input temperature units",
            )

    def test_realistic_values(self):
        """Test that calculated degree days are in realistic ranges."""
        calculator = DegreeDaysCalculator(base_temperature=18.0)
        result = calculator.calculate(self.temperature_celsius)

        annual_hdd = result["annual_hdd"]
        annual_cdd = result["annual_cdd"]

        # Basic sanity checks
        assert (annual_hdd >= 0).all(), "HDD should be non-negative"
        assert (annual_cdd >= 0).all(), "CDD should be non-negative"

        # For our temperature range and base temperature, we should have reasonable values
        # HDD should be larger at higher latitudes
        hdd_by_lat = annual_hdd.mean(dim=["year", "lon"])
        assert hdd_by_lat.isel(lat=-1) > hdd_by_lat.isel(
            lat=0
        ), "More HDD at higher latitudes"

        # Total degree days should be in reasonable range (hundreds to thousands)
        total_hdd = annual_hdd.mean()
        total_cdd = annual_cdd.mean()
        assert 100 < total_hdd < 10000, f"HDD ({total_hdd}) in reasonable range"
        assert 0 < total_cdd < 5000, f"CDD ({total_cdd}) in reasonable range"
