"""
Unit tests for ensemble processing utilities
===========================================

Tests the ensemble processing functions and utilities.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from meteor.impacts.impacts_core import ImpactCalculator, ImpactEnsemble, ImpactResult
from meteor.impacts.ensemble import (
    apply_impact_calculator,
    create_impact_ensemble,
    ensemble_statistics,
)


class MockCalculator(ImpactCalculator):
    """Mock calculator for testing."""

    def __init__(self, name="MockCalculator"):
        super().__init__(name)
        self.call_count = 0

    def calculate(self, climate_data):
        self.call_count += 1
        result_data = {
            "output": climate_data * 2,  # Simple transformation
            "constant": xr.full_like(climate_data, self.call_count),
        }
        return ImpactResult(result_data, {"call_number": self.call_count}, self.name)

    def validate_input(self, climate_data):
        if not isinstance(climate_data, xr.DataArray):
            raise ValueError("Input must be DataArray")


class TestApplyImpactCalculator:
    """Test apply_impact_calculator function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MockCalculator()

        self.single_data = xr.DataArray([1, 2, 3], dims=["x"])

        self.ensemble_list = [
            xr.DataArray([1, 2, 3], dims=["x"]),
            xr.DataArray([2, 3, 4], dims=["x"]),
            xr.DataArray([3, 4, 5], dims=["x"]),
        ]

        self.ensemble_array = xr.concat(self.ensemble_list, dim="ensemble")

    def test_single_dataarray(self):
        """Test with single DataArray."""
        result = apply_impact_calculator(self.calculator, self.single_data)

        assert isinstance(result, ImpactResult)
        assert "output" in result
        np.testing.assert_array_equal(result["output"].values, [2, 4, 6])

    def test_list_of_dataarrays(self):
        """Test with list of DataArrays (ensemble)."""
        results = apply_impact_calculator(self.calculator, self.ensemble_list)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, ImpactResult) for r in results)

        # Check that each member was processed
        np.testing.assert_array_equal(results[0]["output"].values, [2, 4, 6])
        np.testing.assert_array_equal(results[1]["output"].values, [4, 6, 8])
        np.testing.assert_array_equal(results[2]["output"].values, [6, 8, 10])

        # Check ensemble metadata was added
        for i, result in enumerate(results):
            assert result.metadata["ensemble_member"] == i
            assert result.metadata["ensemble_size"] == 3

    def test_dataarray_with_ensemble_dim(self):
        """Test with DataArray that has ensemble dimension."""
        results = apply_impact_calculator(
            self.calculator, self.ensemble_array, ensemble_dim="ensemble"
        )

        assert isinstance(results, list)
        assert len(results) == 3

        # Check processing
        np.testing.assert_array_equal(results[0]["output"].values, [2, 4, 6])
        np.testing.assert_array_equal(results[1]["output"].values, [4, 6, 8])

    def test_invalid_ensemble_dim(self):
        """Test with invalid ensemble dimension name."""
        with pytest.raises(ValueError, match="Ensemble dimension 'bad_dim' not found"):
            apply_impact_calculator(
                self.calculator, self.ensemble_array, ensemble_dim="bad_dim"
            )

    def test_invalid_input_type(self):
        """Test with invalid input type."""
        with pytest.raises(TypeError, match="must be an xarray.DataArray or list"):
            apply_impact_calculator(self.calculator, "invalid_input")

    def test_calculation_error(self):
        """Test error handling during calculation."""
        failing_calc = Mock(spec=ImpactCalculator)
        failing_calc.calculate.side_effect = RuntimeError("Calculation failed")

        with pytest.raises(
            RuntimeError, match="Failed to calculate impacts for ensemble member 0"
        ):
            apply_impact_calculator(failing_calc, self.ensemble_list)


class TestCreateImpactEnsemble:
    """Test create_impact_ensemble convenience function."""

    def test_create_ensemble(self):
        """Test creating impact ensemble."""
        calculator = MockCalculator()
        ensemble_list = [
            xr.DataArray([1, 2, 3], dims=["x"]),
            xr.DataArray([2, 3, 4], dims=["x"]),
        ]

        impact_ensemble = create_impact_ensemble(calculator, ensemble_list)

        assert isinstance(impact_ensemble, ImpactEnsemble)
        assert len(impact_ensemble.results) == 2
        assert impact_ensemble.calculator == calculator


class TestEnsembleStatistics:
    """Test ensemble_statistics function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create ensemble results with known values for testing
        self.results = []
        test_values = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

        for i, values in enumerate(test_values):
            data = {
                "test_var": xr.DataArray(values, dims=["x"]),
                "other_var": xr.DataArray([i] * 3, dims=["x"]),
            }
            result = ImpactResult(data, {"member": i}, f"Calculator{i}")
            self.results.append(result)

    def test_basic_statistics(self):
        """Test basic ensemble statistics."""
        stats = ensemble_statistics(self.results, "test_var")

        assert isinstance(stats, xr.Dataset)
        assert "ensemble_mean" in stats
        assert "ensemble_std" in stats
        assert "ensemble_min" in stats
        assert "ensemble_max" in stats

        # Check mean calculation: ([1,2,3] + [2,3,4] + [3,4,5]) / 3 = [2,3,4]
        np.testing.assert_array_equal(stats["ensemble_mean"].values, [2, 3, 4])

        # Check min/max
        np.testing.assert_array_equal(stats["ensemble_min"].values, [1, 2, 3])
        np.testing.assert_array_equal(stats["ensemble_max"].values, [3, 4, 5])

    def test_custom_statistics(self):
        """Test with custom statistics list."""
        stats = ensemble_statistics(
            self.results,
            "test_var",
            statistics=["mean", "percentile_50", "percentile_90"],
        )

        assert "ensemble_mean" in stats
        assert "ensemble_p50" in stats
        assert "ensemble_p90" in stats
        assert "ensemble_std" not in stats  # Not requested

    def test_percentile_statistics(self):
        """Test percentile statistics."""
        stats = ensemble_statistics(
            self.results,
            "test_var",
            statistics=["percentile_0", "percentile_50", "percentile_100"],
        )

        # percentile_0 should equal min, percentile_100 should equal max
        # For our test data: [1,2,3], [2,3,4], [3,4,5]
        # min = [1,2,3], p50 = [2,3,4], max = [3,4,5]
        expected_min = np.array([1, 2, 3])
        expected_median = np.array([2, 3, 4])
        expected_max = np.array([3, 4, 5])

        np.testing.assert_array_equal(stats["ensemble_p0"].values, expected_min)
        np.testing.assert_array_equal(stats["ensemble_p50"].values, expected_median)
        np.testing.assert_array_equal(stats["ensemble_p100"].values, expected_max)

    def test_empty_results(self):
        """Test with empty results list."""
        with pytest.raises(ValueError, match="No impact results provided"):
            ensemble_statistics([], "test_var")

    def test_missing_variable(self):
        """Test with missing variable name."""
        with pytest.raises(ValueError, match="Variable 'missing_var' not found"):
            ensemble_statistics(self.results, "missing_var")

    def test_invalid_statistic(self):
        """Test with invalid statistic name."""
        with pytest.raises(ValueError, match="Unknown statistic: invalid_stat"):
            ensemble_statistics(self.results, "test_var", statistics=["invalid_stat"])

    def test_invalid_percentile(self):
        """Test with invalid percentile specification."""
        with pytest.raises(ValueError, match="Invalid percentile specification"):
            ensemble_statistics(self.results, "test_var", statistics=["percentile_bad"])

    def test_dataset_attributes(self):
        """Test that output dataset has proper attributes."""
        stats = ensemble_statistics(self.results, "test_var")

        assert stats.attrs["variable"] == "test_var"
        assert stats.attrs["ensemble_size"] == 3
        assert "statistics" in stats.attrs
        assert "calculator" in stats.attrs
