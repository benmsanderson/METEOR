"""
Unit tests for base impact classes
==================================

Tests the core infrastructure: ImpactCalculator, ImpactResult, and ImpactEnsemble.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from meteor.impacts.impacts_core import ImpactCalculator, ImpactEnsemble, ImpactResult


class MockCalculator(ImpactCalculator):
    """Mock calculator for testing."""

    def __init__(self, name="MockCalculator"):
        super().__init__(name)

    def calculate(self, climate_data):
        # Simple mock calculation: square the input
        result_data = {"squared": climate_data**2, "doubled": climate_data * 2}
        return ImpactResult(result_data, {"mock": True}, self.name)

    def validate_input(self, climate_data):
        if not isinstance(climate_data, xr.DataArray):
            raise ValueError("Input must be DataArray")


# ImpactResult tests
def test_impact_result_init():
    """Test ImpactResult initialization."""
    data = {
        "test_var": xr.DataArray([1, 2, 3], dims=["x"]),
        "another_var": xr.DataArray([4, 5, 6], dims=["x"]),
    }
    metadata = {"test": "metadata"}

    result = ImpactResult(data, metadata, "TestCalculator")

    assert result.calculator_name == "TestCalculator"
    assert result.metadata == metadata
    assert "test_var" in result
    assert "another_var" in result
    np.testing.assert_array_equal(result["test_var"].values, [1, 2, 3])


def test_impact_result_getitem():
    """Test accessing variables by name."""
    data = {"var1": xr.DataArray([1, 2, 3], dims=["x"])}
    result = ImpactResult(data)

    retrieved = result["var1"]
    np.testing.assert_array_equal(retrieved.values, [1, 2, 3])


def test_impact_result_contains():
    """Test checking if variables exist."""
    data = {"var1": xr.DataArray([1, 2, 3], dims=["x"])}
    result = ImpactResult(data)

    assert "var1" in result
    assert "var2" not in result


def test_impact_result_keys():
    """Test getting variable names."""
    data = {
        "var1": xr.DataArray([1, 2, 3], dims=["x"]),
        "var2": xr.DataArray([4, 5, 6], dims=["x"]),
    }
    result = ImpactResult(data)

    keys = list(result.keys())
    assert "var1" in keys
    assert "var2" in keys
    assert len(keys) == 2


def test_impact_result_to_dataset():
    """Test conversion to xarray Dataset."""
    data = {
        "var1": xr.DataArray([1, 2, 3], dims=["x"]),
        "var2": xr.DataArray([4, 5, 6], dims=["x"]),
    }
    metadata = {"source": "test"}
    result = ImpactResult(data, metadata)

    dataset = result.to_dataset()

    assert isinstance(dataset, xr.Dataset)
    assert "var1" in dataset
    assert "var2" in dataset
    assert dataset.attrs["source"] == "test"


def test_impact_result_dict_conversion():
    """Test ImpactResult dict conversion by accessing data directly."""
    test_data = {"test_var": np.array([1, 2, 3])}
    result = ImpactResult(data=test_data, metadata={}, calculator_name="test")
    result_dict = dict(result)
    assert "test_var" in result_dict
    assert np.array_equal(result_dict["test_var"], np.array([1, 2, 3]))


# ImpactCalculator tests


def test_impact_calculator_mock_implementation():
    """Test using mock calculator implementation."""
    calc = MockCalculator()

    # Test string representation
    assert "MockCalculator" in str(calc)
    assert "MockCalculator" in repr(calc)

    # Test calculation
    test_data = xr.DataArray([1, 2, 3], dims=["x"])
    result = calc.calculate(test_data)

    assert isinstance(result, ImpactResult)
    assert "squared" in result
    assert "doubled" in result
    np.testing.assert_array_equal(result["squared"].values, [1, 4, 9])
    np.testing.assert_array_equal(result["doubled"].values, [2, 4, 6])


def test_impact_calculator_validation():
    """Test input validation."""
    calc = MockCalculator()

    # Valid input should not raise
    valid_data = xr.DataArray([1, 2, 3], dims=["x"])
    calc.validate_input(valid_data)

    # Invalid input should raise
    with pytest.raises(ValueError):
        calc.validate_input([1, 2, 3])  # Not a DataArray


class TestImpactEnsemble:
    """Test ImpactEnsemble class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MockCalculator()
        self.test_ensemble = [
            xr.DataArray([1, 2, 3], dims=["x"]),
            xr.DataArray([2, 3, 4], dims=["x"]),
            xr.DataArray([3, 4, 5], dims=["x"]),
        ]

    def test_init(self):
        """Test ImpactEnsemble initialization."""
        ensemble = ImpactEnsemble(self.calculator)

        assert ensemble.calculator == self.calculator
        assert ensemble.results == []

    def test_calculate_ensemble(self):
        """Test calculating impacts for ensemble."""
        ensemble = ImpactEnsemble(self.calculator)
        results = ensemble.calculate_ensemble(self.test_ensemble)

        assert len(results) == 3
        assert all(isinstance(r, ImpactResult) for r in results)

        # Check metadata was added
        for i, result in enumerate(results):
            assert result.metadata["ensemble_member"] == i
            assert result.metadata["ensemble_size"] == 3

        # Check calculations
        np.testing.assert_array_equal(results[0]["squared"].values, [1, 4, 9])
        np.testing.assert_array_equal(results[1]["squared"].values, [4, 9, 16])

    def test_ensemble_mean(self):
        """Test ensemble mean calculation."""
        ensemble = ImpactEnsemble(self.calculator)
        ensemble.calculate_ensemble(self.test_ensemble)

        mean_squared = ensemble.ensemble_mean("squared")

        # Expected: mean of [1,4,9], [4,9,16], [9,16,25] = [4.67, 9.67, 16.67]
        expected = np.array([14 / 3, 29 / 3, 50 / 3])
        np.testing.assert_array_almost_equal(mean_squared.values, expected)

    def test_ensemble_std(self):
        """Test ensemble standard deviation calculation."""
        ensemble = ImpactEnsemble(self.calculator)
        ensemble.calculate_ensemble(self.test_ensemble)

        std_squared = ensemble.ensemble_std("squared")

        # Should have 3 elements matching ensemble shape
        assert len(std_squared.values) == 3
        assert all(std_squared.values > 0)  # Standard deviation should be positive

    def test_ensemble_percentiles(self):
        """Test ensemble percentile calculation."""
        ensemble = ImpactEnsemble(self.calculator)
        ensemble.calculate_ensemble(self.test_ensemble)

        # Single percentile
        p50 = ensemble.ensemble_percentiles("squared", 50)
        assert isinstance(p50, xr.DataArray)
        assert len(p50.values) == 3

        # Multiple percentiles
        percentiles = ensemble.ensemble_percentiles("squared", [25, 50, 75])
        assert isinstance(percentiles, dict)
        assert 25 in percentiles
        assert 50 in percentiles
        assert 75 in percentiles

    def test_to_dataset(self):
        """Test conversion to Dataset."""
        ensemble = ImpactEnsemble(self.calculator)
        ensemble.calculate_ensemble(self.test_ensemble)

        dataset = ensemble.to_dataset()

        assert isinstance(dataset, xr.Dataset)
        assert "squared" in dataset
        assert "doubled" in dataset
        assert "ensemble_member" in dataset.dims
        assert dataset.sizes["ensemble_member"] == 3

    def test_errors(self):
        """Test error handling."""
        ensemble = ImpactEnsemble(self.calculator)

        # Error when no results
        with pytest.raises(ValueError, match="No ensemble results"):
            ensemble.ensemble_mean("squared")

        # Error with invalid variable name
        ensemble.calculate_ensemble(self.test_ensemble)
        with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
            ensemble.ensemble_mean("nonexistent")

    def test_calculation_error(self):
        """Test error handling during calculation."""
        # Create a calculator that fails
        failing_calc = Mock(spec=ImpactCalculator)
        failing_calc.calculate.side_effect = RuntimeError("Calculation failed")

        ensemble = ImpactEnsemble(failing_calc)

        with pytest.raises(
            RuntimeError, match="Failed to calculate impacts for ensemble member 0"
        ):
            ensemble.calculate_ensemble(self.test_ensemble)

    def test_additional_edge_cases(self):
        """Test additional edge cases to improve coverage."""
        # Test ensemble error handling for empty results
        ensemble = ImpactEnsemble(self.calculator)
        with pytest.raises(ValueError, match="No ensemble results available"):
            ensemble.to_dataset()

        # Test percentiles with basic data
        ensemble.calculate_ensemble(self.test_ensemble)

        # Should handle basic percentile calculation
        p50 = ensemble.ensemble_percentiles("squared", 50)
        assert isinstance(p50, xr.DataArray)

        # Test dataset conversion after calculation
        dataset = ensemble.to_dataset()
        assert isinstance(dataset, xr.Dataset)
        assert len(dataset.data_vars) > 0

    def test_impact_ensemble_error_conditions(self):
        """Test error handling in ImpactEnsemble methods to hit missing lines."""
        # Create ensemble with dummy calculator
        dummy_calculator = Mock()
        ensemble = ImpactEnsemble(dummy_calculator)

        # Test error when no results available (lines 197, 222, 262)
        with pytest.raises(ValueError, match="No ensemble results available"):
            ensemble.ensemble_percentiles("temperature", 50)

        # Test to_dataset with no results (should hit line 251 in to_dataset)
        with pytest.raises(ValueError, match="No ensemble results available"):
            ensemble.to_dataset(["temperature"])

        # Manually add some results to test variable not found errors
        dummy_data = {"other_var": xr.DataArray([1, 2, 3], dims=["x"])}
        dummy_result = ImpactResult(dummy_data, metadata={"test": "metadata"})
        ensemble.results = [dummy_result]  # Directly set results

        # Test variable not found errors (lines 202, 227)
        with pytest.raises(ValueError, match="Variable 'temperature' not found"):
            ensemble.ensemble_percentiles("temperature", 50)

        # Test to_dataset variable not found (should hit line 262)
        with pytest.raises(ValueError, match="Variable 'temperature' not found"):
            ensemble.to_dataset(["temperature"])
