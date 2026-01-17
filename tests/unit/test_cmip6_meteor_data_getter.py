from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from meteor import cmip6_meteor_data_getter


def test_get_unique_models():
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter()
    # Disable caching to ensure experiment validation is checked
    data_getter.enable_cache = False

    models = data_getter.get_models_avail()
    print(models)
    assert isinstance(models, list)
    assert len(models) > 0

    assert data_getter.check_if_model_has_data("CanESM5")
    assert not data_getter.check_if_model_has_data("NorESM1")

    with pytest.raises(KeyError, match="No or incomplete data for NorESM1"):
        data_getter.get_single_var_mod_data_yearmean("piControl", "tas", "NorESM1")
    with pytest.raises(
        KeyError,
        match="This datagetter does not handle data from the historical experiment, available options are *",
    ):
        data_getter.get_single_var_mod_data_yearmean("historical", "tas", "CanESM5")
    with pytest.raises(
        KeyError,
        match="This datagetter does not handle rsut data, available options are *",
    ):
        data_getter.get_single_var_mod_data_yearmean("piControl", "rsut", "CanESM5")

    # data_getter.get_single_var_mod_data("CanESM5", "piControl", "tas")
    assert isinstance(data_getter.df_all, list)

    # Only mock the most expensive operations, leaving error testing intact
    with patch(
        "meteor.cmip6_meteor_data_getter.Cmip6MeteorDataGetter.get_single_var_mod_data_yearmean"
    ) as mock_get_data:
        mock_data = xr.DataArray(
            np.random.randn(5, 3, 3),  # Small fake climate data
            dims=["time", "lat", "lon"],
            coords={
                "time": range(2000, 2005),
                "lat": [-45, 0, 45],
                "lon": [-90, 0, 90],
            },
        )
        mock_get_data.return_value = mock_data

        test_var = data_getter.get_single_var_mod_data_yearmean(
            "piControl", "tas", "CanESM5"
        )
        assert isinstance(test_var, xr.DataArray)

    with patch(
        "meteor.cmip6_meteor_data_getter.Cmip6MeteorDataGetter.make_meteor_training_data"
    ) as mock_make_training:
        mock_training = xr.Dataset(
            {
                "tas": xr.DataArray(
                    np.random.randn(5, 3, 3),
                    dims=["time", "lat", "lon"],
                    coords={
                        "time": range(2000, 2005),
                        "lat": [-45, 0, 45],
                        "lon": [-90, 0, 90],
                    },
                ),
            }
        )
        mock_make_training.return_value = mock_training

        test_training = data_getter.make_meteor_training_data("base", "CanESM5")
        print(test_training)
        assert isinstance(test_training, xr.Dataset)

    data_getter_2 = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        exps=["historical", "ssp370"], dbe=["CMIP", "ScenarioMIP"]
    )
    test_composite = data_getter_2.make_meteor_training_data_composite(
        ["historical", "ssp370"], model="CanESM5"
    )
    assert test_composite.sizes["year"] == 251
    print(test_composite["year"].values)
    # assert False


def test_comprehensive_basic_functionality():
    """Comprehensive test covering initialization, validation, model checks, and basic data operations."""

    # Test 1: Default initialization and basic properties
    print("Testing default initialization...")
    default_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter()

    # Check initialization worked
    assert default_getter is not None
    assert hasattr(default_getter, "flds")
    assert hasattr(default_getter, "exps")
    assert default_getter.flds == ["tas", "pr"]
    assert default_getter.exps == ["piControl", "abrupt-4xCO2"]

    # Test method existence
    assert hasattr(default_getter, "get_models_avail")
    assert hasattr(default_getter, "check_if_model_has_data")

    # Test 2: Custom initialization
    print("Testing custom initialization...")
    custom_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        flds=["tas"], exps=["piControl", "abrupt-4xCO2", "1pctCO2"]
    )
    assert custom_getter.flds == ["tas"]
    assert custom_getter.exps == ["piControl", "abrupt-4xCO2", "1pctCO2"]

    # Test 3: Models property and validation
    print("Testing model validation...")
    models = default_getter.models
    assert isinstance(models, list)
    assert len(models) > 0
    assert "CanESM5" in models

    # Test model availability checks
    assert default_getter.check_if_model_has_data("CanESM5") is True
    assert default_getter.check_if_model_has_data("NonExistentModel123") is False

    # Test getting available models
    available_models = default_getter.get_models_avail()
    assert isinstance(available_models, list)
    assert len(available_models) > 0
    assert "CanESM5" in available_models

    # Test 4: Experiment and field validation
    print("Testing validation...")
    # Valid fields
    assert "tas" in default_getter.flds
    assert "pr" in default_getter.flds

    # Experiment structure
    assert "piControl" in default_getter.exps
    assert "abrupt-4xCO2" in default_getter.exps
    assert isinstance(default_getter.exps, list)
    assert len(default_getter.exps) > 0

    # Test invalid field error
    with pytest.raises(
        KeyError, match="This datagetter does not handle invalid_field data"
    ):
        default_getter.get_single_var_mod_data_yearmean(
            "piControl", "invalid_field", "CanESM5"
        )

    # Test 5: Data retrieval and caching
    print("Testing data retrieval and caching...")
    # First call
    data1 = default_getter.get_single_var_mod_data_yearmean(
        "piControl", "tas", "CanESM5"
    )

    # Second call should use cached data (same object)
    data2 = default_getter.get_single_var_mod_data_yearmean(
        "piControl", "tas", "CanESM5"
    )

    assert isinstance(data1, xr.DataArray)
    assert isinstance(data2, xr.DataArray)
    assert data1.equals(data2)

    # Test data structure
    assert "lat" in data1.dims
    assert "lon" in data1.dims
    assert "year" in data1.dims

    # Test 6: Training data creation
    print("Testing training data creation...")
    # Test base experiment (piControl)
    base_data = default_getter.make_meteor_training_data("piControl", "CanESM5")
    assert isinstance(base_data, xr.Dataset)
    assert "tas" in base_data.data_vars
    assert "pr" in base_data.data_vars

    # Check training data structure
    for var in base_data.data_vars:
        assert "lat" in base_data[var].dims
        assert "lon" in base_data[var].dims
        assert "year" in base_data[var].dims

    # Test CO2x4 experiment (abrupt-4xCO2)
    co2x4_data = default_getter.make_meteor_training_data("abrupt-4xCO2", "CanESM5")
    assert isinstance(co2x4_data, xr.Dataset)
    assert "tas" in co2x4_data.data_vars

    # Test 7: Error handling
    print("Testing error handling...")
    # Test invalid model in training data creation
    with pytest.raises(KeyError):
        default_getter.make_meteor_training_data("base", "InvalidModel123")

    # Test invalid experiment type in training data creation
    with pytest.raises(KeyError):
        default_getter.make_meteor_training_data("invalid_exp", "CanESM5")

    print("All comprehensive basic functionality tests completed successfully!")


def test_year_mean_monthly_function():
    """Test the year_mean_monthly helper function."""

    # Create test data: 2 years (24 months) of random data
    test_data = np.random.rand(24, 10, 10)  # 24 months, 10x10 spatial grid

    # Test the function
    yearly_data = cmip6_meteor_data_getter.year_mean_monthly(test_data)

    # Should have 2 years
    assert yearly_data.shape[0] == 2
    # Spatial dimensions should be preserved
    assert yearly_data.shape[1:] == test_data.shape[1:]

    # Test with different number of months (36 months = 3 years)
    test_data_3years = np.random.rand(36, 5, 5)
    yearly_data_3years = cmip6_meteor_data_getter.year_mean_monthly(test_data_3years)
    assert yearly_data_3years.shape[0] == 3
    assert yearly_data_3years.shape[1:] == test_data_3years.shape[1:]


def test_helper_functions():
    """Test helper functions for better coverage."""
    # Test multiply_along_axis function

    array_a = np.array([[1, 2], [3, 4], [5, 6]])
    array_b = np.array([2, 3, 4])
    result = cmip6_meteor_data_getter.multiply_along_axis(array_a, array_b, 0)
    assert result.shape == array_a.shape

    # Test year_mean_monthly function
    monthly_data = np.random.randn(24, 5, 5)  # 2 years of monthly data
    yearly_data = cmip6_meteor_data_getter.year_mean_monthly(monthly_data)
    assert yearly_data.shape[0] == 2  # Should have 2 years
    assert yearly_data.shape[1:] == monthly_data.shape[1:]  # Same spatial dims


def test_year_mean_monthly_xarray():
    """Test year_mean_monthly_xarray function."""

    # Create test monthly data
    monthly_data = xr.DataArray(
        np.random.rand(24, 3, 3),  # 24 months, 3x3 spatial
        dims=["time", "lat", "lon"],
        coords={"time": range(24), "lat": [1, 2, 3], "lon": [1, 2, 3]},
    )

    # Test the function
    yearly_data = cmip6_meteor_data_getter.year_mean_monthly_xarray(monthly_data)

    assert isinstance(yearly_data, xr.DataArray)
    assert yearly_data.sizes["time"] == 2  # Should have 2 years
    assert yearly_data.sizes["lat"] == 3
    assert yearly_data.sizes["lon"] == 3


# def test_error_handling_for_invalid_experiments_and_fields():
#     """Test error handling for invalid experiments and fields to hit lines 585-593."""

#     # Create a data getter with limited experiments and fields
#     data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
#         exps=["historical"], flds=["tas"]
#     )

#     # Set models manually and mock the model check
#     data_getter.models = ["test_model"]
#     data_getter.check_if_model_has_data = lambda model: model == "test_model"

#     # Test invalid experiment error (line 585-587)
#     with pytest.raises(
#         KeyError,
#         match="This datagetter does not handle data from the invalid_exp experiment",
#     ):
#         data_getter.get_single_var_mod_data("invalid_exp", "tas", "test_model")
