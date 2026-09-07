import os
import pickle
import shutil
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from meteor import cache_handling, cmip6_meteor_data_getter
from meteor.cache_handling import CacheHandler


class OldFormatFlds:
    name = ""

    def __init__(self, name, flds):
        self.name = name
        self.flds = flds


class OldFormatPatternFlds:
    name = ""

    def __init__(self, name, patternflds):
        self.name = name
        self.patternflds = patternflds


class TrackingCache:
    cache_functioning = True

    def __init__(self):
        self.saved = False

    def get_cmip6_query_catalogue(self):
        return "/tmp/cmip6_catalog.csv"

    def load_cmip6_cached_data(self, *_args, **_kwargs):
        return None

    def save_cmip6_to_cache(self, *_args, **_kwargs):
        self.saved = True


def _make_light_cache_getter(cache_handler, **kwargs):
    defaults = {
        "models": ["CanESM5"],
        "flds": ["tas", "pr"],
        "exps": ["piControl", "abrupt-4xCO2"],
        "enable_cache": True,
        "cache_handler": cache_handler,
    }
    defaults.update(kwargs)
    return cmip6_meteor_data_getter.Cmip6MeteorDataGetter(**defaults)


@pytest.fixture(scope="session")
def light_mock_cache_dir(test_data_dir):
    return os.path.join(test_data_dir, "light_mock_cache")


@pytest.fixture()
def light_cache_handler(light_mock_cache_dir):
    return CacheHandler(cache_dir=light_mock_cache_dir, purpose="cmip6")


def _make_banana_getter(**kwargs):
    defaults = {
        "flds": ["banana"],
        "tabids": ["Lmon"],  # bananas grow on land, not in the atmosphere!
        "exps": ["abrupt-4xCO2"],
    }
    defaults.update(kwargs)
    return cmip6_meteor_data_getter.Cmip6MeteorDataGetter(**defaults)


def _make_empty_catalog_cache(tmp_path):
    cache_handler = CacheHandler(
        cache_dir=str(tmp_path / "empty-cache"), purpose="cmip6"
    )
    empty_df = pd.DataFrame(
        columns=[
            "activity_id",
            "table_id",
            "variable_id",
            "experiment_id",
            "source_id",
            "member_id",
            "zstore",
        ]
    )
    catalog_path = os.path.join(
        cache_handler.cache_dir, "cmip6", "cmip6-zarr-consolidated-stores.csv"
    )
    empty_df.to_csv(catalog_path, index=False)
    return cache_handler


def test_get_unique_models(light_cache_handler):
    data_getter = _make_light_cache_getter(light_cache_handler)
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

    data_getter.enable_cache = True
    test_var = data_getter.get_single_var_mod_data_yearmean(
        "piControl", "tas", "CanESM5"
    )
    assert isinstance(test_var, xr.DataArray)

    test_training = data_getter.make_meteor_training_data("base", "CanESM5")
    assert isinstance(test_training, xr.Dataset)

    data_getter_2 = _make_light_cache_getter(
        light_cache_handler,
        exps=["historical", "ssp370"],
        dbe=["CMIP", "ScenarioMIP"],
    )
    test_composite = data_getter_2.make_meteor_training_data_composite(
        ["historical", "ssp370"], model="CanESM5"
    )
    assert test_composite.sizes["year"] == 251
    print(test_composite["year"].values)
    # assert False


def test_comprehensive_basic_functionality(light_cache_handler):
    """Comprehensive test covering initialization, validation, model checks, and basic data operations."""

    # Test 1: Default initialization and basic properties
    print("Testing default initialization...")
    default_getter = _make_light_cache_getter(light_cache_handler)

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
    custom_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl", "abrupt-4xCO2", "1pctCO2"],
    )
    assert custom_getter.flds == ["tas"]
    assert custom_getter.exps == ["piControl", "abrupt-4xCO2", "1pctCO2"]

    # Test 2b: Parameter defaults and validation in _set_models_flds_tabids_exps_dbe
    print("Testing parameter normalization...")
    normalized_getter = _make_light_cache_getter(
        light_cache_handler,
        models="CanESM5",
        flds="tas",
        tabids="Amon",
        exps="ssp126",
    )
    assert normalized_getter.filter_models == ["CanESM5"]
    assert normalized_getter.flds == ["tas"]
    assert normalized_getter.tabids == ["Amon"]
    assert normalized_getter.exps == ["ssp126"]
    assert normalized_getter._set_models_flds_tabids_exps_dbe(
        models="CanESM5",
        flds="tas",
        tabids="Amon",
        exps="ssp126",
        dbe=None,
    ) == ["ScenarioMIP"]

    with pytest.raises(ValueError, match="tabids should be the same length as flds"):
        _make_light_cache_getter(
            light_cache_handler,
            flds=["tas", "pr"],
            tabids=["Amon"],
            exps=["piControl"],
        )

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


def test_sort_member_ids_with_unparseable_ids():
    member_ids = np.array(
        [
            "r1i1p1f1",  # valid
            "r10i1p1f1",  # valid
            "not_a_member",  # invalid → triggers float("inf")
            "r2i1p1f1",  # valid
        ]
    )

    sorted_ids = cmip6_meteor_data_getter.sort_member_ids_numerically(member_ids)

    # 'not_a_member' should end up at the end
    assert sorted_ids[-1] == "not_a_member"
    # valid IDs should be sorted numerically
    assert sorted_ids[:-1] == ["r1i1p1f1", "r2i1p1f1", "r10i1p1f1"]


def test_default_mdl_skipmbrs_used_when_none():
    flds = ["pr"]
    exps = ["historical"]
    df_dummy = pd.DataFrame(
        {
            "source_id": ["NorESM2-LM", "ModelX"],
            "experiment_id": ["historical", "historical"],
            "member_id": ["r1i1p1f1", "r1i1p1f1"],
            "pr": [0.1, 0.2],
        }
    )
    df_all1 = [[df_dummy.copy() for _ in flds] for _ in exps]

    df_all_none, mdls_none = cmip6_meteor_data_getter.initialise_dataframe_and_models(
        df_all1, flds, exps, mdl_skipmbrs=None
    )
    df_all_explicit, mdls_explicit = (
        cmip6_meteor_data_getter.initialise_dataframe_and_models(
            df_all1,
            flds,
            exps,
            mdl_skipmbrs={"NorESM2-LM": ["r1i1p1f1"]},
        )
    )


def test_init_error_message_no_models(tmp_path):
    cache_handler = _make_empty_catalog_cache(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        _make_banana_getter(enable_cache=True, cache_handler=cache_handler)

    msg = str(excinfo.value)
    assert msg.startswith("No models found that have complete data")
    assert "banana (table_id: Lmon)" in msg
    assert "Experiments: ['abrupt-4xCO2']" in msg


def test_init_error_message_single_model(tmp_path):
    cache_handler = _make_empty_catalog_cache(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        _make_banana_getter(
            models="NonExistentModel",
            enable_cache=True,
            cache_handler=cache_handler,
        )

    msg = str(excinfo.value)
    assert msg.startswith("Model NonExistentModel does not have complete data")
    assert "banana (table_id: Lmon)" in msg
    assert "Experiments: ['abrupt-4xCO2']" in msg


def test_init_error_message_multiple_models(tmp_path):
    cache_handler = _make_empty_catalog_cache(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        _make_banana_getter(
            models=["Model1", "Model2"],
            enable_cache=True,
            cache_handler=cache_handler,
        )

    msg = str(excinfo.value)
    assert msg.startswith(
        "Non of the requested models ['Model1', 'Model2'] have complete data"
    )
    assert "banana (table_id: Lmon)" in msg
    assert "Experiments: ['abrupt-4xCO2']" in msg


def test_get_single_var_mod_data_warns_for_unfiltered_model(
    light_cache_handler, caplog
):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        models=["CanESM5"],
        flds=["tas"],
        exps=["piControl"],
    )

    caplog.set_level("WARNING")
    assert not data_getter.check_if_model_has_data("OtherModel")
    assert "was not among the requested models" in caplog.text


def test_get_single_var_mod_data_cache_hit(light_cache_handler):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )

    result = data_getter.get_single_var_mod_data("piControl", "tas", "CanESM5")
    assert isinstance(result, (xr.Dataset, xr.DataArray))


def test_get_single_var_mod_data_nan_zstore_ref(light_cache_handler):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )
    data_getter.enable_cache = False

    df_entry = data_getter.df_all[data_getter.exps.index("piControl")][
        data_getter.flds.index("tas")
    ]
    df_entry.loc[df_entry["source_id"] == "CanESM5", "zstore"] = np.nan

    with pytest.raises(KeyError, match="No zstore ref for CanESM5"):
        data_getter.get_single_var_mod_data("piControl", "tas", "CanESM5")


def test_get_single_var_mod_data_yearmean_cache_hit(light_cache_handler, caplog):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )

    caplog.set_level("INFO")
    result = data_getter.get_single_var_mod_data_yearmean("piControl", "tas", "CanESM5")
    assert isinstance(result, xr.DataArray)
    assert "Using cached yearly data" in caplog.text


def test_get_single_var_mod_data_yearmean_monthly_none(light_cache_handler):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )
    data_getter.enable_cache = False
    data_getter.get_single_var_mod_data_monthly = lambda *_args, **_kwargs: None

    assert (
        data_getter.get_single_var_mod_data_yearmean("piControl", "tas", "CanESM5")
        is None
    )


def test_get_single_var_mod_data_yearmean_saves_cache(light_cache_handler):
    monthly = xr.DataArray(
        np.zeros((1, 12, 1, 1)),
        dims=["ens", "month", "lat", "lon"],
        coords={"ens": [1], "month": np.arange(12), "lat": [0], "lon": [0]},
    )

    cache = TrackingCache()
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )
    data_getter.cache_handler = cache
    data_getter.enable_cache = True
    data_getter.get_single_var_mod_data_monthly = lambda *_args, **_kwargs: monthly

    result = data_getter.get_single_var_mod_data_yearmean("piControl", "tas", "CanESM5")
    assert result is not None
    assert cache.saved is True


def test_get_single_var_mod_data_monthly_none(light_cache_handler):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )
    data_getter.enable_cache = False
    data_getter.get_single_var_mod_data = lambda *_args, **_kwargs: None

    assert (
        data_getter.get_single_var_mod_data_monthly("piControl", "tas", "CanESM5")
        is None
    )


def test_get_single_var_mod_data_monthly_renames_lat_lon(light_cache_handler):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )

    ds = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.zeros((2, 3, 4)),
                dims=["time", "latitude", "longitude"],
                coords={
                    "time": [0, 1],
                    "latitude": [10, 20, 30],
                    "longitude": [0, 90, 180, 270],
                },
            )
        }
    )
    data_getter.get_single_var_mod_data = lambda *_args, **_kwargs: ds

    monthly = data_getter.get_single_var_mod_data_monthly("piControl", "tas", "CanESM5")
    assert "lat" in monthly.dims
    assert "lon" in monthly.dims
    assert "latitude" not in monthly.dims
    assert "longitude" not in monthly.dims


def test_get_single_var_mod_data_monthly_saves_cache(light_cache_handler):
    ds = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.zeros((2, 2, 2)),
                dims=["time", "lat", "lon"],
                coords={"time": [0, 1], "lat": [0, 1], "lon": [0, 1]},
            )
        }
    )

    cache = TrackingCache()
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )
    data_getter.cache_handler = cache
    data_getter.enable_cache = True
    data_getter.get_single_var_mod_data = lambda *_args, **_kwargs: ds

    result = data_getter.get_single_var_mod_data_monthly("piControl", "tas", "CanESM5")
    assert result is not None
    assert cache.saved is True


def test_make_meteor_training_data_cache_hit(light_cache_handler):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )

    result = data_getter.make_meteor_training_data(
        "base", "CanESM5", exp_mapper=None, monthly=False
    )
    assert isinstance(result, xr.Dataset)


def test_make_meteor_training_data_default_exp_mapper_monthly(light_cache_handler):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )
    data_getter.enable_cache = False

    called = {}

    def _monthly(exp, fld, model):
        called["exp"] = exp
        called["fld"] = fld
        called["model"] = model
        return xr.DataArray([1], dims=["month"])  # minimal

    data_getter.get_single_var_mod_data_monthly = _monthly
    result = data_getter.make_meteor_training_data(
        "base", "CanESM5", exp_mapper=None, monthly=True
    )

    assert called["exp"] == "piControl"
    assert "tas" in result.data_vars


def test_make_meteor_training_data_monthly_exp_in_mapper(light_cache_handler):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )
    data_getter.enable_cache = False

    called = {}

    def _monthly(exp, fld, model):
        called["exp"] = exp
        return xr.DataArray([1], dims=["month"])

    data_getter.get_single_var_mod_data_monthly = _monthly
    result = data_getter.make_meteor_training_data(
        "base", "CanESM5", exp_mapper={"base": "piControl"}, monthly=True
    )

    assert called["exp"] == "piControl"
    assert "tas" in result.data_vars


def test_make_meteor_training_data_monthly_exp_not_in_mapper(light_cache_handler):
    data_getter = _make_light_cache_getter(
        light_cache_handler,
        flds=["tas"],
        exps=["piControl"],
    )
    data_getter.enable_cache = False

    called = {}

    def _monthly(exp, fld, model):
        called["exp"] = exp
        return xr.DataArray([1], dims=["month"])

    data_getter.get_single_var_mod_data_monthly = _monthly
    result = data_getter.make_meteor_training_data(
        "piControl", "CanESM5", exp_mapper={"base": "piControl"}, monthly=True
    )

    assert called["exp"] == "piControl"
    assert "tas" in result.data_vars


def test_caching(tmp_path):
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        flds=["tas", "pr"],
        exps=["piControl"],
    )
    expected_name = "cmip6-ModelA-aer"

    # Missing cache file
    missing_file = tmp_path / "missing.pkl"
    is_valid, cached_model, info = data_getter.validate_pattern_scaling_cache(
        str(missing_file), "ModelA"
    )
    assert is_valid is False
    assert cached_model is None
    assert "Cache file not found" in info["message"]

    # New format with patternflds
    new_format_file = tmp_path / "new_format.pkl"
    with open(new_format_file, "wb") as handle:
        pickle.dump(
            {
                "name": expected_name,
                "patternflds": {"tas": None, "pr": None},
            },
            handle,
        )
    is_valid, cached_model, info = data_getter.validate_pattern_scaling_cache(
        str(new_format_file), "ModelA"
    )
    assert is_valid is True
    assert cached_model is not None
    assert "Cache valid" in info["message"]

    # Old format with flds attribute
    old_flds_file = tmp_path / "old_flds.pkl"
    with open(old_flds_file, "wb") as handle:
        pickle.dump(OldFormatFlds(expected_name, {"tas": None, "pr": None}), handle)
    is_valid, cached_model, info = data_getter.validate_pattern_scaling_cache(
        str(old_flds_file), "ModelA"
    )
    assert is_valid is True
    assert cached_model is not None
    assert "Cache valid" in info["message"]

    # Old format with patternflds attribute only
    old_pattern_file = tmp_path / "old_pattern.pkl"
    with open(old_pattern_file, "wb") as handle:
        pickle.dump(
            OldFormatPatternFlds(expected_name, {"tas": None, "pr": None}),
            handle,
        )
    is_valid, cached_model, info = data_getter.validate_pattern_scaling_cache(
        str(old_pattern_file), "ModelA"
    )
    assert is_valid is True
    assert cached_model is not None
    assert "Cache valid" in info["message"]

    # Missing field information
    missing_fields_file = tmp_path / "missing_fields.pkl"
    with open(missing_fields_file, "wb") as handle:
        pickle.dump({"name": expected_name}, handle)
    is_valid, cached_model, info = data_getter.validate_pattern_scaling_cache(
        str(missing_fields_file), "ModelA"
    )
    assert is_valid is False
    assert cached_model is None
    assert "missing field information" in info["message"]

    # Missing required fields
    missing_required_file = tmp_path / "missing_required.pkl"
    with open(missing_required_file, "wb") as handle:
        pickle.dump({"name": expected_name, "patternflds": {"tas": None}}, handle)
    is_valid, cached_model, info = data_getter.validate_pattern_scaling_cache(
        str(missing_required_file), "ModelA"
    )
    assert is_valid is False
    assert cached_model is None
    assert "Missing required fields" in info["message"]

    # Corrupted cache file
    corrupted_file = tmp_path / "corrupted.pkl"
    corrupted_file.write_text("not a pickle")
    is_valid, cached_model, info = data_getter.validate_pattern_scaling_cache(
        str(corrupted_file), "ModelA"
    )
    assert is_valid is False
    assert cached_model is None
    assert "Error reading cached file" in info["message"]

    # variable= must be honored so that per-variable caches (which
    # MeteorPatternScaling saves under `cmip6-{model}-aer-{variable}` and which
    # only carry that one variable in patternflds) round-trip.
    # Without this, MeteorInterface prepares training data on every call because
    # this check returns "invalid" even when the pkl loads fine by filename.
    per_var_file = tmp_path / "per_variable.pkl"
    per_var_name = "cmip6-ModelA-aer-tas"
    with open(per_var_file, "wb") as handle:
        pickle.dump(
            {"name": per_var_name, "patternflds": {"tas": None}},
            handle,
        )

    # No variable= arg: legacy validator, mismatches per-variable name.
    is_valid, _, info = data_getter.validate_pattern_scaling_cache(
        str(per_var_file), "ModelA"
    )
    assert is_valid is False
    assert "Model name mismatch" in info["message"]

    # variable="tas": matches name AND expected_vars is narrowed to just
    # {"tas"} instead of self.flds, so the single-variable pkl is accepted.
    is_valid, cached_model, info = data_getter.validate_pattern_scaling_cache(
        str(per_var_file), "ModelA", variable="tas"
    )
    assert is_valid is True
    assert cached_model is not None
    assert "Cache valid" in info["message"]


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


# =============================================================================
# Regression tests hardening the data-getter / cache round-trip
# =============================================================================
#
# These tests target the class of bug where a filename convention on one side
# of a cache boundary diverges from the other, so a "cache miss" report from
# the outer view actually corresponds to a real cache hit at the inner loader.
# The catalyst was a per-variable-suffix mismatch that made every generation
# call redundantly fetch CMIP6 data from GCS. We defend against a repeat by:
#
#   1. Verifying make_meteor_training_data_composite stitches historical + ssp
#      into a contiguous, correctly-sized time series.
#   2. Providing a `gcs_zarr_mock` fixture that patches the two network
#      seams (gcsfs.GCSFileSystem and xarray.open_zarr) so tests can exercise
#      the fetch-and-cache path without a network.
#   3. Verifying that after a mocked fetch, a cache file appears at the exact
#      path _generate_cmip6_cache_key predicts, and a follow-up call returns
#      from cache without re-consulting gcsfs.


def test_make_meteor_training_data_composite_stitches_historical_and_ssp_contiguously(
    light_cache_handler,
):
    """Composite of historical (1850-2014) + ssp370 (2015-2100) must produce
    251 contiguous years with no gaps, no duplicates, and the two periods
    fully preserved end-to-end.

    Existing tests check that ``sizes["year"] == 251`` but not that the
    boundary between experiments is clean. A stitching bug (off-by-one at
    the join, or accidentally reordering years) would still pass the size
    check but silently corrupt training data.
    """
    getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["CanESM5"],
        flds=["tas", "pr"],
        exps=["historical", "ssp370"],
        dbe=["CMIP", "ScenarioMIP"],
        enable_cache=True,
        cache_handler=light_cache_handler,
    )

    composite = getter.make_meteor_training_data_composite(
        ["historical", "ssp370"], model="CanESM5"
    )

    years = np.asarray(composite["year"].values)
    assert years.size == 251, f"expected 251 years, got {years.size}"

    # Strictly monotonically increasing by 1 (contiguous, no duplicates, no reorder).
    diffs = np.diff(years)
    assert np.all(diffs == 1), (
        "composite year axis is not strictly contiguous: "
        f"unique diff values {sorted(set(diffs.tolist()))}"
    )

    # Fields promised by the getter must all round-trip.
    for fld in ("tas", "pr"):
        assert fld in composite.data_vars, f"expected {fld!r} in composite"
        # The join should not introduce NaN gaps for the tracked fields.
        assert (
            not composite[fld].isnull().any()
        ), f"unexpected NaNs in composite {fld!r} field"


@pytest.fixture
def fresh_cmip6_cache_dir(tmp_path, light_mock_cache_dir):
    """A pristine cache directory with only the catalog CSV preseeded.

    This mirrors what a real first-run user has: the catalog is available
    (either shipped or previously downloaded), but no data-fetch cache
    files exist yet. Any code path that reads model data must therefore go
    through the fetch route, giving us a clean way to exercise it.
    """
    dst = tmp_path / "fresh_cmip6_cache"
    dst_cmip6 = dst / "cmip6"
    dst_cmip6.mkdir(parents=True)
    src_csv = os.path.join(
        light_mock_cache_dir, "cmip6", "cmip6-zarr-consolidated-stores.csv"
    )
    shutil.copy(src_csv, str(dst_cmip6))
    return str(dst)


@pytest.fixture
def gcs_zarr_mock(monkeypatch):
    """Patch the two network seams the data-getter uses (``gcsfs.GCSFileSystem``
    and ``xarray.open_zarr``) so tests can drive the fetch-and-cache path
    without touching the network.

    Yields a dict that records mapper calls so tests can assert cache-hit
    behavior on subsequent invocations.
    """
    calls = {"get_mapper": []}

    fake_gcs = MagicMock()

    def _fake_get_mapper(zstore_ref):
        calls["get_mapper"].append(zstore_ref)
        # A sentinel string is fine — the patched open_zarr recognizes it.
        return f"MOCK::{zstore_ref}"

    fake_gcs.get_mapper.side_effect = _fake_get_mapper

    monkeypatch.setattr(
        "meteor.cmip6_meteor_data_getter.gcsfs.GCSFileSystem",
        lambda **_: fake_gcs,
    )

    def _fake_open_zarr(mapper, decode_times=False):  # noqa: ARG001
        assert isinstance(mapper, str) and mapper.startswith(
            "MOCK::"
        ), f"gcs_zarr_mock: expected our sentinel mapper, got {mapper!r}"
        # 200 years of monthly-resolution data at a small (2×3) grid.
        # Enough to exercise the piControl>1800-month trim branch too.
        n_months = 200 * 12
        time_axis = np.arange(n_months, dtype="float64")
        n_lat, n_lon = 2, 3
        seasonal = 15.0 + 5.0 * np.sin(2 * np.pi * time_axis / 12.0)
        broadcast = np.broadcast_to(seasonal[:, None, None], (n_months, n_lat, n_lon))
        return xr.Dataset(
            {"tas": (["time", "lat", "lon"], broadcast.astype("float32"))},
            coords={
                "time": time_axis,
                "lat": np.array([-45.0, 45.0]),
                "lon": np.array([0.0, 60.0, 180.0]),
            },
        )

    monkeypatch.setattr("xarray.open_zarr", _fake_open_zarr)
    return calls


def test_get_single_var_mod_data_monthly_writes_cache_at_expected_path(
    fresh_cmip6_cache_dir, gcs_zarr_mock
):
    """A cache-miss fetch through get_single_var_mod_data_monthly must write
    an .nc file at the exact path _generate_cmip6_cache_key predicts, so that
    a subsequent call finds it and does not re-consult gcsfs.

    This is the write-then-read round-trip that the whole caching layer
    depends on. Any divergence between the write filename and the validate/
    read filename (which is what PR #94 was) will show up here as either a
    missing file or a redundant fetch on the second call.
    """
    handler = CacheHandler(cache_dir=fresh_cmip6_cache_dir, purpose="cmip6")
    getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["CanESM5"],
        flds=["tas"],
        exps=["piControl"],
        enable_cache=True,
        cache_handler=handler,
    )

    # First call: cache miss -> goes through gcsfs mock, writes to cache.
    first = getter.get_single_var_mod_data_monthly("piControl", "tas", "CanESM5")
    assert isinstance(first, xr.DataArray)
    assert gcs_zarr_mock["get_mapper"], "gcsfs should have been consulted on cache miss"

    # The written file must live at the exact key the cache-key generator
    # predicts. If this filename convention drifts, PR-#94-style bugs return.
    expected_key = cache_handling._generate_cmip6_cache_key(
        "get_single_var_mod_data_monthly", "piControl", "tas", "CanESM5"
    )
    assert expected_key == "CanESM5_piControl_tas_monthly"
    cache_file = os.path.join(fresh_cmip6_cache_dir, "cmip6", f"{expected_key}.nc")
    assert os.path.exists(cache_file), (
        f"cache miss should have written {cache_file!r} but the file is not there. "
        "Either the write path drifted from the key generator, or save "
        "silently failed."
    )

    # Second call: must hit the cache and NOT touch gcsfs again.
    n_before = len(gcs_zarr_mock["get_mapper"])
    second = getter.get_single_var_mod_data_monthly("piControl", "tas", "CanESM5")
    assert len(gcs_zarr_mock["get_mapper"]) == n_before, (
        "cache-hit call should not re-consult gcsfs; if it does, the loader "
        "is looking at a different filename than save wrote to"
    )
    xr.testing.assert_equal(first, second)
