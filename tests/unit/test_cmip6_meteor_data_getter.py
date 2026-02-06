import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from meteor import cmip6_meteor_data_getter


def _make_min_catalog_df(model, exp, fld, tabid, activity_id="CMIP"):
    return pd.DataFrame(
        [
            {
                "activity_id": activity_id,
                "table_id": tabid,
                "variable_id": fld,
                "experiment_id": exp,
                "source_id": model,
                "member_id": "r1i1p1f1",
                "zstore": f"gs://dummy/{model}/{exp}/{fld}",
            }
        ]
    )


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

    # Test 2b: Parameter defaults and validation in _set_models_flds_tabids_exps_dbe
    print("Testing parameter normalization...")
    normalized_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
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
        cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
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
    member_ids = np.array([
        'r1i1p1f1',     # valid
        'r10i1p1f1',    # valid
        'not_a_member', # invalid → triggers float("inf")
        'r2i1p1f1',     # valid
    ])

    sorted_ids = cmip6_meteor_data_getter.sort_member_ids_numerically(member_ids)

    # 'not_a_member' should end up at the end
    assert sorted_ids[-1] == 'not_a_member'
    # valid IDs should be sorted numerically
    assert sorted_ids[:-1] == ['r1i1p1f1', 'r2i1p1f1', 'r10i1p1f1']

def test_data_query_empty_branch():
    flds = ['pr']
    exps = ['historical']

    # Create dataframe with one member, but mismatch so query returns empty
    df_dummy = pd.DataFrame({
        'source_id': ['ModelX'],
        'experiment_id': ['historical'],
        'member_id': ['r2i1p1f1'],  # will be sorted first as r2
        'pr': [0.1]
    })

    # df_all1: list of list of dataframes (exps x flds)
    df_all1 = [[df_dummy.copy() for _ in flds] for _ in exps]

    # Introduce a mismatch: first sorted member is r1i1p1f1, but only r2 exists
    def fake_sort_member_ids_numerically(member_ids):
        # Force it to return 'r1i1p1f1' first
        return ['r1i1p1f1'] + list(member_ids)

    # Patch the function used inside
    original_sort = cmip6_meteor_data_getter.sort_member_ids_numerically
    cmip6_meteor_data_getter.sort_member_ids_numerically = fake_sort_member_ids_numerically

    try:
        df_all, mdls = cmip6_meteor_data_getter.initialise_dataframe_and_models(df_all1, flds, exps)
        # The model should be excluded because sufficient_data becomes False
        assert 'ModelX' not in mdls
    finally:
        # Restore original function
        cmip6_meteor_data_getter.sort_member_ids_numerically = original_sort


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

    assert mdls_none == mdls_explicit
    assert len(df_all_none) == len(df_all_explicit)
    for exp_idx in range(len(exps)):
        for fld_idx in range(len(flds)):
            pd.testing.assert_frame_equal(
                df_all_none[exp_idx][fld_idx],
                df_all_explicit[exp_idx][fld_idx],
            )


def _patch_empty_catalog(monkeypatch):
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
    monkeypatch.setattr(
        cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: empty_df
    )
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )


def test_init_error_message_no_models(monkeypatch):
    _patch_empty_catalog(monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
            flds=["banana"],
            tabids=["Lmon"],
            exps=["abrupt-4xCO2"],
        )

    msg = str(excinfo.value)
    assert msg.startswith("No models found that have complete data")
    assert "banana (table_id: Lmon)" in msg
    assert "Experiments: ['abrupt-4xCO2']" in msg


def test_init_error_message_single_model(monkeypatch):
    _patch_empty_catalog(monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
            models="NonExistentModel",
            flds=["banana"],
            tabids=["Lmon"],
            exps=["abrupt-4xCO2"],
        )

    msg = str(excinfo.value)
    assert msg.startswith("Model NonExistentModel does not have complete data")
    assert "banana (table_id: Lmon)" in msg
    assert "Experiments: ['abrupt-4xCO2']" in msg


def test_init_error_message_multiple_models(monkeypatch):
    _patch_empty_catalog(monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
            models=["Model1", "Model2"],
            flds=["banana"],
            tabids=["Lmon"],
            exps=["abrupt-4xCO2"],
        )

    msg = str(excinfo.value)
    assert msg.startswith(
        "Non of the requested models ['Model1', 'Model2'] have complete data"
    )
    assert "banana (table_id: Lmon)" in msg
    assert "Experiments: ['abrupt-4xCO2']" in msg


def test_get_single_var_mod_data_warns_for_unfiltered_model(monkeypatch, caplog):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
    )

    caplog.set_level("WARNING")
    assert not data_getter.check_if_model_has_data("OtherModel")
    assert "was not among the requested models" in caplog.text


def test_get_single_var_mod_data_cache_hit(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    class DummyCache:
        cache_functioning = True

        def get_cmip6_query_catalogue(self):
            return "/tmp/cmip6_catalog.csv"

        def load_cmip6_cached_data(self, *_args, **_kwargs):
            return "CACHED"

        def save_cmip6_to_cache(self, *_args, **_kwargs):
            return None

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
        enable_cache=True,
        cache_handler=DummyCache(),
    )

    assert data_getter.get_single_var_mod_data("piControl", "tas", "ModelA") == "CACHED"


def test_get_single_var_mod_data_nan_zstore_ref(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    df.loc[0, "zstore"] = np.nan
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
    )

    with pytest.raises(KeyError, match="No zstore ref for ModelA"):
        data_getter.get_single_var_mod_data("piControl", "tas", "ModelA")


def test_get_single_var_mod_data_yearmean_cache_hit(monkeypatch, caplog):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    cached = xr.DataArray(
        np.zeros((1, 1, 1, 1)),
        dims=["ens", "year", "lat", "lon"],
        coords={"ens": [1], "year": [0], "lat": [0], "lon": [0]},
    )

    class DummyCache:
        cache_functioning = True

        def get_cmip6_query_catalogue(self):
            return "/tmp/cmip6_catalog.csv"

        def load_cmip6_cached_data(self, *_args, **_kwargs):
            return cached

        def save_cmip6_to_cache(self, *_args, **_kwargs):
            return None

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
        enable_cache=True,
        cache_handler=DummyCache(),
    )

    caplog.set_level("INFO")
    result = data_getter.get_single_var_mod_data_yearmean("piControl", "tas", "ModelA")
    assert result is cached
    assert "Using cached yearly data" in caplog.text


def test_get_single_var_mod_data_yearmean_monthly_none(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
    )
    data_getter.get_single_var_mod_data_monthly = lambda *_args, **_kwargs: None

    assert (
        data_getter.get_single_var_mod_data_yearmean("piControl", "tas", "ModelA")
        is None
    )


def test_get_single_var_mod_data_yearmean_saves_cache(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    monthly = xr.DataArray(
        np.zeros((1, 12, 1, 1)),
        dims=["ens", "month", "lat", "lon"],
        coords={"ens": [1], "month": np.arange(12), "lat": [0], "lon": [0]},
    )

    class DummyCache:
        cache_functioning = True
        saved = False

        def get_cmip6_query_catalogue(self):
            return "/tmp/cmip6_catalog.csv"

        def load_cmip6_cached_data(self, *_args, **_kwargs):
            return None

        def save_cmip6_to_cache(self, *_args, **_kwargs):
            self.saved = True

    cache = DummyCache()
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
        enable_cache=True,
        cache_handler=cache,
    )
    data_getter.get_single_var_mod_data_monthly = lambda *_args, **_kwargs: monthly

    result = data_getter.get_single_var_mod_data_yearmean("piControl", "tas", "ModelA")
    assert result is not None
    assert cache.saved is True


def test_get_single_var_mod_data_monthly_none(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
    )
    data_getter.get_single_var_mod_data = lambda *_args, **_kwargs: None

    assert (
        data_getter.get_single_var_mod_data_monthly("piControl", "tas", "ModelA")
        is None
    )


def test_get_single_var_mod_data_monthly_renames_lat_lon(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
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

    monthly = data_getter.get_single_var_mod_data_monthly(
        "piControl", "tas", "ModelA"
    )
    assert "lat" in monthly.dims
    assert "lon" in monthly.dims
    assert "latitude" not in monthly.dims
    assert "longitude" not in monthly.dims


def test_get_single_var_mod_data_monthly_saves_cache(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    class DummyCache:
        cache_functioning = True
        saved = False

        def get_cmip6_query_catalogue(self):
            return "/tmp/cmip6_catalog.csv"

        def load_cmip6_cached_data(self, *_args, **_kwargs):
            return None

        def save_cmip6_to_cache(self, *_args, **_kwargs):
            self.saved = True

    ds = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.zeros((2, 2, 2)),
                dims=["time", "lat", "lon"],
                coords={"time": [0, 1], "lat": [0, 1], "lon": [0, 1]},
            )
        }
    )

    cache = DummyCache()
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
        enable_cache=True,
        cache_handler=cache,
    )
    data_getter.get_single_var_mod_data = lambda *_args, **_kwargs: ds

    result = data_getter.get_single_var_mod_data_monthly("piControl", "tas", "ModelA")
    assert result is not None
    assert cache.saved is True


def test_make_meteor_training_data_cache_hit(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    cached = xr.Dataset({"tas": xr.DataArray([1], dims=["x"])})

    class DummyCache:
        cache_functioning = True

        def get_cmip6_query_catalogue(self):
            return "/tmp/cmip6_catalog.csv"

        def load_cmip6_cached_data(self, *_args, **_kwargs):
            return cached

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
        enable_cache=True,
        cache_handler=DummyCache(),
    )

    result = data_getter.make_meteor_training_data(
        "base", "ModelA", exp_mapper=None, monthly=False
    )
    assert result is cached


def test_make_meteor_training_data_default_exp_mapper_monthly(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
    )

    called = {}

    def _monthly(exp, fld, model):
        called["exp"] = exp
        called["fld"] = fld
        called["model"] = model
        return xr.DataArray([1], dims=["month"])  # minimal

    data_getter.get_single_var_mod_data_monthly = _monthly
    result = data_getter.make_meteor_training_data(
        "base", "ModelA", exp_mapper=None, monthly=True
    )

    assert called["exp"] == "piControl"
    assert "tas" in result.data_vars


def test_make_meteor_training_data_monthly_exp_in_mapper(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
    )

    called = {}

    def _monthly(exp, fld, model):
        called["exp"] = exp
        return xr.DataArray([1], dims=["month"])

    data_getter.get_single_var_mod_data_monthly = _monthly
    result = data_getter.make_meteor_training_data(
        "base", "ModelA", exp_mapper={"base": "piControl"}, monthly=True
    )

    assert called["exp"] == "piControl"
    assert "tas" in result.data_vars


def test_make_meteor_training_data_monthly_exp_not_in_mapper(monkeypatch):
    df = _make_min_catalog_df("ModelA", "piControl", "tas", "Amon")
    monkeypatch.setattr(cmip6_meteor_data_getter.pd, "read_csv", lambda *_, **__: df)
    monkeypatch.setattr(
        cmip6_meteor_data_getter.gcsfs, "GCSFileSystem", lambda *_, **__: object()
    )

    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        models=["ModelA"],
        flds=["tas"],
        tabids=["Amon"],
        exps=["piControl"],
    )

    called = {}

    def _monthly(exp, fld, model):
        called["exp"] = exp
        return xr.DataArray([1], dims=["month"])

    data_getter.get_single_var_mod_data_monthly = _monthly
    result = data_getter.make_meteor_training_data(
        "piControl", "ModelA", exp_mapper={"base": "piControl"}, monthly=True
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
        pickle.dump(
            {"name": expected_name, "patternflds": {"tas": None}}, handle
        )
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
