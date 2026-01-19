"""
Tests for ensemble output containers.

Tests the VariableOutput and EnsembleOutput classes that organize
METEOR ensemble results.
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from meteor.ensemble_output import EnsembleOutput, VariableOutput

# Test VariableOutput container class.


def test_initialization():
    """Test basic initialization."""
    var_out = VariableOutput("tas")

    assert var_out.variable == "tas"
    assert var_out.timeseries == {}
    assert var_out.gridded == {}
    assert var_out.impacts == {}
    assert var_out.metadata == {}


def test_initialization_with_metadata():
    """Test initialization with metadata."""
    metadata = {"scenario": "ssp245", "model": "CESM2"}
    var_out = VariableOutput("pr", metadata=metadata)

    assert var_out.variable == "pr"
    assert var_out.metadata == metadata


def test_list_timeseries_empty():
    """Test list_timeseries when empty."""
    var_out = VariableOutput("tas")
    assert var_out.list_timeseries() == []


def test_list_timeseries_populated():
    """Test list_timeseries with data."""
    var_out = VariableOutput("tas")
    var_out.timeseries["global"] = xr.DataArray([1, 2, 3])
    var_out.timeseries["regional:EAS"] = xr.DataArray([4, 5, 6])

    ts_list = var_out.list_timeseries()
    assert len(ts_list) == 2
    assert "global" in ts_list
    assert "regional:EAS" in ts_list


def test_list_gridded_empty():
    """Test list_gridded when empty."""
    var_out = VariableOutput("tas")
    assert var_out.list_gridded() == []


def test_list_gridded_populated():
    """Test list_gridded with data."""
    var_out = VariableOutput("tas")
    var_out.gridded["annual"] = xr.DataArray(np.zeros((10, 5, 5)))
    var_out.gridded["2050"] = xr.DataArray(np.zeros((5, 5)))

    grid_list = var_out.list_gridded()
    assert len(grid_list) == 2
    assert "annual" in grid_list
    assert "2050" in grid_list


def test_list_impacts_empty():
    """Test list_impacts when empty."""
    var_out = VariableOutput("tas")
    assert var_out.list_impacts() == []


def test_list_impacts_populated():
    """Test list_impacts with data."""
    var_out = VariableOutput("tas")
    var_out.impacts["hdd"] = {"global": xr.DataArray([100, 200])}
    var_out.impacts["cdd"] = {"global": xr.DataArray([50, 75])}

    impact_list = var_out.list_impacts()
    assert len(impact_list) == 2
    assert "hdd" in impact_list
    assert "cdd" in impact_list


def test_repr_empty():
    """Test __repr__ for empty VariableOutput."""
    var_out = VariableOutput("tas")
    repr_str = repr(var_out)

    assert "VariableOutput('tas')" in repr_str
    # Should not mention timeseries/gridded/impacts if empty
    assert "Timeseries" not in repr_str
    assert "Gridded" not in repr_str
    assert "Impacts" not in repr_str


def test_repr_with_data():
    """Test __repr__ with data."""
    var_out = VariableOutput("tas")
    var_out.timeseries["global"] = xr.DataArray([1, 2, 3])
    var_out.timeseries["regional:EAS"] = xr.DataArray([4, 5, 6])
    var_out.gridded["annual"] = xr.DataArray(np.zeros((10, 5, 5)))
    var_out.impacts["hdd"] = {"global": xr.DataArray([100, 200])}

    repr_str = repr(var_out)

    assert "VariableOutput('tas')" in repr_str
    assert "Timeseries: 2 aggregations" in repr_str
    assert "Gridded: 1 time slices" in repr_str
    assert "Impacts: ['hdd']" in repr_str


# Test EnsembleOutput container class.


def test_initialization_empty():
    """Test basic initialization."""
    ensemble = EnsembleOutput()

    assert ensemble.variables == {}
    assert ensemble.metadata == {}


def test_initialization_with_data():
    """Test initialization with variables and metadata."""
    var_tas = VariableOutput("tas")
    var_pr = VariableOutput("pr")
    results = {"tas": var_tas, "pr": var_pr}
    metadata = {"scenario": "ssp245", "n_realizations": 10}

    ensemble = EnsembleOutput(results, metadata)

    assert len(ensemble.variables) == 2
    assert "tas" in ensemble.variables
    assert "pr" in ensemble.variables
    assert ensemble.metadata["scenario"] == "ssp245"
    assert ensemble.metadata["n_realizations"] == 10


def test_getitem():
    """Test accessing variables via indexing."""
    var_tas = VariableOutput("tas")
    ensemble = EnsembleOutput({"tas": var_tas})

    retrieved = ensemble["tas"]
    assert retrieved.variable == "tas"
    assert retrieved is var_tas


def test_getitem_missing_key():
    """Test accessing non-existent variable raises KeyError."""
    ensemble = EnsembleOutput()

    with pytest.raises(KeyError):
        _ = ensemble["nonexistent"]


def test_contains():
    """Test 'in' operator."""
    var_tas = VariableOutput("tas")
    ensemble = EnsembleOutput({"tas": var_tas})

    assert "tas" in ensemble
    assert "pr" not in ensemble


def test_list_variables_empty():
    """Test list_variables when empty."""
    ensemble = EnsembleOutput()
    assert ensemble.list_variables() == []


def test_list_variables_populated():
    """Test list_variables with data."""
    var_tas = VariableOutput("tas")
    var_pr = VariableOutput("pr")
    ensemble = EnsembleOutput({"tas": var_tas, "pr": var_pr})

    var_list = ensemble.list_variables()
    assert len(var_list) == 2
    assert "tas" in var_list
    assert "pr" in var_list


def test_repr_minimal():
    """Test __repr__ with minimal metadata."""
    ensemble = EnsembleOutput()
    repr_str = repr(ensemble)

    assert "EnsembleOutput" in repr_str
    assert "Variables: []" in repr_str


def test_repr_with_metadata():
    """Test __repr__ with full metadata."""
    var_tas = VariableOutput("tas")
    metadata = {
        "scenario": "ssp245",
        "n_realizations": 50,
        "year_range": (2000, 2100),
    }
    ensemble = EnsembleOutput({"tas": var_tas}, metadata)
    repr_str = repr(ensemble)

    assert "EnsembleOutput" in repr_str
    assert "Variables: ['tas']" in repr_str
    assert "Scenario: ssp245" in repr_str
    assert "Realizations: 50" in repr_str
    assert "Years: (2000, 2100)" in repr_str


def test_to_netcdf_basic():
    """Test saving to netCDF with basic data."""
    # Create mock variable output
    var_tas = VariableOutput("tas")
    var_tas.timeseries["global"] = xr.DataArray(
        np.array([290.0, 290.5, 291.0]),
        dims=["time"],
        coords={"time": [2020, 2030, 2040]},
    )

    ensemble = EnsembleOutput({"tas": var_tas})

    # Use temporary file
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    # Patch print to avoid output
    with patch("builtins.print"):
        ensemble.to_netcdf(tmp_path, include_impacts=False)

    # Verify file was created
    assert os.path.exists(tmp_path)

    # Verify we can load it back
    loaded = xr.open_dataset(tmp_path)
    assert "tas_global" in loaded.data_vars
    assert len(loaded["tas_global"]) == 3
    loaded.close()

    # Clean up
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_to_netcdf_with_special_characters():
    """Test that special characters in names are sanitized."""
    var_tas = VariableOutput("tas")
    # Name with special characters that need sanitizing
    var_tas.timeseries["regional:EAS"] = xr.DataArray(
        np.array([290.0, 290.5]), dims=["time"], coords={"time": [2020, 2030]}
    )
    var_tas.timeseries["point:59.9,10.8"] = xr.DataArray(
        np.array([288.0, 288.5]), dims=["time"], coords={"time": [2020, 2030]}
    )

    ensemble = EnsembleOutput({"tas": var_tas})

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    with patch("builtins.print"):
        ensemble.to_netcdf(tmp_path, include_impacts=False)

    # Verify file was created and names were sanitized
    loaded = xr.open_dataset(tmp_path)
    # Colons should be replaced with underscores
    assert "tas_regional_EAS" in loaded.data_vars
    # Dots should be replaced with 'p'
    assert (
        "tas_point_59p9,10p8" in loaded.data_vars
        or "tas_point_59p9_10p8" in loaded.data_vars
    )
    loaded.close()

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_to_netcdf_with_impacts():
    """Test saving with impact metrics included."""
    var_tas = VariableOutput("tas")
    var_tas.timeseries["global"] = xr.DataArray([290.0], dims=["time"])
    var_tas.impacts["hdd"] = {"global": xr.DataArray([1500.0], dims=["time"])}

    ensemble = EnsembleOutput({"tas": var_tas})

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    with patch("builtins.print"):
        ensemble.to_netcdf(tmp_path, include_impacts=True)

    loaded = xr.open_dataset(tmp_path)
    assert "tas_global" in loaded.data_vars
    assert "tas_hdd_global" in loaded.data_vars
    loaded.close()

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_to_netcdf_without_impacts():
    """Test saving with impact metrics excluded."""
    var_tas = VariableOutput("tas")
    var_tas.timeseries["global"] = xr.DataArray([290.0], dims=["time"])
    var_tas.impacts["hdd"] = {"global": xr.DataArray([1500.0], dims=["time"])}

    ensemble = EnsembleOutput({"tas": var_tas})

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    with patch("builtins.print"):
        ensemble.to_netcdf(tmp_path, include_impacts=False)

    loaded = xr.open_dataset(tmp_path)
    assert "tas_global" in loaded.data_vars
    # Impact should not be included
    assert "tas_hdd_global" not in loaded.data_vars
    loaded.close()

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_to_netcdf_with_metadata_attributes():
    """Test that metadata is saved as attributes."""
    var_tas = VariableOutput("tas")
    var_tas.timeseries["global"] = xr.DataArray([290.0], dims=["time"])

    metadata = {
        "scenario": "ssp245",
        "n_realizations": 10,
        "model": "CESM2",
        "some_float": 3.14,
        "some_list": [1, 2, 3],  # Lists won't be saved as attrs
    }
    ensemble = EnsembleOutput({"tas": var_tas}, metadata)

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    with patch("builtins.print"):
        ensemble.to_netcdf(tmp_path, include_impacts=False)

    loaded = xr.open_dataset(tmp_path)
    # String/int/float metadata should be in attributes
    assert loaded.attrs.get("scenario") == "ssp245"
    assert loaded.attrs.get("n_realizations") == 10
    assert loaded.attrs.get("model") == "CESM2"
    assert loaded.attrs.get("some_float") == 3.14
    # List won't be saved (not a simple type)
    assert "some_list" not in loaded.attrs
    loaded.close()

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_to_netcdf_multiple_variables():
    """Test saving multiple variables to one file."""
    var_tas = VariableOutput("tas")
    var_tas.timeseries["global"] = xr.DataArray([290.0], dims=["time"])

    var_pr = VariableOutput("pr")
    var_pr.timeseries["global"] = xr.DataArray([2e-6], dims=["time"])

    ensemble = EnsembleOutput({"tas": var_tas, "pr": var_pr})

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    with patch("builtins.print"):
        ensemble.to_netcdf(tmp_path, include_impacts=False)

    loaded = xr.open_dataset(tmp_path)
    # Both variables should be present
    assert "tas_global" in loaded.data_vars
    assert "pr_global" in loaded.data_vars
    loaded.close()

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
