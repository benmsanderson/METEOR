import os

import pandas as pd
import pytest

from meteor.scm_input_lib import (
    load_emissions_concentrations,
    load_emissions_concentrations_from_name,
    parse_scenario_input,
)


def test_parse_scenario_input():
    """Test the _parse_scenario_input function."""

    # Test with string input
    scenario_str = "ssp245"
    result = parse_scenario_input(scenario_str)
    assert result["name"] == "ssp245"
    assert result["emissions"] is None
    assert result["concentrations"] is None

    # Test with dict input
    scenario_dict = {
        "name": "custom_scenario",
        "emissions": "/path/to/emissions.csv",
        "concentrations": "/path/to/concentrations.csv",
    }
    result = parse_scenario_input(scenario_dict)
    assert result["name"] == "custom_scenario"
    assert result["emissions"] == "/path/to/emissions.csv"
    assert result["concentrations"] == "/path/to/concentrations.csv"
    assert result["type"] == "custom"

    scenario_dict_no_conc = {
        "emissions": "/path/to/emissions.csv",
    }
    result = parse_scenario_input(scenario_dict_no_conc)
    assert result["name"] == "custom"
    assert result["emissions"] == "/path/to/emissions.csv"
    assert result["concentrations"].endswith("default_scm_data/ssp245_conc_RCMIP.txt")

    # Test with dictionary missing emissions key
    with pytest.raises(
        ValueError, match="Custom scenario dict must include 'emissions' key"
    ):
        parse_scenario_input({"Hello": "invalid"})  # Invalid type

    with pytest.raises(TypeError, match="scenario must be str or dict"):
        parse_scenario_input(12345)  # Invalid type


def test_load_emissions_concentrations(test_data_dir):
    """Test the _load_emissions_concentrations function."""
    # Create temporary emissions and concentrations files
    df_min = pd.DataFrame({"year": [2000, 2001], "CO2": [10, 11], "CH4": [1, 1.1]})

    with pytest.raises(TypeError, match="emissions must be str path or DataFrame"):
        load_emissions_concentrations(123, 123)
    with pytest.raises(TypeError, match="concentrations must be str path or DataFrame"):
        load_emissions_concentrations(df_min, 123)
    em_data, conc_data = load_emissions_concentrations(df_min, df_min, verbose=True)
    pd.testing.assert_frame_equal(em_data, df_min)
    pd.testing.assert_frame_equal(conc_data, df_min)
    em_data, conc_data = load_emissions_concentrations(df_min, df_min)
    pd.testing.assert_frame_equal(em_data, df_min)
    pd.testing.assert_frame_equal(conc_data, df_min)

    em_data, conc_data = load_emissions_concentrations(
        os.path.join(test_data_dir, "rcp85_em_RCMIP.txt"),
        os.path.join(test_data_dir, "rcp85_conc_RCMIP.txt"),
    )
    assert "CO2_FF" in em_data.columns
    assert "CH4" in em_data.columns
    assert "CO2" in conc_data.columns
    assert "CH4" in conc_data.columns
    print(em_data)
    print(conc_data)
    # Updated to match actual data dimensions (1750-2500 = 751 years for emissions)
    assert em_data.shape == (751, 40)
    assert conc_data.shape == (801, 30)

    em_data2, conc_data2 = load_emissions_concentrations(
        os.path.join(test_data_dir, "rcp85_em_RCMIP.txt"),
        os.path.join(test_data_dir, "rcp85_conc_RCMIP.txt"),
        verbose=True,
    )
    pd.testing.assert_frame_equal(em_data, em_data2)
    pd.testing.assert_frame_equal(conc_data, conc_data2)


def test_load_emissions_concentrations_from_name(test_data_dir):
    """Test the load_emissions_concentrations_from_name function."""
    with pytest.raises(FileNotFoundError, match="Concentration file not found"):
        load_emissions_concentrations_from_name("nonexistent_scenario")
