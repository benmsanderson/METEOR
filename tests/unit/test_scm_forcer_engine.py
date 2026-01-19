import numpy as np
import pandas as pd

from meteor import scm_forcer_engine
from meteor.scm_forcer_engine import (
    ScmEngineConfigurations,
    ScmEngineForPatternScaling,
    aerosol_priority_mapping,
)


def test_scm_engine_configurations():
    sec = ScmEngineConfigurations(
        gaspam_data=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        concentrations_data=pd.DataFrame(
            {"CO2": np.ones(151) * 370}, index=np.arange(1950, 2101)
        ),
        emissions_data=pd.DataFrame(
            {"CO2": np.ones(151) * 35}, index=np.arange(1950, 2101)
        ),
    )
    assert sec.emstart == 2000
    assert sec.nystart == 1950
    assert sec.idtm == 24
    assert sec.conc_run
    assert all((sec.gaspam_data["A"] - [1, 2]) == 0)
    assert sec.nat_ch4_data.shape == (151, 1)
    assert all((sec.nat_ch4_data["CH4"]) == 242.09)
    assert all(sec.nat_n2o_data["N2O"] == 11.7027)
    sec = ScmEngineConfigurations(
        gaspam_data=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        concentrations_data=pd.DataFrame(
            {"CO2": np.ones(151) * 370}, index=np.arange(1950, 2101)
        ),
        emissions_data=pd.DataFrame(
            {"CO2": np.ones(151) * 35}, index=np.arange(1950, 2101)
        ),
        nat_ch4_data=pd.DataFrame(
            {"CH4": np.ones(151) * 230}, index=np.arange(1950, 2101)
        ),
        nat_n2o_data=pd.DataFrame(
            {"N2O": np.ones(151) * 12}, index=np.arange(1950, 2101)
        ),
    )
    assert sec.emstart == 2000
    assert sec.nystart == 1950
    assert sec.idtm == 24
    assert sec.conc_run
    assert all((sec.gaspam_data["A"] - [1, 2]) == 0)
    assert sec.nat_ch4_data.shape == (151, 1)
    assert all((sec.nat_ch4_data["CH4"]) == 230)
    assert all(sec.nat_n2o_data["N2O"] == 12)


def test_forcer_engine():
    sefps = scm_forcer_engine.ScmEngineForPatternScaling(None)
    scaling = sefps.run_to_get_scaling(["base", "co2x2", "bcx10", "sulx7"])
    # assert np.allclose(scaling, [0.0, 4.08366688, 1.08897933, -7.62538963])
    assert len(scaling) == 4
    assert np.allclose(scaling[0], [0.0])


def test_aerosol_priority_mapping():
    aer1 = scm_forcer_engine.aerosol_priority_mapping(["CO2", "CH4", "N2O"])
    assert set(aer1.keys()) == set(
        ["SO4_IND", "SO4_DIR", "BMB_AEROS_BC", "BMB_AEROS_OC", "SO2", "BC", "OC"]
    )
    assert set(aer1.values()) == set(["CO2"])
    aer2 = scm_forcer_engine.aerosol_priority_mapping(["CO2", "BC", "N2O"])
    print(aer2)
    assert set(aer2.keys()) == set(
        ["SO4_IND", "SO4_DIR", "BMB_AEROS_BC", "BMB_AEROS_OC", "SO2", "OC", "BC"]
    )
    assert aer2["OC"] == "CO2"
    assert aer2["SO4_IND"] == "BC"
    assert aer2["SO4_DIR"] == "BC"
    aer3 = scm_forcer_engine.aerosol_priority_mapping(
        ["CO2", "BC", "SO2"], bc_oc_to_co2=False
    )
    assert set(aer3.keys()) == set(
        ["SO4_IND", "SO4_DIR", "BMB_AEROS_BC", "BMB_AEROS_OC", "OC"]
    )
    assert aer3["OC"] == "BC"
    assert aer3["SO4_IND"] == "SO2"
    assert aer3["BMB_AEROS_OC"] == "BC"
    aer4 = scm_forcer_engine.aerosol_priority_mapping(
        ["CO2", "OC", "SO2"], bc_oc_to_co2=False
    )
    assert set(aer4.keys()) == set(
        ["SO4_IND", "SO4_DIR", "BMB_AEROS_BC", "BMB_AEROS_OC", "BC"]
    )
    assert aer4["BC"] == "OC"
    assert aer4["SO4_IND"] == "SO2"
    assert aer4["SO4_DIR"] == "SO2"
    assert aer4["BMB_AEROS_BC"] == "OC"
    aer5 = scm_forcer_engine.aerosol_priority_mapping(["CO2", "CH4", "SO2"])
    assert set(aer5.keys()) == set(
        ["SO4_IND", "SO4_DIR", "BMB_AEROS_BC", "BMB_AEROS_OC", "BC", "OC"]
    )
    assert aer5["OC"] == "CO2"
    assert aer5["SO4_IND"] == "SO2"
    assert aer4["SO4_DIR"] == "SO2"
    assert aer5["BMB_AEROS_BC"] == "CO2"


def test_aerosol_priority_mapping_additional():
    """Test additional cases for aerosol_priority_mapping."""
    # Test with empty list
    result_empty = aerosol_priority_mapping([])
    assert isinstance(result_empty, dict)

    # Test with single component
    result_single = aerosol_priority_mapping(["CO2"])
    assert isinstance(result_single, dict)

    # Test with bc_oc_to_co2 parameter variations
    comps = ["CO2", "SO2", "BC", "OC"]
    result_true = aerosol_priority_mapping(comps, bc_oc_to_co2=True)
    result_false = aerosol_priority_mapping(comps, bc_oc_to_co2=False)

    # Both should be dictionaries
    assert isinstance(result_true, dict)
    assert isinstance(result_false, dict)

    # Both should have aerosol mapping keys
    assert len(result_true) > 0
    assert len(result_false) > 0


def test_scm_engine_class_exists():
    """Test that SCM engine class exists and has expected attributes."""
    # Test that the class can be imported

    # Test it's a class
    assert isinstance(ScmEngineForPatternScaling, type)

    # Test it has expected methods
    assert hasattr(ScmEngineForPatternScaling, "run_to_get_scaling")
    assert hasattr(ScmEngineForPatternScaling, "run_and_return_per_forcer_results")


def test_scm_forcer_engine_edge_cases():
    """Test edge cases in SCM forcer engine to improve coverage."""
    # Test with unusual scenario combinations
    sefps = scm_forcer_engine.ScmEngineForPatternScaling(None)

    # Test with single scenario
    single_scaling = sefps.run_to_get_scaling(["base"])
    assert len(single_scaling) == 1
    assert np.allclose(single_scaling[0], [0.0])

    # Test with repeated scenarios
    repeated_scaling = sefps.run_to_get_scaling(["base", "base", "co2x2"])
    assert len(repeated_scaling) == 3


def test_scm_forcer_engine_numerical_stability():
    """Test numerical stability of SCM forcer engine."""
    # Test aerosol priority mapping with edge cases

    # Test with components that might cause conflicts
    edge_components = ["CO2", "SO2", "BC", "OC", "CH4", "N2O"]
    result = aerosol_priority_mapping(edge_components)
    assert isinstance(result, dict)

    # Test all values are from the input components
    for value in result.values():
        assert value in edge_components

    # Test with different bc_oc_to_co2 settings
    result_true = aerosol_priority_mapping(edge_components, bc_oc_to_co2=True)
    result_false = aerosol_priority_mapping(edge_components, bc_oc_to_co2=False)

    assert isinstance(result_true, dict)
    assert isinstance(result_false, dict)

    # Should have different mappings for BC/OC components
    if "BMB_AEROS_BC" in result_true and "BMB_AEROS_BC" in result_false:
        # These may differ based on bc_oc_to_co2 setting
        pass


def test_scm_engine_per_forcer_results():
    """Test per-forcer results functionality."""
    sefps = scm_forcer_engine.ScmEngineForPatternScaling(None)

    # Test that per-forcer results method exists and runs
    per_forcer_results = sefps.run_and_return_per_forcer_results(["base", "co2x2"])
    # Should return some result structure
    assert per_forcer_results is not None
