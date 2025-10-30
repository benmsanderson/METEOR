import numpy as np

from meteor import scm_forcer_engine
from meteor.scm_forcer_engine import aerosol_priority_mapping


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
    from meteor.scm_forcer_engine import ScmEngineForPatternScaling

    # Test it's a class
    assert isinstance(ScmEngineForPatternScaling, type)

    # Test it has expected methods
    assert hasattr(ScmEngineForPatternScaling, "run_to_get_scaling")
    assert hasattr(ScmEngineForPatternScaling, "run_and_return_per_forcer_results")
