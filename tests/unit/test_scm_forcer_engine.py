import numpy as np

from meteor import scm_forcer_engine


def test_forcer_engine():
    sefps = scm_forcer_engine.ScmEngineForPatternScaling(None)
    scaling = sefps.run_to_get_scaling(["base", "co2x2", "bcx10", "sulx7"])
    print(scaling)
    assert np.allclose(scaling, [0.0, 4.08366688, 1.08897933, -7.62538963])


def test_aerosol_priority_mapping():
    aer1 = scm_forcer_engine.aerosol_priority_mapping(["CO2", "CH4", "N2O"])
    assert set(aer1.keys()) == set(["SO4_IND", "BMB_AEROS_BC", "BMB_AEROS_OC"])
    assert set(aer1.values()) == set(["CO2"])
    aer2 = scm_forcer_engine.aerosol_priority_mapping(["CO2", "BC", "N2O"])
    assert set(aer2.keys()) == set(["SO4_IND", "BMB_AEROS_BC", "BMB_AEROS_OC"])
    assert aer2["BMB_AEROS_OC"] == "CO2"
    assert aer2["SO4_IND"] == "CO2"
    assert aer2["BMB_AEROS_BC"] == "BC"
    aer3 = scm_forcer_engine.aerosol_priority_mapping(["CO2", "BC", "SO2"])
    assert set(aer3.keys()) == set(["SO4_IND", "BMB_AEROS_BC", "BMB_AEROS_OC"])
    assert aer3["BMB_AEROS_BC"] == "BC"
    assert aer3["SO4_IND"] == "SO2"
    assert aer3["BMB_AEROS_OC"] == "CO2"
