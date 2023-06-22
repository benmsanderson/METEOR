import numpy as np

from meteor import scm_forcer_engine


def test_forcer_engine():
    sefps = scm_forcer_engine.ScmEngineForPatternScaling(None)
    scaling = sefps.run_to_get_scaling(["base", "co2x2", "bcx10", "sulx7"])
    print(scaling)
    assert np.allclose(scaling, [0., 4.08366688, 1.08897933, -7.62538963])
