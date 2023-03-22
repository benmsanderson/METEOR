import os

import numpy as np

from meteor import MeteorPatternScaling


def test_meteor_scaling(test_data_dir):
    canesm_basic_pattern = MeteorPatternScaling(
        "pdrmip-CanESM2-basic",
        {"tas": 2, "pr": 10},
        lambda exp: os.path.join(test_data_dir, f"pdrmip-{exp}_T42_ANN.nc"),
        exp_list={"base": 0, "co2x2": 3.71},
    )
    assert canesm_basic_pattern.name == "pdrmip-CanESM2-basic"
    assert "base" in canesm_basic_pattern.pattern_dict
    assert "tas" in canesm_basic_pattern.pattern_dict["co2x2"]
    assert "outp" in canesm_basic_pattern.pattern_dict["co2x2"]["pr"]

    f = 5 * np.exp(-np.square(np.arange(0, 300) - 150) / 5000)
    Xsim = canesm_basic_pattern.predict_from_forcing_profile(f, "tas")
    assert Xsim.shape[0] == len(f)
