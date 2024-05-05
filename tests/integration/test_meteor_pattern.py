import os
from functools import partial

import numpy as np
import pytest
from ciceroscm import input_handler

from meteor import Cmip6MeteorDataGetter, MeteorPatternScaling


def test_meteor_scaling(test_data_dir):
    canesm_basic_pattern = MeteorPatternScaling(
        "pdrmip-CanESM2-basic",
        {"tas": 2, "pr": 10},
        lambda exp: os.path.join(test_data_dir, f"pdrmip-{exp}_T42_ANN.nc"),
        exp_list=["base", "co2x2"],
    )
    assert canesm_basic_pattern.name == "pdrmip-CanESM2-basic"
    assert "base" in canesm_basic_pattern.pattern_dict
    assert "tas" in canesm_basic_pattern.pattern_dict["co2x2"]
    assert "outp" in canesm_basic_pattern.pattern_dict["co2x2"]["pr"]

    f = 5 * np.exp(-np.square(np.arange(0, 300) - 150) / 5000)
    Xsim = canesm_basic_pattern.predict_from_forcing_profile(f, "tas")
    assert Xsim.shape[0] == len(f)


def test_meteor_scaling_scm_timseries(test_data_dir):
    canesm_basic_pattern = MeteorPatternScaling(
        "pdrmip-CanESM2-basic",
        {"tas": 2, "pr": 10},
        lambda exp: os.path.join(test_data_dir, f"pdrmip-{exp}_T42_ANN.nc"),
        exp_list=["base", "co2x2"],
    )
    assert canesm_basic_pattern.name == "pdrmip-CanESM2-basic"
    assert "base" in canesm_basic_pattern.pattern_dict
    assert "tas" in canesm_basic_pattern.pattern_dict["co2x2"]
    assert "outp" in canesm_basic_pattern.pattern_dict["co2x2"]["pr"]
    conc_data = input_handler.read_inputfile(
        os.path.join(test_data_dir, "rcp85_conc_RCMIP.txt")
    )
    ih = input_handler.InputHandler({})
    em_data = ih.read_emissions(os.path.join(test_data_dir, "rcp85_em_RCMIP.txt"))

    patterns = canesm_basic_pattern.predict_from_combined_experiment(
        em_data, conc_data, ["pr", "tas"]
    )
    assert set(patterns.keys()) == set(["pr", "tas"])


def test_pattern_from_cmip6(test_data_dir):

    datagetter = Cmip6MeteorDataGetter()
    canesm_basic_pattern = MeteorPatternScaling(
        "cmip6-CanESM5-basic",
        {"tas": 2, "pr": 2},
        partial(datagetter.make_meteor_training_data, model="CanESM5"),
        from_file=False,
        exp_list=["base", "co2x4"],
    )
    assert canesm_basic_pattern.name == "cmip6-CanESM5-basic"
    assert "base" in canesm_basic_pattern.pattern_dict
    assert "tas" in canesm_basic_pattern.pattern_dict["co2x4"]
    assert "outp" in canesm_basic_pattern.pattern_dict["co2x4"]["pr"]
    conc_data = input_handler.read_inputfile(
        os.path.join(test_data_dir, "rcp85_conc_RCMIP.txt")
    )
    ih = input_handler.InputHandler({})
    em_data = ih.read_emissions(os.path.join(test_data_dir, "rcp85_em_RCMIP.txt"))

    patterns = canesm_basic_pattern.predict_from_combined_experiment(
        em_data, conc_data, ["pr", "tas"]
    )
    assert set(patterns.keys()) == set(["pr", "tas"])


def test_sulfate_from_residual_functionality(test_data_dir):
    datagetter = Cmip6MeteorDataGetter(
        exps=["piControl", "abrupt-4xCO2", "historical", "ssp245"],
        dbe=["CMIP", "CMIP", "CMIP", "ScenarioMIP"],
    )
    training_data = {
        "base": datagetter.make_meteor_training_data("base", "CanESM5"),
        "co2x4": datagetter.make_meteor_training_data("co2x4", "CanESM5"),
        "sulxanom": datagetter.make_meteor_training_data_composite(
            ["historical", "ssp245"], "CanESM5"
        ),
        "bcxanom": datagetter.make_meteor_training_data_composite(
            ["historical", "ssp245"], "CanESM5"
        ),
    }
    with pytest.raises(RuntimeError):
        canesm_anomsulf_pattern = MeteorPatternScaling(
            "cmip6-CanESM5-anomsulf",
            {"tas": 2, "pr": 2},
            lambda key: training_data[key],
            from_file=False,
            exp_list=["base", "co2x4", "sulxanom", "bcxanom"],
        )
    canesm_anomsulf_pattern = MeteorPatternScaling(
        "cmip6-CanESM5-anomsulf",
        {"tas": 2, "pr": 2},
        lambda key: training_data[key],
        from_file=False,
        exp_list=["base", "co2x4", "sulxanom"],
    )
    assert canesm_anomsulf_pattern.name == "cmip6-CanESM5-anomsulf"
    assert "sulxanom" in canesm_anomsulf_pattern.pattern_dict
    assert "tas" in canesm_anomsulf_pattern.pattern_dict["co2x4"]
    assert "outp" in canesm_anomsulf_pattern.pattern_dict["sulxanom"]["pr"]
