import os
from meteor import MeteorPatternScaling
    

def test_meteor_scaling(test_data_dir):

    
    canesm_basic_pattern = MeteorPatternScaling("pdrmip-CanESM2-basic", {"tas":2, "pr":10}, lambda exp : os.path.join(test_data_dir, f"pdrmip-{exp}_T42_ANN_CanESM2.nc"), exp_list = ["base", "co2x2"] )
    assert canesm_basic_pattern.name == "pdrmip-CanESM2-basic"
    assert "base" in canesm_basic_pattern.od
    assert "tas" in canesm_basic_pattern.od["co2x2"]
    assert "outp" in canesm_basic_pattern.od["co2x2"]["pr"]
