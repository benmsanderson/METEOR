import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from functools import partial
from ciceroscm import input_handler

from meteor import MeteorPatternScaling
from meteor import prpatt
from meteor import Cmip6MeteorDataGetter


cscm_data_dir = "/mnt/c/Users/masan/Downloads/Input_for_scenarios/"
#plt.show()
# 
# ,
#
# 'CMIP',
data_getter = Cmip6MeteorDataGetter(exps=["piControl", "abrupt-4xCO2", "historical", "ssp245"], dbe=['CMIP','CMIP','CMIP', 'ScenarioMIP'])
models = data_getter.models
models = ["CanESM5", "NorESM2-LM"]
flds = ["tas", "pr"]

#For predictions: 
conc_data = input_handler.read_inputfile(
    os.path.join(cscm_data_dir, "ssp245_conc_RCMIP.txt")
)
ih = input_handler.InputHandler({})
em_data = ih.read_emissions(os.path.join(cscm_data_dir, "ssp245_em_RCMIP.txt"))

fig, axs = plt.subplots(nrows=len(flds), ncols=len(models), sharex=True)
years_total = np.arange(1750, 2101)
years_hist = np.arange(1850, 2015)
years_ssp = np.arange(2015, 2101)
for i, model in enumerate(models):
    # Build pattern 
    model_basic_pattern = MeteorPatternScaling(
        f"cmip6-{model}-basic",
        {"tas": 2, "pr": 10},
        partial(data_getter.make_meteor_training_data, model=model),
        from_file=False,
        exp_list=["base", "co2x4"],
        )
    # Predict for ssp
    pattern_ssp = model_basic_pattern.predict_from_combined_experiment(
        em_data, conc_data, ["pr", "tas"]
        )
    # Get true output
    truth_historical = data_getter.make_meteor_training_data("historical", model)
    #sys.exit(4)
    truth_ssp = data_getter.make_meteor_training_data("ssp245", model)
    for j, fld in enumerate(flds):
        hist_time = prpatt.global_mean(truth_historical[fld]).values[0]
        hist_time_start = np.mean(hist_time[:50])
        print(f"Variable: {fld} and model {model}: {hist_time}")
        axs[j,i].plot(years_total, prpatt.global_mean(pattern_ssp[fld]).values, color="blue", label="METEOR")
        axs[j,i].plot(years_hist, hist_time - hist_time_start, color="red", label="CMIP6")
        axs[j,i].plot(years_ssp, prpatt.global_mean(truth_ssp[fld]).values[0] - hist_time_start, color="red")
        axs[j,i].set_xlim(left=1850, right=2100)
        axs[j,i].set_xlabel("Years")
        if j == 0:
            axs[j,i].set_title(f"Results for {model}")
        if i == 0:
            axs[j,i].set_ylabel(f"Global mean of {fld}")
        axs[i,j].legend()
plt.savefig("meteor_patterns_cmip6.png")
