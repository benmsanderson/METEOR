import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict, dataclass

from functools import partial
from ciceroscm import input_handler

from meteor import MeteorPatternScaling
from meteor import prpatt
from meteor import Cmip6MeteorDataGetter
from meteor import scm_forcer_engine


cscm_data_dir = "/mnt/c/Users/masan/Downloads/Input_for_scenarios/"
#plt.show()
# 
# ,
#
# 'CMIP',
data_getter = Cmip6MeteorDataGetter(exps=["piControl", "abrupt-4xCO2", "historical", "ssp245"], dbe=['CMIP','CMIP','CMIP', 'ScenarioMIP'])
models = data_getter.models
models = ["CanESM5", "IPSL-CM6A-LR"]
flds = ["tas", "pr"]

#For predictions: 
conc_data = input_handler.read_inputfile(
    os.path.join(cscm_data_dir, "ssp245_conc_RCMIP.txt")
)
ih_temp = input_handler.InputHandler({})
em_data = ih_temp.read_emissions(os.path.join(cscm_data_dir, "ssp245_em_RCMIP.txt"))

cfg = {
    "conc_run": False,
    "nystart": em_data.index[0],
    "emstart": em_data.index[0] + 100,
    "nyend": 2100,
    "concentrations_data": conc_data,
    "emissions_data": em_data,
    }
sefps = scm_forcer_engine.ScmEngineForPatternScaling(cfg)
print(asdict(sefps.cfg)['nat_n2o_data'])
#sys.exit(4)
ih = input_handler.InputHandler(asdict(sefps.cfg))
#forcing_series = sefps.run_and_return_per_forcer_results(['base', 'co2x4'])
#print(forcing_series)
#plt.plot(range(em_data.index[0], 2101), forcing_series['co2x4'])
#plt.xlim(1750,1900)
#plt.savefig("Forcing_timeseries.png")
forc_orig =  scm_forcer_engine.run_single_experiment(asdict(sefps.cfg), ih)
print(forc_orig)
for comp, values in forc_orig.items():
    if np.abs(values[-1]) > 0.1:
        plt.plot(range(em_data.index[0], 2101), values, label=comp)
plt.legend()
plt.savefig(f"Forcing_timeseries_components.png")
#sys.exit(4)

"""
models_total = data_getter.get_models_avail()
print(models_total)
to_remove = ['CNRM-CM6-1', 'CNRM-CM6-1-HR','EC-Earth3-CC', 'EC-Earth3-Veg']#, 'CNRM-CM6-1-HR', "CNRM-ESM2-1"]
for model in to_remove:
    if model in models_total:
        models_total.remove(model)
print(models_total)
tot_mod_num = len(models_total)
print(tot_mod_num)
#sys.exit(4)
for k in range(np.ceil(tot_mod_num/4.).astype('int')):
    if (k+1)*4 < tot_mod_num:
        models = models_total[k*4:(k+1)*4]
    else:
        models = models_total[k*4:]
    filename = f"meteor_test_cmip6_{'_'.join(models)}.png"

    if os.path.exists(filename):
        continue
    """
fig, axs = plt.subplots(nrows=len(flds), ncols=len(models), sharex=True)
years_total = np.arange(1750, 2101)
years_hist = np.arange(1850, 2015)
years_ssp = np.arange(2015, 2101)
for i, model in enumerate(models):
    # Build pattern 
    model_basic_pattern = MeteorPatternScaling(
        f"cmip6-{model}-basic",
        {"tas": 2, "pr": 4},
        partial(data_getter.make_meteor_training_data, model=model),
        from_file=False,
        exp_list=["base", "co2x4"],
        )
    print(model_basic_pattern.exp_forc_dict)
    #sys.exit(4)
    # Predict for ssp
    pattern_ssp = model_basic_pattern.predict_from_combined_experiment(
        em_data, conc_data, ["pr", "tas"]
        )
    for j, fld in enumerate(flds):
        zero_val = np.mean(prpatt.global_mean(data_getter.get_single_var_mod_data_yearmean("piControl", fld, model)).values)
        print(zero_val)
        hist_time = prpatt.global_mean(data_getter.get_single_var_mod_data_yearmean("historical", fld, model)).values[0]
        #print(f"Variable: {fld} and model {model}: {hist_time}")
        axs[j,i].plot(years_total, prpatt.global_mean(pattern_ssp[fld]).values, color="blue", label="METEOR")
        print(model)
        axs[j,i].plot(years_hist[:len(hist_time)], hist_time - zero_val, color="red", label="CMIP6")
        axs[j,i].plot(years_ssp, prpatt.global_mean(data_getter.get_single_var_mod_data_yearmean("ssp245", fld, model)).values[0] - zero_val, color="red")
        #axs[j,i].set_xlim(left=1850, right=2100)
        axs[j,i].set_xlabel("Years")
        if j == 0:
            axs[j,i].set_title(f"Results for {model}")
        if i == 0:
            axs[j,i].set_ylabel(f"Global mean of {fld}")
        axs[i,j].legend()
    #print(model_basic_pattern.exp_forc_dict)
plt.savefig("CanESM5_NorESM2_ssp245_pred.png")
#plt.savefig(filename)


