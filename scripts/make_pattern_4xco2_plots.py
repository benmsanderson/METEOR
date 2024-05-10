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
from meteor import meteor_plot_utils


cscm_data_dir = "/mnt/c/Users/masan/Downloads/Input_for_scenarios/"
#plt.show()
# 
# ,
#
# 'CMIP',
data_getter = Cmip6MeteorDataGetter(exps=["piControl", "abrupt-4xCO2"], dbe=['CMIP','CMIP'])
models = data_getter.models
models = ["CanESM5", "NorESM2-LM"]
flds = ["tas"]#], "pr"]

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
fig, axs = plt.subplots(nrows=len(flds)*len(models), ncols=3, sharex=True)
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
    for j, fld in enumerate(flds):
        truth_abrupt = prpatt.global_mean(data_getter.get_single_var_mod_data_yearmean("abrupt-4xCO2", fld, model)).values[0]
        truth_picontrol = prpatt.global_mean(data_getter.get_single_var_mod_data_yearmean("piControl", fld, model)).values[0]
        years_total = range(len(truth_abrupt))
        years_pic = range(len(truth_picontrol))
        data_in_pattern = prpatt.global_mean(model_basic_pattern.dacanom[fld][model_basic_pattern.exp_list.index('co2x4'), :100, :, :])
        #print(f"Variable: {fld} and model {model}: {hist_time}")
        axs[i*len(flds)+j,0].plot(years_total, truth_abrupt - truth_picontrol[:len(truth_abrupt)], color="red", label="CMIP6_abrupt")
        axs[i*len(flds)+j,2].plot(years_pic, truth_picontrol - truth_picontrol[0], color="red", label="CMIP6_piControl")
        meteor_plot_utils.plot_global_mean_values(model_basic_pattern, axs[i*len(flds)+j,0], fld, 'co2x4')
        axs[i*len(flds)+j,1].plot(range(100), truth_abrupt[:100] - data_in_pattern)
        meteor_plot_utils.plot_global_mean_values(model_basic_pattern, axs[i*len(flds)+j,2], fld, 'base')
        axs[i*len(flds)+j,0].set_ylabel(f"{model} and {fld}")
        
        axs[i,j].set_xlim(-5,50)
axs[0,0].set_title(f"Global mean direct")
axs[0,1].set_title(f"Diff data in pattern and outside")
axs[0,0].legend()
#axs[0,1].set_title(f"Automatic global mean plot")
#axs[0,2].set_title(f"Truth map")
#axs[0,2].set_title(f"Principal component map 1")
plt.tight_layout()
plt.savefig("CanESM5_NorESM_abrupt-4xco2_reconstruction.png")
