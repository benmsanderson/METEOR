import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from functools import partial
from ciceroscm import input_handler

from meteor import prpatt
from meteor import Cmip6MeteorDataGetter
from meteor.cmip6_meteor_data_getter import multiply_along_axis

# plt.show()
#
# ,
#
# 'CMIP',
fld = "tas"
exps_to_show = ["abrupt-4xCO2", "historical", "ssp245"]
exps = ["abrupt-4xCO2", "historical", "ssp245", "piControl"]
data_getter = Cmip6MeteorDataGetter(
    flds=[fld], exps=exps, dbe=["CMIP", "CMIP", "ScenarioMIP", "CMIP"]
)  #'CMIP',
models_total = data_getter.get_models_avail()
print(models_total)
to_remove = [
    "CNRM-CM6-1",
    "CNRM-CM6-1-HR",
    "EC-Earth3-CC",
    "EC-Earth3-Veg",
]  # , 'CNRM-CM6-1-HR', "CNRM-ESM2-1"]
for model in to_remove:
    models_total.remove(model)
print(models_total)
# sys.exit(4)
tot_mod_num = len(models_total)
print(tot_mod_num)
# sys.exit(4)
for k in range(np.ceil(tot_mod_num / 4.0).astype("int")):
    if (k + 1) * 4 < tot_mod_num:
        models = models_total[k * 4 : (k + 1) * 4]
    else:
        models = models_total[k * 4 :]
    filename = f"test_input_data_cmip6_{'_'.join(models)}.png"
    if os.path.exists(filename):
        continue
    fig, axs = plt.subplots(nrows=len(models), ncols=len(exps_to_show), sharey=True)
    # years_total = np.arange(1750, 2101)
    # years_hist = np.arange(1850, 2015)
    # years_ssp = np.arange(2015, 2101)
    for i, model in enumerate(models):
        # Build pattern
        print(model)
        zero_val = np.mean(
            prpatt.global_mean(
                data_getter.get_single_var_mod_data_yearmean("piControl", fld, model)
            ).values
        )
        for j, exp in enumerate(exps_to_show):

            glob_mean = prpatt.global_mean(
                data_getter.make_meteor_training_data(exp, model)[fld]
            ).values[0]
            print(exp)
            years = np.arange(len(glob_mean))
            months = np.linspace(years[0], years[-1] + 11 / 12.0, num=len(years) * 12)
            monthly_glob_mean = prpatt.global_mean(
                data_getter.get_single_var_mod_data(exp, fld, model)
            )[fld].values
            axs[i, j].plot(months, monthly_glob_mean - zero_val, label="monthly")
            axs[i, j].plot(years, glob_mean - zero_val, label="glob_mean")
            # axs[i,j].plot(years, prpatt.global_mean(full_data_yearly).values[0], label="yearmean_single")
            # axs[i,j].plot(years, year_mean_monthly_1d(monthly_glob_mean), label="mean_of_monthly")
            if i == 0:
                axs[i, j].set_title(exp)
            if j == 0:
                axs[i, j].set_ylabel(model)
    axs[-1, -1].legend()
    plt.savefig(f"test_input_data_cmip6_{'_'.join(models)}.png")
