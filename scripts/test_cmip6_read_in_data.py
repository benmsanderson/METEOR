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

def year_mean_monthly_1d(monthly_data):
    """
    Calculate yearmean from monthly data

    Weighting by days in month and calculating the yearly mean of an array of
    monthly data. The data are assumed to be January to December per year, and
    will assume for simplicity that the data follows a no-leap calendar

    Parameters
    ----------
    monthly_data : np.ndarray
        1 or multiple dimensional np.ndarray with the first dimension being time
        and on monthly resolutions running from January to December for each year

    Returns
    -------
    np.ndarray
        Weighted year averages for the monthly_data, the output-dimension will be
        the same as for the monthly_data, except that the first time dimension
        will be 1/12th as long as before including only yearly mean values
    """
    month_weights = np.tile(
        np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) / 365.0*12.,
        monthly_data.shape[0] // 12,
    )
    print(month_weights)
    print(sum(month_weights))
    mul_weigths = multiply_along_axis(monthly_data, month_weights, 0)
    print(mul_weigths)
    return np.mean(
        mul_weigths.reshape(
            -1,
            12,
        ),
        axis=1,
    )

test_1 = np.ones(24)
print(test_1)
test_1_mean = year_mean_monthly_1d(test_1)
print(test_1_mean)
#sys.exit(4)
#plt.show()
# 
# ,
#
# 'CMIP',
exps = ["piControl", "abrupt-4xCO2", "historical", "ssp245"]
data_getter = Cmip6MeteorDataGetter(flds=['tas'], exps=exps, dbe=['CMIP','CMIP','CMIP', 'ScenarioMIP'])
print(data_getter.models)
models = ["CanESM5", "NorESM2-LM"]

fig, axs = plt.subplots(nrows=len(models), ncols=len(exps))
#years_total = np.arange(1750, 2101)
#years_hist = np.arange(1850, 2015)
#years_ssp = np.arange(2015, 2101)
for i, model in enumerate(models):
    # Build pattern 
    for j, exp in enumerate(exps):
        full_data = data_getter.make_meteor_training_data(exp, model)
        glob_mean = prpatt.global_mean(full_data['tas']).values[0]
        full_data_yearly = data_getter.get_single_var_mod_data_yearmean(exp, 'tas', model)
        full_data_monthly = data_getter.get_single_var_mod_data(exp, 'tas', model)
        #print(full_data.shape)
        print(full_data_yearly)
        years = np.arange(len(glob_mean))
        months = np.linspace(years[0], years[-1]+11/12., num = len(years)*12)
        axs[i, j].plot(years,glob_mean, label ="glob_mean")
        axs[i,j].plot(years, prpatt.global_mean(full_data_yearly).values[0], label="yearmean_single")
        monthly_glob_mean = prpatt.global_mean(full_data_monthly)['tas'].values
        axs[i,j].plot(months, monthly_glob_mean, label="monthly")
        axs[i,j].plot(years, year_mean_monthly_1d(monthly_glob_mean), label="mean_of_monthly")
        if i == 0:
            axs[i, j].set_title(exp)
        if j == 0:
            axs[i, j].set_ylabel(model)
axs[-1,-1].legend()
plt.savefig("test_input_data_cmip6.png")