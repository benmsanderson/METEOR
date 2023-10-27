import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from ciceroscm import input_handler

from meteor import MeteorPatternScaling
from meteor import prpatt


def make_pattern(name, exp_list):
    
    pattern = MeteorPatternScaling(name, {"tas":2, "pr":10}, lambda exp: os.path.join("/div/no-backup/users/masan/temp", f"pdrmip-{exp}_T42_ANN_NorESM1.nc"), exp_list = exp_list)
    return pattern

def predict_per_experiment(pattern, experiment):
    conc_data = input_handler.read_inputfile(
    os.path.join("/div/amoc/CSCM/SCM_Linux_v2019/RCMIP/input", f"{experiment}_conc_RCMIP.txt")
)
    ih = input_handler.InputHandler({})
    em_data = ih.read_emissions(os.path.join("../tests/test-data", f"{experiment}_em_RCMIP.txt"))
    patterns = pattern.predict_from_combined_experiment(em_data, conc_data, ["pr", "tas"])
    return patterns
#noresm_pattern = make_pattern("pdrmip-NorESM1-all", ["base", "co2x2", "bcx10", "sulx5"])
#patterns = predict_per_experiment(noresm_pattern, "rcp85")

def add_rcp_data(ax, comp, exp, top_figure_ax):
    data_hist = xr.open_dataset(f"/div/no-backup/users/masan/temp/{comp}_yr_NorESM1-M_historical_r1i1p1_185001-200512.nc")[comp]
    data_rcp = xr.open_dataset(f"/div/no-backup/users/masan/temp/{comp}_yr_NorESM1-M_{exp}_r1i1p1_200601-210012.nc")[comp]
    hist_ts = prpatt.global_mean(data_hist).values
    tot_ts = np.concatenate((hist_ts - hist_ts[0], prpatt.global_mean(data_rcp).values - hist_ts[0]))
    top_figure_ax.plot(np.arange(1850,2101), tot_ts, label="CMIP5")
    mean_change = data_rcp[-20:,:,:].mean(dim="time") - data_hist[:50,:,:].mean(dim="time")
    if comp == "tas":
        mean_change.plot(ax=ax, cmap="bwr", vmin = 0)
    else:
        mean_change.plot(ax=ax, cmap="bwr")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    

def add_pattern_data(patterns, ax, comp, top_figure_ax, label):
    top_figure_ax.plot(np.arange(1750,2101), prpatt.global_mean(patterns[comp]), linestyle = "--", alpha= 0.8, label=label)
    mean_change = patterns[comp][-20:,:,:].mean(dim="time") - patterns[comp][100:150,:,:].mean(dim="time")
    mean_change.plot(ax=ax, cmap="bwr")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    
def make_labels_and_suptitles(exp_lists):
    suptitles = ["Global mean change"]
    labels = []
    for def_exp in exp_lists:
        label = ""
        strn = "Pattern with "
        for i, sub_e in enumerate(def_exp):
            if i == 0:
                continue
            compound = sub_e.split("x")[0]
            if i == 1:
                label = f"{label}{compound.upper()}"
                strn = f"{strn}{compound}"
            elif i < len(def_exp)-1:
                label = f"{label}+{compound.upper()}"
                strn = f"{strn}, {compound}"
            else:
                label = f"{label}+{compound.upper()}"
                strn = f"{strn} and {compound}"
        labels.append(label)
        suptitles.append(strn)
    suptitles.append("CMIP5 1850-1900 vs 2081-2100")
    return suptitles, labels

experiments = ["rcp85", "rcp60", "rcp45", "rcp26"]
fig = plt.figure(constrained_layout=True, figsize=(40,30))
fig.suptitle("METEOR Pattern scaling for NorESM1-M")
exp_lists = [["base","co2x2"], ["base","co2x2", "bcx10"],  ["base","co2x2", "sulx5"], ["base","co2x2", "bcx10", "sulx5"]] # ,
rownum = len(exp_lists) + 2
subfigs = fig.subfigures(nrows = rownum, ncols=len(experiments))
suptitles, labels = make_labels_and_suptitles(exp_lists)
components = ["tas", "pr"]
top_figures = []

for row in range(rownum):
    if row >2 and row < (rownum -1):
        n_pattern = make_pattern("pdrmip-NorESM1-all", exp_lists[row-1])
    for column in range(len(experiments)):
        exp = experiments[column]
        if row >0 and row < rownum -1:
            patterns = predict_per_experiment(n_pattern, exp)     
        subfig = subfigs[row,column]
        subfig.suptitle(f"{suptitles[row]} {exp}")
        axs = subfig.subplots(nrows = 1, ncols = 2)
        for col, ax in enumerate(axs):
            comp = components[col]
            if row == 0:
                top_figures.append(ax)
                ax.plot()
                ax.set_xlabel("Years")
            elif row == rownum -1:
                add_rcp_data(ax, comp, exp, top_figures[column*2 + col])
            else:
                add_pattern_data(patterns, ax, comp, top_figures[column*2+col], labels[row-1])
            ax.set_title(comp)

for ax in top_figures:
    ax.legend()
plt.savefig("predicted_noresm_all_exp.png")
#plt.show()
