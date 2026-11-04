import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec

from functools import partial
from ciceroscm import input_handler

from meteor import MeteorPatternScaling
from meteor import Cmip6MeteorDataGetter


cscm_data_dir = "/mnt/c/Users/masan/Downloads/Input_for_scenarios/"
#plt.show()
# 
# ,
#
# 'CMIP',
data_getter = Cmip6MeteorDataGetter(exps=["piControl", "abrupt-4xCO2", "historical", "ssp245"], dbe=['CMIP','CMIP', 'CMIP', 'ScenarioMIP'])
models = data_getter.models
models = ["CanESM5"]#, "NorESM2-LM"]
flds = ["tas", "pr"]

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
fig = plt.figure(figsize=(15,15))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.1)
fig, axs = plt.subplots(nrows=len(flds)*len(models), ncols=2, sharex=True)#, projection=ccrs.PlateCarree(central_longitude=180), figsize=(8, 10))
years_total = np.arange(1750, 2101)
years_hist = np.arange(1850, 2015)
years_ssp = np.arange(2015, 2101)
training_data = {
        "base": data_getter.make_meteor_training_data("base", "CanESM5"),
        "co2x4": data_getter.make_meteor_training_data("co2x4", "CanESM5"),
        "sulxanom": data_getter.make_meteor_training_data_composite(
            ["historical", "ssp245"], "CanESM5"
        ),
    }
print(training_data["base"])
lats = training_data["base"].lat.values
lons = training_data["base"].lon.values
#sys.exit(4)
conc_data = input_handler.read_inputfile(
    os.path.join("../tests/test-data", "rcp85_conc_RCMIP.txt")
)
ih = input_handler.InputHandler({})
em_data = ih.read_emissions(os.path.join("/home/masan/gitrepos/cscm-input-data-generation", "baseline_diamond_em_gases_vupdate_2024_WMO_added_new_em_DiamondBaseline.txt"))

# for ax in axs.flatten():
#     ax.set_xticks(np.arange(60, 301, 60), crs=ccrs.PlateCarree())
#     ax.set_xticklabels(["120°W","60°W","0°","60°E","120°E"], size=8)
#     ax.set_yticks(np.arange(-60, 61, 30), crs=ccrs.PlateCarree())
#     ax.set_yticklabels(["60°S","30°S","0°","30°N","60°N"], size=8)
#     ax.coastlines(lw=0.5)
#     ax.set_aspect(1.1)
#     ax.annotate('f', xy=(-0.04, 1.1), xycoords='axes fraction', fontsize=13, ha='left', va='center', weight="bold")
fld_spec_vals = {
    "tas": (np.arange(-10,10.1,1), "RdBu_r", "Temperature change (K)"),
    "pr": (np.arange(-3e-5,3.1e-5,0.5e-5), "BrBG", "Precipitation change (kg / m2 s)"),
    }

for i, model in enumerate(models):
    # Build pattern 
    model_basic_pattern = MeteorPatternScaling(
        f"cmip6-{model}-basic",
        {"tas": 3, "pr": 3},
        lambda key: training_data[key],
        from_file=False,
        exp_list=["base", "co2x4", "sulxanom"],
        )
    patterns = model_basic_pattern.predict_from_combined_experiment(
    em_data, conc_data, ["pr", "tas"]
    )


    for j, fld in enumerate(flds):
        print(patterns[fld])
        print(patterns[fld].values.shape)

        sub1 = fig.add_subplot(gs[i*len(flds)+j, 0], projection=ccrs.PlateCarree(central_longitude=180))
        sub1.set_xticks(np.arange(60, 301, 60), crs=ccrs.PlateCarree())
        sub1.set_xticklabels(["120°W","60°W","0°","60°E","120°E"], size=8)
        sub1.set_yticks(np.arange(-60, 61, 30), crs=ccrs.PlateCarree())
        sub1.set_yticklabels(["60°S","30°S","0°","30°N","60°N"], size=8)
        sub1.coastlines(lw=0.5)
        sub1.set_aspect(1.1)
        #sub1.annotate('f', xy=(-0.04, 1.1), xycoords='axes fraction', fontsize=13, ha='left', va='center', weight="bold")
        value_2050 = np.nanmean(patterns[fld].values[-60:-40, :, :], axis=0) - np.nanmean(patterns[fld].values[:50, :, :], axis=0)
        #print(value_2050.shape)
        #sys.exit(4)
        cont = sub1.contourf(lons, lats, value_2050, projection=ccrs.PlateCarree(central_longitude=180), cmap = fld_spec_vals[fld][1], levels = fld_spec_vals[fld][0], extend="both", transform=ccrs.PlateCarree())
        sub1.set_aspect(1.1)
        sub1.annotate(f"{chr(97 + i*len(flds)+2*j)}) 2050 {fld} change", xy=(-0.04, 1.1), xycoords='axes fraction', fontsize=13, ha='left', va='center', weight="bold")
        sub1.grid(False)
        #sub1.annotate(f'METEOR predict DIAMOND baseline 2050 {fld}', xy=(-0.2, 0.5), xycoords='axes fraction', fontsize=13, ha='left', va='center', weight="bold", rotation=90)
        sub2 = fig.add_subplot(gs[i*len(flds)+j, 1], projection=ccrs.PlateCarree(central_longitude=180))
        sub2.set_xticks(np.arange(60, 301, 60), crs=ccrs.PlateCarree())
        sub2.set_xticklabels(["120°W","60°W","0°","60°E","120°E"], size=8)
        sub2.set_yticks(np.arange(-60, 61, 30), crs=ccrs.PlateCarree())
        sub2.set_yticklabels(["60°S","30°S","0°","30°N","60°N"], size=8)
        sub2.coastlines(lw=0.5)
        sub2.set_aspect(1.1)
        sub2.annotate(f"{chr(97 + i*len(flds)+2*j +1)}) End of century {fld}", xy=(-0.04, 1.1), xycoords='axes fraction', fontsize=13, ha='left', va='center', weight="bold")

        value_last_20 = np.nanmean(patterns[fld].values[-20:, :, :], axis=0) - np.nanmean(patterns[fld].values[:50, :, :], axis=0)

        sub2.contourf(lons, lats, value_last_20, projection=ccrs.PlateCarree(central_longitude=180), cmap = fld_spec_vals[fld][1], levels = fld_spec_vals[fld][0], extend="both", transform=ccrs.PlateCarree())
        sub2.set_aspect(1.1)
        sub1.grid(False)
        if fld == "pr":
            cbar_ax = fig.add_axes([0.25, 0.015, 0.65, 0.03])
            cb = fig.colorbar(cont, cax=cbar_ax,orientation='horizontal')
            fig.text(0.15,0.015,"$\Delta$P [kg m-2 s-1]",ha="center")
        else:
            cbar_ax = fig.add_axes([0.2, 0.490, 0.7, 0.03])
            cb = fig.colorbar(cont, cax=cbar_ax,orientation='horizontal')
            fig.text(0.13,0.49,"$\Delta$T [K]",ha="center")        # sub2.annotate('a', xy=(-0.04, 1.1), xycoords='axes fraction', fontsize=13, ha='left', va='center', weight="bold")
fig.suptitle(f'METEOR predict DIAMOND baseline', fontsize=16, ha='center', va='top', weight="bold")
for ax in fig.get_axes():
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
#"Diff data in pattern and outside"
#axs[0,0].legend()
#axs[0,1].set_title(f"Automatic global mean plot")
#axs[0,2].set_title(f"Truth map")
#axs[0,2].set_title(f"Principal component map 1")
plt.tight_layout()
plt.savefig("CanESM5_NorESM_diamond_reconstruction.png", bbox_inches='tight', dpi=300)
