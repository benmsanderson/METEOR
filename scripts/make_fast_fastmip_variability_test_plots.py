
import xarray as xr
import matplotlib.pyplot as plt
import glob

from meteor.geo_data_utils import global_mean as glb_mean

full_ds = None
scenario = "M"
#scenario="omnia"
variable = "tas"
for infile in glob.glob(f"../data/FASTMIP_phase2/METEOR_emulations/raw/METEOR_CanESM5_{scenario}_scaledtoFAIRens_*.nc"):
#for infile in glob.glob(f"../data/DIAMOND/meteor_impacts/raw/METEOR_CanESM5_esm-allGHG-baseline_v1_2_new+softlink_omnia.nc"):
    print(infile)
    ds = xr.open_dataset(infile)
    if len(ds["month"]) > 1000:
        print(f"Skipping {infile} because it has less than 1500 years of data.")
        continue
    if full_ds is None:
        full_ds = ds
    else:
        full_ds = xr.concat([full_ds, ds], dim="member")
CanESM_H_unscaled_noisetrue = full_ds
CanESM_H_unscaled_noisetrue_NH = full_ds.where(full_ds.lat >= 0, drop=True)
CanESM_H_unscaled_noisetrue_SH = full_ds.where(full_ds.lat <= 0, drop=True)

tas_annual_glb_mean = glb_mean(CanESM_H_unscaled_noisetrue[f"{variable}_grid_annual"])
tas_monthly = CanESM_H_unscaled_noisetrue[f"{variable}_global"]
tas_monthly_annual = tas_monthly.coarsen(month=12, boundary='trim').mean('month')
tas_annual_glb_mean_nh = glb_mean(CanESM_H_unscaled_noisetrue_NH[f"{variable}_grid_annual"])
tas_annual_glb_mean_sh = glb_mean(CanESM_H_unscaled_noisetrue_SH[f"{variable}_grid_annual"])

fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout="constrained", sharey=True)
print(tas_monthly.shape, tas_monthly_annual.shape, tas_annual_glb_mean.shape)
if tas_monthly.shape[0] > 1:
    for ens in range(tas_monthly.shape[0]):
        print(f"Ensemble member {ens}")
        axs[0].plot(tas_monthly['month'], tas_monthly.T.values[:,:, ens])
else:
        axs[0].plot(tas_monthly['month'], tas_monthly.T.values[:,:])
axs[0].set_title(f'{variable}_global (monthly global mean)')
axs[0].set_xlabel('Time (months)')
axs[0].set_ylabel(f'{variable} anomaly (K)')
if tas_monthly_annual.shape[0] > 1:
    for ens in range(tas_monthly_annual.shape[0]):
        print(f"Ensemble member {ens}")
        axs[1].plot(tas_monthly_annual['month'], tas_monthly_annual.T.values[:,:, ens])
else: 
    axs[1].plot(tas_monthly_annual['month'], tas_monthly_annual.T.values[:,:])
axs[1].set_title(f'{variable}_global annual mean')
if tas_annual_glb_mean.shape[0] > 1:
    for ens in range(tas_annual_glb_mean.shape[0]):
        print(f"Ensemble member {ens}")
        axs[2].plot(tas_annual_glb_mean['year'], tas_annual_glb_mean.values[ens, :])
else:
    axs[2].plot(tas_annual_glb_mean['year'], tas_annual_glb_mean.values[:,:])
axs[2].set_title(f'{variable}_grid_annual (global mean of annual gridded field)')
axs[2].set_xlabel('Time (years)')
fig.suptitle(f'CanESM5 {scenario} scenario, not scaled to FAIR emulation, 10 ensemble members')
fig.savefig(f"{variable}_CanESM5_{scenario}_unscaled_noisetrue_timeseries_gen_update_pr_transform_flatfair.png")

plt.clf()
fig, axs = plt.subplots(1, 2, figsize=(12, 6), layout="constrained", sharey=True)
if tas_annual_glb_mean_nh.shape[0] > 1:
    for ens in range(tas_annual_glb_mean_nh.shape[0]):
        print(f"Ensemble member {ens}")
        axs[0].plot(tas_annual_glb_mean_nh['year'], tas_annual_glb_mean_nh.values[ens, :])
else:
    axs[0].plot(tas_annual_glb_mean_nh['year'], tas_annual_glb_mean_nh.values[:,:])
axs[0].set_title(f'{variable}_grid_annual (global mean of annual gridded field) - Northern Hemisphere')
axs[0].set_xlabel('Time (years)')
if tas_annual_glb_mean_sh.shape[0] > 1:
    for ens in range(tas_annual_glb_mean_sh.shape[0]):
        print(f"Ensemble member {ens}")
        axs[1].plot(tas_annual_glb_mean_sh['year'], tas_annual_glb_mean_sh.values[ens, :])
else:
    axs[1].plot(tas_annual_glb_mean_sh['year'], tas_annual_glb_mean_sh.values[:,:])
axs[1].set_title(f'{variable}_grid_annual (global mean of annual gridded field) - Southern Hemisphere')
axs[1].set_xlabel('Time (years)')
fig.suptitle(f'CanESM5 {scenario} scenario, not scaled to FAIR emulation, 10 ensemble members')
fig.savefig(f"{variable}_CanESM5_{scenario}_unscaled_noisetrue_timeseries_gen_update_pr_transform_flatfair_NH_SH.png")