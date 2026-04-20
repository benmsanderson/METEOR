import os
import numpy as np
import xarray as xr
import pandas as pd
import regionmask
import glob
import random
from cdo import Cdo

# =========================================================
# SETUP
# =========================================================

ESM_list = ['ACCESS-ESM1-5','CanESM5', 'IPSL-CM6A-LR', 'MPI-ESM1-2-LR', 'MIROC6']  #(Tier 1 list)
scenario_list = ['Low - SSP2 (Marker)','Medium - SSP2 (Marker)','High - SSP3 (Marker)','Very Low - SSP1 (Marker)']
scenarios_short = ['L','M','H','VL']
quantile_list = [0.01, 0.025, 0.05, 0.33, 0.5, 0.67, 0.95, 0.975, 0.99] # 1%, 2.5%, 5%, 33%, median, 67%, 95%, 97.5%, 99%

output_dir = '../data/FASTMIP_phase2/METEOR_emulations/raw/'
aggregate_dir = '../data/FASTMIP_phase2/METEOR_emulations/aggregated/'
processed_dir = '../data/FASTMIP_phase2/METEOR_emulations/processed/'

os.makedirs(aggregate_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)


# =========================================================
# Some utils
# =========================================================

def add_coords(ds):
    filename = os.path.basename(ds.encoding["source"])
    esm = filename.split('_')[1]
    fair_realisation = filename.split('_')[-1].split('.')[0]
    ds = ds.expand_dims(esm=[esm], fair_realisation=[fair_realisation])
    ds = ds.rename({'year': 'time', 'realization': 'noise_realisation', 'tas_grid_annual': 'tas', 'pr_grid_annual': 'pr'})
    ds = ds.drop_vars(['tas_global', 'pr_global'])
    ds['time'] = pd.to_datetime(ds['time'].values, format="%Y")
    ds.coords['lat'].attrs['units'] = 'degrees_north'
    ds.coords['lon'].attrs['units'] = 'degrees_east'
    ds.time.encoding = {
    "units": "days since 2015-01-01 00:00:00",
    "calendar": "proleptic_gregorian"
    }
    return ds

def check_esm_weights(cdo, target_grid, esm, template_ds):
    weights_file = f"{aggregate_dir}cdo_weights_{esm}_to_g025.nc"
    if not os.path.exists(weights_file):
        cdo.gencon(target_grid, input=template_ds, output=weights_file)
    return weights_file

def regrid_file_with_weights(cdo, input_ds, esm, scenario, weights_file):
    target_grid = '../data/FASTMIP_phase2/g025.txt'
    tmp_out = f"{aggregate_dir}{esm}_{scenario}_tmp.nc"
    cdo.remap(f"{target_grid},{weights_file}", input=input_ds, output=tmp_out)
    return tmp_out



def open_all_files(scenarios_short_list):
    ds_list = []
    for scenario in scenarios_short_list:
        scenario_file = f"{aggregate_dir}METEOR_{scenario}_scaledtoFAIR_combinedESMs_regridded.nc"
        ds = xr.open_dataset(scenario_file)
        ds_list.append(ds)
    combined_ds = xr.concat(ds_list, dim=xr.Variable('scenario', data=scenarios_short_list))
    return combined_ds


# calculate AR6 regional means (re-written from provided aggregate_to_regions.py).
def compute_regional_means(ds, ar6_mask):
    lat = ds["lat"]
    lon = ds["lon"]
    weights = np.cos(np.deg2rad(lat))
    weights.name = "area_weight"
    weights_expanded = weights.expand_dims(lon=lon).transpose("lon", "lat")

    da_ar6 = (ds*weights).groupby(ar6_mask).mean(dim=['stacked_lon_lat']) / weights_expanded.groupby(ar6_mask).mean(dim='stacked_lon_lat')

    da_global = ds.weighted(weights).mean(dim=['lon', 'lat']).expand_dims(mask=[-1])

    da_regions = xr.concat([da_ar6, da_global], dim="mask")

    flag_values = np.concatenate([ar6.numbers, [-1]])
    flag_meanings = " ".join(ar6.names + ["GLOBAL"])

    da_regions["mask"].attrs = {
        "standard_name": "region",
        "flag_values": flag_values,
        "flag_meanings": flag_meanings,
    }

    return da_regions


def save_outputs(tas, pr, processed_dir, aggregation, quantity, scenario_short_name=None, fair_realisation_numbers=None):

    # add METEOR metadata:
    tas = tas.drop_attrs() # first clear old ones
    pr = pr.drop_attrs()
    tas = tas.assign_attrs({
        'model': 'METEOR (tag v1.6.0-10-g95e4345)',
        'scenario': f'{aggregation} {quantity} for scenario {scenario_short_name if scenario_short_name is not None else "cross-scenario"}',
        'reference': 'https://doi.org/10.5194/gmd-18-8269-2025 UPDATE TO v1.6 WHEN AVAILABLE',
        'ensemble_info': 'Full ensemble is 200 members per ESM per scenario (20 randomly chosen FAIR GSAT ensemble members x 10 METEOR noise/internal variability realisations). FAIR realisations are shared across ESMs and scenarios, noise realisations are independent.',
        'FAIR ensemble members': fair_realisation_numbers.tolist(),
        'units': 'K'
    })
    pr = pr.assign_attrs(tas.attrs)
    pr = pr.assign_attrs({'units': 'kg m-2 s-1'})

    tas = tas.compute()
    pr = pr.compute()

    # save per-scenario output.
    if scenario_short_name is not None:
        tas.to_netcdf(f"{processed_dir}tas_{scenario_short_name}_METEOR_{aggregation}_{quantity}.nc")
        pr.to_netcdf(f"{processed_dir}pr_{scenario_short_name}_METEOR_{aggregation}_{quantity}.nc")
    # save across-scenario output (use 'cross-scenario' in filename to distinguish):
    else:
        tas.to_netcdf(f"{processed_dir}tas_cross-scenario_METEOR_{aggregation}_{quantity}.nc")
        pr.to_netcdf(f"{processed_dir}pr_cross-scenario_METEOR_{aggregation}_{quantity}.nc")


# =========================================================
# FASTMIP OUTPUT CALCULATIONS
# =========================================================

# 1 Save 10 randomly selected ensemble members across both FAIR and noise realisations:
def random_subset(ds, n):
    ds_stacked = ds.stack(realisation=['noise_realisation', 'fair_realisation'])

    # check that we didn't accidentally choose all the same FAIR realisation (i.e. that we have some variability across both noise and FAIR realisations):
    len_fair_sampled = 1 
    while len_fair_sampled == 1:
        idx = random.sample(range(ds_stacked.sizes['realisation']), n)
        subset_ds = ds_stacked.isel(realisation=idx)
        len_fair_sampled = len(np.unique(subset_ds['fair_realisation']))

    return subset_ds.reset_index('realisation')


# 2, 3, 4, 5 Gridwise and regional means and quantiles by ESM, and across ESMs:
def compute_quantiles(ds, quantile_list, quantity, aggregation, ar6_mask=None):

    if quantity == 'quantiles-by-ESM':
        group_dims = ['fair_realisation', 'noise_realisation']
    elif quantity == 'quantiles-across-ESM':
        group_dims = ['fair_realisation', 'noise_realisation', 'esm']

    if aggregation == 'gridcell':
        mean = ds.mean(dim=group_dims).rename({'tas':'tas_mean', 'pr':'pr_mean'})
        quant = ds.quantile(quantile_list, dim=group_dims)
    elif aggregation == 'regional':
        regional_means = compute_regional_means(ds, ar6_mask)
        mean = regional_means.mean(dim=group_dims).rename({'tas':'tas_mean', 'pr':'pr_mean'})
        quant = regional_means.quantile(quantile_list, dim=group_dims)

    tas = xr.merge([mean['tas_mean'], quant['tas']])
    pr = xr.merge([mean['pr_mean'], quant['pr']])

    return tas, pr


# 6, 7 Gridwise and regional uncertainty decomposition per scenario:
def compute_uncertainty_per_scenario(ds, aggregation, ar6_mask=None):

    if aggregation == 'gridcell':
        var_esm = ds.mean(dim='noise_realisation').var(dim='esm').mean(dim='fair_realisation')
        var_glb = ds.mean(dim='noise_realisation').var(dim='fair_realisation').mean(dim='esm')
        var_int = ds.var(dim='noise_realisation').mean(dim=['esm', 'fair_realisation'])
    elif aggregation == 'regional':
        regional_means = compute_regional_means(ds, ar6_mask)
        var_esm = regional_means.mean(dim='noise_realisation').var(dim='esm').mean(dim='fair_realisation')
        var_glb = regional_means.mean(dim='noise_realisation').var(dim='fair_realisation').mean(dim='esm')
        var_int = regional_means.var(dim='noise_realisation').mean(dim=['esm', 'fair_realisation'])

    tas = xr.merge([var_esm['tas'].rename('var_esm'), var_glb['tas'].rename('var_glb'), var_int['tas'].rename('var_int')])
    pr = xr.merge([var_esm['pr'].rename('var_esm'), var_glb['pr'].rename('var_glb'), var_int['pr'].rename('var_int')])

    return tas, pr

# 8, 9 Gridwise and regional uncertainty decomposition across scenarios:
def compute_uncertainty_across_scenarios(ds, aggregation, ar6_mask=None):

    if aggregation == 'gridcell':
        var_int = ds.var(dim='noise_realisation').mean(dim=['esm', 'fair_realisation', 'scenario'])
        var_glb = ds.mean(dim='noise_realisation').var(dim=['fair_realisation']).mean(dim=['esm', 'scenario'])
        var_esm = ds.mean(dim='noise_realisation').var(dim=['esm']).mean(dim=['fair_realisation', 'scenario'])
        var_scenario = ds.mean(dim='noise_realisation').var(dim='scenario').mean(dim=['esm', 'fair_realisation'])
    elif aggregation == 'regional':
        regional_means = compute_regional_means(ds, ar6_mask)
        var_int = regional_means.var(dim='noise_realisation').mean(dim=['esm', 'fair_realisation', 'scenario'])
        var_glb = regional_means.mean(dim='noise_realisation').var(dim=['fair_realisation']).mean(dim=['esm', 'scenario'])
        var_esm = regional_means.mean(dim='noise_realisation').var(dim=['esm']).mean(dim=['fair_realisation', 'scenario'])
        var_scenario = regional_means.mean(dim='noise_realisation').var(dim='scenario').mean(dim=['esm', 'fair_realisation'])

    tas = xr.merge([var_int['tas'].rename('var_int'), var_glb['tas'].rename('var_glb'), var_esm['tas'].rename('var_esm'), var_scenario['tas'].rename('var_scenario')])
    pr = xr.merge([var_int['pr'].rename('var_int'), var_glb['pr'].rename('var_glb'), var_esm['pr'].rename('var_esm'), var_scenario['pr'].rename('var_scenario')])

    return tas, pr


# =========================================================
# MAIN
# =========================================================

# Regridding and combining ESM outputs is done in a separate loop per scenario to save memory, and intermediate regridded files are saved to disk to avoid having to keep all ESMs in memory at once. The final outputs are then calculated from the combined regridded files for each scenario, and saved to disk. Finally, the across-scenario outputs are calculated from the combined per-scenario files, and saved to disk.

for scenario_short_name in scenarios_short:
# Part 1: regrid and combine all ESMs for this scenario, and save as intermediate file (skip if already exists to save time and memory):
    scenario_long_name = scenario_list[scenarios_short.index(scenario_short_name)]
    scenario_filename = f"{aggregate_dir}METEOR_{scenario_short_name}_scaledtoFAIR_combinedESMs_regridded.nc"

    # if aggregate file already exists, skip
    if os.path.exists(scenario_filename):
        print(f"File already exists, skip making: {scenario_filename}")
        continue

    all_esms = []

    # regrid each ESM seperately, then combine per scenario
    for esm in ESM_list:

        # get list of files for this ESM and scenario:
        esm_files = sorted(glob.glob(f"{output_dir}METEOR_{esm}_{scenario_short_name}_scaledtoFAIRens_*.nc"))

        # check for cdo weights file for this ESM, and create if it doesn't exist
        cdo = Cdo()
        target_grid = '../data/FASTMIP_phase2/g025.txt'
        template_ds = add_coords(xr.open_dataset(esm_files[0])).stack(sample=['esm', 'fair_realisation', 'noise_realisation']).reset_index('sample').transpose('time', ...)
        weights_file = check_esm_weights(cdo, target_grid, esm, template_ds)

        # get stacked coordinates for this esm:
        esm_ds = xr.open_mfdataset(esm_files, concat_dim='fair_realisation', combine='nested', preprocess=add_coords)
        stacked = esm_ds.stack(sample=['esm', 'fair_realisation', 'noise_realisation'])
        stacked_index = stacked['sample'].to_index()

        # re-grid and save temporary file
        esm_tmp = regrid_file_with_weights(cdo, stacked.reset_index('sample'), esm, scenario_short_name, weights_file)

        # open temporary file and unstack coordinates
        esm_regridded = xr.open_dataset(esm_tmp).assign_coords(sample=stacked_index).unstack('sample')

        # combine all re-gridded esm datasets for the scenario
        all_esms.append(esm_regridded)

    # save combined file for this scenario, and delete temporary files
    scenario_regridded = xr.concat(all_esms, dim='esm')
    scenario_regridded = scenario_regridded.assign_coords(time=scenario_regridded.time.dt.year).rename({'time': 'year'})
    scenario_regridded.to_netcdf(scenario_filename)

    for tmp_file in glob.glob(f"{aggregate_dir}*tmp.nc"):
        os.remove(tmp_file)



for scenario_short_name in scenarios_short:
# Part 2: for each scenario, calculate the scenario-specific FastMIP outputs:
    scenario_filename = f"{aggregate_dir}METEOR_{scenario_short_name}_scaledtoFAIR_combinedESMs_regridded.nc"
    with xr.open_dataset(scenario_filename) as ds_regridded:

        ar6=regionmask.defined_regions.ar6.land
        ar6_mask=ar6.mask(ds_regridded.lat, ds_regridded.lon).persist()

        # get FAIR ensemble member numbers (the ensemble itself is also saved in 
        # /METEOR/data/FASTMIP_phase2/FAIR_data/ in a .pkl).
        all_fair_realisation_numbers = ds_regridded['fair_realisation'].values.astype(int)

        # 1. Random subset of 10 members across both FAIR and noise realisations:
        print('starting scenario ', scenario_short_name)
        subset_ds = random_subset(ds_regridded, n=10)
        save_outputs(subset_ds['tas'].to_dataset(promote_attrs=True), subset_ds['pr'].to_dataset(promote_attrs=True), processed_dir, 'gridcell', 'selected-realisations', scenario_short_name, np.unique(subset_ds['fair_realisation'].values.astype(int)))


        # 2. Gridwise mean and quantiles by ESM:
        tas, pr = compute_quantiles(ds_regridded, quantile_list, 'quantiles-by-ESM', 'gridcell')
        save_outputs(tas, pr, processed_dir, 'gridcell', 'quantiles-by-ESM', scenario_short_name, all_fair_realisation_numbers)

        # 3. Regional mean and quantiles by ESM:
        tas, pr = compute_quantiles(ds_regridded, quantile_list, 'quantiles-by-ESM', 'regional', ar6_mask=ar6_mask)
        save_outputs(tas, pr, processed_dir, 'regional', 'quantiles-by-ESM', scenario_short_name, all_fair_realisation_numbers)

        # 4. Gridwise mean and quantiles across ESMs:
        tas, pr = compute_quantiles(ds_regridded, quantile_list, 'quantiles-across-ESM', 'gridcell')
        save_outputs(tas, pr, processed_dir, 'gridcell', 'quantiles-across-ESM', scenario_short_name, all_fair_realisation_numbers)

        # 5. Regional mean and quantiles across ESMs:
        tas, pr = compute_quantiles(ds_regridded, quantile_list, 'quantiles-across-ESM', 'regional', ar6_mask=ar6_mask)
        save_outputs(tas, pr, processed_dir, 'regional', 'quantiles-across-ESM', scenario_short_name, all_fair_realisation_numbers)

        # 6. Uncertainty decomposition by region:
        tas, pr = compute_uncertainty_per_scenario(ds_regridded, 'regional', ar6_mask=ar6_mask)
        save_outputs(tas, pr, processed_dir, 'regional', 'uncertainty', scenario_short_name, all_fair_realisation_numbers)

        # 7. Uncertainty decomposition gridwise:
        tas, pr = compute_uncertainty_per_scenario(ds_regridded, 'gridcell')
        save_outputs(tas, pr, processed_dir, 'gridcell', 'uncertainty', scenario_short_name, all_fair_realisation_numbers)


# Part 3: calculate FastMIP outputs across scenarios:
combined_ds = open_all_files(scenarios_short)

# 8. Uncertainty decomposition by region, across scenarios:
tas, pr = compute_uncertainty_across_scenarios(combined_ds, 'regional', ar6_mask=ar6_mask)
save_outputs(tas, pr, processed_dir, 'regional', 'across-scenario-uncertainty', scenario_short_name=None, fair_realisation_numbers=all_fair_realisation_numbers)

# 9. Uncertainty decomposition gridwise, across scenarios:
tas, pr = compute_uncertainty_across_scenarios(combined_ds, 'gridcell')
save_outputs(tas, pr, processed_dir, 'gridcell', 'across-scenario-uncertainty', scenario_short_name=None, fair_realisation_numbers=all_fair_realisation_numbers)

