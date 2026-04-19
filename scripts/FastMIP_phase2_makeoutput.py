import os
import numpy as np
import xarray as xr
import pandas as pd
import regionmask
import glob
import random
import tempfile
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
    ds = ds.expand_dims(esm=[esm], fair_realisation=[fair_realisation]).rename({'year': 'time'})
    ds['time'] = pd.to_datetime(ds['time'].values, format="%Y")
    ds.coords['lat'].attrs['units'] = 'degrees_north'
    ds.coords['lon'].attrs['units'] = 'degrees_east'
    ds.time.encoding = {
    "units": "days since 2015-01-01 00:00:00",
    "calendar": "proleptic_gregorian"
    }
    return ds


def open_per_scenario_files(files):
    ds = xr.open_mfdataset(
        files,
        concat_dim=['esm', 'fair_realisation'],
        combine='nested',
        preprocess=add_coords
    )

    return ds.rename({
        'realization': 'noise_realisation',
        'tas_grid_annual': 'tas',
        'pr_grid_annual': 'pr'
    })


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


def build_scenario_file_list(scenario_short_name, ESM_list=ESM_list, output_dir=output_dir):
    files = []
    for esm in ESM_list:
        files_sub = sorted(glob.glob(f"{output_dir}METEOR_{esm}_{scenario_short_name}_scaledtoFAIRens_*.nc"))
        files.append(files_sub)
    return files


def make_scenario_aggregate(scenario_short_name, cdo, ESM_list=ESM_list, output_dir=output_dir, aggregate_dir=aggregate_dir):

    scenario_filename = f"{aggregate_dir}METEOR_{scenario_short_name}_scaledtoFAIR_combinedESMs_regridded.nc"

    # if aggregate file already exists, skip
    if os.path.exists(scenario_filename):
        print(f"File already exists, skip making: {scenario_filename}")
        return scenario_filename

    for esm in ESM_list:
        esm_files = sorted(glob.glob(f"{output_dir}METEOR_{esm}_{scenario_short_name}_scaledtoFAIRens_*.nc"))

        # check for cdo weights file for this ESM, and create if it doesn't exist
        weights_file = check_esm_weights(cdo, esm, esm_files[0])

        esm_ds = xr.open_mfdataset(esm_files, concat_dim='fair_realisation', combine='nested', preprocess=add_coords).rename({
        'realization': 'noise_realisation',
        'tas_grid_annual': 'tas',
        'pr_grid_annual': 'pr'})



        esm_dataset = []
    for f in files_sub:
        tmp_file = regrid_file_with_weights(cdo, f, weights_file)
        try:
            with xr.open_dataset(tmp_file) as ds:
                ds = add_coords(ds)
                ds = ds.rename({'realization': 'noise_realisation', 'tas_grid_annual': 'tas', 'pr_grid_annual': 'pr'})
                fair_datasets.append(ds[['tas', 'pr']].load())
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    if not fair_datasets:
        return None

    esm_ds = xr.concat(fair_datasets, dim='fair_realisation')
    esm_ds.to_netcdf(esm_output)
    return esm_output




    return scenario_filename

def regrid_to_common_grid(ds):
    cdo = Cdo()
    target_grid = '../data/FASTMIP_phase2/g025.txt'
    # CDO can only handle up to 4 dimensions, so stack everything appart from time, lat, and lon
    stacked = ds.stack(sample=['esm', 'fair_realisation', 'noise_realisation'])
    stacked_index = stacked['sample'].to_index()
    regridded = cdo.remapcon(target_grid, input=stacked.reset_index('sample').transpose('time', ...), returnXDataset=True)
    regridded = regridded.assign_coords(sample=stacked_index)
    regridded_ds = regridded.unstack('sample')

    return regridded_ds

def check_esm_weights(cdo, esm, template_file):
    target_grid = '../data/FASTMIP_phase2/g025.txt'
    weights_file = f"{aggregate_dir}cdo_weights_{esm}_to_g025.nc"
    if not os.path.exists(weights_file):
        cdo.gencon(target_grid, input=template_file, output=weights_file)
    return weights_file


def regrid_file_with_weights(cdo, input_file, weights_file):
    target_grid = '../data/FASTMIP_phase2/g025.txt'
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False, dir=aggregate_dir) as tmp:
        tmp_out = tmp.name
    cdo.remap(f"{target_grid},{weights_file}", input=input_file, output=tmp_out)
    return tmp_out


def build_esm_aggregate(scenario_short_name, esm, files_sub, cdo):
    esm_output = get_esm_aggregate_path(scenario_short_name, esm)
    if os.path.exists(esm_output):
        return esm_output

    if not files_sub:
        return None

    weights_file = check_esm_weights(cdo, esm, files_sub[0])

    fair_datasets = []
    for f in files_sub:
        tmp_file = regrid_file_with_weights(cdo, f, weights_file)
        try:
            with xr.open_dataset(tmp_file) as ds:
                ds = add_coords(ds)
                ds = ds.rename({'realization': 'noise_realisation', 'tas_grid_annual': 'tas', 'pr_grid_annual': 'pr'})
                fair_datasets.append(ds[['tas', 'pr']].load())
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    if not fair_datasets:
        return None

    esm_ds = xr.concat(fair_datasets, dim='fair_realisation')
    esm_ds.to_netcdf(esm_output)
    return esm_output


def build_scenario_aggregate(scenario_short_name):
    scenario_output = get_scenario_aggregate_path(scenario_short_name)
    if os.path.exists(scenario_output):
        return scenario_output

    files_by_esm = build_scenario_file_list(scenario_short_name)
    cdo = Cdo()

    esm_files = []
    for esm, files_sub in zip(ESM_list, files_by_esm):
        esm_file = build_esm_aggregate(scenario_short_name, esm, files_sub, cdo)
        if esm_file is not None:
            esm_files.append(esm_file)

    if not esm_files:
        raise ValueError(f"No input files found for scenario {scenario_short_name}")

    esm_datasets = [xr.open_dataset(path) for path in esm_files]
    try:
        scenario_ds = xr.concat(esm_datasets, dim='esm')
        scenario_ds.to_netcdf(scenario_output)
    finally:
        for ds in esm_datasets:
            ds.close()

    return scenario_output


def save_outputs(tas, pr, processed_dir, aggregation, quantity, scenario_long_name=None, fair_realisation_numbers=None):
    # reformat time coordinates to years:
    tas = tas.assign_coords(time=tas.time.dt.year)
    tas = tas.rename({'time': 'year'})
    pr = pr.assign_coords(time=pr.time.dt.year)
    pr = pr.rename({'time': 'year'})

    # add METEOR metadata:
    tas = tas.drop_attrs() # first clear old ones
    pr = pr.drop_attrs()
    tas = tas.assign_attrs({
        'model': 'METEOR (tag v1.6.0-10-g95e4345)',
        'scenario': f'{aggregation} {quantity} for scenario {scenario_long_name if scenario_long_name is not None else "cross-scenario"}',
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
    if scenario_long_name is not None:
        tas.to_netcdf(f"{processed_dir}tas_{scenario_long_name}_METEOR_{aggregation}_{quantity}.nc")
        pr.to_netcdf(f"{processed_dir}pr_{scenario_long_name}_METEOR_{aggregation}_{quantity}.nc")
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

# Workflow:
# For each scenario: 
#   1. Get all files for a given ESM
#   2. Build regridder once per ESM using the first file as a template
#   3. Regrid and eagerly load each file to avoid a large dask graph
#   5. combine into aggregate file per scenario (save, clear memory, load this one to do calculations)

# outputs per scenario:
for scenario_short_name in scenarios_short:

    scenario_long_name = scenario_list[scenarios_short.index(scenario_short_name)]

    scenario_filename = make_scenario_aggregate(scenario_short_name)
    ds_regridded = xr.open_dataset(scenario_filename).persist()

    ar6=regionmask.defined_regions.ar6.land
    ar6_mask=ar6.mask(ds_regridded.lat, ds_regridded.lon).persist()

    # get FAIR ensemble member numbers (the ensemble itself is also saved in 
    # /METEOR/data/FASTMIP_phase2/FAIR_data/ in a .pkl).
    all_fair_realisation_numbers = ds_regridded['fair_realisation'].values.astype(int)

    # 1. Random subset of 10 members across both FAIR and noise realisations:
    print('starting scenario ', scenario_short_name, ': saving random subset of 10 members across both FAIR and noise realisations...')
    subset_ds = random_subset(ds_regridded, n=10)
    save_outputs(subset_ds['tas'], subset_ds['pr'], processed_dir, 'gridcell', 'selected-realisations', scenario_short_name, scenario_long_name, np.unique(subset_ds['fair_realisation'].values.astype(int)))

    # 2. Gridwise mean and quantiles by ESM:
    tas, pr = compute_quantiles(ds_regridded, quantile_list, 'quantiles-by-ESM', 'gridcell')
    save_outputs(tas, pr, processed_dir, 'gridcell', 'quantiles-by-ESM', scenario_short_name, scenario_long_name, all_fair_realisation_numbers)

    # 3. Regional mean and quantiles by ESM:
    tas, pr = compute_quantiles(ds_regridded, quantile_list, 'quantiles-by-ESM', 'regional', ar6_mask=ar6_mask)
    save_outputs(tas, pr, processed_dir, 'regional', 'quantiles-by-ESM', scenario_short_name, scenario_long_name, all_fair_realisation_numbers)

    # 4. Gridwise mean and quantiles across ESMs:
    tas, pr = compute_quantiles(ds_regridded, quantile_list, 'quantiles-across-ESM', 'gridcell')
    save_outputs(tas, pr, processed_dir, 'gridcell', 'quantiles-across-ESM', scenario_short_name, scenario_long_name, all_fair_realisation_numbers)

    # 5. Regional mean and quantiles across ESMs:
    tas, pr = compute_quantiles(ds_regridded, quantile_list, 'quantiles-across-ESM', 'regional', ar6_mask=ar6_mask)
    save_outputs(tas, pr, processed_dir, 'regional', 'quantiles-across-ESM', scenario_short_name, scenario_long_name, all_fair_realisation_numbers)

    # 6. Uncertainty decomposition by region:
    tas, pr = compute_uncertainty_per_scenario(ds_regridded, 'regional', ar6_mask=ar6_mask)
    save_outputs(tas, pr, processed_dir, 'regional', 'uncertainty', scenario_short_name, scenario_long_name, all_fair_realisation_numbers)

    # 7. Uncertainty decomposition gridwise:
    tas, pr = compute_uncertainty_per_scenario(ds_regridded, 'gridcell')
    save_outputs(tas, pr, processed_dir, 'gridcell', 'uncertainty', scenario_short_name, scenario_long_name, all_fair_realisation_numbers)


# outputs across scenarios:
combined_ds = open_all_files(scenarios_short)

# 8. Uncertainty decomposition by region, across scenarios:
tas, pr = compute_uncertainty_across_scenarios(combined_ds, 'regional', ar6_mask=ar6_mask)
save_outputs(tas, pr, processed_dir, 'regional', 'across-scenario-uncertainty', scenario_short_name=None, scenario_long_name=None, fair_realisation_numbers=all_fair_realisation_numbers)

# 9. Uncertainty decomposition gridwise, across scenarios:
tas, pr = compute_uncertainty_across_scenarios(combined_ds, 'gridcell')
save_outputs(tas, pr, processed_dir, 'gridcell', 'across-scenario-uncertainty', scenario_short_name=None, scenario_long_name=None, fair_realisation_numbers=all_fair_realisation_numbers)



'''

import os
#import datetime
import numpy as np
import xarray as xr
#import pandas as pd
import xesmf as xe
#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#from meteor import MeteorInterface
import random
import regionmask
import glob

# -------
# SETUP
# TODO: add as optional inputs with these below as defaults (for ESM_list, num_fair, n_members)

ESM_list = ['ACCESS-ESM1-5','CanESM5', 'IPSL-CM6A-LR', 'MPI-ESM1-2-LR', 'MIROC6']  # Tier 1 list
scenario_list = ['Low - SSP2 (Marker)','Medium - SSP2 (Marker)','High - SSP3 (Marker)','Very Low - SSP1 (Marker)']
scenarios_short = ['L','M','H','VL']

# check if output directories exist, if not, create them:
output_dir = '../data/FASTMIP_phase2/METEOR_emulations/raw/'
aggregate_dir = '../data/FASTMIP_phase2/METEOR_emulations/aggregated/'
processed_dir = '../data/FASTMIP_phase2/METEOR_emulations/processed/'
if not os.path.exists(aggregate_dir):
    os.makedirs(aggregate_dir)
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# -------
# SOME UTILS

def save_if_not_exists(dataset, filename):
    if not os.path.exists(filename):
        dataset.to_netcdf(filename)
        print(f"File created: {filename}")
    else:
        print(f"File already exists, skipping: {filename}")

# get esm name file path and add as coordinates to dataset
def add_coords(ds):
    filename = os.path.basename(ds.encoding["source"])
    esm_str = filename.split('_')[1]
    return ds.expand_dims(esm=[esm_str])

# calculate AR6 regional means (re-written from provided aggregate_to_regions.py).
def compute_regional_means(base_ds):
    lat = base_ds["lat"]
    lon = base_ds["lon"]
    ar6=regionmask.defined_regions.ar6.land
    ar6_mask=ar6.mask(base_ds.lat, base_ds.lon)
    weights = np.cos(np.deg2rad(lat))
    weights.name = "area_weight"
    weights_expanded = weights.expand_dims(lon=lon).transpose("lon", "lat")

    da_ar6 = (base_ds*weights).groupby(ar6_mask).mean(dim=['stacked_lon_lat']) / weights_expanded.groupby(ar6_mask).mean(dim='stacked_lon_lat')

    da_global = base_ds.weighted(weights).mean(dim=['lon', 'lat']).expand_dims(mask=[-1])

    da_regions = xr.concat([da_ar6, da_global], dim="mask")

    flag_values = np.concatenate([ar6.numbers, [-1]])
    flag_meanings = " ".join(ar6.names + ["GLOBAL"])

    da_regions["mask"].attrs = {
        "standard_name": "region",
        "flag_values": flag_values,
        "flag_meanings": flag_meanings,
    }

    return da_regions

# regrid
def regrid_to_common_grid(ds):
    xsize    = 144
    ysize    = 72
    xfirst   = 1.25
    xinc     = 2.5
    yfirst   = -88.75
    yinc     = 2.5

    ds_out = xe.util.cf_grid_2d(
    lon0_b=xfirst, lon1_b=xfirst + xsize*xinc, d_lon=xinc,
    lat0_b=yfirst, lat1_b=yfirst + ysize*yinc, d_lat=yinc)

    regridder = xe.Regridder(ds, ds_out, "conservative")
    return regridder(ds)

# -------
# -------
# GENERATE FASTMIP OUTPUTS

# First set of outputs are per scenario. 
for s_idx, scenario in enumerate(scenario_list):
    files = []
    for esm in ESM_list:
        files_sub = glob.glob(f"{output_dir}METEOR_{esm}_{scenarios_short[s_idx]}_scaledtoFAIRens_*.nc")
        files.append(files_sub)

    combined_ds = xr.open_mfdataset(files, concat_dim=['esm','fair_realisation'], combine='nested', preprocess=add_coords).rename({'realization':'noise_realisation','tas_grid_annual':'tas', 'pr_grid_annual':'pr'})

# 0. Regrid to common grid:
    combined_ds_regridded = regrid_to_common_grid(combined_ds)
    #file_name = f"{aggregate_dir}METEOR_{scenarios_short[s_idx]}_scaledtoFAIR_combinedESMs_regridded.nc"
    #save_if_not_exists(combined_ds_regridded, file_name)

# 1. Save 10 randomly selected ensemble members across both FAIR and noise realisations:
    aggregation = 'gridcell'
    quantity = 'selected-realisations'
    # stack the realisation dimensions to make it easier to select random members across both dimensions:
    data_stacked = combined_ds_regridded.stack(realisation=['noise_realisation', 'fair_realisation'])
    # randomly select 10 members across both dimensions:
    if data_stacked.sizes['realisation'] > 1:
        rand_index = random.sample(range(data_stacked.sizes['realisation']), 10)
        subset_data = data_stacked.isel(realisation=rand_index)
    # save tas and pr to separate files:
    subset_data['tas'].to_netcdf(f"{processed_dir}tas_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")
    subset_data['pr'].to_netcdf(f"{processed_dir}pr_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")

# 2. Gridwise mean and quantiles by ESM, using all ensemble members:
    quantity = 'quantiles-by-ESM'
    aggregation = 'gridcell'
    # gridwise mean and quantiles:
    quantile_list = [0.01, 0.025, 0.05, 0.33, 0.5, 0.67, 0.95, 0.975, 0.99] # 1%, 2.5%, 5%, 33%, median, 67%, 95%, 97.5%, 99%
    # calculate gridwise means and quantiles:
    mean_ds = combined_ds_regridded.mean(dim=['fair_realisation', 'noise_realisation']).rename({'tas':'tas_mean', 'pr':'pr_mean'})
    quantiles_ds = combined_ds_regridded.quantile(quantile_list, dim=['fair_realisation', 'noise_realisation'])
    # save tas and pr to separate files:
    tas_ds = xr.merge([mean_ds['tas_mean'], quantiles_ds['tas']])
    pr_ds = xr.merge([mean_ds['pr_mean'], quantiles_ds['pr']])
    tas_ds.to_netcdf(f"{processed_dir}tas_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")
    pr_ds.to_netcdf(f"{processed_dir}pr_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")

# First, scenario-wide outputs:

for s_idx, scenario in enumerate(scenario_list):
    # get all the data for this scenario (all ESMs, all FAIR and noise realisations):
    base_ds = xr.open_dataset(f"{aggregate_dir}METEOR_{scenarios_short[s_idx]}_scaledtoFAIR_combinedESMs_regridded.nc")

    # 1. -----
    # 10 Ensemble members per ESM:
    # Split tas and pr to separate files. 
    # Randomly select 10 members across both 'fair_realisation' (FAIR) and 'noise_realisation' (METEOR noise) dimensions.
    aggregation = 'gridcell'
    quantity = 'selected-realisations'
    # stack the realisation dimensions to make it easier to select random members across both dimensions:
    data_stacked = base_ds.stack(realisation=['noise_realisation', 'fair_realisation'])
    # randomly select 10 members across both dimensions:
    if data_stacked.sizes['realisation'] > 1:
        rand_index = random.sample(range(data_stacked.sizes['realisation']), 10)
        subset_data = data_stacked.isel(realisation=rand_index)
    # save tas and pr to separate files:
    subset_data['tas'].to_netcdf(f"{processed_dir}tas_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")
    subset_data['pr'].to_netcdf(f"{processed_dir}pr_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")

    # 2. -----
    # FastMIP outputs per ESM (gridwise mean and quantiles).
    # These are calculated using all ensemble members.
    quantity = 'quantiles-by-ESM'
    aggregation = 'gridcell'
    # gridwise mean and quantiles:
    quantile_list = [0.01, 0.025, 0.05, 0.33, 0.5, 0.67, 0.95, 0.975, 0.99] # 1%, 2.5%, 5%, 33%, median, 67%, 95%, 97.5%, 99%
    # calculate gridwise means and quantiles:
    mean_ds = base_ds.mean(dim=['fair_realisation', 'noise_realisation']).rename({'tas':'tas_mean', 'pr':'pr_mean'})
    quantiles_ds = base_ds.quantile(quantile_list, dim=['fair_realisation', 'noise_realisation'])
    # save tas and pr to separate files:
    tas_ds = xr.merge([mean_ds['tas_mean'], quantiles_ds['tas']])
    pr_ds = xr.merge([mean_ds['pr_mean'], quantiles_ds['pr']])
    tas_ds.to_netcdf(f"{processed_dir}tas_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")
    pr_ds.to_netcdf(f"{processed_dir}pr_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")

    # 3. -----
    # FastMIP outputs per ESM (regional mean and quantiles).
    # These are calculated using all the METEOR noise realisations.
    quantity = 'quantiles-by-ESM'
    aggregation = 'regional'
    # aggregate to regional level, than calculate statistics:
    regional_means = compute_regional_means(base_ds)
    mean_ds = regional_means.mean(dim=['fair_realisation', 'noise_realisation']).rename({'tas':'tas_mean', 'pr':'pr_mean'})
    quantiles_ds = regional_means.quantile(quantile_list, dim=['fair_realisation', 'noise_realisation'])

    # save tas and pr to separate files:
    tas_ds = xr.merge([mean_ds['tas_mean'], quantiles_ds['tas']])
    pr_ds = xr.merge([mean_ds['pr_mean'], quantiles_ds['pr']])
    tas_ds.to_netcdf(f"{processed_dir}tas_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")
    pr_ds.to_netcdf(f"{processed_dir}pr_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")

    # 4. -----
    # FastMIP outputs across ESMs (gridwise mean and quantiles).
    # These are calculated using all the METEOR noise realisations.
    quantity = 'quantiles-across-ESM'
    aggregation = 'gridcell'
    # calculate and save gridwise quantiles:
    mean_ds = base_ds.mean(dim=['fair_realisation', 'noise_realisation', 'esm']).rename({'tas':'tas_mean', 'pr':'pr_mean'})
    quantiles_ds = base_ds.quantile(quantile_list, dim=['fair_realisation', 'noise_realisation', 'esm'])

    # save tas and pr to separate files:
    tas_ds = xr.merge([mean_ds['tas_mean'], quantiles_ds['tas']])
    pr_ds = xr.merge([mean_ds['pr_mean'], quantiles_ds['pr']])
    tas_ds.to_netcdf(f"{processed_dir}tas_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")
    pr_ds.to_netcdf(f"{processed_dir}pr_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")

    # 5. -----
    # FastMIP outputs across ESMs (regional mean and quantiles).
    # These are calculated using all the METEOR noise realisations.
    quantity = 'quantiles-across-ESM'
    aggregation = 'regional'
    # aggregate to regional level, than calculate statistics:
    regional_means = compute_regional_means(base_ds)
    mean_ds = regional_means.mean(dim=['fair_realisation', 'noise_realisation', 'esm']).rename({'tas':'tas_mean', 'pr':'pr_mean'})
    quantiles_ds = regional_means.quantile(quantile_list, dim=['fair_realisation', 'noise_realisation', 'esm'])

    # save tas and pr to separate files:
    tas_ds = xr.merge([mean_ds['tas_mean'], quantiles_ds['tas']])
    pr_ds = xr.merge([mean_ds['pr_mean'], quantiles_ds['pr']])
    tas_ds.to_netcdf(f"{processed_dir}tas_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")
    pr_ds.to_netcdf(f"{processed_dir}pr_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")

    # 6. -----
    # FastMIP uncertainty decomposition by region, per scenario.
    aggregation = 'regional'
    quantity = 'uncertainty'
    # calculate regional means then statistics:
    regional_means = compute_regional_means(base_ds)
    var_esm = regional_means.mean(dim='noise_realisation').var(dim='esm').mean(dim='fair_realisation')
    var_glb = regional_means.mean(dim='noise_realisation').var(dim='fair_realisation').mean(dim='esm')
    var_int = regional_means.var(dim='noise_realisation').mean(dim=['esm', 'fair_realisation'])

    # save tas and pr to separate files:
    tas_var = xr.merge([var_esm['tas'].rename('var_esm'), var_glb['tas'].rename('var_glb'), var_int['tas'].rename('var_int')])
    pr_var = xr.merge([var_esm['pr'].rename('var_esm'), var_glb['pr'].rename('var_glb'), var_int['pr'].rename('var_int')])
    tas_var.to_netcdf(f"{processed_dir}tas_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")
    pr_var.to_netcdf(f"{processed_dir}pr_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")

    # 7. -----
    #FastMIP uncertainty decomposition gridwise, per scenario.
    aggregation = 'gridcell'
    quantity = 'uncertainty'
    # variance gridwise:
    var_esm = base_ds.mean(dim='noise_realisation').var(dim='esm').mean(dim='fair_realisation')
    var_glb = base_ds.mean(dim='noise_realisation').var(dim='fair_realisation').mean(dim='esm')
    var_int = base_ds.var(dim='noise_realisation').mean(dim=['esm', 'fair_realisation'])

    # save tas and pr to separate files:
    tas_var = xr.merge([var_esm['tas'].rename('var_esm'), var_glb['tas'].rename('var_glb'), var_int['tas'].rename('var_int')])
    pr_var = xr.merge([var_esm['pr'].rename('var_esm'), var_glb['pr'].rename('var_glb'), var_int['pr'].rename('var_int')])
    tas_var.to_netcdf(f"{processed_dir}tas_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")
    pr_var.to_netcdf(f"{processed_dir}pr_{scenarios_short[s_idx]}_METEOR_{aggregation}_{quantity}.nc")

# across scenario outputs:

# FastMIP uncertainty decomposition by region, across scenarios.
aggregation = 'regional'
quantity = 'across-scenario-uncertainty'

# combine all scenarios:
base_ds_list = []
for s_idx, scenario in enumerate(scenario_list):
    base_ds = xr.open_dataset(f"{aggregate_dir}METEOR_{scenarios_short[s_idx]}_scaledtoFAIR_combinedESMs_regridded.nc")
    base_ds_list.append(base_ds)
combined_ds = xr.concat(base_ds_list, dim='scenario', coords='minimal')

# 8. -----
# aggregate to regional level, than calculate variance components:
regional_means = compute_regional_means(combined_ds)

var_int = regional_means.var(dim='noise_realisation').mean(dim=['esm', 'fair_realisation', 'scenario'])
var_glb = regional_means.mean(dim='noise_realisation').var(dim=['fair_realisation']).mean(dim=['esm', 'scenario'])
var_esm = regional_means.mean(dim='noise_realisation').var(dim=['esm']).mean(dim=['fair_realisation', 'scenario'])
var_scenario = regional_means.mean(dim='noise_realisation').var(dim='scenario').mean(dim=['esm', 'fair_realisation'])

# save tas and pr to separate files:
tas_var = xr.merge([var_int['tas'].rename('var_int'), var_glb['tas'].rename('var_glb'), var_esm['tas'].rename('var_esm'), var_scenario['tas'].rename('var_scenario')])
pr_var = xr.merge([var_int['pr'].rename('var_int'), var_glb['pr'].rename('var_glb'), var_esm['pr'].rename('var_esm'), var_scenario['pr'].rename('var_scenario')])
tas_var.to_netcdf(f"{processed_dir}tas_cross-scenario_METEOR_{aggregation}_{quantity}.nc")
pr_var.to_netcdf(f"{processed_dir}pr_cross-scenario_METEOR_{aggregation}_{quantity}.nc")

# 9. -----
# FastMIP uncertainty decomposition gridwise, across scenarios.
aggregation = 'gridcell'
quantity = 'across-scenario-uncertainty'
var_int = combined_ds.var(dim='noise_realisation').mean(dim=['esm', 'fair_realisation', 'scenario'])
var_glb = combined_ds.mean(dim='noise_realisation').var(dim=['fair_realisation']).mean(dim=['esm', 'scenario'])
var_esm = combined_ds.mean(dim='noise_realisation').var(dim=['esm']).mean(dim=['fair_realisation', 'scenario'])
var_scenario = combined_ds.mean(dim='noise_realisation').var(dim='scenario').mean(dim=['esm', 'fair_realisation'])

# save tas and pr to separate files:
tas_var = xr.merge([var_int['tas'].rename('var_int'), var_glb['tas'].rename('var_glb'), var_esm['tas'].rename('var_esm'), var_scenario['tas'].rename('var_scenario')])
pr_var = xr.merge([var_int['pr'].rename('var_int'), var_glb['pr'].rename('var_glb'), var_esm['pr'].rename('var_esm'), var_scenario['pr'].rename('var_scenario')])
tas_var.to_netcdf(f"{processed_dir}tas_cross-scenario_METEOR_{aggregation}_{quantity}.nc")
pr_var.to_netcdf(f"{processed_dir}pr_cross-scenario_METEOR_{aggregation}_{quantity}.nc")
'''