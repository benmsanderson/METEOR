import os
import datetime
import pandas as pd
from meteor import MeteorInterface
import random

# =========================================================
# SETUP
# =========================================================

base_dir = '/div/no-backup-nac/users/maurad/METEOR/'

ESM_list = ['CanESM5','ACCESS-ESM1-5', 'IPSL-CM6A-LR', 'MPI-ESM1-2-LR', 'MIROC6']  # Tier 1 list
scenarios = ['SSP2 - Low Emissions', 'SSP2 - Medium Emissions', 'SSP3 - High Emissions', 'SSP1 - Very Low Emissions'] #scenario names for FAIR forced
#scenarios = ['Low - SSP2 (Marker)', 'Medium - SSP2 (Marker)','High - SSP3 (Marker)','Very Low - SSP1 (Marker)'] #scneario names for FAIR full variability.  
scenarios_short = ['L','M','H','VL']


# FAIR ensemble samples:
FAIRens = pd.read_csv(f'{base_dir}/data/FASTMIP_phase2/FAIR_data/climate_assessment_forced_v20260325.csv')
#FAIRens = pd.read_csv(f'{base_dir}/data/FASTMIP_phase2/FAIR_data/climate_assessment_full.csv')
num_fair = 20

# METEOR setup:
start_year = 2015
end_year = 2100
n_members = 10 # Meteor realizations per FAIR ensemble member
emiss_dir = f'{base_dir}/data/FASTMIP_phase2/scenario_data/'

# check if output directories exist, if not, create them:
output_dir = f'{base_dir}/data/FASTMIP_phase2/METEOR_emulations/raw/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# =========================================================
# Some utils
# =========================================================

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

# =========================================================
# GENERATE FAIR SUBSAMPLE
# TODO: add if wrapper about generating new FAIR subsample, or using existing one. 
# If re-using subsample (ie. for additional scenarios), then user needs to give a datetime to identify which subsample to use. 
# =========================================================

# create new random FAIR subsample
all_ensemble_members = FAIRens['ensemble_member'].unique().tolist()
random_subsample = random.sample(all_ensemble_members, num_fair)
FAIR_subsample = FAIRens.loc[FAIRens['ensemble_member'].isin(random_subsample) & (FAIRens['scenario'].isin(scenarios)) & (FAIRens['variable'] == 'Climate Assessment|Surface Temperature (GSAT)')].sort_values('ensemble_member')

# Or load previous FAIR subsample pickle:
#FAIR_subsample = pd.read_pickle('../data/FASTMIP_phase2/FAIR_data/subsample_20members_20260617-225409.pkl')

# save with timestamp for a unique name
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
FAIR_subsample.to_pickle(f'{base_dir}/data/FASTMIP_phase2/FAIR_data/subsample_{num_fair}members_{timestamp}.pkl')

# =========================================================
# train METEOR for list of ESMs (uses cached models if available)
# and generate annual gridded output scaled to selected FAIR ensemble members
# =========================================================

for esm in ESM_list:
    print(f"Creating emulator for {esm}...")
    emulator = MeteorInterface(
        model=esm,
        variables=['tas', 'pr'],
        cache_dir=f'{base_dir}/cache'
    )
   
    print(f"Training emulator for {emulator.model}...")
    emulator.train(verbose=True)

    # For each scenario, generate METEOR realizations scaled to randomly selected FAIR GSAT timeseries
    for s_idx,scenario in enumerate(scenarios):
        print(f"Emulating scenario {scenario}...")

        # Construct emissions and concentrations dict
        emissions_path = f"/div/no-backup-nac/users/masan/GRAFITE/temp_indata/scen7-{scenarios_short[s_idx]}_em_gases_vupdate_2022_AR6.txt"
        concentrations_path = f"{emiss_dir}scen7-{scenarios_short[s_idx]}_conc_gases_vupdate_2022_AR6.txt"
        scenario_dict = {'emissions': emissions_path, 'concentrations': concentrations_path}

        # Get FAIR subsample for this scenario:
        FAIR_sample_for_scenario = FAIR_subsample.loc[FAIR_subsample['scenario'] == scenario]

        # Loop through FAIR ensemble members and create METEOR realizations:
        for fair_idx in FAIR_sample_for_scenario.index:

            # Extract the timeseries for this member:
            scaling_ts = FAIR_sample_for_scenario.loc[fair_idx, str(start_year):str(end_year)].rename(int).to_xarray().rename('tas').rename({'index':'year'})
            # Extract FAIR ensemble member number for naming:
            fair_mem_num = FAIR_sample_for_scenario.loc[fair_idx, 'ensemble_member']

            # Create METEOR noise realizations scaled to this FAIR member's GSAT timeseries:
            ensemble_scaled = emulator.generate_ensemble_outputs(
                scenario=scenario_dict,
                start_year=start_year,
                end_year=end_year,
                n_realizations=n_members,
                timeseries=['global'], 
                gridded={'annual': list(range(start_year, end_year + 1)), 'monthly': list(range(start_year, end_year + 1))},
                temp_scaling_ts=scaling_ts,
                save_to=f"{output_dir}METEOR_{esm}_{scenarios_short[s_idx]}_scaledtoFAIRens_{fair_mem_num}.nc"
            ) 