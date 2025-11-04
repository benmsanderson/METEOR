# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%

# %%


# conda activate meteor
# # rm -r venv              
# make first-venv
# make clean
# make virtual-environment
# source venv/bin/activate
# then select the kernel 'venv' in jupyter notebook

import os,sys

import numpy as np
import xarray as xr
import pandas as pd
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))
warnings.filterwarnings("ignore", message=".*not in pamset.*")
 
from meteor import MeteorPatternScaling, meteor
from meteor import prpatt
from meteor import Cmip6MeteorDataGetter
from ciceroscm import input_handler



# %% [markdown]
# # Computing ClimateBench-like metrics for METEOR
#
# Metrics calculations as in ClimateBench 
#
# > Watson-Parris, D., Rao, Y., Olivié, D., Seland, Ø., Nowack, P., Camps-Valls, G., Stier, P., Bouabid, S., Dewey, M., Fons, E.,
# Gonzalez, J., Harder, P., Jeggle, K., Lenhardt, J., Manshausen, P., Novitasari, M., Ricard, L., and Roesch, C.: ClimateBench
# v1.0: A Benchmark for Data-Driven Climate Projections, Journal of Advances in Modeling Earth Systems, 14, e2021MS002 954,
# #https://doi.org/https://doi.org/10.1029/2021MS002954, e2021MS002954 2021MS002954, 2022
#
# ## Functions for calculations

# %%


def global_mean(ds):
    """
    Wrapper for calculating global mean values
    """
    try:
        gm = prpatt.global_mean(ds)
    except RuntimeError:   
        lat = ds[prpatt.get_lat_name(ds)]
        weight = np.cos(np.deg2rad(lat))
        weight = weight / weight.mean()
        other_dims = set(ds.dims) - {"ens"}
        gm = (ds * weight).mean(other_dims, skipna=True)
    return gm


# %%
# Function to calculate nrmse_values as done in ClimateBench 
# Watson-Parris, D., Rao, Y., Olivié, D., Seland, Ø., Nowack, P., Camps-Valls, G., Stier, P., Bouabid, S., Dewey, M., Fons, E.,
# Gonzalez, J., Harder, P., Jeggle, K., Lenhardt, J., Manshausen, P., Novitasari, M., Ricard, L., and Roesch, C.: ClimateBench
# v1.0: A Benchmark for Data-Driven Climate Projections, Journal of Advances in Modeling Earth Systems, 14, e2021MS002 954,
# https://doi.org/https://doi.org/10.1029/2021MS002954, e2021MS002954 2021MS002954, 2022

def nrmse_values(pred: xr.DataArray, true: xr.DataArray, alpha=5):
    """
    Compute global, spatial and combined NRMSE values
    """

    gm_truth = global_mean(true).mean("time")

    nrmse_spatial = np.sqrt(global_mean(np.square(pred.mean("time") - true.mean("time"))))/np.abs(gm_truth)
    nrmse_global = np.sqrt(np.square(global_mean(pred) - global_mean(true)).mean("time"))/np.abs(gm_truth)
    nrmse_tot = nrmse_spatial + nrmse_global * alpha
    return nrmse_spatial, nrmse_global, nrmse_tot


# %% [markdown]
# ## Ready input data for emissions to forcing

# %%


cscm_data_dir = os.path.join(os.getcwd(), "..", "src", "meteor", "default_scm_data")
just_NorESMLM = False
just_ssp534 = True


# # Data preparations

# ### Getting some data that can be used by the forcer engine and ciceroscm

# SSP245 for aerorol residual from abrupt-4xCO2

# %%


ih = input_handler.InputHandler({})
# NBVAL_IGNORE_OUTPUT

#For predictions: 
conc_data = input_handler.read_inputfile(os.path.join(cscm_data_dir, "ssp245_conc_RCMIP.txt"))
em_data = ih.read_emissions(os.path.join(cscm_data_dir, "ssp245_em_RCMIP.txt"))


# standard SSP scenarios

# %%


if just_NorESMLM:
    scenarios = [ "ssp126", "ssp245", "ssp370"] #, "ssp585"
else:
    scenarios = [ "ssp126", "ssp245", "ssp370", "ssp585"]


# %%


#For predictions: read in SSP concentrations and emissions
#scenarios = ["ssp126","ssp245","ssp370","ssp585"]

ih = input_handler.InputHandler({})
# NBVAL_IGNORE_OUTPUT

conc_data_fwd=[]
em_data_fwd=[]
for i,s in enumerate(scenarios):
    conc_data_fwd.append(input_handler.read_inputfile(os.path.join(cscm_data_dir, s+"_conc_RCMIP.txt")))
    em_data_fwd.append(ih.read_emissions(os.path.join(cscm_data_dir, s+"_em_RCMIP.txt")))


# SSP534-over

# %%


#For predictions: read in SSP concentrations and emissions

ih = input_handler.InputHandler({})
# NBVAL_IGNORE_OUTPUT

conc_data_ssp534=[]
em_data_ssp534=[]
for i,s in enumerate(['ssp534-over']):
    conc_data_ssp534.append(input_handler.read_inputfile(os.path.join(cscm_data_dir, s+"_conc_RCMIP.txt")))
    em_data_ssp534.append(ih.read_emissions(os.path.join(cscm_data_dir, s+"_em_RCMIP.txt")))


# ### Getting data for standard scenarios

# Getting CMIP6 data from the zarrstore for tas

# %% [markdown]
# ## Preparing data access from zarrstore

# %%
# Variables to emulate

flds = ["tas","pr"]


# %%

models = []
data_getter = None
if just_NorESMLM:
    data_getter = Cmip6MeteorDataGetter(exps=["piControl", "abrupt-4xCO2", "historical", "ssp245", "ssp126", "ssp370"], #"ssp126", "ssp245", "ssp370", "ssp585"], 
                                        flds = flds, 
                                        dbe=['CMIP','CMIP','CMIP', 'ScenarioMIP', 'ScenarioMIP','ScenarioMIP', 'ScenarioMIP'])
    models = data_getter.models
    len(models)
    print(models)
    models = ['NorESM2-LM']
    print(models)
elif not just_ssp534:
    data_getter = Cmip6MeteorDataGetter(exps=["piControl", "abrupt-4xCO2", "historical", "ssp245", "ssp126", "ssp370", "ssp585"], #"ssp126", "ssp245", "ssp370", "ssp585"], 
                                        flds = flds, 
                                        dbe=['CMIP','CMIP','CMIP', 'ScenarioMIP', 'ScenarioMIP','ScenarioMIP', 'ScenarioMIP'])
    models = data_getter.models
    len(models)
    #print(models)  

#for property, value in vars(data_getter).items():
#    print(property)


# %%


# remove problematic models when creating the pattern
if 'IITM-ESM' in models: models.pop(models.index('IITM-ESM'))  # drop because bad data
if 'IPSL-CM6A-LR' in models: models.pop(models.index('IPSL-CM6A-LR'))  # drop because of data issue
if 'TaiESM1' in models: models.pop(models.index('TaiESM1'))  # drop because of data issue
if 'EC-Earth3' in models: models.pop(models.index('EC-Earth3'))  # drop because of data issue in pr
if 'GISS-E2-1-G' in models: models.pop(models.index('GISS-E2-1-G'))  # drop temperature fit doesn't work
if 'MIROC6' in models: models.pop(models.index('MIROC6'))
if 'MPI-ESM1-2-LR' in models: models.pop(models.index('MPI-ESM1-2-LR'))
if 'FGOALS-g3' in models: models.pop(models.index('FGOALS-g3'))    # data offset between piC and hist
if 'MRI-ESM2-0' in models: models.pop(models.index('MRI-ESM2-0'))
print(models)
#sys.exit(4)

# ### Getting data for SSP534-over

# %%


if not just_NorESMLM and just_ssp534:
    data_getter_ssp534 = Cmip6MeteorDataGetter(exps=["piControl",  "abrupt-4xCO2", "historical", "ssp534-over", "ssp245", "ssp585"], 
                                        flds = flds, 
                                        dbe=['CMIP','CMIP', 'CMIP', 'ScenarioMIP', 'ScenarioMIP', 'ScenarioMIP'])

    models_ssp534 = data_getter_ssp534.models
    len(models_ssp534)
    print(models_ssp534)
    if 'UKESM1-0-LL' in models_ssp534: models_ssp534.pop(models_ssp534.index('UKESM1-0-LL'))  # ominous data
    if 'EC-Earth3' in models_ssp534: models_ssp534.pop(models_ssp534.index('EC-Earth3'))   # data issue (with pr?)
    if 'IPSL-CM6A-LR' in models_ssp534: models_ssp534.pop(models_ssp534.index('IPSL-CM6A-LR'))   # nan data issue
    if 'FGOALS-g3' in models_ssp534: models_ssp534.pop(models_ssp534.index('FGOALS-g3'))  # data offset between piC and hist
    if 'MIROC6' in models_ssp534: models_ssp534.pop(models_ssp534.index('MIROC6'))
    if 'GISS-E2-1-G' in models_ssp534: models_ssp534.pop(models_ssp534.index('GISS-E2-1-G'))


# # Train patterns and timescales for all models

# %% [markdown]
# ### Help functions to handle and get data from cache or zarrstore, and do training

# %%
def get_data_from_cache_or_download(mname, exp, data_getter=data_getter):
    if os.path.exists(f"cache/{mname}_{exp}_training_data.nc"):
        print(f"Getting {mname} {exp} data from cache")
        return xr.open_dataset(f"cache/{mname}_{exp}_training_data.nc")
    print(f"Getting {mname} {exp} data from datagetter")
    if exp in ["base", "co2x4"]:
        return data_getter.make_meteor_training_data(exp, mname)
    elif exp == "ssp245":
        return data_getter.make_meteor_training_data_composite(["historical", exp], mname)

def get_training_data_and_train(mname, data_getter=data_getter):
    # NBVAL_IGNORE_OUTPUT
    training_data = {
        "base": get_data_from_cache_or_download(mname, "base", data_getter=data_getter).isel(year=slice(-200, None)),
        "co2x4": get_data_from_cache_or_download(mname, "co2x4", data_getter=data_getter),#.isel(year=slice(0, 150)),
        "sulxanom": get_data_from_cache_or_download(mname, "ssp245", data_getter=data_getter),
    }
    ts_dict = {}
    for fld in flds:
        ts_dict[fld] = 3
        for exp, data in training_data.items():
            print(f"{exp} -- {fld}")
            print("---------------")
            print(training_data[exp].sizes)

            gm_tas = global_mean(data[fld])
            y_nan = np.isnan(gm_tas.values).sum()
            last_tas = np.isnan(gm_tas.values[-y_nan:]).sum()
            print(f"{exp}: total {np.isnan(data[fld].values).sum()}, years: {y_nan}, last:{last_tas}, mean: {np.mean(gm_tas)}")
            if last_tas > 0:
                training_data[exp] = training_data[exp].isel(year=slice(0,-last_tas))
    #sys.exit(4)
    # create GHG modes and spatial pattern
    CMIP_pattern = MeteorPatternScaling(
            mname,
            {"tas": 3, "pr": 3},
            lambda key: training_data[key],
            from_file=False,
            exp_list=["base", "co2x4", "sulxanom"],
            anom_timescales={"tas": 3, "pr": 3}
        )
    return CMIP_pattern


# # Get test data from file

# %% [markdown]
# ### Get ClimateBench data for comparison
#
# Downloaded from *Watson-Parris, D.: ClimateBench, https://doi.org/10.5281/zenodo.7064308, 2021.*

# %%


CB_test_data = xr.open_dataset("outputs_ssp245.nc")
#print(CB_test_data["pr"].values)
#print(global_mean(CB_test_data["pr"]))
#print(global_mean(CB_test_data["tas"]))


# # Predict and score:

# capture the spatial response of the last 20 years (2080--2100)

# %% [markdown]
# ## Emulate and compare
#
# ### Full loop for ClimateBench or full set of standard scenarios

# %%

if not just_ssp534:
    cols = []
    err_measures = ["NRMSE_spatial", "NRMSE_global", "NRMSE_total"]
    skip_for_now = [] #["EC-Earth3-Veg", "NorESM2-MM"]# "FGOALS-f3-L""CESM2", "CNRM-CM6-1-HR", , "EC-Earth3", "FGOALS-f3-L"]

    for nv, var in enumerate(flds): 
        for err_measure in err_measures:
            cols.append(f"{err_measure} for {var}")
    for nm, mname in enumerate(models):
        rowhs = []
        if os.path.exists(f"CB_scoring_{mname}_latex.csv") or mname in skip_for_now:
            continue
        scoring_table_data = np.zeros((len(scenarios) , 3*len(flds)))
        print(mname)
        #sys.exit(4)
        try:
            CMIP_pattern = get_training_data_and_train(mname)
        except ValueError:
            print(f"Pattern making not working for {mname}")
            continue
        for nv, var in enumerate(flds):
            shift_data = data_getter.get_single_var_mod_data_yearmean("piControl", var, mname).isel(year = slice(-200, None)).mean("year")
            for nsc, sc in enumerate(scenarios):
                # Get native data as an xarray DataArray
                prediction_ds = CMIP_pattern.predict_from_combined_experiment(
                    em_data_fwd[nsc], conc_data_fwd[nsc], [var]
                )[var]
                if sc == "ssp245" and just_NorESMLM:
                    truth_ds = CB_test_data[var]
                    if var == "pr":
                        truth_ds = truth_ds + shift_data
                    truth_ds = truth_ds.mean("member")
                    #print(truth_ds.time)
                    truth_ds = truth_ds.isel(time= slice(65, 86))



                    #sys.exit(4)
                else:
                    if os.path.exists(f"cache/{mname}_{sc}_training_data.nc"):
                        truth_ds = xr.open_dataset(f"cache/{mname}_{sc}_training_data.nc")[var].isel(year=slice(230,251))
                    else:
                        truth_ds = data_getter.get_single_var_mod_data_yearmean(sc, var, mname)  
                        truth_ds = truth_ds.isel(year= slice(65, 86))
                        truth_ds = truth_ds - shift_data
                    truth_ds = truth_ds.rename({"year": "time"})
                ##print(prediction_ds.time.shape)
                t_trying = np.arange(len(prediction_ds["time"]))
                #print(t_trying)
                prediction_ds = prediction_ds.assign_coords(time= list(t_trying))
                prediction_ds = prediction_ds.isel(time= slice(330, 351))
                if var == "pr" and (sc != "ssp245" or not just_NorESMLM) and not os.path.exists(f"cache/{mname}_{sc}_training_data.nc"):
                    prediction_ds = prediction_ds + shift_data                
                #print(f"{var}, {sc}, gm: {global_mean(prediction_ds).mean('time').values}, gm_truth: {global_mean(truth_ds).mean('time').values}")

                #print(truth_ds["time"])
                #print(prediction_ds["time"])
                try:
                    prediction_ds["time"] = truth_ds["time"]
                except ValueError:
                    prediction_ds = prediction_ds.isel(time= slice(0, len(truth_ds["time"])))
                    prediction_ds["time"] = truth_ds["time"]
                #print(var)
                nrmse_spatial, nrmse_global, nrmse_tot = nrmse_values(prediction_ds, truth_ds)
                scoring_table_data[nsc, nv*3] = nrmse_spatial
                scoring_table_data[nsc, nv*3 + 1] = nrmse_global
                scoring_table_data[nsc, nv*3 + 2] = nrmse_tot

        #for mname in models:
        for sc in scenarios:
            rowhs.append(f"{mname} {sc}")

        scoring_df = pd.DataFrame(
            data = scoring_table_data,
            columns = cols,
            index = rowhs

        )
        scoring_df
        if just_NorESMLM:
            scoring_df_CB = pd.DataFrame(
                data = np.array([[0.109, 0.074, 0.478, 2.341, 0.341, 4.048],
                        [0.107, 0.044, 0.327, 2.128, 0.209, 3.175],
                        [0.108, 0.058, 0.400, 2.524, 0.502, 5.035],
                        [0.080, 0.048, 0.320, 2.006, 0.331, 3.662],
                        [0.052, 0.072, 0.414, 1.350, 0.268, 2.691],
                        [0.258, 0.177, 1.141, 1.994, 0.389, 3.940]
                    ]),
                columns = cols,
                index = ["Gaussian Process CB", "Neural Network CB", "Random Forest CB", "Pattern Scaling CB", "Variability CB", "CMIP6 CB"]
            )
            """
            scoring_df["Gaussian Process CB"] = np.array([0.109, 0.074, 0.478, 2.341, 0.341, 4.048])
            scoring_df["Neural Network CB"] = np.array([0.107, 0.044, 0.327, 2.128, 0.209, 3.175])
            scoring_df["Random Forest CB"] = np.array([0.108, 0.058, 0.400, 2.524, 0.502, 5.035])
            scoring_df["Pattern Scaling CB"] = np.array([0.080, 0.048, 0.320, 2.006, 0.331, 3.662])
            scoring_df["Variability CB"] = np.array([0.052, 0.072, 0.414, 1.350, 0.268, 2.691])
            scoring_df["CMIP6 CB"] = np.array([0.258, 0.177, 1.141, 1.994, 0.389, 3.940])
            """
            scoring_df = pd.concat((scoring_df, scoring_df_CB))
            scoring_df.to_csv("CB_scoring_NorESM_LM_latex.csv")
            scoring_df.to_latex("CB_scoring_NorESM-LM_latex.txt", bold_rows=True, float_format="%.3f", label="CB_comprison_errors", caption = "METEOR performance on NorESM-LM evaluated using and compared to the Climate bench evalutaion suite. Note that comparison to non-ssp245 scenarios are to single scenarios, so variability driven errors are stronger in these.")
        else:
            scoring_df.to_csv(f"CB_scoring_{mname}_latex.csv")
            #scoring_df.to_latex(f"CB_scoring_{mname}_latex.txt", bold_rows=True, float_format="{{:0.3f}}".format, label="CB_comprison_errors_all", caption = "METEOR performance for all models evaluated using the Climate bench evalutaion metrics. Note that comparison is to single ensemble members, so variability driven errors are included.")


# %%
# Put the results into a latex-formatted table to put into overleaf

if not just_NorESMLM and not just_ssp534:
    list_scoring_dfs = []
    for mname in models:
        if mname in skip_for_now:
            continue
        scoring_df = pd.read_csv(f"CB_scoring_{mname}_latex.csv")
        list_scoring_dfs.append(scoring_df)
    full_scoring_df = pd.concat(list_scoring_dfs)
    full_scoring_df.set_index("Unnamed: 0", inplace=True)
    #print(full_scoring_df)
    full_scoring_df.to_latex("CB_scoring_all_latex.txt", float_format="%.3f", label="CB_comprison_errors_all", caption = "METEOR performance for all models evaluated using the Climate bench evaluation metrics. Note that comparison is to single ensemble members, so variability driven errors are included.")


# # Application to new scenarios

# ### SSP534-over

# %% [markdown]
# ## Loop over overshoot-scenario models

# %%


if not just_NorESMLM and just_ssp534:
    scenarios_534 = ["ssp534-over"]
    
    cols = []
    err_measures = ["NRMSE_spatial", "NRMSE_global", "NRMSE_total"]
    skip_for_now_534 = []#["MIROC-ES2L"]#["CESM2-WACCM"]
    rowhs = []
    for nv, var in enumerate(flds): 
        for err_measure in err_measures:
            cols.append(f"{err_measure} for {var}")   
    for nm, mname in enumerate(models_ssp534):
        print(mname)
        if mname in skip_for_now_534:
            continue
        if os.path.exists(f"CB_scoring_ssp534_{mname}.csv"):
            continue
        scoring_table_data_ssp534 = np.zeros((len(scenarios_534) , 3*len(flds)))

        CMIP_pattern = get_training_data_and_train(mname, data_getter=data_getter_ssp534)
        for nv, var in enumerate(flds):
            if os.path.exists(f"cache/{mname}_base_training_data.nc"):
                print(f"cache/{mname}_base_training_data.nc")
                shift_data = xr.open_dataset(f"cache/{mname}_base_training_data.nc")[var].isel(year=slice(-200,None)).mean("year")
            else:
                shift_data = data_getter_ssp534.get_single_var_mod_data_yearmean("piControl", var, mname).isel(year=slice(-200,None)).mean("year")
            for nsc, sc in enumerate(scenarios_534):
                # Get native data as an xarray DataArray
                prediction_ds = CMIP_pattern.predict_from_combined_experiment(
                    em_data_fwd[nsc], conc_data_fwd[nsc], [var]
                )[var]
                print(mname)
                print(sc)
                if os.path.exists(f"cache/{mname}_{sc}_training_data.nc"):
                    print(f"cache/{mname}_{sc}_training_data.nc")
                    truth_ds = xr.open_dataset(f"cache/{mname}_{sc}_training_data.nc")[var].isel(year=slice(230,251))
                else:
                    truth_ds = data_getter_ssp534.get_single_var_mod_data_yearmean(sc, var, mname)  
                    truth_ds = truth_ds#.isel(year= slice(65, 86))
                if var == "tas":
                    truth_ds = truth_ds - shift_data
                #sys.exit(4)
                if len(truth_ds.year) == 61:
                    truth_ds = truth_ds.isel(year= slice(40, 61))
                elif len(truth_ds.year) == 86:
                    truth_ds = truth_ds.isel(year= slice(65, 86))
                else:
                    print(len(truth_ds.year))
                ##print(prediction_ds.time.shape)
                truth_ds = truth_ds.rename({"year": "time"})
                t_trying = np.arange(len(prediction_ds["time"]))
                #print(t_trying)
                prediction_ds = prediction_ds.assign_coords(time= list(t_trying))
                prediction_ds = prediction_ds.isel(time= slice(330, 351))
                if var == "pr":
                    prediction_ds = prediction_ds + shift_data                
                print(f"{var}, {sc}, gm: {global_mean(prediction_ds).mean('time').values}, gm_truth: {global_mean(truth_ds).mean('time').values}")

                #print(truth_ds["time"])
                #print(prediction_ds["time"])
                print(mname)
                print(truth_ds)
                prediction_ds["time"] = truth_ds["time"]
                #print(var)
                
                nrmse_spatial, nrmse_global, nrmse_tot = nrmse_values(prediction_ds, truth_ds)
                scoring_table_data_ssp534[0, nv*3] = nrmse_spatial
                scoring_table_data_ssp534[0, nv*3 +1] = nrmse_global
                scoring_table_data_ssp534[0, nv*3+2] = nrmse_tot
        scoring_df_534 = pd.DataFrame(
            data = scoring_table_data_ssp534,
            columns = cols,
            index = [f"{mname} {sc}"]
        )
        scoring_df_534.to_csv(f"CB_scoring_ssp534_{mname}.csv")
    for mname in models_ssp534:
        for sc in scenarios_534:
            rowhs.append(f"{mname} {sc}")
    """
    scoring_df_534 = pd.DataFrame(
        data = scoring_table_data_ssp534,
        columns = cols,
        index = rowhs
    )
    scoring_df_534.to_csv("CB_scoring_ssp534.csv")
    scoring_df_534.to_latex("CB_scoring_ssp534_latex.txt", bold_rows=True, float_format="{{:0.3f}}".format, label="CB_comprison_errors_ssp534", caption = "METEOR performance for the overshoot scenario ssp534 evaluated using the Climate bench evalutaion metrics. Note that comparison is to single ensemble members, so variability driven errors are included.")
    """

# %%
# Put the results into a latex-formatted table to put into overleaf
if not just_NorESMLM and just_ssp534:

    list_scoring_dfs_534 = []
    for mname in models_ssp534:
        if mname in skip_for_now_534:
            continue
        scoring_df = pd.read_csv(f"CB_scoring_ssp534_{mname}.csv")
        list_scoring_dfs_534.append(scoring_df)
    full_scoring_df = pd.concat(list_scoring_dfs_534)
    full_scoring_df.set_index("Unnamed: 0", inplace=True)
    print(full_scoring_df)
    full_scoring_df.to_latex("CB_scoring_all_534_latex.txt", bold_rows=True, float_format="%.3f", label="CB_comprison_errors_all", caption = "METEOR performance for all models evaluated using the Climate bench evalutaion metrics. Note that comparison is to single ensemble members, so variability driven errors are included.")


# # Application to new scenarios

# ### SSP534-over

# %%
