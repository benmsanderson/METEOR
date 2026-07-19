"""
Module to get CMIP6 data and convert to format that can be used for METEOR
"""

import logging
import os
import pickle  # nosec B403
import re
from typing import Iterable

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

from .cache_handling import CacheHandler

cmip6_to_meteor_exp_remapper = {
    "base": "piControl",
    "co2x4": "abrupt-4xCO2",
    "co2x8": "abrupt-4xCO2",
    "co2x16": "abrupt-4xCO2",
    "1pc": "1pctCO2",
}


def sort_member_ids_numerically(member_ids: Iterable[str]) -> list[str]:
    """
    Sort CMIP6 member IDs numerically by the realization number.

    Member IDs follow the pattern rXiYpZfW where X, Y, Z, W are integers.
    String sorting would put r10 before r1, but we want numerical order.

    Parameters
    ----------
    member_ids : Iterable[str]
        List of member IDs like ['r1i1p1f1', 'r10i1p1f1', 'r2i1p1f1']

    Returns
    -------
    list[str]
        Member IDs sorted numerically by realization number (r value)

    Examples
    --------
    >>> sort_member_ids_numerically(['r10i1p1f1', 'r1i1p1f1', 'r2i1p1f1'])
    ['r1i1p1f1', 'r2i1p1f1', 'r10i1p1f1']
    """

    def extract_realization_number(member_id: str) -> int | float:
        """Extract the realization number (r value) from member_id."""
        match = re.match(r"r(\d+)i", member_id)
        if match:
            return int(match.group(1))
        return float("inf")  # Put unparseable IDs at the end

    return sorted(member_ids, key=extract_realization_number)


def multiply_along_axis(array_a, array_b, axis):
    """
    Multiply to arrays along a given axis

    Pure infrastructure function to make multiplying along an axis
    that is not the last axis along two np.ndarray objects without
    encountering broadcasting issues

    Parameters
    ----------
    array_a : np.ndarray
        First array to multiply
    array_b : np.ndarray
        Second array to multiply
    axis : int
        Axis to multiply along

    Returns
    -------
    np.ndarray
        Array/matrix with result of multiplication, and the
        multiplication axis back where it was
    """
    return np.swapaxes(np.swapaxes(array_a, axis, -1) * array_b, -1, axis)


def year_mean_monthly(monthly_data: np.ndarray) -> np.ndarray:
    """
    Calculate yearmean from monthly data

    Weighting by days in month and calculating the yearly mean of an array of
    monthly data. The data are assumed to be January to December per year, and
    will assume for simplicity that the data follows a no-leap calendar

    Parameters
    ----------
    monthly_data : np.ndarray
        multiple dimensional np.ndarray with the first dimension being
        time and on monthly resolutions running from January to December
        for each year

    Returns
    -------
    np.ndarray
        Weighted year averages for the monthly_data, the output-dimension will
        be the same as for the monthly_data, except that the first time
        dimension will be 1/12th as long as before including only yearly mean
        values
    """
    month_weights = np.tile(
        (np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) / 365.0 * 12.0),
        monthly_data.shape[0] // 12,
    )
    mul_weigths = multiply_along_axis(monthly_data, month_weights, 0)
    return np.mean(
        mul_weigths.reshape(
            -1,
            12,
            monthly_data.shape[1],
            monthly_data.shape[2],
        ),
        axis=1,
    )


def year_mean_monthly_xarray(monthly_xarray: xr.DataArray) -> xr.DataArray:
    """
    Calculate yearmean from monthly xarray dataarray

    Essentially just a wrapper for xarray around the numpy based
    year_mean_monthly function above

    Parameters
    ----------
    monthly_xarray : xr.DataArray
        with the first dimension being time followed by lat and lon and on
        monthly resolutions running from January to December for each year

    Returns
    -------
    xr.DataArray
        with same dimensions as before but with the time dimension 12 times
        fewer entries, corresponding to yearly values, this coordinate now
        also no longer has an associated coordinate as it has changed
    """
    return xr.apply_ufunc(
        year_mean_monthly,
        monthly_xarray,
        input_core_dims=[
            ["time", "lat", "lon"],
        ],
        output_core_dims=[
            ["time", "lat", "lon"],
        ],
        exclude_dims=set(("time",)),
        dask="allowed",
    )


def make_xarray_with_correct_dims(fld_names: list, fld_values: list) -> xr.Dataset:
    """
    Make a dataset for a list of dataArrays over the same dimensions

    Parameters
    ----------
    fld_names : list
        containing names of the fields, the ordering should be
        the same as for the fld_values
    fld_values : list
        containing the xr.DataArrays for each of the fields in the
        fld_names list

    Returns
    -------
    xr.Dataset
    """
    ds = xr.Dataset(
        data_vars={fld_names[i]: fld_values[i] for i in range(len(fld_names))}
    )
    return ds


def initialise_dataframe_and_models(
    df_all1, flds, exps, mdl_skipmbrs=None
):  # pylint: disable=too-many-locals, too-many-branches
    """
    Intialise a dataframe with complete data for the first full data ensemble
    member from a cmip6 data list

    Parameters
    ----------
    df_all1: list
        Two-dimensional list of pd.DataFrame where the first dimension runs
        over the experiments and the second over the fields in fld
    flds: list
        of names of fields
    exps: list
        of names of experiments
    mdl_skipmbrs : dict
        Of models and a list of ensemble members to skip for that model. If
        none is sent, it defaults to skipping NorESM2-Lm which has
        insufficient data for pr for the historical experiment even though
        there is a file
    Returns
    -------
    list
        Consisting of first a list with dataframes with models and first
        ensemble members with full data per experiment and field, and second a
        list of the models that have the full data
    """
    mdls1 = df_all1[0][0].source_id.unique()
    mdls1 = sorted(mdls1)
    # Collect rows as lists, then build DataFrames at end
    df_all_rows = []
    if mdl_skipmbrs is None:
        mdl_skipmbrs = {"NorESM2-LM": ["r1i1p1f1"]}
    for i in range(len(exps)):
        tmp = [[] for j in range(len(flds))]
        df_all_rows.append(tmp)

    mdls = []
    missing_data_info = []  # Track which models are missing which data

    for mdl in mdls1:  # pylint: disable=too-many-nested-blocks

        # Test that one ensemble member has all data:
        sufficient_data = True
        model_missing_data = []
        model_rows = []
        for i, _ in enumerate(exps):
            model_rows.append([None] * len(flds))

            # find first variable for expt/model
            for j, _ in enumerate(flds):
                if "historical" in exps:
                    ii = exps.index("historical")
                    query_str = f"source_id=='{mdl}' &" f"experiment_id == 'historical'"
                    hist_tmp = df_all1[ii][j].query(query_str)
                    hmb = sort_member_ids_numerically(hist_tmp.member_id.unique())
                else:
                    hmb = []
                tmp = df_all1[i][j].query(f"source_id=='{mdl}'")
                mmbs = tmp.member_id.unique()
                if len(mmbs) > 0:
                    # Sort member IDs numerically to prefer r1 over r10, r2,
                    # etc.
                    mmbs_sorted = sort_member_ids_numerically(mmbs)
                    mmb = mmbs_sorted[0]
                    if len(hmb) > 0:
                        if hmb[0] in mmbs:
                            mmb = hmb[0]

                    tt = df_all1[i][j].query(f"source_id=='{mdl}' & member_id=='{mmb}'")

                    if len(tt) == 0:
                        # Data query returned empty - this shouldn't happen
                        # given mmbs check
                        model_missing_data.append(f"{flds[j]}@{exps[i]}")
                        sufficient_data = False
                    model_rows[i][j] = tt.iloc[0].to_dict()
                else:
                    # No member data found for this field/experiment
                    # combination
                    mmb = -1
                    model_rows[i][j] = None
                    model_missing_data.append(f"{flds[j]}@{exps[i]}")
                    sufficient_data = False
            # add model to final list

        if sufficient_data:
            mdls.append(mdl)
            for i in range(len(exps)):
                for j in range(len(flds)):
                    df_all_rows[i][j].append(model_rows[i][j])
        else:
            missing_data_info.append(f"{mdl}: missing {', '.join(model_missing_data)}")

    # Log which models were excluded due to missing data
    if missing_data_info:
        logging.info(
            "Excluded %d models due to incomplete data:\n%s",
            len(missing_data_info),
            # Show first 10
            "\n".join(f"  - {info}" for info in missing_data_info),
        )
    # Build DataFrames from collected rows
    df_all = []
    for i in range(len(exps)):
        tmp = []
        for j in range(len(flds)):
            if len(df_all_rows[i][j]) > 0:
                tmp.append(pd.DataFrame(df_all_rows[i][j]))
            else:
                tmp.append(pd.DataFrame(columns=df_all1[0][0].columns))
        df_all.append(tmp)

    return df_all, mdls


class Cmip6MeteorDataGetter:  # pylint: disable=too-many-instance-attributes
    """
    Cmip6MeteorDataGetter class

    Objects of this class holds a shortlist of CMIP6 data links to data for
    one ensemble member for all models that have complete data for the full
    combination of the classes fields and experiments all monthly data.

    In various methods it can return yearly mean data for a single
    dataset, or a full METEOR training dataset

    Attributes
    ----------
    flds: list
        List of variable fields for the instance to consider
    exps: list
        List of experiments for the instance to consider


    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    def __init__(
        self,
        models=None,
        flds=None,
        tabids=None,
        exps=None,
        dbe=None,
        cache_dir=None,
        cache_handler=None,
        enable_cache=False,
        enable_compression=True,
        compression_level=6,
    ):
        """
        Initialise data getter object

        Parameters
        ----------
        models : list
            List of models to include. If nothing is sent the object will
            include all models that have data for the full combination of
            fields and experiments
        flds : list
            Of variable fields to include (monthly data)
            If nothing is sent for this it will be set to
            ['tas', 'pr']
        tabids : list[str], optional
            List of table_id names to use for each field.
            This should be the same length as flds, and have values
            corresponding to each field in the same order.
            If nothing is sent for this it will be set to a list of all
            entries equal to 'Amon' with the same length as flds.
        exps : list
            Of experiments to include. If nothing is sent for
            this it will be set to ['piControl', 'abrupt4xCO2']
        dbe : list
            Of CMIP6 project names for each of the experiments. This should be
            the same length as exps, and have values corresponding to each
            experiment in the same order. If nothing is set, it will be set to
            a list of all entries equal to 'CMIP' with the same length as
            exps except for experiments starting with 'ssp' which will be set
            to 'ScenarioMIP'
        cache_dir : str, optional
            Directory to store cached data. If None, the cache directory will
            be set to a default location depending on the installation type:
             - For development installations: `<repo_root>/.cache`
             - For pip-installed packages: `~/.meteor/cache`
        enable_cache : bool, optional
            Whether to enable automatic caching. Default is False.
            Set to True to cache downloaded data locally for faster subsequent
            access.
        enable_compression : bool, optional
            Whether to enable netCDF4/zlib compression for cached files.
            Default is True. This can significantly reduce file sizes
            (typically 70-90% compression).
        compression_level : int, optional
            Compression level for netCDF4/zlib compression (1-9). Default is
            6. Higher values provide better compression but slower performance:
            - 1: Fastest compression, larger files
            - 6: Good balance of speed vs compression (recommended)
            - 9: Best compression, slowest performance
        df_all : list
            Two dimensional list along experiments and fields, every entry
            is a pandas.DataSet with lines with data placement for one
            ensemble member for each model that has data for the full
            combination of fields and experiments
        gcs : gcfs.GCSFileSystem
              A GCSFileSystem to load data
        """
        dbe = self._set_models_flds_tabids_exps_dbe(models, flds, tabids, exps, dbe)

        # Set up caching
        self.enable_cache = enable_cache
        if self.enable_cache:
            if cache_handler is None:
                self.cache_handler = CacheHandler(
                    cache_dir=cache_dir,
                    purpose="cmip6",
                    enable_compression=enable_compression,
                    compression_level=compression_level,
                )
            else:
                self.cache_handler = cache_handler
            self.enable_cache = self.cache_handler.cache_functioning

            # Load CMIP6 catalog (with caching to avoid network calls when
            # possible)
            catalog_cache_file = self.cache_handler.get_cmip6_query_catalogue()
        else:
            catalog_cache_file = None

        if self.enable_cache and os.path.exists(catalog_cache_file):
            # Use cached catalog
            logging.info("Loading CMIP6 catalog from cache: %s", catalog_cache_file)
            df = pd.read_csv(catalog_cache_file, low_memory=False)
        else:
            # Download catalog from Google Cloud Storage
            try:
                logging.info("Downloading CMIP6 catalog from Google Cloud Storage...")
                df = pd.read_csv(
                    "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
                    low_memory=False,
                )
                # Cache the catalog if caching is enabled
                if self.enable_cache:
                    try:
                        df.to_csv(catalog_cache_file, index=False)
                        logging.info(
                            "Cached CMIP6 catalog to: %s",
                            catalog_cache_file,
                        )
                    except OSError as e:
                        logging.warning("Failed to cache CMIP6 catalog: %s", e)
            except Exception as e:  # pylint: disable=broad-exception-caught
                # If download fails and we have cache enabled, check if there's
                # an old catalog
                if self.enable_cache and os.path.exists(catalog_cache_file):
                    logging.warning(
                        "Failed to download CMIP6 catalog (%s), using cached "
                        "version.",
                        e,
                    )
                    df = pd.read_csv(catalog_cache_file, low_memory=False)
                else:
                    # No cache available and download failed
                    raise RuntimeError(
                        "Failed to load CMIP6 catalog from Google Cloud "
                        f"Storage and no cached version available: {e}"
                    ) from e
        model_filter = (
            "" if self.filter_models is None else "|".join(self.filter_models)
        )
        df_all1 = []
        for i, exp in enumerate(self.exps):
            df_ta1 = []
            for fld, tabid in zip(self.flds, self.tabids):
                query_str = (
                    f"activity_id=='{dbe[i]}' &"
                    f"table_id == '{tabid}' &"
                    f"variable_id == '{fld}' &"
                    f"experiment_id == '{exp}'"
                )
                if model_filter:
                    query_str += f" & source_id.str.contains('{model_filter}')"
                tmp = df.query(query_str)
                df_ta1.append(tmp)
            df_all1.append(df_ta1)

        self.df_all, self.models = initialise_dataframe_and_models(
            df_all1, flds=self.flds, exps=self.exps
        )

        # Validate that we found at least one model with complete data
        if len(self.models) == 0:
            if self.filter_models is None:
                model_str = "No models found that "
            else:
                if len(self.filter_models) == 1:
                    model_str = f"Model {self.filter_models[0]} does not "
                else:
                    model_str = f"Non of the requested models {self.filter_models} "
            error_msg = model_str + (
                "have complete data for the requested data combination:\n"
                "  Fields and Table IDs:\n"
                + "\n".join(
                    f"    - {fld} (table_id: {tabid})"
                    for fld, tabid in zip(self.flds, self.tabids)
                )
                + "\n"
                f"  Experiments: {self.exps}\n\n"
            )
            raise ValueError(error_msg)

        self.gcs = gcsfs.GCSFileSystem(token="anon")  # nosec

    def _set_models_flds_tabids_exps_dbe(self, models, flds, tabids, exps, dbe):
        """
        Private method to set flds and exps and dbe and
        take care of if they are not set then have defaults

        Parameters
        ----------
        models : list
            List of models to include. If None is sent for this it will
            be set to None and not used for filtering, and the class will
            include all models that have data for the full combination of
            fields and experiments
        flds : list
            Of variable fields to include (monthly data)
            If None is sent for this it will be set to
            ['tas', 'pr']
        tabid : list[str]
            List of table_id names to use for each field if flds.
            This should be the same length as flds,
            if None is sent for this it will be set to a list of all
            entries equal to 'Amon' with the same length as flds.
        exps : list
            Of experiments to include. If None is sent for
            this it will be set to ['piControl', 'abrupt4xCO2']
        dbe : list
            Of CMIP6 project names for each of the experiments. This should be
            the same length as exps, and have values corresponding to each
            experiment in the same order. If None is sent, it will be set
            based on the experiment type:
            - CMIP experiments: 'CMIP'
            - SSP scenarios: 'ScenarioMIP'
            - 1pctCO2: 'CMIP'

        Returns
        -------
        list
            dbe, If None was sent, a list with correct activity_ids for each
            experiment
        """
        if not models:
            models = None  # No filtering on models
        if isinstance(models, str):
            models = [models]
        if not flds:
            flds = ["tas", "pr"]
        if isinstance(flds, str):
            flds = [flds]
        if not tabids:
            tabids = ["Amon"] * len(flds)
        if isinstance(tabids, str):
            tabids = [tabids] * len(flds)
        if len(tabids) != len(flds):
            raise ValueError("tabids should be the same length as flds")
        if not exps:
            exps = ["piControl", "abrupt-4xCO2"]
        if isinstance(exps, str):
            exps = [exps]
        if not dbe:
            # Map experiments to correct activity_ids
            dbe = []
            for exp in exps:
                if exp.startswith("ssp"):
                    # SSP scenarios are in ScenarioMIP
                    dbe.append("ScenarioMIP")
                else:
                    # Default experiments (1pct, piControl, abrupt-4xCO2,
                    # historical) are in CMIP
                    dbe.append("CMIP")
        self.filter_models = models
        self.flds = flds
        self.tabids = tabids
        self.exps = exps
        return dbe

    def get_models_avail(self):
        """
        Get list of available models

        Returns
        -------
        list
            List of available models. They have full data for the dataGetters
            flds and experiment combinations for at least one ensemble member.
        """
        return self.models.copy()

    def check_if_model_has_data(self, model):
        """
        Check if a model has full data for this dataGetter object

        Parameters
        ----------
        model: str
            Name of model

        Returns
        -------
        bool
            True if the model has full data for at least on ensemble member
            False otherwise
        """
        if self.filter_models is not None and model not in self.filter_models:
            logging.warning(
                "Model %s was not among the requested models provided in "
                "%s initialisation. Available ones are %s.\n"
                "Reinitialise the object to include "
                "this %s if you want to use it.",
                model,
                self.__class__.__name__,
                self.filter_models,
                model,
            )
        if model in self.models:
            return True
        return False

    def get_single_var_mod_data(self, exp, fld, model):
        """
        Get mean data for a single variable, model and experiment combination

        Parameters
        ----------
        exp: str
            Name of experiment for which you want data
        fld: str
            Name of field for which you want data
        model: str
            Name of model for which you want data

        Returns
        -------
        xr.DataArrray
            Data downloaded from zstore

        Raises
        ------
        KeyError
            If either exp, or fld is not in the instances self.exps and
            self.flds or if the model is not among the models that has
            complete data for all the combinations of experiments
            and fields that the instance holds.
        """
        # Try to load from cache first if caching is enabled
        if self.enable_cache:
            cached_data = self.cache_handler.load_cmip6_cached_data(
                "get_single_var_mod_data", exp, fld, model
            )
            if cached_data is not None:
                return cached_data

        # Original logic for data fetching
        if exp not in self.exps:
            raise KeyError(
                "This datagetter does not handle data from the "
                f"{exp} experiment, available options are {self.exps}"
            )
        if fld not in self.flds:
            raise KeyError(
                "This datagetter does not handle "
                f"{fld} data, available options are {self.flds}"
            )
        if not self.check_if_model_has_data(model):
            raise KeyError(f"No or incomplete data for {model}")
        zstore_ref = (
            self.df_all[self.exps.index(exp)][self.flds.index(fld)]
            .loc[self.models.index(model)]
            .zstore
        )

        if zstore_ref is np.nan:
            raise KeyError(f"No zstore ref for {model}")

        mapper = self.gcs.get_mapper(zstore_ref)
        fld_data = xr.open_zarr(mapper, decode_times=False).sortby("time")

        # Apply limit for piControl experiments to save storage space while
        # maintaining sufficient data for pattern fitting. Pattern fitting
        # requires enough data points to accurately estimate temporal response
        # parameters.
        # Use 150 years which balances storage efficiency with statistical
        # robustness
        # 150 years * 12 months
        if exp == "piControl" and len(fld_data.time) > 1800:
            print(
                "   Limiting piControl data to first 150 years "
                f"(was {len(fld_data.time) // 12} years)"
            )
            fld_data = fld_data.isel(
                time=slice(0, 1800)
            )  # First 150 years (1800 months)

        # No longer cache raw data - we only cache processed monthly data
        # to avoid redundancy and save storage space
        # if self.enable_cache:
        #     self._save_to_cache(cache_key, fld_data)

        return fld_data

    def get_single_var_mod_data_yearmean(self, exp, fld, model):
        """
        Get yearly mean data for a single variable, model and experiment
        combination

        Parameters
        ----------
        exp: str
            Name of experiment for which you want data
        fld: str
            Name of field for which you want data
        model: str
            Name of model for which you want data

        Returns
        -------
        xr.DataArrray
            Data converted from monthly to yearly mean data and including an
            extra flat ens dimension
        """
        # Try to load from cache first if caching is enabled
        if self.enable_cache:
            cached_data = self.cache_handler.load_cmip6_cached_data(
                "get_single_var_mod_data_yearmean",
                exp,
                fld,
                model,
                expected_type="DataArray",
            )
            if cached_data is not None:
                logging.info("✓ Using cached yearly data for %s/%s/%s", model, exp, fld)
                return cached_data

        # Get monthly data (which may be cached) and compute yearly mean
        logging.info(
            "Computing yearly mean from monthly data for %s/%s/%s...",
            model,
            exp,
            fld,
        )
        var_monthly = self.get_single_var_mod_data_monthly(exp, fld, model)
        if var_monthly is None:
            return None

        # Convert monthly to yearly mean
        # var_monthly has dimensions (ens, month, lat, lon)
        # We need to reshape to compute yearly means
        n_years = var_monthly.sizes["month"] // 12
        var_monthly_subset = var_monthly.isel(month=slice(0, n_years * 12))

        # Reshape and compute yearly mean
        var_yearly = var_monthly_subset.coarsen(month=12, boundary="trim").mean()
        var_yearly = var_yearly.assign_coords({"month": np.arange(n_years)}).rename(
            {"month": "year"}
        )

        # Cache the yearly data to avoid recomputing
        if self.enable_cache:
            self.cache_handler.save_cmip6_to_cache(
                var_yearly,
                "get_single_var_mod_data_yearmean",
                exp,
                fld,
                model,
            )

        return var_yearly

    def get_single_var_mod_data_monthly(self, exp, fld, model):
        """
        Get monthly data for a single variable, model and experiment
        combination

        Parameters
        ----------
        exp: str
            Name of experiment for which you want data
        fld: str
            Name of field for which you want data
        model: str
            Name of model for which you want data

        Returns
        -------
        xr.DataArray
            Monthly data with time dimension renamed to "month"
            including an extra flat ens dimension
        """
        # Try to load from cache first if caching is enabled
        if self.enable_cache:
            cached_data = self.cache_handler.load_cmip6_cached_data(
                "get_single_var_mod_data_monthly",
                exp,
                fld,
                model,
                expected_type="DataArray",
            )
            if cached_data is not None:
                logging.info(
                    "✓ Using cached monthly data for %s/%s/%s", model, exp, fld
                )
                return cached_data

        # Original logic
        logging.info(
            "Downloading data from Google Cloud for %s/%s/%s...",
            model,
            exp,
            fld,
        )
        ds = self.get_single_var_mod_data(exp, fld, model)
        if ds is None:
            return None

        # Extract the variable and standardize dimension names
        var_monthly = ds[fld]

        # Standardize dimension names for compatibility
        # Some models use 'latitude'/'longitude', others use 'lat'/'lon'
        dim_mapping = {}
        if "latitude" in var_monthly.dims:
            dim_mapping["latitude"] = "lat"
        if "longitude" in var_monthly.dims:
            dim_mapping["longitude"] = "lon"

        if dim_mapping:
            var_monthly = var_monthly.rename(dim_mapping)

        var_monthly = var_monthly.assign_coords(
            {"time": np.arange(len(ds.time.values))}
        ).rename({"time": "month"})
        var_monthly = var_monthly.expand_dims(dim={"ens": np.array([1])})

        # Save to cache if caching is enabled
        if self.enable_cache:
            self.cache_handler.save_cmip6_to_cache(
                var_monthly, "get_single_var_mod_data_monthly", exp, fld, model
            )

        return var_monthly

    def make_meteor_training_data(
        self,
        exp,
        model,
        exp_mapper=None,
        monthly=False,
    ):
        """
        Make xr.dataset with data and format used for meteor

        Parameters
        ----------
        exp : str
            Experiment name used by METEOR, this will be remapped to an
            experiment in the cmip6 set by the exp_mapper dictionary
        model : str
            Name of model for which to find and format training data
        exp_mapper : dict
            Dictionary that maps experiments of the METEOR type (keys) to
            experiments getable by the CMIP6DataGetter (values)
        monthly : bool, optional
            If True, return monthly data instead of yearly averages. Default
            is False.

        Returns
        -------
        xr.Dataset
            Dataset with yearly data on the format usable for METEOR (or
            monthly if monthly=True)
        """
        # Try to load from cache first if caching is enabled
        if self.enable_cache:
            cached_data = self.cache_handler.load_cmip6_cached_data(
                "make_meteor_training_data", exp, model, exp_mapper, monthly=monthly
            )
            if cached_data is not None:
                return cached_data
        # Original logic
        if not exp_mapper:
            exp_mapper = cmip6_to_meteor_exp_remapper
        fld_values = []
        for fld in self.flds:
            if exp in exp_mapper:
                if monthly:
                    fld_values.append(
                        self.get_single_var_mod_data_monthly(
                            exp_mapper[exp], fld, model
                        )
                    )
                else:
                    fld_values.append(
                        self.get_single_var_mod_data_yearmean(
                            exp_mapper[exp], fld, model
                        )
                    )
            elif monthly:
                fld_values.append(self.get_single_var_mod_data_monthly(exp, fld, model))
            else:
                fld_values.append(
                    self.get_single_var_mod_data_yearmean(exp, fld, model)
                )
        training_data = make_xarray_with_correct_dims(self.flds, fld_values)

        # No longer cache training data - generate on-the-fly from
        # variable-specific caches
        # to avoid redundancy and enable flexible variable combinations
        # if self.enable_cache:
        #     self._save_to_cache(cache_key, training_data)

        return training_data

    # pylint: disable=too-many-nested-blocks,too-many-branches,too-many-locals
    def make_meteor_training_data_composite(
        self, exps, model, overlap=None, monthly=False
    ):
        """
        Make xr.dataset with data and format used for meteor

        Parameters
        ----------
        exps : list
            Lists with experiments to be glued together, should be ordered
            the same order as the experiments are meant to be concatenated
        model : str
            Name of model for which to find and format training data
        overlap : dict
            If any of the experiments are not supposed to just be glued
            one after the other, this can be specified using this dictionary
            The keyword should be the latter of the experiments to glue
            together
            If Full-back is chosen as the value for this, a cut will be made
            to the former dataset to make room for the latter dataset. If the
            latter dataset has more data than for 200 years (i.e. ssp running
            beyond 2100) and the dataset to overlap over this does not have
            such a long dataset, the last 200 years of the dataset will be
            cut. For more control you can specify a number of years (int) to
            cut in the previous dataset. Currently cutting from the last
            dataset is not implemented, but may be added later.
        monthly : bool, optional
            If True, return monthly data instead of yearly averages. Default
            is False. When True, overlap cuts are multiplied by 12 to account
            for monthly timesteps.

        Returns
        -------
        xr.Dataset
            Dataset with yearly data on the format usable for METEOR (or
            monthly if monthly=True)
        """
        # Try to load from cache first if caching is enabled
        if self.enable_cache:
            cached_data = self.cache_handler.load_cmip6_cached_data(
                "make_meteor_training_data_composite",
                exps,
                model,
                overlap,
                monthly=monthly,
            )
            if cached_data is not None:
                return cached_data

        # Original logic - TODO: Add support for cutting forward.
        fld_values = []
        time_dim = "month" if monthly else "year"

        for fld in self.flds:
            value = None
            for exp in exps:
                if value is None:
                    if monthly:
                        value = self.get_single_var_mod_data_monthly(exp, fld, model)
                    else:
                        value = self.get_single_var_mod_data_yearmean(exp, fld, model)
                else:
                    if monthly:
                        next_dataset = self.get_single_var_mod_data_monthly(
                            exp, fld, model
                        )
                    else:
                        next_dataset = self.get_single_var_mod_data_yearmean(
                            exp, fld, model
                        )

                    if overlap is not None:
                        if exp in overlap:
                            if overlap[exp] == "Full-back":
                                cut = len(next_dataset[time_dim].values)
                                # Hacky fix for if ssp experiments run
                                if (
                                    exp.startswith("ssp")
                                    and cut
                                    > (
                                        2400 if monthly else 200
                                    )  # 200 years * 12 months
                                    and len(value[time_dim].values)
                                    <= (
                                        4224 if monthly else 352
                                    )  # 352 years * 12 months
                                ):
                                    cut = cut - (
                                        2400 if monthly else 200
                                    )  # 200 years * 12 months
                            else:
                                cut = overlap[exp] * (
                                    12 if monthly else 1
                                )  # Convert years to months if needed
                            value = value.sel(
                                **{
                                    time_dim: slice(
                                        0, len(value[time_dim].values) - cut - 1
                                    )
                                }
                            )
                    start_time = value[time_dim].values[-1] + 1

                    end_time_plus = start_time + next_dataset.sizes[time_dim]
                    next_dataset = next_dataset.assign_coords(
                        {time_dim: np.arange(start_time, end_time_plus)}
                    )
                    value = xr.concat(
                        [value, next_dataset],
                        dim=time_dim,
                    )
            fld_values.append(value)

        result = make_xarray_with_correct_dims(self.flds, fld_values)

        # No longer cache composite training data - generate on-the-fly
        # to avoid redundancy and save storage space
        # if self.enable_cache:
        #     self._save_to_cache(cache_key, result)

        return result

    def prepare_pattern_scaling_training_data(
        self, model_name, scenario_train="ssp245"
    ):
        """
        Prepare complete training data dictionary for pattern scaling.

        Creates a standardized training data dictionary with the required
        experiments for METEOR pattern scaling: base (piControl), co2x4
        (abrupt-4xCO2), scenario (historical+SSP), and sulxanom (aerosol
        anomaly).

        Parameters
        ----------
        model_name : str
            Name of the CMIP6 model to prepare data for
        scenario_train : str, optional
            SSP scenario to use for training. Default is "ssp245". Common
            options: "ssp126", "ssp245", "ssp370", "ssp585"

        Returns
        -------
        dict
            Dictionary with keys: 'base', 'co2x4', scenario_train, 'sulxanom'.
            Each value is an xr.Dataset with training data for that experiment

        Examples
        --------
        >>> data_getter = Cmip6MeteorDataGetter(
        ...     exps=["piControl", "ssp245", "historical", "abrupt-4xCO2"],
        ...     flds=["tas"]
        ... )
        >>> training_data = data_getter.prepare_pattern_scaling_training_data(
        ...     "CESM2"
        ... )
        >>> print(training_data.keys())
        dict_keys(['base', 'co2x4', 'ssp245', 'sulxanom'])
        """
        print(f"🔧 Preparing pattern scaling training data for {model_name}...")

        # Prepare training data dictionary
        training_data = {
            "base": self.make_meteor_training_data("base", model_name),
            "co2x4": self.make_meteor_training_data("co2x4", model_name),
            scenario_train: self.make_meteor_training_data_composite(
                ["historical", scenario_train], model_name
            ),
        }

        # Use the scenario as the aerosol anomaly experiment (sulxanom)
        training_data["sulxanom"] = training_data[scenario_train]

        print(
            "   ✅ Training data prepared for experiments: ", list(training_data.keys())
        )
        return training_data

    def validate_pattern_scaling_cache(
        self, cache_file, model_name, scenario="aer", variable=None
    ):
        """
        Validate a cached pattern scaling model file.

        Checks if the cached pickle file exists, can be loaded, and contains
        the expected model name and variable fields.

        Parameters
        ----------
        cache_file : str
            Path to the cached model file
        model_name : str
            Expected model name
        scenario : str, optional
            Scenario suffix for expected model name. Default is "aer".
        variable : str, optional
            Variable suffix appended to the expected model name (e.g. "tas").
            Must match the name ``MeteorPatternScaling`` uses when saving the
            per-variable model. When omitted, the legacy (no-suffix) form is
            used, but this will mismatch caches saved with per-variable names.

        Returns
        -------
        tuple
            (is_valid, cached_model, info_dict) where:
            - is_valid: bool indicating if cache is valid
            - cached_model: loaded model object if valid, None otherwise
            - info_dict: dict with 'message', 'expected_name', 'found_name',
              'expected_fields', 'found_fields'

        Examples
        --------
        >>> data_getter = Cmip6MeteorDataGetter(
        ...     exps=["piControl"], flds=["tas"]
        ... )
        >>> cache_file = data_getter.get_pattern_scaling_cache_path("CESM2")
        >>> is_valid, model, info = data_getter.validate_pattern_scaling_cache(
        ...     cache_file, "CESM2"
        ... )
        >>> if is_valid:
        ...     print(f"✅ {info['message']}")
        """
        expected_name = f"cmip6-{model_name}-{scenario}"
        if variable is not None:
            expected_name = f"{expected_name}-{variable}"
        expected_vars = set(self.flds)

        info = {
            "expected_name": expected_name,
            "expected_fields": expected_vars,
            "found_name": None,
            "found_fields": set(),
            "message": "",
        }

        # Check if file exists
        if not os.path.exists(cache_file):
            info["message"] = f"Cache file not found: {cache_file}"
            return False, None, info

        # Try to load and validate
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)  # nosec B301

            # The MeteorPatternScaling.save_model() saves a dict,
            # not the object itself
            # Check if we loaded a dict (new format) or object (old format)
            if isinstance(cached_data, dict):
                # New format: dictionary with model data
                info["found_name"] = cached_data.get("name", "unknown")
                if "patternflds" in cached_data:
                    info["found_fields"] = set(cached_data["patternflds"].keys())
            else:
                # Old format: try to get attributes from object
                info["found_name"] = getattr(cached_data, "name", "unknown")
                if hasattr(cached_data, "flds"):
                    info["found_fields"] = set(cached_data.flds.keys())
                elif hasattr(cached_data, "patternflds"):
                    info["found_fields"] = set(cached_data.patternflds.keys())

            # Validate model name
            if info["found_name"] != expected_name:
                info["message"] = (
                    f"Model name mismatch: expected '{expected_name}', "
                    f"found '{info['found_name']}'"
                )
                return False, None, info

            # Validate fields exist
            if not info["found_fields"]:
                info["message"] = "Cached model missing field information"
                return False, None, info

            # Validate all expected fields are present
            if not expected_vars.issubset(info["found_fields"]):
                missing = expected_vars - info["found_fields"]
                info["message"] = (
                    f"Missing required fields: {missing}. "
                    f"Expected {expected_vars}, found {info['found_fields']}"
                )
                return False, None, info

            # Cache is valid
            info["message"] = (
                f"Cache valid: model={info['found_name']}, "
                f"fields={list(info['found_fields'])}"
            )
            return True, cached_data, info

        except Exception as e:  # pylint: disable=broad-exception-caught
            info["message"] = f"Error reading cached file {cache_file}: {e}"
            return False, None, info
