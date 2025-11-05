"""
Module to get CMIP6 data and convert to format that can be used for METEOR
"""

import hashlib
import logging
import os

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

cmip6_to_meteor_exp_remapper = {
    "base": "piControl",
    "co2x4": "abrupt-4xCO2",
    "co2x8": "abrupt-4xCO2",
    "co2x16": "abrupt-4xCO2",
    "1pc": "1pctCO2",
}


def multiply_along_axis(array_a, array_b, axis):
    """
    Multiply to arrays along a given axis

    Pure infrastructure function to make multiplying along an axis that is not the last
    axis along two np.ndarray objects without encountering broadcasting issues

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
        Array/matrix with result of multiplication, and the multiplication axis
        back where it was
    """
    return np.swapaxes(np.swapaxes(array_a, axis, -1) * array_b, -1, axis)


def year_mean_monthly(monthly_data):
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
        np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) / 365.0 * 12.0,
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


def year_mean_monthly_xarray(monthly_xarray):
    """
    Calculate yearmean from monthly xarray dataarray

    Essentially just a wrapper for xarray around the numpy based
    year_mean_monthly function above

    Parameters
    ----------
    monthly_xarray : xr.dataArray
        with the first dimension being time followed by lat and lon
        and on monthly resolutions running from January to December for each year

    Returns
    -------
    xr.dataArray
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


def make_xarray_with_correct_dims(fld_names, fld_values):
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
        data_vars={fld_names[i]: fld_values[i] for i in range(len(fld_names))},
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
        two-dimensional list of pd.DataFrame where the first dimension runs over the
        experiments and the second over the fields in fld
    flds: list
        of names of fields
    exps: list
        of names of experiments
    mdl_skipmbrs : dict
        of models and a list of ensemble members to skip for that model. If none
        is sent, it defaults to skipping NorESM2-Lm which has insufficient data
        for pr for the historical experiment even though there is a file
    Returns
    -------
    list
        Consisting of first a list with dataframes with models and first ensemble members
        with full data per experiment and field, and second a list of the models
        that have the full data
    """
    mdls1 = df_all1[0][0].source_id.unique()
    mdls1.sort()
    df_all = []
    cnames = df_all1[0][0].columns
    if mdl_skipmbrs is None:
        mdl_skipmbrs = {"NorESM2-LM": ["r1i1p1f1"]}
    for i in range(len(exps)):
        # tmp = []
        # for fld in flds:
        #    tmp.append(pd.DataFrame(columns=cnames))
        tmp = [pd.DataFrame(columns=cnames) for j in range(len(flds))]
        df_all.append(tmp)

    mdls = []

    n = 0
    for mdl in mdls1:  # pylint: disable=too-many-nested-blocks

        # Test that one ensemble member has all data:
        sufficient_data = True
        for i in range(len(exps)):
            # find first variable for expt/model
            for j in range(len(flds)):
                if "historical" in exps:
                    ii = exps.index("historical")
                    hist_tmp = df_all1[ii][j].query(
                        "source_id=='" + mdl + "' & experiment_id == 'historical'"
                    )
                    hmb = hist_tmp.member_id.unique()
                else:
                    hmb = []
                tmp = df_all1[i][j].query("source_id=='" + mdl + "'")
                mmbs = tmp.member_id.unique()
                if len(mmbs) > 0:
                    mmb = mmbs[0]
                    if len(hmb) > 0:
                        if hmb[0] in mmbs:
                            mmb = hmb[0]

                    tt = df_all1[i][j].query(f"source_id=='{mdl}' & member_id=='{mmb}'")
                    df_all[i][j].loc[n] = tt.values[0]
                else:
                    mmb = -1
                    df_all[i][j].loc[n] = None
                    sufficient_data = False
            # add model to final list

        if sufficient_data:
            mdls.append(mdl)
            n = n + 1
            # print(f"Model {mdl} has full data")

    return df_all, mdls


class Cmip6MeteorDataGetter:  # pylint: disable=too-many-instance-attributes
    """
    Cmip6MeteorDataGetter class

    Objects of this class holds a shortlist of CMIP6 data links to
    data for one ensemble member for all models that have complete data for
    the full combination of the classes fields and experiments all monthly
    data.

    In various methods it can return yearly mean data for a single
    dataset, or a full METEOR training dataset

    Attributes
    ----------
    flds: list
        List of variable fields for the instance to consider
    exps: list
        List of experiments for the instance to consider


    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        flds=None,
        exps=None,
        dbe=None,
        cache_dir=None,
        enable_cache=False,
        enable_compression=True,
        compression_level=6,
    ):
        """
        Initialise data getter object

        Parameters
        ----------
        flds : list
            Of variable fields to include (monthly data)
            If nothing is sent for this it will be set to
            ['tas', 'pr']
        exps : list
            Of experiments to include. If nothing is sent for
            this it will be set to ['piControl', 'abrupt4xCO2']
        dbe : list
            Of CMIP6 project names for each of the experiments. This should be the same length
            as exps, and have values corresponding to ecah experiment in the same order.
            If nothing is set, it will be set to a list of all entries equal to
            'CMIP' with the same lenght as exps
        cache_dir : str, optional
            Directory to store cached data. If None, defaults to ~/.meteor/cmip6_cache
        enable_cache : bool, optional
            Whether to enable automatic caching. Default is False.
            Set to True to cache downloaded data locally for faster subsequent access.
        enable_compression : bool, optional
            Whether to enable netCDF4/zlib compression for cached files. Default is True.
            This can significantly reduce file sizes (typically 70-90% compression).
        compression_level : int, optional
            Compression level for netCDF4/zlib compression (1-9). Default is 6.
            Higher values provide better compression but slower performance:
            - 1: Fastest compression, larger files
            - 6: Good balance of speed vs compression (recommended)
            - 9: Best compression, slowest performance
        df_all : list
            Two dimensional list along experiments and fields, every entry
            is a pandas.DataSet with lines with data placement for one
            ensemble memeber for each model that has data for the full
            combination of fields and experiments
        models : list
            List of models with full data available
        gcs : gcfs.GCSFileSystem
              A GCSFileSystem to load data
        """
        dbe = self._set_fld_exps_dbe(flds, exps, dbe)

        # Set up caching
        self.enable_cache = enable_cache
        self.enable_compression = enable_compression
        self.compression_level = max(
            1, min(9, compression_level)
        )  # Clamp to valid range 1-9
        if cache_dir is None:
            # Default to .cache folder in the repository root
            # Find the repository root by looking for setup.py or other marker files
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = current_dir
            while repo_root != os.path.dirname(repo_root):  # Stop at filesystem root
                if any(
                    os.path.exists(os.path.join(repo_root, marker))
                    for marker in ["setup.py", ".git", "README.md"]
                ):
                    break
                repo_root = os.path.dirname(repo_root)
            cache_dir = os.path.join(repo_root, ".cache", "cmip6")
        self.cache_dir = cache_dir

        if self.enable_cache:
            try:

                os.makedirs(self.cache_dir, exist_ok=True)
                logging.info(
                    "Setting up local cache for CMIP6 data at: %s. "
                    "This will improve performance by storing downloaded data locally.",
                    self.cache_dir,
                )
            except OSError as e:
                logging.warning(
                    "Failed to create cache directory at %s: %s. "
                    "Disabling cache functionality.",
                    self.cache_dir,
                    e,
                )
                self.enable_cache = False

        df = pd.read_csv(
            "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
            low_memory=False,
        )
        df_all1 = []
        for i, exp in enumerate(self.exps):
            df_ta1 = []
            for fld in self.flds:
                tmp = df.query(
                    f"activity_id=='{dbe[i]}' & table_id == 'Amon' & variable_id == '{fld}' & experiment_id == '{exp}'"
                )
                df_ta1.append(tmp)
            df_all1.append(df_ta1)
        self.df_all, self.models = initialise_dataframe_and_models(
            df_all1, flds=self.flds, exps=self.exps
        )
        self.gcs = gcsfs.GCSFileSystem(token="anon")  # nosec

    def _generate_cache_key(self, method_name, *args, **kwargs):
        """
        Generate a descriptive cache filename for a method call.

        Parameters
        ----------
        method_name : str
            Name of the method being cached
        *args : tuple
            Positional arguments to the method
        **kwargs : dict
            Keyword arguments to the method

        Returns
        -------
        str
            Descriptive cache filename (without extension)
        """
        if method_name == "get_single_var_mod_data":
            exp, fld, model = args
            return f"{model}_{exp}_{fld}_raw"
        if method_name == "get_single_var_mod_data_yearmean":
            exp, fld, model = args
            return f"{model}_{exp}_{fld}_yearly"
        if method_name == "get_single_var_mod_data_monthly":
            exp, fld, model = args
            return f"{model}_{exp}_{fld}_monthly"
        if method_name == "make_meteor_training_data":
            exp, model = args[:2]  # pylint: disable=unbalanced-tuple-unpacking
            monthly = kwargs.get("monthly", False)
            suffix = "_monthly" if monthly else "_yearly"
            return f"{model}_{exp}_training{suffix}"
        if method_name == "make_meteor_training_data_composite":
            exps, model = args[:2]  # pylint: disable=unbalanced-tuple-unpacking
            monthly = kwargs.get("monthly", False)
            exp_str = "_".join(exps)
            suffix = "_monthly" if monthly else "_yearly"
            return f"{model}_{exp_str}_composite{suffix}"
        # Fallback to hash-based naming
        key_data = {
            "method": method_name,
            "args": args,
            "kwargs": kwargs,
        }
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(
            key_str.encode()
        ).hexdigest()  # nosec - Used for cache key generation, not security

    def is_cached(self, method_name, *args, **kwargs):
        """
        Check if data is already cached and valid for a given method call.

        Parameters
        ----------
        method_name : str
            Name of the method being checked
        *args : tuple
            Positional arguments to the method
        **kwargs : dict
            Keyword arguments to the method

        Returns
        -------
        bool
            True if valid cached data exists, False otherwise
        """
        if not self.enable_cache:
            return False

        cache_key = self._generate_cache_key(method_name, *args, **kwargs)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.nc")

        # Check if file exists
        if not os.path.exists(cache_file):
            return False

        # Validate the cached data
        try:
            dataset = xr.open_dataset(cache_file)

            # For methods that expect specific variables, extract variable name from args
            expected_variable = None
            if method_name in [
                "get_single_var_mod_data",
                "get_single_var_mod_data_yearmean",
                "get_single_var_mod_data_monthly",
            ]:
                if len(args) >= 2:
                    expected_variable = args[
                        1
                    ]  # fld parameter is usually second argument
            elif method_name == "make_meteor_training_data":
                # For training data, we expect variables from self.flds
                # Since we can't easily determine which specific variable to check,
                # we'll validate that the dataset has at least one data variable
                # and that each variable in self.flds exists if we're checking a specific scenario
                pass  # Basic validation below should catch empty datasets

            is_valid = self._validate_cached_data(dataset, expected_variable)
            dataset.close()

            if not is_valid:
                # Remove invalid cache file
                try:
                    os.remove(cache_file)
                    logging.debug("Removed invalid cache file: %s", cache_file)
                except OSError:
                    pass
                return False

            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Cache file is corrupted, remove it
            try:
                os.remove(cache_file)
                logging.debug("Removed corrupted cache file: %s (%s)", cache_file, e)
            except OSError:
                pass
            return False

    def _get_cache_path(self, cache_key):
        """
        Get the full path for a cache file.

        Parameters
        ----------
        cache_key : str
            Cache key

        Returns
        -------
        str
            Full path to cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.nc")

    def _load_from_cache(self, cache_key, expected_type=None, expected_variable=None):
        """
        Load data from cache if it exists and is valid.

        Parameters
        ----------
        cache_key : str
            Cache key
        expected_type : str, optional
            Expected return type ('Dataset' or 'DataArray'). If None, uses metadata from cache.
        expected_variable : str, optional
            Expected variable name to validate presence in cached data.

        Returns
        -------
        object or None
            Cached data if exists and is valid, None otherwise
        """
        # pylint: disable=too-many-return-statements
        if not self.enable_cache:
            return None

        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                dataset = xr.open_dataset(cache_path)

                # Validate the cached data
                if not self._validate_cached_data(dataset, expected_variable):
                    logging.warning(
                        "Cached data at %s failed validation. Removing and re-downloading.",
                        cache_path,
                    )
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
                    return None

                # Determine return type based on expected_type or metadata
                if expected_type == "DataArray":
                    # Force return as DataArray
                    if len(dataset.data_vars) == 1:
                        var_name = list(dataset.data_vars)[0]
                        return dataset[var_name]
                    # Multiple variables, can't convert to DataArray safely
                    return None
                if expected_type == "Dataset":
                    # Force return as Dataset
                    return dataset
                # Use metadata to determine type (backward compatibility)
                original_type = dataset.attrs.get("original_type")
                if original_type == "DataArray" and len(dataset.data_vars) == 1:
                    var_name = list(dataset.data_vars)[0]
                    return dataset[var_name]
                # Default to Dataset (safer for zarr data)
                return dataset
            except (OSError, ValueError, KeyError):
                # Cache file corrupted, remove it
                logging.warning(
                    "Cached data at %s is corrupted. Removing and re-downloading.",
                    cache_path,
                )
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        return None

    def _validate_cached_data(  # pylint: disable=too-many-return-statements
        self, dataset, expected_variable=None
    ):
        """
        Validate that cached data is not corrupted and contains expected content.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to validate
        expected_variable : str, optional
            Expected variable name

        Returns
        -------
        bool
            True if data is valid, False otherwise
        """
        try:
            # Check 1: Dataset should have data variables
            if len(dataset.data_vars) == 0:
                logging.debug("Validation failed: No data variables in cached dataset")
                return False

            # Check 2: If we expect a specific variable, it should be present
            if expected_variable and expected_variable not in dataset.data_vars:
                logging.debug(
                    "Validation failed: Expected variable '%s' not found in cached dataset",
                    expected_variable,
                )
                return False

            # Check 3: Data variables should have reasonable dimensions
            for var_name, var_data in dataset.data_vars.items():
                if len(var_data.dims) == 0:
                    logging.debug(
                        "Validation failed: Variable '%s' has no dimensions", var_name
                    )
                    return False

                # Check that dimensions have reasonable sizes (not empty)
                for dim in var_data.dims:
                    if dim in dataset.sizes and dataset.sizes[dim] == 0:
                        logging.debug(
                            "Validation failed: Dimension '%s' has size 0", dim
                        )
                        return False

            # Check 4: Essential coordinate variables should exist
            # Most climate data should have time coordinate
            if "time" in dataset.sizes and "time" not in dataset.coords:
                logging.debug(
                    "Validation failed: 'time' dimension exists but no time coordinate"
                )
                return False

            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.debug("Validation failed with exception: %s", e)
            return False

    def _save_to_cache(self, cache_key, data):
        """
        Save data to cache.

        Parameters
        ----------
        cache_key : str
            Cache key
        data : xr.Dataset or xr.DataArray
            Data to cache
        """
        if not self.enable_cache:
            return

        cache_path = self._get_cache_path(cache_key)
        try:
            # Convert DataArray to Dataset if needed and mark the original type
            if isinstance(data, xr.DataArray):
                # Use the variable name if available, otherwise use 'data'
                var_name = data.name if data.name else "data"
                data_to_save = data.to_dataset(name=var_name)
                # Mark that this was originally a DataArray
                data_to_save.attrs["original_type"] = "DataArray"
            else:
                data_to_save = data
                # Mark that this was originally a Dataset
                data_to_save.attrs["original_type"] = "Dataset"

            # Add metadata about when this was cached
            data_to_save.attrs["cached_by_meteor"] = (
                "true"  # Use string instead of boolean
            )
            data_to_save.attrs["cache_timestamp"] = pd.Timestamp.now().isoformat()

            # Clean up problematic variables before saving
            # Drop time_bnds if it exists as it often has conflicting fill values
            # and is not needed for METEOR analysis
            variables_to_drop = []
            if "time_bnds" in data_to_save.variables:
                variables_to_drop.append("time_bnds")

            if variables_to_drop:
                data_to_save = data_to_save.drop_vars(variables_to_drop)
                logging.debug(
                    "Dropped variables %s before caching to avoid encoding conflicts",
                    variables_to_drop,
                )

            # Set up compression options
            if self.enable_compression:
                # Use zlib compression with user-specified level and shuffling for better compression ratios
                encoding = {}
                for var_name in data_to_save.data_vars:
                    encoding[var_name] = {
                        "zlib": True,
                        "complevel": self.compression_level,
                        "shuffle": True,
                        "fletcher32": False,  # Skip checksum for speed
                    }
                # Also compress coordinate variables if they exist
                for coord_name in data_to_save.coords:
                    if coord_name not in encoding:
                        encoding[coord_name] = {
                            "zlib": True,
                            "complevel": self.compression_level,
                            "shuffle": True,
                            "fletcher32": False,
                        }

                data_to_save.to_netcdf(cache_path, encoding=encoding)
                logging.debug(
                    "Saved compressed cache file (level %s) to %s",
                    self.compression_level,
                    cache_path,
                )
            else:
                data_to_save.to_netcdf(cache_path)
                logging.debug("Saved uncompressed cache file to %s", cache_path)
        except (OSError, ValueError) as e:
            # Failed to cache, but don't raise error
            logging.warning(
                "Failed to save data to cache at %s: %s. "
                "Continuing without caching.",
                cache_path,
                e,
            )

    def clear_cache(self):
        """
        Clear all cached data for this data getter.
        """
        if not self.enable_cache or not os.path.exists(self.cache_dir):
            return

        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".nc"):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except OSError:
                    pass

    def _set_fld_exps_dbe(self, flds, exps, dbe):
        """
        Private method to set flds and exps and dbe and
        take care of if they are not set then have defaults

        Parameters
        ----------
        flds : list
            Of variable fields to include (monthly data)
            If None is sent for this it will be set to
            ['tas', 'pr']
        exps : list
            Of experiments to include. If None is sent for
            this it will be set to ['piControl', 'abrupt4xCO2']
        dbe : list
            Of CMIP6 project names for each of the experiments. This should be the same length
            as exps, and have values corresponding to ecah experiment in the same order.
            If None is sent, it will be set based on the experiment type:
            - CMIP experiments: 'CMIP'
            - SSP scenarios: 'ScenarioMIP'
            - 1pctCO2: 'CMIP'

        Returns
        -------
        list
            dbe, If None was sent, a list with correct activity_ids for each experiment
        """
        if not flds:
            flds = ["tas", "pr"]
        if not exps:
            exps = ["piControl", "abrupt-4xCO2"]
        if not dbe:
            # Map experiments to correct activity_ids
            dbe = []
            for exp in exps:
                if exp.startswith("ssp"):
                    # SSP scenarios are in ScenarioMIP
                    dbe.append("ScenarioMIP")
                elif exp in ["1pctCO2"]:
                    # 1pctCO2 is in CMIP
                    dbe.append("CMIP")
                else:
                    # Default experiments (piControl, abrupt-4xCO2, historical) are in CMIP
                    dbe.append("CMIP")
        self.flds = flds
        self.exps = exps
        return dbe

    def get_models_avail(self):
        """
        Get list of available models

        Returns
        -------
        list
            List of available models. They have full data for the dataGetters flds and
            experiment combinations for at least one ensemble member.
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
            If either exp, or fld is not in the instances self.exps and self.flds or if the model
            is not among the models that has complete data for all the combinations of experiments
            and fields that the instance holds.
        """
        # Try to load from cache first if caching is enabled
        if self.enable_cache:
            cache_key = self._generate_cache_key(
                "get_single_var_mod_data", exp, fld, model
            )
            cached_data = self._load_from_cache(cache_key, expected_variable=fld)
            if cached_data is not None:
                return cached_data

        # Original logic for data fetching
        if exp not in self.exps:
            raise KeyError(
                f"This datagetter does not handle data from the {exp} experiment, available options are {self.exps} "
            )
        if fld not in self.flds:
            raise KeyError(
                f"This datagetter does not handle {fld} data, available options are {self.flds}"
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

        # Apply 50-year limit for piControl experiments to save storage space
        # piControl data is only used for baseline calculation (mean value)
        # so limiting to first 50 years has no impact on scientific results
        if exp == "piControl" and len(fld_data.time) > 600:  # 50 years * 12 months
            print(
                f"   Limiting piControl data to first 50 years (was {len(fld_data.time) // 12} years)"
            )
            fld_data = fld_data.isel(time=slice(0, 600))  # First 50 years (600 months)

        # No longer cache raw data - we only cache processed monthly data
        # to avoid redundancy and save storage space
        # if self.enable_cache:
        #     self._save_to_cache(cache_key, fld_data)

        return fld_data

    def get_single_var_mod_data_yearmean(self, exp, fld, model):
        """
        Get yearly mean data for a single variable, model and experiment combination

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
            Data converted from monthly to yearly mean data and including and extra flat ens dimension
        """
        # Try to load from cache first if caching is enabled
        if self.enable_cache:
            cache_key = self._generate_cache_key(
                "get_single_var_mod_data_yearmean", exp, fld, model
            )
            cached_data = self._load_from_cache(
                cache_key, expected_type="DataArray", expected_variable=fld
            )
            if cached_data is not None:
                return cached_data

        # Get monthly data from cache and compute yearly mean
        monthly_data = self.get_single_var_mod_data_monthly(exp, fld, model)
        if monthly_data is None:
            return None

        # Convert monthly data back to time-based indexing for yearly averaging
        # The monthly data has dimensions like ('ens', 'month', 'lat', 'lon')
        # We need to reshape it to ('time', 'lat', 'lon') for yearly averaging

        # Remove the ens dimension and rename month back to time
        var_data = monthly_data.squeeze("ens").rename({"month": "time"})

        var_yearly = year_mean_monthly_xarray(var_data)
        var_yearly = var_yearly.assign_coords(
            {"time": np.arange(len(var_data.time.values) // 12)}
        ).rename({"time": "year"})
        var_yearly = var_yearly.expand_dims(
            dim={"ens": np.array([1])}
        )  # .assign_coords({'ens':1})

        # No longer cache yearly data - compute on-the-fly from monthly cache to save storage
        # if self.enable_cache:
        #     self._save_to_cache(cache_key, var_yearly)

        return var_yearly

    def get_single_var_mod_data_monthly(self, exp, fld, model):
        """
        Get monthly data for a single variable, model and experiment combination

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
            Monthly data with time dimension preserved and including an extra flat ens dimension
        """
        # Try to load from cache first if caching is enabled
        if self.enable_cache:
            cache_key = self._generate_cache_key(
                "get_single_var_mod_data_monthly", exp, fld, model
            )
            cached_data = self._load_from_cache(
                cache_key, expected_type="DataArray", expected_variable=fld
            )
            if cached_data is not None:
                return cached_data

        # Original logic
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
            self._save_to_cache(cache_key, var_monthly)

        return var_monthly

    def make_meteor_training_data(self, exp, model, exp_mapper=None, monthly=False):
        """
        Make xr.dataset with data and format used for meteor

        Parameters
        ----------
        exp : str
            Experiment name used by METEOR, this will be remapped to an experiment
            in the cmip6 set by the exp_mapper dictionary
        model : str
            Name of model for which to find and format training data
        exp_mapper : dict
            Dictionary that maps experiments of the METEOR type (keys) to experiments
            getable by the CMIP6DataGetter (values)
        monthly : bool, optional
            If True, return monthly data instead of yearly averages. Default is False.

        Returns
        -------
        xr.Dataset
            Dataset with yearly data on the format usable for METEOR (or monthly if monthly=True)
        """
        # Try to load from cache first if caching is enabled
        if self.enable_cache:
            cache_key = self._generate_cache_key(
                "make_meteor_training_data", exp, model, exp_mapper, monthly=monthly
            )
            cached_data = self._load_from_cache(cache_key)
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

        # No longer cache training data - generate on-the-fly from variable-specific caches
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
            The keyword should be the latter of the experiments to glue together
            If Full-back is chosen as the value for this, a cut will be made to
            the former dataset to make room for the latter dataset. If the latter dataset
            has more data than for 200 years (i.e. ssp running beyond 2100) and the
            dataset to overlap over this does not have such a long dataset, the last
            200 years of the dataset will be cut. For more control you can
            you can specify a number of years (int) to cut in the previous dataset.
            Currently cutting from the last dataset is not implemented, but
            may be added later.
        monthly : bool, optional
            If True, return monthly data instead of yearly averages. Default is False.
            When True, overlap cuts are multiplied by 12 to account for monthly timesteps.

        Returns
        -------
        xr.Dataset
            Dataset with yearly data on the format usable for METEOR (or monthly if monthly=True)
        """
        # Try to load from cache first if caching is enabled
        if self.enable_cache:
            cache_key = self._generate_cache_key(
                "make_meteor_training_data_composite",
                exps,
                model,
                overlap,
                monthly=monthly,
            )
            cached_data = self._load_from_cache(cache_key)
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
