"""
Module for handling caching of data in METEOR.
"""

import hashlib
import logging
import os

import pandas as pd
import xarray as xr


def find_suitable_cache_location():
    """
    Determine a suitable cache directory location based on the
    installation type. Attempts to locate the repository root
    by searching for common markers (setup.py, .git, README.md)
    in the directory hierarchy starting from the current file's location.
    This approach works well for development environments (git clones).

    If a repository root is found, the cache directory is created within the
    repository at `.cache/`. If no repository root is found (e.g.,
    when the package is installed via pip), the cache directory
    falls back to the user's home directory at `~/.meteor/cache/`.

    Returns
    -------
    str
        The absolute path to the suitable cache directory.
        - For development installations: `<repo_root>/.cache`
        - For pip-installed packages: `~/.meteor/cache`
    """
    # Try to locate the repository root by looking for setup.py, .git, etc.
    # This works well for development environments (git clones).
    # If not found (e.g., pip-installed package), fall back to home directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = current_dir
    found_repo_root = False
    while repo_root != os.path.dirname(repo_root):  # Stop at filesystem root
        if any(
            os.path.exists(os.path.join(repo_root, marker))
            for marker in ["setup.py", ".git", "README.md"]
        ):
            found_repo_root = True
            break
        repo_root = os.path.dirname(repo_root)

    if found_repo_root:
        # Development: cache in repository root
        cache_dir = os.path.join(repo_root, ".cache")
    else:
        # Pip-installed: cache in user home directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".meteor", "cache")
    return cache_dir


def _generate_cmip6_cache_key(method_name, *args, **kwargs):
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
    return hashlib.md5(key_str.encode()).hexdigest()  # nosec - Used for cache key generation, not security)


def _find_expected_variable_from_args(method_name, *args):
    """
    Extract expected variable name from method arguments.

    Parameters
    ----------
    method_name : str
        Name of the method being called
    *args : tuple
        Positional arguments to the method

    Returns
    -------
    str or None
        Expected variable name, or None if not applicable
    """
    if method_name in [
        "get_single_var_mod_data",
        "get_single_var_mod_data_yearmean",
        "get_single_var_mod_data_monthly",
    ]:
        if len(args) >= 2:
            return args[1]  # fld parameter is usually second argument
    return None


class CacheHandler:
    """
    Class to handle caching of data.

    """

    def __init__(
        self,
        cache_dir=None,
        purpose="general",
        enable_compression=True,
        compression_level=6,
    ):
        """
        Initialize the CacheHandler.

        Parameters
        ----------
        cache_dir : str, optional
            Directory to store cache files. If None, a suitable location is determined.
        purpose : str, optional
            Purpose of the cache handler. Options are 'classic', 'noise', or 'general'.
            Determines which sub-caches to set up. Default is 'general'.
        """
        if cache_dir is None:
            cache_dir = find_suitable_cache_location()  # pragma: no cover
        self.cache_dir = cache_dir

        self.sub_caches = ["cmip6"]
        if purpose == "classic":
            self.sub_caches.append("pattern_scaling")
        elif purpose == "noise":
            self.sub_caches.append("noise_models")
        elif purpose == "general":
            self.sub_caches.extend(["pattern_scaling", "noise_models"])
        try:
            self.setup_cache_tree()
            self.cache_functioning = True
        except OSError as e:
            logging.warning(
                "Failed to create cache directory at %s: %s. "
                "Disabling cache functionality.",
                self.cache_dir,
                e,
            )
            self.cache_functioning = False
        self.enable_compression = enable_compression
        self.compression_level = max(1, min(9, compression_level))

    def setup_cache_tree(self):
        """
        Set up the directory structure for caching.

        Raises
        ------
        OSError
            If the cache directory cannot be created.
        """
        # Logic to set up cache tree structure
        try:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
        except Exception as e:
            raise OSError(e) from e
        for sub_cache in self.sub_caches:
            sub_cache_path = os.path.join(self.cache_dir, sub_cache)
            if not os.path.exists(sub_cache_path):
                os.makedirs(sub_cache_path)
        logging.info("Cache tree structure set up.")

    def get_cmip6_query_catalogue(self):
        """
        Get the path to the CMIP6 query catalogue.

        Returns
        -------
        str
            Path to the CMIP6 query catalogue CSV file.
        """
        return os.path.join(
            self.cache_dir, "cmip6", "cmip6-zarr-consolidated-stores.csv"
        )

    def check_if_cmip6_cached(self, method_name, *args, **kwargs):
        """
        Check if data for a CMIP6 method call is cached and valid.

        If cached data is found but fails validation, it is removed.

        Parameters
        ----------
        method_name : str
            Name of the method being called
        *args : tuple
            Positional arguments to the method
        **kwargs : dict
            Keyword arguments to the method

        Returns
        -------
        bool
            True if data is cached and valid, False otherwise
        """
        if not self.cache_functioning:
            return False
        cache_key = _generate_cmip6_cache_key(method_name, *args, **kwargs)
        cache_file = os.path.join(self.cache_dir, "cmip6", f"{cache_key}.nc")
        # Check if file exists
        if not os.path.exists(cache_file):
            return False

        # Validate the cached data
        try:
            dataset = xr.open_dataset(cache_file)

            # For methods that expect specific variables, extract variable name from args
            expected_variable = _find_expected_variable_from_args(method_name, *args)
            # For training data, we expect variables from self.flds
            # Since we can't easily determine which specific variable to check,
            # we'll validate that the dataset has at least one data variable
            # and that each variable in self.flds exists if we're checking a specific scenario
            # Basic validation below should catch empty datasets or weird dimensions

            is_valid = self._validate_cached_data(dataset, expected_variable)
            dataset.close()

            if not is_valid:
                # Remove invalid cache file
                try:
                    os.remove(cache_file)
                    logging.debug("Removed invalid cache file: %s", cache_file)
                except OSError:  # pragma: no cover
                    pass
                return False

            return True

        except (  # pragma: no cover
            Exception  # pylint: disable=broad-exception-caught
        ) as e:
            # Cache file is corrupted, remove it
            try:
                os.remove(cache_file)
                logging.debug("Removed corrupted cache file: %s (%s)", cache_file, e)
            except OSError:
                pass
            return False

    def load_cmip6_cached_data(self, method_name, *args, **kwargs):
        """
        Load cached data for a CMIP6 method call.

        Parameters
        ----------
        method_name : str
            Name of the method being called
        *args : tuple
            Positional arguments to the method
        **kwargs : dict
            Keyword arguments to the method

        Returns
        -------
        xr.Dataset or xr.DataArray or None
            Cached data if exists and is valid, None otherwise
        """
        if not self.cache_functioning:
            raise RuntimeError("Cache is not functioning.")
        cache_key = _generate_cmip6_cache_key(method_name, *args, **kwargs)
        expected_variable = _find_expected_variable_from_args(method_name, *args)
        if "expected_type" in kwargs:
            expected_type = kwargs.pop("expected_type")
        else:
            expected_type = None
        dataset = self._load_from_cmip6_cache(
            cache_key,
            expected_type=expected_type,
            expected_variable=expected_variable,
        )
        return dataset

    # TODO: Deal with subcaches properly here
    def _get_cache_path(self, cache_key, subcache="cmip6"):
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
        return os.path.join(self.cache_dir, subcache, f"{cache_key}.nc")

    def _load_from_cmip6_cache(
        self, cache_key, expected_type=None, expected_variable=None
    ):
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
        if not self.cache_functioning:
            return None

        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                logging.info(
                    "Loading data from cache: %s", os.path.basename(cache_path)
                )
                dataset = xr.open_dataset(cache_path)

                # Validate the cached data
                if not self._validate_cached_data(dataset, expected_variable):
                    logging.warning(
                        "Cached data at %s failed validation. Removing and re-downloading.",
                        cache_path,
                    )
                    try:
                        os.remove(cache_path)
                    except OSError:  # pragma: no cover
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
                except OSError:  # pragma: no cover
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
            if (
                "time" in dataset.sizes and "time" not in dataset.coords
            ):  # pragma: no cover
                logging.debug(
                    "Validation failed: 'time' dimension exists but no time coordinate"
                )
                return False

            return True

        except (  # pragma: no cover
            Exception  # pylint: disable=broad-exception-caught
        ) as e:
            logging.debug("Validation failed with exception: %s", e)
            return False

    def save_cmip6_to_cache(self, data, method_name, *args, **kwargs):
        """
        Save data to cache for a CMIP6 method call.

        Parameters
        ----------
        method_name : str
            Name of the method being called
        *args : tuple
            Positional arguments to the method
        **kwargs : dict
            Keyword arguments to the method
        """
        if not self.cache_functioning:
            return
        cache_key = _generate_cmip6_cache_key(method_name, *args, **kwargs)
        if data is None:
            logging.warning("No data provided to save to cache for key: %s", cache_key)
            return
        self._save_cmip6_to_cache(cache_key, data)

    def _save_cmip6_to_cache(self, cache_key, data):
        """
        Save data to cache.

        Parameters
        ----------
        cache_key : str
            Cache key
        data : xr.Dataset or xr.DataArray
            Data to cache
        """
        if not self.cache_functioning:  # pragma: no cover
            return
        cache_path = self._get_cache_path(cache_key, subcache="cmip6")
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
        except (OSError, ValueError) as e:  # pragma: no cover
            # Failed to cache, but don't raise error
            logging.warning(
                "Failed to save data to cache at %s: %s. "
                "Continuing without caching.",
                cache_path,
                e,
            )

    def _clear_single_cache(self, sub_cache):
        """
        Clear all cached data for a specific sub-cache.

        Parameters
        ----------
        sub_cache : str
            Name of the sub-cache to clear
        """
        sub_cache_path = os.path.join(self.cache_dir, sub_cache)
        if os.path.exists(sub_cache_path):
            for filename in os.listdir(sub_cache_path):
                # Previously this tested specifically for netcdf or being the cmip6 catalogue
                # Now we just do a clearing of all files if we can
                if os.path.isfile(os.path.join(sub_cache_path, filename)):
                    try:
                        os.remove(os.path.join(sub_cache_path, filename))
                    except OSError:  # pragma: no cover
                        pass

    def clear_cache(self, sub_cache="all"):
        """
        Clear all cached data for this data getter.

        Parameters
        ----------
        sub_cache : str, optional
            Name of the sub-cache to clear. If "all", clears all sub-caches. Default is "all".
        """
        if not self.cache_functioning:  # pragma: no cover
            return
        if sub_cache == "all":
            for sub_cache_name in self.sub_caches:
                self._clear_single_cache(sub_cache_name)
        else:
            if sub_cache in self.sub_caches:
                self._clear_single_cache(sub_cache)
            else:
                logging.warning(
                    "Sub-cache %s not recognized. Available sub-caches: %s",
                    sub_cache,
                    self.sub_caches,
                )

    def get_pattern_scaling_cache_path(self, model_name, scenario="aer", variable=None):
        """
        Get the standardized cache file path for a pattern scaling model.

        Parameters
        ----------
        model_name : str
            Name of the CMIP6 model
        cache_dir : str, optional
            Directory for cache files. If None, uses default cache location.
        scenario : str, optional
            Scenario suffix for the model name. Default is "aer" (aerosol-inclusive).
        variable : str, optional
            Variable name (e.g., 'tas', 'pr'). If provided, included in filename.

        Returns
        -------
        str
            Full path to the cache file

        Examples
        --------
        >>> data_getter = Cmip6MeteorDataGetter(exps=["piControl"], flds=["tas"])
        >>> cache_path = data_getter.get_pattern_scaling_cache_path("CESM2")
        >>> print(cache_path)
        /path/to/.cache/trained_pattern_scaling_models/cmip6-CESM2-aer_pattern_scaling.pkl
        """
        if not self.cache_functioning:
            return None
        cache_dir = os.path.join(self.cache_dir, "pattern_scaling")

        os.makedirs(cache_dir, exist_ok=True)
        if variable:
            return os.path.join(
                cache_dir,
                f"cmip6-{model_name}-{scenario}-{variable}_pattern_scaling.pkl",
            )
        return os.path.join(
            cache_dir, f"cmip6-{model_name}-{scenario}_pattern_scaling.pkl"
        )

    def get_subdir(self, name):
        """Return the path to a named subdirectory within the cache root.

        The directory is created on demand if it does not already exist.
        Useful for non-CMIP6 data (e.g. GGCM coefficient files) that should
        live alongside the standard cache structure.

        Parameters
        ----------
        name : str
            Subdirectory name (e.g. ``'ggcm'``).

        Returns
        -------
        str
            Absolute path to the subdirectory.
        """
        subdir = os.path.join(self.cache_dir, name)
        os.makedirs(subdir, exist_ok=True)
        return subdir

    def get_noise_model_cache_path(self, model_name, variable_name):
        """
        Get the standardized cache file path for a noise model.

        Parameters
        ----------
        model_name : str
            Name of the CMIP6 model
        variable_name : str
            Variable name (e.g., 'tas', 'pr')

        Returns
        -------
        str
            Full path to the cache file

        Examples
        --------
        >>> data_getter = Cmip6MeteorDataGetter(exps=["piControl"], flds=["tas"])
        >>> cache_path = data_getter.get_noise_model_cache_path("CESM2", "tas")
        >>> print(cache_path)
        /path/to/noise_cache/CESM2_tas_noise_model.pkl
        """
        if not self.cache_functioning:
            return None
        cache_dir = os.path.join(self.cache_dir, "noise_models")

        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{model_name}_{variable_name}_noise_model.pkl")
