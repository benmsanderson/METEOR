"""
Cache management utilities for METEOR models.

Provides functions for validating and managing cached pattern scaling
and noise models, separate from data acquisition concerns.
"""

import os
import pickle  # nosec - Used for trusted model serialization only


def get_pattern_scaling_cache_path(
    model_name, cache_dir=None, scenario="aer", variable=None
):
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
    >>> cache_path = get_pattern_scaling_cache_path("CESM2")
    >>> print(cache_path)
    /path/to/.cache/trained_pattern_scaling_models/cmip6-CESM2-aer_pattern_scaling.pkl
    """
    if cache_dir is None:
        # Use default cache location in repository root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = current_dir
        while repo_root != os.path.dirname(repo_root):
            if any(
                os.path.exists(os.path.join(repo_root, marker))
                for marker in ["setup.py", ".git", "README.md"]
            ):
                break
            repo_root = os.path.dirname(repo_root)
        cache_dir = os.path.join(repo_root, ".cache", "trained_pattern_scaling_models")

    os.makedirs(cache_dir, exist_ok=True)
    if variable:
        return os.path.join(
            cache_dir,
            f"cmip6-{model_name}-{scenario}-{variable}_pattern_scaling.pkl",
        )
    return os.path.join(cache_dir, f"cmip6-{model_name}-{scenario}_pattern_scaling.pkl")


def validate_pattern_scaling_cache(
    cache_file, model_name, expected_fields, scenario="aer"
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
    expected_fields : set or list
        Expected variable fields that should be in the model (e.g., ['tas', 'pr'])
    scenario : str, optional
        Scenario suffix for expected model name. Default is "aer".

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
    >>> cache_file = get_pattern_scaling_cache_path("CESM2")
    >>> is_valid, model, info = validate_pattern_scaling_cache(
    ...     cache_file, "CESM2", expected_fields=["tas", "pr"]
    ... )
    >>> if is_valid:
    ...     print(f"✅ {info['message']}")
    """
    expected_name = f"cmip6-{model_name}-{scenario}"
    expected_vars = set(expected_fields)

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

        # The MeteorPatternScaling.save_model() saves a dict, not the object itself
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

    except Exception as e:
        info["message"] = f"Error reading cache: {e}"
        return False, None, info


def get_noise_model_cache_path(model_name, variable_name, cache_dir=None):
    """
    Get the standardized cache file path for a noise model.

    Parameters
    ----------
    model_name : str
        Name of the CMIP6 model
    variable_name : str
        Variable name (e.g., 'tas', 'pr')
    cache_dir : str, optional
        Directory for cache files. If None, uses default noise cache location.

    Returns
    -------
    str
        Full path to the cache file

    Examples
    --------
    >>> cache_path = get_noise_model_cache_path("CESM2", "tas")
    >>> print(cache_path)
    /path/to/noise_cache/CESM2_tas_noise_model.pkl
    """
    if cache_dir is None:
        # Use noise_cache in current working directory (notebook convention)
        cache_dir = os.path.join(os.getcwd(), "noise_cache")

    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{model_name}_{variable_name}_noise_model.pkl")


def validate_noise_model_cache(
    cache_file,
    variable_name,
    n_modes=40,
    lag_order=2,
):
    """
    Validate a cached noise model file.

    Checks if the cached pickle file exists, can be loaded, and contains
    the expected configuration (n_modes, lag_order, variable_name) and
    required attributes (pca, varx_results).

    Parameters
    ----------
    cache_file : str
        Path to the cached noise model file
    variable_name : str
        Expected variable name (e.g., 'tas', 'pr')
    n_modes : int, optional
        Expected number of PCA modes. Default is 40.
    lag_order : int, optional
        Expected temporal lag order. Default is 2.

    Returns
    -------
    tuple
        (is_valid, cached_model, info_dict) where:
        - is_valid: bool indicating if cache is valid
        - cached_model: loaded MeteorNoiseGenerator if valid, None otherwise
        - info_dict: dict with 'message', 'expected', 'found' information

    Examples
    --------
    >>> cache_file = get_noise_model_cache_path("CESM2", "tas")
    >>> is_valid, model, info = validate_noise_model_cache(
    ...     cache_file, "tas", n_modes=40, lag_order=2
    ... )
    >>> if is_valid:
    ...     print(f"✅ {info['message']}")
    """
    # Import here to avoid circular dependency
    from meteor.noise_generator import MeteorNoiseGenerator

    info = {
        "expected": {
            "variable_name": variable_name,
            "n_modes": n_modes,
            "lag_order": lag_order,
        },
        "found": {},
        "message": "",
    }

    # Check if file exists
    if not os.path.exists(cache_file):
        info["message"] = f"Cache file not found: {cache_file}"
        return False, None, info

    # Try to load and validate
    try:
        noise_model = MeteorNoiseGenerator(n_modes=n_modes, lag_order=lag_order)
        noise_model.load_model(cache_file)

        # Extract found information
        info["found"]["n_modes"] = getattr(noise_model, "n_modes", None)
        info["found"]["lag_order"] = getattr(noise_model, "lag_order", None)
        info["found"]["variable_name"] = getattr(noise_model, "variable_name", None)

        # Validate n_modes
        if not hasattr(noise_model, "n_modes") or noise_model.n_modes != n_modes:
            info["message"] = (
                f"n_modes mismatch: expected {n_modes}, "
                f"found {info['found']['n_modes']}"
            )
            return False, None, info

        # Validate lag_order
        if not hasattr(noise_model, "lag_order") or noise_model.lag_order != lag_order:
            info["message"] = (
                f"lag_order mismatch: expected {lag_order}, "
                f"found {info['found']['lag_order']}"
            )
            return False, None, info

        # Validate variable_name
        if (
            not hasattr(noise_model, "variable_name")
            or noise_model.variable_name != variable_name
        ):
            info["message"] = (
                f"variable_name mismatch: expected '{variable_name}', "
                f"found '{info['found']['variable_name']}'"
            )
            return False, None, info

        # Validate required attributes
        required_attrs = ["pca", "varx_results"]
        missing_attrs = [
            attr for attr in required_attrs if not hasattr(noise_model, attr)
        ]
        if missing_attrs:
            info["message"] = (
                f"Cached model missing required attributes: {missing_attrs}"
            )
            return False, None, info

        # Cache is valid
        info["message"] = (
            f"Cache valid: variable={variable_name}, "
            f"n_modes={n_modes}, lag_order={lag_order}"
        )
        return True, noise_model, info

    except Exception as e:
        info["message"] = f"Error reading cache: {e}"
        return False, None, info
