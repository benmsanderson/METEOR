"""
Precipitation Distribution Transform Module

This module provides tools for transforming precipitation data to ensure physical
realism (positivity) while matching target distributions from climate models.

Key Features:
- Gaussian-to-Gamma quantile mapping
- Support for 1D (regional/global means) and 3D (gridded) data
- Multiple target distributions (Gamma, Weibull, Log-Normal, Generalized Gamma)
- Empirical (non-parametric) quantile mapping

Author: METEOR Development Team
"""

import numpy as np
import xarray as xr
from scipy import stats

# =============================================================================
# PATH 1: 1D Transform (for regional/global mean time series)
# =============================================================================


def fit_distribution_parameters_1d(timeseries_data, distribution="gamma"):
    """
    Fit distribution to 1D time series (for regional/global means).

    This is the efficient path when working with generate_regional_mean_realizations(),
    which produces (n_realizations, n_time) output without computing gridded fields.

    Parameters
    ----------
    timeseries_data : np.ndarray or xr.DataArray
        1D time series or ensemble of time series
        Shape: (n_time,) or (n_ensemble, n_time) or (n_time, n_ensemble)
        Should be absolute precipitation (positive values for non-Gaussian)
    distribution : str
        Distribution to fit: 'gaussian', 'gamma', 'weibull', 'lognorm', 'gengamma'

    Returns
    -------
    dict
        Dictionary with scalar parameters
        - Gaussian: {'mean': float, 'std': float}
        - Gamma: {'shape': float, 'scale': float}
        - Weibull: {'shape': float, 'scale': float}
        - LogNormal: {'shape': float, 'scale': float}
        - GenGamma: {'a': float, 'c': float, 'scale': float}
    """
    # Convert to numpy if needed
    if isinstance(timeseries_data, xr.DataArray):
        data = timeseries_data.values
    else:
        data = timeseries_data

    # Flatten all data (combine all realizations and time)
    data_flat = data.flatten()

    # Remove NaNs
    data_clean = data_flat[~np.isnan(data_flat)]

    if distribution == "gaussian":
        params = {"mean": np.mean(data_clean), "std": np.std(data_clean)}

    elif distribution == "gamma":
        # Ensure data is positive
        if np.any(data_clean < 0):
            raise ValueError(
                "Gamma distribution requires positive data. "
                "Ensure you're using absolute precipitation, not anomalies."
            )

        # Fit gamma with location fixed at 0
        try:
            shape, loc, scale = stats.gamma.fit(data_clean, floc=0)
        except Exception:
            # If fit fails, use method of moments
            mean_val = np.mean(data_clean)
            var_val = np.var(data_clean)
            if var_val > 0:
                shape = (mean_val**2) / var_val
                scale = var_val / mean_val
            else:
                shape = 1.0
                scale = mean_val

        params = {"shape": shape, "scale": scale}

    elif distribution == "weibull":
        # Weibull distribution (good for different tail behaviors)
        if np.any(data_clean < 0):
            raise ValueError("Weibull distribution requires positive data.")

        try:
            shape, loc, scale = stats.weibull_min.fit(data_clean, floc=0)
            params = {"shape": shape, "scale": scale}
        except Exception:
            # Fallback to Gamma
            mean_val = np.mean(data_clean)
            params = {"shape": 1.5, "scale": mean_val / 1.5}

    elif distribution == "lognorm":
        # Log-normal distribution (good for right-skewed data with long tails)
        if np.any(data_clean <= 0):
            # Add small offset to handle zeros
            data_clean = data_clean + 1e-6

        try:
            shape, loc, scale = stats.lognorm.fit(data_clean, floc=0)
            params = {
                "shape": shape,  # This is sigma (std of log)
                "scale": scale,  # This is exp(mu) where mu is mean of log
            }
        except Exception:
            # Fallback
            log_data = np.log(data_clean)
            params = {"shape": np.std(log_data), "scale": np.exp(np.mean(log_data))}

    elif distribution == "gengamma":
        # Generalized Gamma (3 parameters - most flexible)
        if np.any(data_clean < 0):
            raise ValueError("Generalized Gamma distribution requires positive data.")

        try:
            a, c, loc, scale = stats.gengamma.fit(data_clean, floc=0)
            params = {
                "a": a,  # First shape parameter
                "c": c,  # Second shape parameter
                "scale": scale,
            }
        except Exception:
            # Fallback to regular Gamma
            shape, loc, scale = stats.gamma.fit(data_clean, floc=0)
            params = {"a": shape, "c": 1.0, "scale": scale}

    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            "Supported: 'gaussian', 'gamma', 'weibull', 'lognorm', 'gengamma'"
        )

    return params


# =============================================================================
# PATH 2: 3D Transform (for gridded fields)
# =============================================================================


def fit_distribution_parameters_3d(spatial_data, distribution="gamma"):
    """
    Fit distribution at each grid point (for gridded fields).

    This is for use with generate_realization(), which produces full spatial fields.
    Fits a separate distribution at each (lat, lon) location.

    Parameters
    ----------
    spatial_data : np.ndarray or xr.DataArray
        Spatial precipitation data
        Shape: (n_time, n_lat, n_lon) or (n_ensemble, n_time, n_lat, n_lon)
        Should be absolute precipitation (positive values for non-Gaussian)
    distribution : str
        Distribution to fit: 'gaussian' or 'gamma'

    Returns
    -------
    dict
        Dictionary with spatial arrays of parameters (n_lat, n_lon)
        - For Gaussian: {'mean': array, 'std': array}
        - For Gamma: {'shape': array, 'scale': array}
    """
    # Convert to numpy if needed
    if isinstance(spatial_data, xr.DataArray):
        data_array = spatial_data.values
    else:
        data_array = spatial_data

    # Handle different input shapes
    if data_array.ndim == 3:
        # (n_time, n_lat, n_lon)
        n_time, n_lat, n_lon = data_array.shape
        data_reshaped = data_array.reshape(n_time, n_lat * n_lon)
    elif data_array.ndim == 4:
        # (n_ensemble, n_time, n_lat, n_lon) - flatten ensemble and time
        n_ensemble, n_time, n_lat, n_lon = data_array.shape
        data_reshaped = data_array.reshape(n_ensemble * n_time, n_lat * n_lon)
    else:
        raise ValueError(
            "Data must be 3D (n_time, n_lat, n_lon) or "
            f"4D (n_ensemble, n_time, n_lat, n_lon), got shape {data_array.shape}"
        )

    n_spatial = n_lat * n_lon

    if distribution == "gaussian":
        # Fit Gaussian: simple mean and std at each grid point
        mean_params = np.mean(data_reshaped, axis=0).reshape(n_lat, n_lon)
        std_params = np.std(data_reshaped, axis=0).reshape(n_lat, n_lon)

        params = {"mean": mean_params, "std": std_params}

    elif distribution == "gamma":
        # Fit Gamma distribution at each grid point
        shape_params = np.zeros(n_spatial)
        scale_params = np.zeros(n_spatial)

        print(f"Fitting Gamma distribution to {n_spatial} grid points...")
        for i in range(n_spatial):
            grid_data = data_reshaped[:, i]
            # Remove any NaNs
            grid_data_clean = grid_data[~np.isnan(grid_data)]

            if len(grid_data_clean) > 0 and np.all(grid_data_clean >= 0):
                # Fit gamma with location fixed at 0
                try:
                    shape, loc, scale = stats.gamma.fit(grid_data_clean, floc=0)
                    shape_params[i] = shape
                    scale_params[i] = scale
                except Exception:
                    # If fit fails, use method of moments
                    mean_val = np.mean(grid_data_clean)
                    var_val = np.var(grid_data_clean)
                    if var_val > 0:
                        shape_params[i] = (mean_val**2) / var_val
                        scale_params[i] = var_val / mean_val
                    else:
                        shape_params[i] = 1.0
                        scale_params[i] = mean_val
            else:
                # Default values for invalid data
                shape_params[i] = 1.0
                scale_params[i] = 0.01

            if (i + 1) % 5000 == 0:  # pragma no cover
                print(f"  Processed {i + 1}/{n_spatial} grid points...")

        params = {
            "shape": shape_params.reshape(n_lat, n_lon),
            "scale": scale_params.reshape(n_lat, n_lon),
        }

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return params


# =============================================================================
# Unified Transform Function (auto-detects 1D vs 3D)
# =============================================================================


def apply_distribution_transform(
    gaussian_data, gaussian_params, target_params, target_dist="gamma"
):
    """
    Transform Gaussian-distributed data to target distribution.

    Automatically detects whether to use 1D or 3D transform based on parameter shapes.

    Parameters
    ----------
    gaussian_data : np.ndarray or xr.DataArray
        Generated precipitation data from noise model
        - 1D case: (n_realizations, n_time) from generate_regional_mean_realizations()
        - 3D case: (n_time, n_lat, n_lon) or (n_realizations, n_time, n_lat, n_lon)
    gaussian_params : dict
        Dictionary with 'mean' and 'std'
        - 1D case: scalar values
        - 3D case: (n_lat, n_lon) arrays
    target_params : dict
        Parameters for target distribution
        - Gamma: {'shape': ..., 'scale': ...}
        - Weibull: {'shape': ..., 'scale': ...}
        - LogNorm: {'shape': ..., 'scale': ...}
        - GenGamma: {'a': ..., 'c': ..., 'scale': ...}
    target_dist : str
        Target distribution: 'gamma', 'weibull', 'lognorm', 'gengamma'

    Returns
    -------
    np.ndarray or xr.DataArray
        Transformed data with same shape and type as input
    """
    # Preserve xarray structure if input is xarray
    is_xarray = isinstance(gaussian_data, xr.DataArray)
    if is_xarray:
        coords = gaussian_data.coords
        dims = gaussian_data.dims
        data = gaussian_data.values
    else:
        data = gaussian_data

    # Detect 1D vs 3D based on parameter shape
    is_scalar_params = np.isscalar(gaussian_params["mean"])

    if is_scalar_params:
        # ===== 1D TRANSFORM =====
        if data.ndim != 2:
            raise ValueError(
                f"For scalar parameters, expected 2D data (n_realizations, n_time), "
                f"got shape {data.shape}"
            )

        # Transform: Gaussian → Uniform → Target
        uniform = stats.norm.cdf(
            data, loc=gaussian_params["mean"], scale=gaussian_params["std"]
        )
        uniform = np.clip(uniform, 1e-10, 1 - 1e-10)

        # Map to target distribution
        if target_dist == "gamma":
            transformed = stats.gamma.ppf(
                uniform, a=target_params["shape"], scale=target_params["scale"]
            )
        elif target_dist == "weibull":
            transformed = stats.weibull_min.ppf(
                uniform, c=target_params["shape"], scale=target_params["scale"]
            )
        elif target_dist == "lognorm":
            transformed = stats.lognorm.ppf(
                uniform, s=target_params["shape"], scale=target_params["scale"]
            )
        elif target_dist == "gengamma":
            transformed = stats.gengamma.ppf(
                uniform,
                a=target_params["a"],
                c=target_params["c"],
                scale=target_params["scale"],
            )
        else:
            raise ValueError(f"Unknown target distribution: {target_dist}")

    else:
        # ===== 3D TRANSFORM =====
        if data.ndim == 3:
            n_time, n_lat, n_lon = data.shape
            data_reshaped = data.reshape(n_time, n_lat * n_lon)
        elif data.ndim == 4:
            n_real, n_time, n_lat, n_lon = data.shape
            data_reshaped = data.reshape(n_real * n_time, n_lat * n_lon)
        else:
            raise ValueError(
                f"For spatial parameters, expected 3D or 4D data, got shape {data.shape}"
            )

        # Flatten parameter grids
        mean_grid = gaussian_params["mean"].flatten()
        std_grid = gaussian_params["std"].flatten()

        # Transform: Gaussian → Uniform
        uniform = stats.norm.cdf(
            data_reshaped, loc=mean_grid[None, :], scale=std_grid[None, :]
        )
        uniform = np.clip(uniform, 1e-10, 1 - 1e-10)

        # Transform: Uniform → Target (only Gamma supported for 3D currently)
        if target_dist == "gamma":
            shape_grid = target_params["shape"].flatten()
            scale_grid = target_params["scale"].flatten()
            transformed_flat = stats.gamma.ppf(
                uniform, a=shape_grid[None, :], scale=scale_grid[None, :]
            )
        else:
            raise ValueError(
                f"Only 'gamma' distribution supported for 3D data, got {target_dist}"
            )

        # Reshape back to original
        transformed = transformed_flat.reshape(data.shape)

    # Return in same format as input
    if is_xarray:
        return xr.DataArray(transformed, coords=coords, dims=dims)
    else:
        return transformed


# =============================================================================
# Empirical Quantile Mapping (non-parametric alternative)
# =============================================================================


def apply_empirical_quantile_mapping(generated_data, target_data):
    """
    Apply empirical quantile mapping to transform generated data to match target distribution.

    This is a non-parametric approach that doesn't assume any distribution shape.
    Works better for complex distributions like point-scale precipitation.

    Parameters
    ----------
    generated_data : np.ndarray or xr.DataArray
        Generated precipitation data (can have negative values)
        Shape: (n_realizations, n_time) or similar
    target_data : np.ndarray or xr.DataArray
        Target CMIP6 data to match
        Shape: (n_ensemble, n_time) or (n_time,)

    Returns
    -------
    transformed_data : same type as input
        Data transformed to match target distribution
    """
    # Preserve xarray structure
    is_xarray = isinstance(generated_data, xr.DataArray)
    if is_xarray:
        coords = generated_data.coords
        dims = generated_data.dims
        # Convert to numpy
        gen_values = generated_data.values
    else:
        gen_values = generated_data

    if isinstance(target_data, xr.DataArray):
        target_values = target_data.values
    else:
        target_values = target_data

    # Flatten both datasets
    gen_flat = gen_values.flatten()
    target_flat = target_values.flatten()

    # Remove NaNs
    gen_flat = gen_flat[~np.isnan(gen_flat)]
    target_flat = target_flat[~np.isnan(target_flat)]

    # Sort both to get empirical CDFs
    gen_sorted = np.sort(gen_flat)
    target_sorted = np.sort(target_flat)

    # Interpolate: for each value in generated data, find its quantile,
    # then map to corresponding quantile in target data
    transformed_flat = np.interp(
        gen_values.flatten(),
        gen_sorted,
        target_sorted,
        left=target_sorted[0],  # Extrapolate with min/max for out-of-sample
        right=target_sorted[-1],
    )

    # Reshape back
    transformed = transformed_flat.reshape(gen_values.shape)

    # Return in same format as input
    if is_xarray:
        return xr.DataArray(transformed, coords=coords, dims=dims)
    else:
        return transformed
