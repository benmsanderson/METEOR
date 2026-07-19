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

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
from scipy import stats
from scipy.special import (  # pylint: disable=no-name-in-module
    digamma,
    gammaincinv,
    polygamma,
)


def _resolve_gamma_ppf_threads():
    """Pick thread count for :func:`_threaded_gamma_ppf_3d`.

    Reads the ``METEOR_GAMMA_PPF_THREADS`` environment variable so HPC users
    can pin threads without touching Python (same pattern as ``OMP_NUM_THREADS``).
    Falls back to ``min(8, cpu_count())`` — bounded because the underlying
    scipy special function has diminishing returns past ~8 threads on typical
    hardware. Set to ``1`` to disable threading entirely.
    """
    env = os.environ.get("METEOR_GAMMA_PPF_THREADS")
    if env:
        try:
            n = int(env)
            if n >= 1:
                return n
        except ValueError:
            pass
    return min(8, os.cpu_count() or 1)


def _threaded_gamma_ppf_3d(u, shape_flat, scale_flat, n_threads=None):
    """gamma.ppf across the trailing spatial axis, threaded.

    scipy's ``gammaincinv`` (which backs ``stats.gamma.ppf``) releases the GIL,
    so we chunk along the spatial axis and evaluate in parallel threads. For
    the METEOR gridded workload this dominates :func:`apply_distribution_transform`.

    Parameters
    ----------
    u : ndarray, shape (..., n_spatial)
        Uniform-scale quantiles (output of ``norm.cdf``).
    shape_flat, scale_flat : ndarray, shape (n_spatial,)
    n_threads : int, optional
        Explicit thread count. If ``None`` (default), consults the
        ``METEOR_GAMMA_PPF_THREADS`` env var, falling back to
        ``min(8, cpu_count())``. Bypasses threading for small problems where
        thread setup would dominate.
    """
    n_spatial = shape_flat.shape[0]
    if n_threads is None:
        n_threads = _resolve_gamma_ppf_threads()
    if n_threads <= 1 or n_spatial < 4096:
        return scale_flat * gammaincinv(shape_flat, u)

    chunks = np.array_split(np.arange(n_spatial), n_threads)
    out = np.empty_like(u)

    def _work(idx):
        return idx, scale_flat[idx] * gammaincinv(shape_flat[idx], u[..., idx])

    with ThreadPoolExecutor(n_threads) as ex:
        for idx, res in ex.map(_work, chunks):
            out[..., idx] = res
    return out


def _vectorized_gamma_mle(data, max_iter=8, tol=1e-8):
    """MLE fit of Gamma(shape, scale) with location fixed at 0, vectorized.

    Same estimator scipy.stats.gamma.fit(x, floc=0) uses internally, but
    applied to a batch of independent samples in one call. For each column j
    of ``data`` (shape ``(n_obs, n_series)``), solve

        log(k) - psi(k) = log(mean(x)) - mean(log(x))       (Choi & Wette 1969)
        theta = mean(x) / k

    by Newton's method on k, seeded with the Choi-Wette initial guess.

    Non-positive samples are treated as invalid and masked out; series with
    fewer than 2 positive samples fall back to shape=1.0, scale=mean.

    Parameters
    ----------
    data : ndarray (n_obs, n_series)
    max_iter : int
    tol : float
        Convergence tolerance on |Δk| / k.

    Returns
    -------
    shape, scale : ndarray (n_series,)
    """
    x = np.asarray(data, dtype=np.float64)

    valid = (x > 0) & np.isfinite(x)
    n_valid = valid.sum(axis=0)

    # Sum and log-sum with invalid entries zeroed out
    x_masked = np.where(valid, x, 0.0)
    logx_masked = np.where(valid, np.log(np.maximum(x, np.finfo(float).tiny)), 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = x_masked.sum(axis=0) / n_valid
        mean_logx = logx_masked.sum(axis=0) / n_valid
        s = np.log(mean_x) - mean_logx  # >= 0 with equality iff constant

    # Guard against degenerate series
    fittable = (n_valid >= 2) & np.isfinite(s) & (s > 0)

    # Choi-Wette initial guess for k (only used where fittable)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = (3.0 - s + np.sqrt(np.maximum((s - 3.0) ** 2 + 24.0 * s, 0.0))) / (12.0 * s)
    k = np.where(fittable, k, 1.0)
    k = np.clip(k, 1e-6, 1e6)

    # Newton iterations
    for _ in range(max_iter):
        f = np.log(k) - digamma(k) - s
        fp = 1.0 / k - polygamma(1, k)  # trigamma
        step = f / fp
        k_new = np.clip(k - step, 1e-6, 1e6)
        if np.max(np.abs(k_new - k) / np.maximum(k, 1e-12)) < tol:
            k = k_new
            break
        k = k_new

    shape = np.where(fittable, k, 1.0)
    scale = np.where(fittable, mean_x / shape, np.where(n_valid > 0, mean_x, 0.01))
    return shape, scale


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
            shape, loc, scale = stats.gamma.fit(  # pylint: disable=unused-variable
                data_clean, floc=0
            )
        except (ValueError, RuntimeError, RuntimeWarning):
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
        except (ValueError, RuntimeError, RuntimeWarning):
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
        except (ValueError, RuntimeError, RuntimeWarning):
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
        except (ValueError, RuntimeError, RuntimeWarning):
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

    if distribution == "gaussian":
        # Fit Gaussian: simple mean and std at each grid point
        mean_params = np.mean(data_reshaped, axis=0).reshape(n_lat, n_lon)
        std_params = np.std(data_reshaped, axis=0).reshape(n_lat, n_lon)

        params = {"mean": mean_params, "std": std_params}

    elif distribution == "gamma":
        # Vectorized MLE fit across all gridpoints in one shot.
        shape_flat, scale_flat = _vectorized_gamma_mle(data_reshaped)
        params = {
            "shape": shape_flat.reshape(n_lat, n_lon),
            "scale": scale_flat.reshape(n_lat, n_lon),
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
            transformed_flat = _threaded_gamma_ppf_3d(uniform, shape_grid, scale_grid)
        else:
            raise ValueError(
                f"Only 'gamma' distribution supported for 3D data, got {target_dist}"
            )

        # Reshape back to original
        transformed = transformed_flat.reshape(data.shape)

    # Return in same format as input
    if is_xarray:
        return xr.DataArray(transformed, coords=coords, dims=dims)
    return transformed


# =============================================================================
# Seasonal (per-month-of-year) variants
# =============================================================================
#
# Fitting/applying one quantile map across all months conflates the seasonal
# cycle with internal variability: the Gaussian std becomes dominated by the
# seasonal swing (very large for variables like precipitation), so the CDF
# squashes inter-realization noise into a narrow quantile band and the gamma
# PPF then maps it to a narrow output band. Fitting per month-of-year removes
# the seasonal contribution from the variance the quantile map sees, so the
# transform actually preserves the within-month internal variability.


def _month_of_year_indices(n_time):
    """
    Return a list of 12 index arrays selecting each month-of-year

    Return a list of 12 index arrays selecting each month-of-year from a
    contiguous monthly time axis of length ``n_time``. Assumes the series
    starts in January; partial trailing years are fine (the last month-of-year
    bins will just have one fewer sample).

    Parameters
    ----------
    n_time : int
        Length of the time axis (must be a multiple of 12).

    Returns
    -------
    list of np.ndarray
        Each element is a 1D array of indices selecting the corresponding month-of-year.
    """
    return [np.arange(m, n_time, 12) for m in range(12)]


def fit_distribution_parameters_1d_seasonal(timeseries_data, distribution="gamma"):
    """
    Fit a separate distribution per month-of-year (12 fits) to a 1D-time series.

    Parameters
    ----------
    timeseries_data : np.ndarray or xr.DataArray
        Monthly data. Last axis is time and must be a multiple of 12. Earlier
        axes (realization / ensemble) are pooled into each per-month fit.
    distribution : str
        Distribution to fit at each month-of-year.

    Returns
    -------
    dict
        Each parameter key maps to a 1D array of length 12 (Jan..Dec).
    """
    if isinstance(timeseries_data, xr.DataArray):
        data = timeseries_data.values
    else:
        data = timeseries_data

    n_time = data.shape[-1]
    month_idx = _month_of_year_indices(n_time)

    per_month = [
        fit_distribution_parameters_1d(data[..., idx], distribution=distribution)
        for idx in month_idx
    ]

    keys = per_month[0].keys()
    return {k: np.array([p[k] for p in per_month]) for k in keys}


def fit_distribution_parameters_3d_seasonal(spatial_data, distribution="gamma"):
    """
    Fit per-gridpoint distributions separately for each month-of-year.

    Parameters
    ----------
    spatial_data : np.ndarray or xr.DataArray
        Shape (n_time, n_lat, n_lon) or (n_ensemble, n_time, n_lat, n_lon).
        ``n_time`` must be a multiple of 12.
    distribution : str
        'gaussian' or 'gamma'.

    Returns
    -------
    dict
        Each parameter key maps to an array of shape (12, n_lat, n_lon).
    """
    if isinstance(spatial_data, xr.DataArray):
        data = spatial_data.values
    else:
        data = spatial_data

    if data.ndim == 3:
        time_axis = 0
        n_time = data.shape[0]
    elif data.ndim == 4:
        time_axis = 1
        n_time = data.shape[1]
    else:
        raise ValueError(
            "Data must be 3D (n_time, n_lat, n_lon) or "
            f"4D (n_ensemble, n_time, n_lat, n_lon), got shape {data.shape}"
        )

    month_idx = _month_of_year_indices(n_time)

    per_month = [
        fit_distribution_parameters_3d(
            np.take(data, idx, axis=time_axis), distribution=distribution
        )
        for idx in month_idx
    ]

    keys = per_month[0].keys()
    return {k: np.stack([p[k] for p in per_month], axis=0) for k in keys}


def apply_distribution_transform_seasonal(
    gaussian_data, gaussian_params, target_params, target_dist="gamma"
):
    """
    Apply a per-month-of-year quantile transform.

    Automatically detects 1D vs 3D based on the rank of the parameter arrays
    (``ndim == 1`` -> 1D scalar-per-month; ``ndim == 3`` -> 3D per-gridpoint).

    Parameters
    ----------
    gaussian_data : np.ndarray or xr.DataArray
        Shape (n_realizations, n_time) for 1D or
        (n_realizations, n_time, n_lat, n_lon) / (n_time, n_lat, n_lon) for 3D.
        ``n_time`` must be a multiple of 12.
    gaussian_params, target_params : dict
        Per-month-of-year parameter arrays. For 1D: shape (12,) each.
        For 3D: shape (12, n_lat, n_lon) each.
    target_dist : str
        Target distribution name.

    Returns
    -------
    Same type/shape as input.
    """
    is_xarray = isinstance(gaussian_data, xr.DataArray)
    if is_xarray:
        coords = gaussian_data.coords
        dims = gaussian_data.dims
        data = gaussian_data.values
    else:
        data = gaussian_data

    sample_param = gaussian_params["mean"]
    if sample_param.ndim == 1:
        # 1D scalar-per-month
        if data.ndim != 2:
            raise ValueError(
                "For 1D seasonal transform, expected 2D data (n_real, n_time), "
                f"got shape {data.shape}"
            )
        time_axis = 1
    elif sample_param.ndim == 3:
        # 3D per-gridpoint-per-month: params shape (12, n_lat, n_lon)
        if data.ndim == 3:
            time_axis = 0
        elif data.ndim == 4:
            time_axis = 1
        else:
            raise ValueError(
                "For 3D seasonal transform, expected 3D or 4D data, "
                f"got shape {data.shape}"
            )
    else:
        raise ValueError(
            "Seasonal params must have ndim 1 (scalar per month) or 3 "
            f"(per gridpoint per month), got ndim={sample_param.ndim}"
        )

    n_time = data.shape[time_axis]
    month_idx = _month_of_year_indices(n_time)

    transformed = np.empty_like(data, dtype=np.float64)
    for m, idx in enumerate(month_idx):
        data_m = np.take(data, idx, axis=time_axis)

        if sample_param.ndim == 1:
            # Extract Python scalars so the underlying apply_distribution_transform
            # takes the 1D-scalar path (which uses np.isscalar).
            gp_m = {k: float(v[m]) for k, v in gaussian_params.items()}
            tp_m = {k: float(v[m]) for k, v in target_params.items()}
        else:
            gp_m = {k: v[m] for k, v in gaussian_params.items()}
            tp_m = {k: v[m] for k, v in target_params.items()}

        transformed_m = apply_distribution_transform(
            data_m, gp_m, tp_m, target_dist=target_dist
        )

        # Write back into the appropriate slice of ``transformed``.
        if time_axis == 0:
            transformed[idx] = transformed_m
        else:
            # time_axis == 1 covers both 2D and 4D layouts.
            transformed[:, idx] = transformed_m

    if is_xarray:
        return xr.DataArray(transformed, coords=coords, dims=dims)
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
    return transformed
