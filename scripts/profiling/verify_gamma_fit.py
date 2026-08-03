"""Verify the vectorized gamma MLE matches scipy.stats.gamma.fit(x, floc=0).

Draws synthetic gamma samples at a batch of gridpoints and compares the
vectorized MLE to scipy's per-gridpoint MLE. Then runs against a real slice
of NorESM2-MM precipitation to check the actual data path.
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from scipy import stats
import xarray as xr

from meteor.precipitation_transform import (
    _vectorized_gamma_mle,
    fit_distribution_parameters_3d,
)


def synthetic_check():
    rng = np.random.default_rng(0)
    n_spatial = 20_000
    n_obs = 3012
    true_shape = rng.uniform(0.5, 5.0, size=n_spatial)
    true_scale = rng.uniform(0.1, 10.0, size=n_spatial)

    data = np.stack([
        rng.gamma(k, s, size=n_obs) for k, s in zip(true_shape, true_scale)
    ], axis=1)  # (n_obs, n_spatial)

    t0 = time.perf_counter()
    scipy_shape = np.zeros(n_spatial)
    scipy_scale = np.zeros(n_spatial)
    for i in range(n_spatial):
        k, _, s = stats.gamma.fit(data[:, i], floc=0)
        scipy_shape[i] = k
        scipy_scale[i] = s
    t_scipy = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_shape, vec_scale = _vectorized_gamma_mle(data)
    t_vec = time.perf_counter() - t0

    dk = np.abs(vec_shape - scipy_shape) / scipy_shape
    ds = np.abs(vec_scale - scipy_scale) / scipy_scale
    print(f"synthetic: n={n_spatial}, obs={n_obs}")
    print(f"  scipy fit: {t_scipy:.3f}s   vec fit: {t_vec:.3f}s   "
          f"speedup: {t_scipy/t_vec:.0f}×")
    print(f"  shape rel err: max={dk.max():.3g}, median={np.median(dk):.3g}")
    print(f"  scale rel err: max={ds.max():.3g}, median={np.median(ds):.3g}")


def real_data_check():
    pr_path = REPO_ROOT / "cache" / "cmip6" / "NorESM2-MM_ssp245_pr_monthly.nc"
    ds = xr.open_dataset(pr_path)
    var = [v for v in ds.data_vars if v.lower() == "pr"][0]
    data = ds[var].values.squeeze()  # collapse trivial dims
    if data.ndim == 4:
        data = data[0]
    # Sub-slice for a manageable comparison (~2400 gridpoints)
    data = data[:1200, :40, :60]
    print(f"\nreal data shape: {data.shape}, ({data.size:,} points)")

    t0 = time.perf_counter()
    params_vec = fit_distribution_parameters_3d(data, distribution="gamma")
    t_vec = time.perf_counter() - t0

    n_time, n_lat, n_lon = data.shape
    reshaped = data.reshape(n_time, n_lat * n_lon)
    t0 = time.perf_counter()
    scipy_shape = np.zeros(n_lat * n_lon)
    scipy_scale = np.zeros(n_lat * n_lon)
    for i in range(n_lat * n_lon):
        col = reshaped[:, i]
        col = col[col > 0]
        if len(col) < 2:
            scipy_shape[i] = 1.0
            scipy_scale[i] = 0.01
            continue
        try:
            k, _, s = stats.gamma.fit(col, floc=0)
            scipy_shape[i] = k
            scipy_scale[i] = s
        except Exception:
            scipy_shape[i] = 1.0
            scipy_scale[i] = 0.01
    t_scipy = time.perf_counter() - t0

    scipy_shape = scipy_shape.reshape(n_lat, n_lon)
    scipy_scale = scipy_scale.reshape(n_lat, n_lon)

    # Compare where scipy converged to something plausible
    ok = np.isfinite(scipy_shape) & (scipy_shape > 0) & (scipy_scale > 0)
    dk = np.abs(params_vec["shape"] - scipy_shape) / scipy_shape
    ds = np.abs(params_vec["scale"] - scipy_scale) / scipy_scale
    print(f"  scipy fit: {t_scipy:.3f}s   vec fit: {t_vec:.3f}s   "
          f"speedup: {t_scipy/t_vec:.0f}×")
    print(f"  shape rel err: max={dk[ok].max():.3g}, "
          f"median={np.median(dk[ok]):.3g}")
    print(f"  scale rel err: max={ds[ok].max():.3g}, "
          f"median={np.median(ds[ok]):.3g}")


if __name__ == "__main__":
    synthetic_check()
    real_data_check()
