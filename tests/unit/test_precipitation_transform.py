"""
Unit tests for precipitation transform functions.
"""

import os
import warnings

import numpy as np
import pytest
import xarray as xr
from scipy import stats

from meteor.precipitation_transform import (
    _GAMMA_PPF_THREAD_LIMIT_FACTOR,
    _resolve_gamma_ppf_threads,
    _threaded_gamma_ppf_3d,
    _vectorized_gamma_mle,
    apply_distribution_transform,
    apply_empirical_quantile_mapping,
    fit_distribution_parameters_1d,
    fit_distribution_parameters_3d,
)


def test_fit_distribution_parameters_1d():
    """Test fitting distribution parameters for 1D data."""

    data = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    params = fit_distribution_parameters_1d(data, distribution="gamma")
    assert "shape" in params and "scale" in params
    assert params["shape"] > 0
    assert params["scale"] > 0
    data_neg = np.array([-1.0, 0.5, 1.0])
    with pytest.raises(
        ValueError,
        match="Gamma distribution requires positive data. "
        "Ensure you're using absolute precipitation, not anomalies.",
    ):
        params = fit_distribution_parameters_1d(data_neg, distribution="gamma")
    with pytest.raises(
        ValueError, match="Weibull distribution requires positive data."
    ):
        params = fit_distribution_parameters_1d(data_neg, distribution="weibull")
    with pytest.raises(
        ValueError, match="Generalized Gamma distribution requires positive data."
    ):
        params = fit_distribution_parameters_1d(data_neg, distribution="gengamma")

    params = fit_distribution_parameters_1d(data, distribution="weibull")
    assert np.allclose(params["shape"], 1.1054055920260153)
    assert np.allclose(params["scale"], 1.3668311960654322)

    params = fit_distribution_parameters_1d(data, distribution="lognorm")
    assert np.allclose(params["shape"], 1.1983190724023456)
    assert np.allclose(params["scale"], 0.7860030855966228)

    params = fit_distribution_parameters_1d(data, distribution="gengamma")
    assert np.allclose(params["a"], 0.0002357964916252947)
    assert np.allclose(params["c"], 3154.006251745962)
    assert np.allclose(params["scale"], 3.0065872015276325)

    with pytest.raises(
        ValueError,
        match="Unknown distribution: unknown_dist. "
        "Supported: 'gaussian', 'gamma', 'weibull', 'lognorm', 'gengamma'",
    ):
        params = fit_distribution_parameters_1d(data, distribution="unknown_dist")


def test_fit_distribution_parameters_3d():
    """Test fitting distribution parameters for 3D data."""

    data = np.ones(1)
    with pytest.raises(
        ValueError,
        match=r"Data must be 3D \(n_time, n_lat, n_lon\) or "
        r"4D \(n_ensemble, n_time, n_lat, n_lon\), got shape \(1,\)",
    ):
        params = fit_distribution_parameters_3d(data, distribution="gamma")

    data = np.ones((1, 1, 3, 2)) * 4  # 4D data: (n_ensemble, n_time, n_lat, n_lon)
    with pytest.raises(ValueError, match="Unknown distribution: unknown_dist"):
        params = fit_distribution_parameters_3d(data, distribution="unknown_dist")
    params = fit_distribution_parameters_3d(data, distribution="gamma")
    assert params["shape"].shape == (3, 2)
    assert params["scale"].shape == (3, 2)
    assert np.all(params["shape"] > 0)
    assert np.all(params["scale"] > 0)
    data_xarray = xr.DataArray(
        np.ones((1, 2, 2)) * 5,
        dims=("ntime", "nlat", "nlon"),
        coords={
            "ntime": np.arange(1),
            "nlat": np.arange(2),
            "nlon": np.arange(2),
        },
    )
    params = fit_distribution_parameters_3d(data_xarray, distribution="gaussian")
    print(params)
    assert params["mean"].shape == (2, 2)
    assert np.allclose(params["mean"], 5.0)
    assert params["std"].shape == (2, 2)
    assert np.allclose(params["std"].all(), 0.0)


def test_vectorized_gamma_mle_matches_scipy():
    """The vectorized MLE must agree with scipy's per-gridpoint MLE fit.

    Both implementations use the same estimator (Choi-Wette initial guess
    plus Newton iteration on log(k) - psi(k) = log(mean(x)) - mean(log(x))),
    so on well-behaved synthetic gamma data they should agree to
    floating-point precision.
    """
    rng = np.random.default_rng(42)
    n_series = 200
    n_obs = 500
    true_shape = rng.uniform(0.5, 5.0, size=n_series)
    true_scale = rng.uniform(0.1, 10.0, size=n_series)
    data = np.stack(
        [rng.gamma(k, s, size=n_obs) for k, s in zip(true_shape, true_scale)],
        axis=1,
    )  # (n_obs, n_series)

    vec_shape, vec_scale = _vectorized_gamma_mle(data)

    scipy_shape = np.empty(n_series)
    scipy_scale = np.empty(n_series)
    for i in range(n_series):
        k, _, s = stats.gamma.fit(data[:, i], floc=0)
        scipy_shape[i] = k
        scipy_scale[i] = s

    np.testing.assert_allclose(vec_shape, scipy_shape, rtol=1e-6)
    np.testing.assert_allclose(vec_scale, scipy_scale, rtol=1e-6)


def test_vectorized_gamma_mle_handles_invalid_columns():
    """Columns with fewer than two positive samples must fall back cleanly
    rather than propagating NaN or crashing.
    """
    n_obs = 100
    data = np.column_stack(
        [
            np.random.default_rng(0).gamma(2.0, 1.0, size=n_obs),
            np.zeros(n_obs),  # all zeros -> not fittable
            np.full(n_obs, np.nan),  # all NaN -> not fittable
            np.array([1.0] + [0.0] * (n_obs - 1)),  # single positive value
        ]
    )

    shape, scale = _vectorized_gamma_mle(data)

    assert np.all(np.isfinite(shape))
    assert np.all(np.isfinite(scale))
    assert shape[0] > 0 and scale[0] > 0
    # Degenerate columns fall back to the safe defaults from the doc.
    assert shape[1] == 1.0
    assert shape[2] == 1.0
    assert shape[3] == 1.0


def test_vectorized_gamma_mle_excludes_negatives_without_warning():
    """Negative, -inf and NaN samples must be excluded from the fit, not merely
    tolerated, and must not trip a log-domain warning on the way through.

    The estimator masks before taking logs, so `np.log` is never handed a
    non-positive value. Asserting on warnings here is the point: a refactor that
    took the log first and masked afterwards would still return correct numbers
    (the mask discards the NaNs) but would emit invalid-value RuntimeWarnings,
    and under `-W error` would crash outright.
    """
    rng = np.random.default_rng(0)
    n_obs = 400
    positives = rng.gamma(2.0, 3.0, size=n_obs)
    mixed = np.concatenate(
        [rng.gamma(2.0, 3.0, size=n_obs - 50), -rng.gamma(2.0, 3.0, size=50)]
    )
    with_inf = np.concatenate([rng.gamma(2.0, 3.0, size=n_obs - 1), [-np.inf]])
    all_negative = -rng.gamma(2.0, 3.0, size=n_obs)

    data = np.column_stack([positives, mixed, with_inf, all_negative])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        shape, scale = _vectorized_gamma_mle(data)

    assert np.all(np.isfinite(shape))
    assert np.all(np.isfinite(scale))

    # A column whose negatives were *excluded* must match scipy fitted on the
    # surviving positives alone. A column whose negatives merely became NaN and
    # got averaged in would not.
    survivors = mixed[mixed > 0]
    exp_shape, _, exp_scale = stats.gamma.fit(survivors, floc=0)
    np.testing.assert_allclose(shape[1], exp_shape, rtol=1e-6)
    np.testing.assert_allclose(scale[1], exp_scale, rtol=1e-6)

    # -inf is dropped the same way.
    exp_shape, _, exp_scale = stats.gamma.fit(with_inf[np.isfinite(with_inf)], floc=0)
    np.testing.assert_allclose(shape[2], exp_shape, rtol=1e-6)
    np.testing.assert_allclose(scale[2], exp_scale, rtol=1e-6)

    # Nothing positive left to fit -> documented fallback, no NaN.
    assert shape[3] == 1.0


def test_threaded_gamma_ppf_3d_matches_scipy():
    """Threaded ppf must reproduce scipy.stats.gamma.ppf bitwise."""
    rng = np.random.default_rng(0)
    n_time, n_spatial = 200, 5_000  # n_spatial > 4096 to trigger threading
    shape = rng.uniform(0.5, 5.0, size=n_spatial)
    scale = rng.uniform(0.1, 10.0, size=n_spatial)
    u = rng.uniform(0.01, 0.99, size=(n_time, n_spatial))

    expected = stats.gamma.ppf(u, a=shape[None, :], scale=scale[None, :])
    got = _threaded_gamma_ppf_3d(u, shape, scale)

    np.testing.assert_array_equal(expected, got)


def test_threaded_gamma_ppf_3d_single_thread_bypass():
    """n_threads=1 should return the same result via the direct (non-thread) path."""
    rng = np.random.default_rng(1)
    shape = rng.uniform(0.5, 5.0, size=5_000)
    scale = rng.uniform(0.1, 10.0, size=5_000)
    u = rng.uniform(0.01, 0.99, size=(100, 5_000))

    r_direct = _threaded_gamma_ppf_3d(u, shape, scale, n_threads=1)
    r_threaded = _threaded_gamma_ppf_3d(u, shape, scale, n_threads=4)
    np.testing.assert_array_equal(r_direct, r_threaded)


def test_resolve_gamma_ppf_threads_env_var(monkeypatch):
    """METEOR_GAMMA_PPF_THREADS overrides the default; invalid values fall back."""
    monkeypatch.delenv("METEOR_GAMMA_PPF_THREADS", raising=False)
    default = _resolve_gamma_ppf_threads()
    assert default >= 1

    monkeypatch.setenv("METEOR_GAMMA_PPF_THREADS", "3")
    assert _resolve_gamma_ppf_threads() == 3

    monkeypatch.setenv("METEOR_GAMMA_PPF_THREADS", "1")
    assert _resolve_gamma_ppf_threads() == 1

    # Non-integer input silently falls back so batch scripts with a typo
    # don't crash METEOR at import time.
    monkeypatch.setenv("METEOR_GAMMA_PPF_THREADS", "garbage")
    assert _resolve_gamma_ppf_threads() == default

    # Zero (i.e. "no threads") is also invalid; fall back.
    monkeypatch.setenv("METEOR_GAMMA_PPF_THREADS", "0")
    assert _resolve_gamma_ppf_threads() == default


def test_resolve_gamma_ppf_threads_clamps_absurd_values(monkeypatch):
    """An absurd thread count is clamped with a warning, not honoured.

    One task is submitted per chunk, so an unclamped value would try to spawn
    that many OS threads and fail with "can't start new thread". Modest
    oversubscription is deliberately left alone -- measured scaling is flat
    rather than harmful past the CPU count.
    """
    n_cpu = os.cpu_count() or 1
    ceiling = _GAMMA_PPF_THREAD_LIMIT_FACTOR * n_cpu

    # Modest oversubscription is honoured verbatim.
    monkeypatch.setenv("METEOR_GAMMA_PPF_THREADS", str(n_cpu + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve_gamma_ppf_threads() == n_cpu + 1

    # Absurd values are clamped, and say so.
    monkeypatch.setenv("METEOR_GAMMA_PPF_THREADS", "100000")
    with pytest.warns(RuntimeWarning, match="clamping"):
        assert _resolve_gamma_ppf_threads() == ceiling


def test_threaded_gamma_ppf_3d_never_exceeds_gridpoints():
    """Thread count is bounded by the number of gridpoints, so no worker is
    handed an empty chunk, and the result is unaffected."""
    rng = np.random.default_rng(1)
    n_time, n_spatial = 8, 5_000  # > 4096 so the threaded path is taken
    shape = rng.uniform(0.5, 5.0, size=n_spatial)
    scale = rng.uniform(0.1, 10.0, size=n_spatial)
    u = rng.uniform(0.01, 0.99, size=(n_time, n_spatial))

    expected = stats.gamma.ppf(u, a=shape[None, :], scale=scale[None, :])
    got = _threaded_gamma_ppf_3d(u, shape, scale, n_threads=n_spatial * 10)

    np.testing.assert_array_equal(expected, got)


def test_apply_distribution_transform():
    """Test applying distribution transform to data."""

    data = np.array([[0.1, 0.5, 1.0, 2.0, 3.0]])
    params_gauss = fit_distribution_parameters_1d(data, distribution="gaussian")
    params_gamma = fit_distribution_parameters_1d(data, distribution="gamma")
    transformed = apply_distribution_transform(
        data, params_gauss, params_gamma, target_dist="gamma"
    )
    assert transformed.shape == data.shape
    assert np.all(transformed >= 0)

    data_3d = np.ones((2, 2, 2)) * 4  # 3D data: (n_time, n_lat, n_lon)
    params_gauss_3d = fit_distribution_parameters_3d(data_3d, distribution="gaussian")
    params_3d = fit_distribution_parameters_3d(data_3d, distribution="gamma")
    transformed_3d = apply_distribution_transform(
        data_3d, params_gauss_3d, params_3d, target_dist="gamma"
    )
    assert transformed_3d.shape == data_3d.shape

    with pytest.raises(
        ValueError, match="Only 'gamma' distribution supported for 3D data, got weibull"
    ):
        apply_distribution_transform(
            data_3d, params_gauss_3d, params_3d, target_dist="weibull"
        )

    with pytest.raises(
        ValueError,
        match=r"For spatial parameters, expected 3D or 4D data, got shape \(2,\)",
    ):
        apply_distribution_transform(
            np.array([1.0, 2.0]), params_gauss_3d, params_3d, target_dist="gamma"
        )


def test_apply_empirical_quantile_mapping():
    """Test applying empirical quantile mapping."""

    data = np.array([[0.1, 0.5, 1.0, 2.0, 3.0]])
    reference = np.array([[0.2, 0.6, 1.5, 2.5, 4.0]])
    transformed = apply_empirical_quantile_mapping(data, reference)
    assert transformed.shape == data.shape
    assert np.all(transformed >= 0)

    data = np.ones((2, 2)) * 4  # 3D data: (n_time, n_lat, n_lon)
    reference = np.ones((2, 2)) * 5
    transformed = apply_empirical_quantile_mapping(data, reference)
    assert transformed.shape == data.shape

    data_xarray = xr.DataArray(
        np.ones((2, 2)) * 5,
        dims=("n_realisations", "n_time"),
        coords={
            "n_realisations": np.arange(2),
            "n_time": np.arange(2),
        },
    )
    transformed = apply_empirical_quantile_mapping(data_xarray, reference)
