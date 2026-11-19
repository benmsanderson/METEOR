"""Tests for per-gridcell, per-calendar-month scaling of the noise."""

import numpy as np
import pytest
import xarray as xr

from meteor import noise_generator
from meteor.meteor_interface import _get_default_config
from meteor.noise_generator import MeteorNoiseGenerator, project_pcs

N_LAT, N_LON = 5, 6
LATS = np.linspace(-60, 60, N_LAT)
LONS = np.linspace(0, 300, N_LON)
N_SPACE = N_LAT * N_LON


def month_of_year(n_time):
    return np.arange(n_time) % 12


def _seasonal_spread(months):
    """Anomaly spread by calendar month: 1.6 in January, 0.4 in July."""
    return 1.0 + 0.6 * np.cos(2 * np.pi * months / 12)


def _dataset(n_years=40, seed=0):
    """AR(1) anomalies scaled by a calendar-month spread, on a mean cycle."""
    rng = np.random.default_rng(seed)
    n_time = 12 * n_years
    months = month_of_year(n_time)
    shared = np.zeros(n_time)
    local = np.zeros((n_time, N_SPACE))
    for t in range(1, n_time):
        shared[t] = 0.5 * shared[t - 1] + rng.standard_normal()
        local[t] = 0.3 * local[t - 1] + rng.standard_normal(N_SPACE)
    anomalies = (0.7 * shared[:, None] + 0.7 * local) * _seasonal_spread(months)[
        :, None
    ]
    cycle = 10 * np.sin(2 * np.pi * np.arange(n_time) / 12)
    field = cycle[:, None] + anomalies
    da = xr.DataArray(
        field.reshape(n_time, N_LAT, N_LON, 1),
        coords={"month": np.arange(n_time), "lat": LATS, "lon": LONS, "ens": [0]},
        dims=["month", "lat", "lon", "ens"],
    )
    return xr.Dataset({"tas": da})


def _fit(seasonal_scaling, n_modes=N_SPACE, **fit_kwargs):
    ds = _dataset(**fit_kwargs.pop("data", {}))
    model = MeteorNoiseGenerator(
        n_modes=n_modes, lag_order=1, seasonal_scaling=seasonal_scaling
    )
    model.fit(ds, "tas", custom_global_temp=np.zeros(ds.sizes["month"]), **fit_kwargs)
    return model


def _monthly_sd(series):
    """Spread of (realization, month) series for each calendar month."""
    months = month_of_year(series.shape[-1])
    return np.array([series[..., months == m].std() for m in range(12)])


def test_defaults_on_except_for_pr():
    assert MeteorNoiseGenerator().seasonal_scaling is True
    assert _get_default_config("tas")["seasonal_scaling"] is True
    assert _get_default_config("pr")["seasonal_scaling"] is False
    assert _fit(seasonal_scaling=False).seasonal_sd is None


def test_scaling_reproduces_seasonal_cycle_of_variance():
    """With scaling, January stays ~4x as variable as July; without, it does not."""
    traj = np.zeros(240)
    ratios = {}
    for flag in (False, True):
        model = _fit(seasonal_scaling=flag)
        rng = np.random.default_rng(3)
        pcs = model.generate_stochastic_pcs(traj, n_realizations=200, rng=rng)
        point = model.generate_regional_mean_realizations(
            traj,
            lat=LATS[2],
            lon=LONS[3],
            stochastic_pcs=pcs,
            noise_only=True,
            return_numpy=True,
        )
        sd = _monthly_sd(point - point.mean(axis=0, keepdims=True))
        ratios[flag] = sd[0] / sd[6]
    # Training spread ratio is 1.6 / 0.4 = 4
    assert ratios[True] == pytest.approx(4.0, rel=0.2)
    assert ratios[False] < 1.5


def test_constant_offset_does_not_leak_into_the_basis():
    """A baseline offset is removed before dividing by a month-dependent scale."""
    plain = _fit(seasonal_scaling=True)
    shifted = _fit(seasonal_scaling=True, picontrol_baseline=287.0)
    np.testing.assert_allclose(shifted.seasonal_sd, plain.seasonal_sd)
    np.testing.assert_allclose(
        np.abs(shifted.eof_components), np.abs(plain.eof_components), atol=1e-8
    )


def test_regional_point_and_gridded_paths_agree():
    """Every path applies the same per-month rescaling."""
    model = _fit(seasonal_scaling=True, n_modes=8)
    traj = np.linspace(0, 1, 36)
    pcs = model.generate_stochastic_pcs(
        traj, n_realizations=2, rng=np.random.default_rng(1)
    )
    grids = model.generate_realization(traj, noise_only=True, stochastic_pcs=pcs)
    for i, grid in enumerate(grids):
        values = grid.values.reshape(len(traj), -1)
        w = model.region_weight_vector(region="global")
        regional = model.generate_regional_mean_realizations(
            traj,
            region="global",
            noise_only=True,
            stochastic_pcs=pcs[i],
            return_numpy=True,
        )
        np.testing.assert_allclose(regional, values @ w, rtol=1e-10, atol=1e-10)
        point = model.generate_regional_mean_realizations(
            traj,
            lat=LATS[1],
            lon=LONS[4],
            noise_only=True,
            stochastic_pcs=pcs[i],
            return_numpy=True,
        )
        np.testing.assert_allclose(point, grid.values[:, 1, 4], rtol=1e-10)

    # The anomaly part alone is reconstruct_anomalies, month by month.
    anomalies = model.reconstruct_anomalies(pcs)
    projection = model.location_projection(model.region_weight_vector(region="global"))
    assert projection.shape == (12, 8)
    np.testing.assert_allclose(
        project_pcs(pcs, projection),
        anomalies @ model.region_weight_vector(region="global"),
        rtol=1e-10,
        atol=1e-12,
    )


def test_cell_without_variance_reconstructs_as_zero():
    ds = _dataset()
    values = ds["tas"].values.copy()
    months = month_of_year(values.shape[0])
    # Cell (0, 0) never varies in July (a dry season, say)
    values[months == 6, 0, 0, 0] = 1.0
    ds["tas"].values[:] = values
    model = MeteorNoiseGenerator(n_modes=8, lag_order=1, seasonal_scaling=True)
    model.fit(ds, "tas", custom_global_temp=np.zeros(values.shape[0]))
    assert model.seasonal_sd[6, 0] == pytest.approx(0.0, abs=1e-12)
    pcs = model.generate_stochastic_pcs(
        np.zeros(24), n_realizations=2, rng=np.random.default_rng(0)
    )
    anomalies = model.reconstruct_anomalies(pcs)
    assert np.isfinite(anomalies).all()
    assert np.all(anomalies[:, month_of_year(24) == 6, 0] == 0.0)


def test_pickle_round_trip_and_cache_validation(tmp_path):
    model = _fit(seasonal_scaling=True, n_modes=6)
    path = str(tmp_path / "m.pkl")
    model.save_model(path)

    loaded = MeteorNoiseGenerator()
    loaded.load_model(path)
    assert loaded.seasonal_scaling is True
    np.testing.assert_array_equal(loaded.seasonal_sd, model.seasonal_sd)
    traj = np.zeros(48)
    a = model.generate_regional_mean_realizations(
        traj, region="global", n_realizations=1, random_seed=5, return_numpy=True
    )
    b = loaded.generate_regional_mean_realizations(
        traj, region="global", n_realizations=1, random_seed=5, return_numpy=True
    )
    np.testing.assert_array_equal(a, b)

    ok, _, info = noise_generator.validate_noise_model_cache(
        path, "tas", n_modes=6, lag_order=1, seasonal_scaling=False
    )
    assert not ok and "seasonal_scaling mismatch" in info["message"]
    ok, _, _ = noise_generator.validate_noise_model_cache(
        path, "tas", n_modes=6, lag_order=1, seasonal_scaling=True
    )
    assert ok
