"""Tests for the local, seasonally varying coupling of one variable's noise to another's."""

import numpy as np
import pytest
import xarray as xr

from meteor.meteor_interface import _coupling_matches
from meteor.noise_generator import MeteorNoiseGenerator, project_pcs
from meteor.portable_artifact import export_noise_model, load_noise_model

N_LAT, N_LON = 4, 5
N_SPACE = N_LAT * N_LON
LATS = np.linspace(-50, 50, N_LAT)
LONS = np.linspace(0, 280, N_LON)


def _true_slope(months):
    """pr per unit tas: -2 in January, 0 in July."""
    return -1.0 - np.cos(2 * np.pi * months / 12)


def _dataset(n_years=60, seed=0):
    """tas with spatially shared AR(1) anomalies; pr = slope(month) * tas + noise."""
    rng = np.random.default_rng(seed)
    n_time = 12 * n_years
    months = np.arange(n_time) % 12
    shared = np.zeros(n_time)
    for t in range(1, n_time):
        shared[t] = 0.6 * shared[t - 1] + rng.standard_normal()
    tas = shared[:, None] + 0.5 * rng.standard_normal((n_time, N_SPACE))
    pr = _true_slope(months)[:, None] * tas + 0.5 * rng.standard_normal(
        (n_time, N_SPACE)
    )
    cycle = 5 * np.sin(2 * np.pi * months / 12)[:, None]

    def da(x):
        return xr.DataArray(
            x.reshape(n_time, N_LAT, N_LON, 1),
            coords={"month": np.arange(n_time), "lat": LATS, "lon": LONS, "ens": [0]},
            dims=["month", "lat", "lon", "ens"],
        )

    return xr.Dataset({"tas": da(cycle + tas), "pr": da(cycle + 10 + pr)})


def _fit_pair(seasonal_scaling=True):
    ds = _dataset()
    t_glob = np.zeros(ds.sizes["month"])
    kwargs = {"n_modes": N_SPACE, "lag_order": 1, "seasonal_scaling": seasonal_scaling}
    tas = MeteorNoiseGenerator(**kwargs)
    tas.fit(ds, "tas", custom_global_temp=t_glob)
    pr = MeteorNoiseGenerator(**kwargs)
    pr.fit(ds, "pr", custom_global_temp=t_glob, coupling_parent=tas)
    return tas, pr


def _pcs(tas, pr, n_time=600, n_realizations=20, seed=1):
    rng = np.random.default_rng(seed)
    traj = np.zeros(n_time)
    tas_pcs = tas.generate_stochastic_pcs(traj, n_realizations, rng=rng)
    pr_pcs = pr.generate_stochastic_pcs(
        traj, n_realizations, rng=rng, parent_pcs=tas_pcs
    )
    return tas_pcs, pr_pcs


@pytest.mark.parametrize("seasonal_scaling", [True, False])
def test_recovers_the_seasonal_slope(seasonal_scaling):
    tas, pr = _fit_pair(seasonal_scaling)
    assert pr.coupling_parent == "tas"
    assert pr.coupling_components.shape == (N_SPACE, N_SPACE)
    tas_sd = tas.seasonal_sd if seasonal_scaling else 1.0
    slope = pr.coupling_scale / tas_sd
    np.testing.assert_allclose(
        np.median(slope, axis=1), _true_slope(np.arange(12)), atol=0.1
    )


def test_generated_pr_follows_tas_month_by_month():
    """January is strongly anti-correlated, July not at all; uncoupled is neither."""
    tas, pr = _fit_pair()
    tas_pcs, pr_pcs = _pcs(tas, pr)
    t = tas.reconstruct_anomalies(tas_pcs)[..., 7]
    p = pr.reconstruct_anomalies(pr_pcs)[..., 7]
    months = np.arange(t.shape[-1]) % 12
    corr = [
        np.corrcoef(t[:, months == m].ravel(), p[:, months == m].ravel())[0, 1]
        for m in (0, 6)
    ]
    assert corr[0] < -0.8
    assert abs(corr[1]) < 0.15

    uncoupled = MeteorNoiseGenerator(n_modes=N_SPACE, lag_order=1)
    uncoupled.fit(_dataset(), "pr", custom_global_temp=np.zeros(720))
    q = uncoupled.reconstruct_anomalies(
        uncoupled.generate_stochastic_pcs(
            np.zeros(600), 20, rng=np.random.default_rng(2)
        )
    )[..., 7]
    assert (
        abs(np.corrcoef(t[:, months == 0].ravel(), q[:, months == 0].ravel())[0, 1])
        < 0.15
    )


def test_regional_point_and_gridded_paths_agree():
    tas, pr = _fit_pair()
    _, pr_pcs = _pcs(tas, pr, n_time=36, n_realizations=2)
    realizations = pr.generate_realization(
        np.zeros(36), noise_only=True, stochastic_pcs=pr_pcs
    )
    grids = np.stack([r.values.reshape(36, -1) for r in realizations])
    w = pr.region_weight_vector(region="global")
    regional = pr.generate_regional_mean_realizations(
        np.zeros(36),
        region="global",
        noise_only=True,
        stochastic_pcs=pr_pcs[0],
        return_numpy=True,
    )
    np.testing.assert_allclose(regional, grids[0] @ w, atol=1e-10)
    anomalies = pr.reconstruct_anomalies(pr_pcs)
    np.testing.assert_allclose(
        project_pcs(pr_pcs, pr.location_projection(w)), anomalies @ w, atol=1e-10
    )
    point = pr.generate_regional_mean_realizations(
        np.zeros(36),
        lat=LATS[2],
        lon=LONS[1],
        noise_only=True,
        stochastic_pcs=pr_pcs[1],
        return_numpy=True,
    )
    np.testing.assert_allclose(point, grids[1][:, 2 * N_LON + 1], atol=1e-10)


def test_needs_the_parent_pcs():
    tas, pr = _fit_pair()
    with pytest.raises(ValueError, match="coupled to tas"):
        pr.generate_stochastic_pcs(np.zeros(24), 2)


def test_round_trips_and_cache_check(tmp_path):
    tas, pr = _fit_pair()
    _, pr_pcs = _pcs(tas, pr, n_time=24, n_realizations=2)
    expected = pr.reconstruct_anomalies(pr_pcs)

    pr.save_model(str(tmp_path / "pr.pkl"))
    pickled = MeteorNoiseGenerator()
    pickled.load_model(str(tmp_path / "pr.pkl"))
    np.testing.assert_array_equal(pickled.reconstruct_anomalies(pr_pcs), expected)

    path = str(tmp_path / "pr.nc")
    export_noise_model(pr, path, dtype=np.float64)
    with xr.open_dataset(path) as ds:
        assert ds.attrs["schema_version"] == 3
        assert ds.attrs["coupling_parent"] == "tas"
    np.testing.assert_allclose(
        load_noise_model(path).reconstruct_anomalies(pr_pcs), expected
    )

    # A cached coupled model is reused only with the parent it was fitted to
    assert _coupling_matches(pr, tas)
    assert not _coupling_matches(pr, None)
    assert not _coupling_matches(tas, tas)
    retrained = MeteorNoiseGenerator(n_modes=N_SPACE - 1, lag_order=1)
    retrained.fit(_dataset(), "tas", custom_global_temp=np.zeros(720))
    assert not _coupling_matches(pr, retrained)
