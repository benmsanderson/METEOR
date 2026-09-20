"""Tests for the portable (non-pickle) emulator artifact format."""

import numpy as np
import pytest
import xarray as xr

from meteor.noise_generator import MeteorNoiseGenerator
from meteor.portable_artifact import (
    NOISE_FORMAT,
    SCHEMA_VERSION,
    export_noise_model,
    is_portable_artifact,
    load_noise_model,
)


def _fit_small_model(use_exog="none", weight_eofs=True, n_modes=3):
    """
    Fit a noise generator on a small synthetic grid.

    Deterministic so the export round-trip can assert bitwise identity without
    depending on CMIP6 downloads.

    Parameters
    ----------
    use_exog : str
        VARX exogenous-variable setting.
    weight_eofs : bool
        Whether to area-weight the EOF basis.
    n_modes : int
        Number of PCA modes (must not exceed the gridcell count).

    Returns
    -------
    MeteorNoiseGenerator
        Fitted generator.
    """
    rng = np.random.default_rng(20240917)
    time = np.arange(240)
    lats = np.linspace(-80, 80, 5)
    lons = np.linspace(0, 300, 6)

    field = np.zeros((len(time), len(lats), len(lons), 1))
    for i, t in enumerate(time):
        seasonal = 8.0 * np.sin(2 * np.pi * t / 12.0)
        trend = 0.008 * t
        for j, lat in enumerate(lats):
            for k, lon in enumerate(lons):
                field[i, j, k, 0] = (
                    seasonal + trend + 0.08 * lat + 0.02 * lon + rng.normal(0, 0.4)
                )

    ds = xr.Dataset(
        {
            "tas": xr.DataArray(
                field,
                coords={"month": time, "lat": lats, "lon": lons, "ens": np.array([1])},
                dims=["month", "lat", "lon", "ens"],
            )
        }
    )
    model = MeteorNoiseGenerator(
        n_modes=n_modes, lag_order=2, use_exog=use_exog, weight_eofs=weight_eofs
    )
    model.fit(ds, "tas")
    return model


def _generate_all(model, seed=4242, n_time=180):
    """
    Exercise every generation path and return the raw arrays.

    Parameters
    ----------
    model : MeteorNoiseGenerator
        Fitted or artifact-loaded generator.
    seed : int
        Seed applied before each call so paths are independently reproducible.
    n_time : int
        Trajectory length in months.

    Returns
    -------
    dict of str to np.ndarray
        Generated outputs keyed by path name.
    """
    trajectory = np.linspace(0.0, 2.5, n_time)
    out = {}
    np.random.seed(seed)
    out["pcs_single"] = model.generate_stochastic_pcs(trajectory, n_realizations=1)
    np.random.seed(seed)
    out["pcs_batched"] = model.generate_stochastic_pcs(trajectory, n_realizations=4)
    np.random.seed(seed)
    out["global"] = model.generate_regional_mean_realizations(
        trajectory, region="global", n_realizations=3, return_numpy=True
    )
    np.random.seed(seed)
    out["global_noise_only"] = model.generate_regional_mean_realizations(
        trajectory,
        region="global",
        n_realizations=3,
        noise_only=True,
        return_numpy=True,
    )
    np.random.seed(seed)
    out["point"] = model.generate_regional_mean_realizations(
        trajectory, lat=40.0, lon=120.0, n_realizations=2, return_numpy=True
    )
    np.random.seed(seed)
    out["gridded"] = np.asarray(
        model.generate_realization(trajectory, n_realizations=1).values
    )
    return out


@pytest.mark.parametrize("weight_eofs", [True, False])
def test_export_reload_generates_identical_ensemble(tmp_path, weight_eofs):
    """A reloaded artifact reproduces in-memory generation bit for bit."""
    model = _fit_small_model(weight_eofs=weight_eofs)
    before = _generate_all(model)

    path = str(tmp_path / "noise.nc")
    export_noise_model(model, path, cmip6_model="TEST-ESM", training_scenario="ssp245")

    reloaded = load_noise_model(path)
    after = _generate_all(reloaded)

    assert set(before) == set(after)
    for key, expected in before.items():
        # Identical, not close: the same arrays drive the same arithmetic.
        assert np.array_equal(expected, after[key]), f"{key} differs after round-trip"


def test_reloaded_artifact_carries_no_library_objects(tmp_path):
    """Generation works with the statsmodels/sklearn objects absent."""
    model = _fit_small_model()
    path = str(tmp_path / "noise.nc")
    export_noise_model(model, path)

    reloaded = load_noise_model(path)
    assert reloaded.varx_results is None
    assert reloaded.seasonal_model is None
    assert reloaded.pca is None
    assert reloaded.fitted is True

    result = reloaded.generate_regional_mean_realizations(
        np.linspace(0, 1, 60), region="global", n_realizations=2, return_numpy=True
    )
    assert result.shape == (2, 60)
    assert np.all(np.isfinite(result))


def test_exported_artifact_metadata(tmp_path):
    """Schema version and provenance survive the round trip."""
    model = _fit_small_model()
    path = str(tmp_path / "noise.nc")
    export_noise_model(
        model,
        path,
        cmip6_model="TEST-ESM",
        training_scenario="ssp370",
        training_config={"n_modes_noise": 3, "use_exog": "none"},
    )

    with xr.open_dataset(path) as ds:
        assert ds.attrs["format"] == NOISE_FORMAT
        assert ds.attrs["schema_version"] == SCHEMA_VERSION
        assert ds.attrs["cmip6_model"] == "TEST-ESM"
        assert ds.attrs["training_scenario"] == "ssp370"
        assert "n_modes_noise" in ds.attrs["training_config"]
        assert ds.attrs["variable_name"] == "tas"
        assert ds.attrs["created"]
        assert ds.attrs["meteor_version"]
        # Arrays, not objects.
        assert set(ds.data_vars) >= {
            "varx_intercept",
            "varx_A",
            "varx_residual_cov",
            "seasonal_coef",
            "seasonal_intercept",
            "eof_components",
        }

    assert is_portable_artifact(path)


def test_exog_model_round_trips(tmp_path):
    """Models with exogenous regressors export and reload identically."""
    model = _fit_small_model(use_exog="all")
    assert model.varx_B is not None
    before = _generate_all(model)

    path = str(tmp_path / "noise_exog.nc")
    export_noise_model(model, path)
    reloaded = load_noise_model(path)
    assert reloaded.varx_B is not None
    after = _generate_all(reloaded)

    for key, expected in before.items():
        assert np.array_equal(expected, after[key]), f"{key} differs after round-trip"


def test_load_rejects_foreign_and_future_files(tmp_path):
    """Non-artifacts and newer schema versions are refused, not mis-parsed."""
    foreign = str(tmp_path / "foreign.nc")
    xr.Dataset({"x": ("t", np.arange(3.0))}).to_netcdf(foreign)
    with pytest.raises(ValueError, match="not a METEOR noise-model artifact"):
        load_noise_model(foreign)
    assert not is_portable_artifact(foreign)

    model = _fit_small_model()
    future = str(tmp_path / "future.nc")
    export_noise_model(model, future)
    with xr.open_dataset(future) as ds:
        bumped = ds.load()
    bumped.attrs["schema_version"] = SCHEMA_VERSION + 1
    bumped.to_netcdf(future)
    with pytest.raises(ValueError, match="Upgrade METEOR"):
        load_noise_model(future)


def test_export_requires_fitted_model(tmp_path):
    """Exporting an unfitted model is an explicit error."""
    with pytest.raises(ValueError, match="must be fitted"):
        export_noise_model(MeteorNoiseGenerator(), str(tmp_path / "x.nc"))


def test_is_portable_artifact_on_missing_and_pickle_paths(tmp_path):
    """The cache-routing helper tolerates absent and non-netCDF files."""
    assert not is_portable_artifact(str(tmp_path / "nope.nc"))
    junk = tmp_path / "legacy.pkl"
    junk.write_bytes(b"\x80\x04not-a-netcdf")
    assert not is_portable_artifact(str(junk))


def _make_small_pattern_model(n_modes=2, n_lat=4, n_lon=5):
    """
    Build a minimal pattern-scaling model without touching CMIP6 data.

    Parameters
    ----------
    n_modes : int
        Number of step-response modes.
    n_lat, n_lon : int
        Grid size.

    Returns
    -------
    MeteorPatternScaling
        Model with hand-filled patterns, usable for prediction.
    """
    from meteor.meteor import MeteorPatternScaling

    rng = np.random.default_rng(11)
    lat = np.linspace(-60, 60, n_lat)
    lon = np.linspace(0, 288, n_lon)

    model = MeteorPatternScaling.__new__(MeteorPatternScaling)
    model.name = "test-pattern"
    model.patternflds = {"tas": n_modes}
    model.anom_timescales = {"tas": n_modes}
    model.exp_list = ["base", "co2x4", "sulxanom"]
    model.exp_forc_dict = {"base": 0.0, "co2x4": 8.3858, "sulxanom": 1.0}
    model.dacanom = None
    model.pattern_dict = {}
    for k, exp in enumerate(model.exp_list):
        v = xr.DataArray(
            rng.normal(0, 1.0, (n_modes, n_lat, n_lon)),
            coords={"mode": np.arange(n_modes), "lat": lat, "lon": lon},
            dims=("mode", "lat", "lon"),
        )
        outp = {}
        for i in range(n_modes):
            outp[f"t{i}"] = 2.0 + 10.0 * (i + 1) + k
            outp[f"s{i}"] = 1.5 - 0.3 * i + 0.1 * k
        model.pattern_dict[exp] = {"tas": {"pattern_full": {"v": v}, "outp": outp}}
    return model


def test_pattern_scaling_round_trip_predicts_identically(tmp_path):
    """A reloaded pattern artifact predicts bit for bit like the original."""
    from meteor.portable_artifact import export_pattern_scaling, load_pattern_scaling

    model = _make_small_pattern_model()
    path = str(tmp_path / "pattern.nc")
    export_pattern_scaling(model, path, cmip6_model="TEST-ESM")

    reloaded = load_pattern_scaling(path)
    assert reloaded.exp_list == model.exp_list
    assert reloaded.patternflds == model.patternflds
    assert reloaded.exp_forc_dict == pytest.approx(model.exp_forc_dict)
    # Training-only state is deliberately absent.
    assert reloaded.dacanom is None

    forcing = np.linspace(0.0, 5.0, 120)
    for exp in ["co2x4", "sulxanom"]:
        expected = model.predict_from_forcing_profile(forcing, "tas", exp=exp)
        actual = reloaded.predict_from_forcing_profile(forcing, "tas", exp=exp)
        assert np.array_equal(
            np.asarray(expected.values), np.asarray(actual.values)
        ), f"prediction for {exp} differs after round-trip"


def test_pattern_artifact_holds_no_lmfit_objects(tmp_path):
    """The exported step response is plain numbers, not an lmfit object."""
    from meteor.portable_artifact import (
        PATTERN_FORMAT,
        export_pattern_scaling,
        load_pattern_scaling,
    )

    model = _make_small_pattern_model()
    path = str(tmp_path / "pattern.nc")
    export_pattern_scaling(model, path)

    with xr.open_dataset(path) as ds:
        assert ds.attrs["format"] == PATTERN_FORMAT
        assert ds.attrs["dacanom_included"] == 0
        assert set(ds.data_vars) >= {
            "pattern_v",
            "step_coeffs",
            "step_timescales",
            "exp_forc",
        }

    reloaded = load_pattern_scaling(path)
    outp = reloaded.pattern_dict["co2x4"]["tas"]["outp"]
    assert isinstance(outp, dict)
    assert all(isinstance(value, float) for value in outp.values())


def test_pattern_artifact_rejects_foreign_file(tmp_path):
    """A noise artifact is not mistaken for a pattern artifact."""
    from meteor.portable_artifact import load_pattern_scaling

    path = str(tmp_path / "noise.nc")
    export_noise_model(_fit_small_model(), path)
    with pytest.raises(ValueError, match="not a METEOR pattern-scaling artifact"):
        load_pattern_scaling(path)


def test_publication_metadata_is_recorded(tmp_path):
    """DOI and source URL travel with the artifact so a copy stays traceable."""
    model = _fit_small_model()
    path = str(tmp_path / "noise.nc")
    export_noise_model(
        model, path, doi="10.5281/zenodo.0000000", source_url="https://example.org/dep"
    )
    with xr.open_dataset(path) as ds:
        assert ds.attrs["doi"] == "10.5281/zenodo.0000000"
        assert ds.attrs["source_url"] == "https://example.org/dep"


def test_artifacts_without_publication_metadata_are_quiet(tmp_path, recwarn):
    """A scratch export does not nag about the working tree being dirty."""
    model = _fit_small_model()
    export_noise_model(model, str(tmp_path / "noise.nc"))
    assert not [w for w in recwarn if "uncommitted changes" in str(w.message)]
    with xr.open_dataset(str(tmp_path / "noise.nc")) as ds:
        assert ds.attrs["doi"] == ""
        assert ds.attrs["source_url"] == ""


def test_depositing_from_a_dirty_tree_warns(tmp_path):
    """Publication metadata plus an unreproducible version is flagged."""
    import meteor.portable_artifact as pa

    model = _fit_small_model()
    original = pa.__version__
    pa.__version__ = "1.6.1+6.gdeadbee.dirty"
    try:
        with pytest.warns(UserWarning, match="uncommitted changes"):
            export_noise_model(
                model, str(tmp_path / "n.nc"), doi="10.5281/zenodo.0000000"
            )
    finally:
        pa.__version__ = original
