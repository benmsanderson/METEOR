"""Tests for the compact per-location timeseries bundle."""

import numpy as np
import pytest
import xarray as xr

from meteor.meteor import MeteorPatternScaling
from meteor.noise_generator import MeteorNoiseGenerator
from meteor.timeseries_bundle import (
    BUNDLE_FORMAT,
    bundle_summary,
    export_timeseries_bundle,
    forced_response_from_bundle,
    load_timeseries_bundle,
    parse_location,
)

# 5-degree global grid: coarse enough to keep the tests fast, fine enough that
# the AR6 regions used below actually contain gridpoints.
N_LAT, N_LON = 36, 72
LATS = np.linspace(-87.5, 87.5, N_LAT)
LONS = np.linspace(2.5, 357.5, N_LON)
N_SPACE = N_LAT * N_LON
N_MODES = 4
N_PATTERN_MODES = 3


def _make_noise_model(seed=5):
    """
    Build a noise generator directly from arrays, skipping the fit.

    The bundle tests exercise the linear reduction from gridded state to
    per-location coefficients, which does not depend on how the state was
    fitted -- so the arrays are synthesised rather than trained, keeping the
    tests fast and free of CMIP6 access.

    Parameters
    ----------
    seed : int
        Seed for the synthetic arrays.

    Returns
    -------
    MeteorNoiseGenerator
        Generator with fitted state populated.
    """
    rng = np.random.default_rng(seed)
    model = MeteorNoiseGenerator(
        n_modes=N_MODES, lag_order=2, use_exog="none", weight_eofs=False
    )
    model.coords = {"lat": LATS, "lon": LONS}
    model.varx_params = rng.normal(0, 0.04, (1 + 2 * N_MODES, N_MODES))
    model.varx_sigma_u = np.eye(N_MODES) * 0.25
    model._decompose_varx_params()  # pylint: disable=protected-access
    model.seasonal_coef = rng.normal(0, 1.0, (N_SPACE, 9))
    model.seasonal_intercept = rng.normal(285.0, 4.0, N_SPACE)
    model.eof_components = rng.normal(0, 0.02, (N_MODES, N_SPACE))
    model.eof_weights = None
    model.variable_name = "tas"
    model.fitted = True
    return model


def _make_pattern_model(seed=6):
    """
    Build a pattern-scaling model on the same grid, without training data.

    Parameters
    ----------
    seed : int
        Seed for the synthetic patterns.

    Returns
    -------
    MeteorPatternScaling
        Model usable for prediction.
    """
    rng = np.random.default_rng(seed)
    model = MeteorPatternScaling.__new__(MeteorPatternScaling)
    model.name = "test-pattern"
    model.patternflds = {"tas": N_PATTERN_MODES}
    model.anom_timescales = {"tas": N_PATTERN_MODES}
    model.exp_list = ["base", "co2x4", "sulxanom"]
    model.exp_forc_dict = {"base": 0.0, "co2x4": 8.3858, "sulxanom": 1.0}
    model.dacanom = None
    model.pattern_dict = {}
    for k, exp in enumerate(model.exp_list):
        patterns = xr.DataArray(
            rng.normal(0, 1.0, (N_PATTERN_MODES, N_LAT, N_LON)),
            coords={"mode": np.arange(N_PATTERN_MODES), "lat": LATS, "lon": LONS},
            dims=("mode", "lat", "lon"),
        )
        outp = {}
        for i in range(N_PATTERN_MODES):
            outp[f"t{i}"] = 2.0 + 10.0 * (i + 1) + k
            outp[f"s{i}"] = 1.5 - 0.3 * i + 0.1 * k
        model.pattern_dict[exp] = {
            "tas": {"pattern_full": {"v": patterns}, "outp": outp}
        }
    return model


def _harmonic_features(n_time, t_glob):
    """
    Rebuild the harmonic design matrix from bundle-documented terms only.

    Mirrors what a non-Python client must implement from the schema, rather
    than calling METEOR's private helper.

    Parameters
    ----------
    n_time : int
        Number of months.
    t_glob : np.ndarray
        Global mean temperature trajectory.

    Returns
    -------
    np.ndarray
        Design matrix, shape (n_time, 9).
    """
    time = np.arange(n_time)
    annual_cos = np.cos(2 * np.pi * time / 12)
    annual_sin = np.sin(2 * np.pi * time / 12)
    semi_cos = np.cos(4 * np.pi * time / 12)
    semi_sin = np.sin(4 * np.pi * time / 12)
    return np.vstack(
        [
            t_glob,
            annual_cos,
            annual_sin,
            semi_cos,
            semi_sin,
            t_glob * annual_cos,
            t_glob * annual_sin,
            t_glob * semi_cos,
            t_glob * semi_sin,
        ]
    ).T


LOCATIONS = ["global", "regional:NEU", "regional:SEA", "point:59.9,10.8"]


def test_parse_location_grammar():
    """Location specifiers follow the aggregation grammar used elsewhere."""
    assert parse_location("global")["kind"] == "global"
    assert parse_location("regional:NEU")["region"] == "NEU"
    point = parse_location("point:59.9,10.8")
    assert point["kind"] == "point"
    assert point["lat"] == pytest.approx(59.9)
    assert point["lon"] == pytest.approx(10.8)
    for bad in ["", "region:NEU", "regional:", "somewhere"]:
        with pytest.raises(ValueError, match="Unsupported location"):
            parse_location(bad)


def test_bundle_reproduces_meteor_regional_series(tmp_path):
    """Per-location coefficients reproduce METEOR's own regional output."""
    noise = _make_noise_model()
    pattern = _make_pattern_model()
    path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(
        noise, pattern, path, LOCATIONS, variable="tas", dtype=np.float64
    )
    bundle = load_timeseries_bundle(path)

    n_time = 240
    t_glob = np.linspace(0.2, 3.0, n_time)
    np.random.seed(17)
    pcs = noise.generate_stochastic_pcs(t_glob, n_realizations=3)
    design = _harmonic_features(n_time, t_glob)

    for spec in LOCATIONS:
        location = parse_location(spec)
        kwargs = (
            {"lat": location["lat"], "lon": location["lon"]}
            if location["kind"] == "point"
            else {"region": location["region"]}
        )
        expected = noise.generate_regional_mean_realizations(
            t_glob, n_realizations=3, stochastic_pcs=pcs, return_numpy=True, **kwargs
        )
        idx = list(bundle["location"].values).index(spec)
        seasonal = (
            design @ bundle["seasonal_coef"].values[idx]
            + bundle["seasonal_intercept"].values[idx]
        )
        actual = seasonal[None, :] + pcs @ bundle["eof_projection"].values[idx]
        # float64 storage: the reduction itself is exact, so only the
        # summation order of the reduced matmul can differ.
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_bundle_forced_term_matches_pattern_model(tmp_path):
    """The bundled kernel reproduces the pattern model's forced response."""
    from meteor.geo_data_utils import global_mean, regional_mean

    noise = _make_noise_model()
    pattern = _make_pattern_model()
    path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(
        noise, pattern, path, LOCATIONS, variable="tas", dtype=np.float64
    )
    bundle = load_timeseries_bundle(path)

    forcing = np.linspace(0.0, 6.0, 160)
    gridded = pattern.predict_from_forcing_profile(
        forcing, "tas", exp="co2x4", year_0=1850
    )

    for spec in ["global", "regional:NEU", "regional:SEA"]:
        actual = forced_response_from_bundle(
            bundle, spec, {"co2x4": forcing}, year_0=1850
        )
        if spec == "global":
            expected = global_mean(gridded).values
        else:
            expected = regional_mean(gridded, region_code=spec.split(":")[1]).values
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_bundle_float32_default_is_accurate_enough(tmp_path):
    """The default float32 wire format stays within single precision."""
    noise = _make_noise_model()
    pattern = _make_pattern_model()
    path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(noise, pattern, path, LOCATIONS, variable="tas")
    bundle = load_timeseries_bundle(path)

    idx = list(bundle["location"].values).index("regional:NEU")
    expected = noise._get_regional_eof_projection(
        "NEU"
    )  # pylint: disable=protected-access
    actual = bundle["eof_projection"].values[idx]
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_bundle_is_small_and_excludes_gridded_state(tmp_path):
    """A many-location bundle stays in the tens of kilobytes."""
    noise = _make_noise_model()
    pattern = _make_pattern_model()
    specs = ["global"] + [
        f"regional:{code}"
        for code in ["NEU", "WCE", "MED", "SEA", "WNA", "CNA", "ENA", "NWS", "SES"]
    ]
    path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(noise, pattern, path, specs, variable="tas")

    summary = bundle_summary(path)
    assert summary["n_locations"] == len(specs)
    assert summary["size_bytes"] < 100 * 1024

    bundle = load_timeseries_bundle(path)
    # The EOF maps and per-gridpoint seasonal coefficients are the whole point
    # of the reduction; they must not leak into the bundle.
    assert "eof_components" not in bundle
    assert "lat" not in bundle.dims
    assert "lon" not in bundle.dims
    assert bundle.attrs["gridded_output"] == 0
    assert bundle.attrs["custom_emissions"] == 0
    assert bundle.attrs["format"] == BUNDLE_FORMAT


def test_bundle_rejects_foreign_file(tmp_path):
    """A non-bundle netCDF is refused rather than mis-parsed."""
    path = str(tmp_path / "foreign.nc")
    xr.Dataset({"x": ("t", np.arange(3.0))}).to_netcdf(path)
    with pytest.raises(ValueError, match="not a METEOR timeseries bundle"):
        load_timeseries_bundle(path)


def test_region_weight_vector_rejects_unresolvable_region():
    """A region with no gridpoints fails loudly instead of returning NaN."""
    coarse = _make_noise_model()
    coarse.coords = {"lat": np.array([-45.0, 45.0]), "lon": np.array([0.0, 180.0])}
    with pytest.raises(ValueError, match="covers no gridpoints"):
        coarse.region_weight_vector(region="NEU")


def test_golden_fixture_is_reproducible_and_self_consistent(tmp_path):
    """A fixture replays exactly, and its series follow from its own arrays."""
    from meteor.timeseries_bundle import GOLDEN_FORMAT, export_golden_fixture

    noise = _make_noise_model()
    pattern = _make_pattern_model()
    bundle_path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(
        noise, pattern, bundle_path, LOCATIONS, variable="tas", dtype=np.float64
    )

    n_time = 120
    t_glob = np.linspace(0.3, 2.4, n_time)
    forcing = {"co2x4": np.linspace(0.0, 5.0, 100)}

    first = str(tmp_path / "golden_a.nc")
    second = str(tmp_path / "golden_b.nc")
    for path in (first, second):
        export_golden_fixture(
            bundle_path,
            path,
            noise,
            t_glob,
            seed=7,
            n_realizations=2,
            forcing_by_exp=forcing,
            # float64 so the self-consistency assertion below can be exact;
            # the default float32 is covered separately.
            dtype=np.float64,
        )

    with xr.open_dataset(first) as ds_a, xr.open_dataset(second) as ds_b:
        fixture, repeat = ds_a.load(), ds_b.load()

    assert fixture.attrs["format"] == GOLDEN_FORMAT
    assert fixture.attrs["seed"] == 7
    # Same seed, same fixture: the explicit generator makes this hold across
    # processes, unlike the global numpy.random state.
    assert np.array_equal(
        fixture["stochastic_pcs"].values, repeat["stochastic_pcs"].values
    )
    assert np.array_equal(fixture["series"].values, repeat["series"].values)

    # The stored series must be derivable from the stored PCs and the bundle,
    # which is exactly what a reimplementation has to reproduce.
    bundle = load_timeseries_bundle(bundle_path)
    design = _harmonic_features(n_time, t_glob)
    pcs = fixture["stochastic_pcs"].values
    for i, spec in enumerate(str(s) for s in fixture["location"].values):
        idx = list(bundle["location"].values).index(spec)
        seasonal = (
            design @ bundle["seasonal_coef"].values[idx]
            + bundle["seasonal_intercept"].values[idx]
        )
        rebuilt = seasonal[None, :] + pcs @ bundle["eof_projection"].values[idx]
        np.testing.assert_allclose(
            rebuilt, fixture["series"].values[i], rtol=1e-12, atol=1e-10
        )

    assert "forced_response" in fixture
    assert fixture["forced_response"].shape == (len(LOCATIONS), 100)


def test_golden_fixture_defaults_to_float32(tmp_path):
    """The default precision keeps a fixture small and still self-consistent.

    A fixture validates a port that reads float32 bundle arrays, so it cannot
    be held to a tighter tolerance than single precision -- and the PC array
    dominates the file size.
    """
    from meteor.timeseries_bundle import export_golden_fixture

    noise = _make_noise_model()
    pattern = _make_pattern_model()
    bundle_path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(
        noise, pattern, bundle_path, LOCATIONS, variable="tas", dtype=np.float64
    )

    n_time = 120
    t_glob = np.linspace(0.3, 2.4, n_time)
    path = str(tmp_path / "golden32.nc")
    export_golden_fixture(bundle_path, path, noise, t_glob, seed=7, n_realizations=2)

    with xr.open_dataset(path) as opened:
        fixture = opened.load()
    assert fixture["stochastic_pcs"].dtype == np.float32
    assert fixture["series"].dtype == np.float32

    bundle = load_timeseries_bundle(bundle_path)
    design = _harmonic_features(n_time, t_glob)
    pcs = fixture["stochastic_pcs"].values.astype(np.float64)
    for i, spec in enumerate(str(s) for s in fixture["location"].values):
        idx = list(bundle["location"].values).index(spec)
        seasonal = (
            design @ bundle["seasonal_coef"].values[idx]
            + bundle["seasonal_intercept"].values[idx]
        )
        rebuilt = seasonal[None, :] + pcs @ bundle["eof_projection"].values[idx]
        np.testing.assert_allclose(
            rebuilt, fixture["series"].values[i], rtol=1e-6, atol=1e-4
        )


def test_bundle_bakes_pr_reference_per_location(tmp_path):
    """The pr reference is stored already aggregated, so pr needs no network."""
    noise = _make_noise_model()
    pattern = _make_pattern_model()

    rng = np.random.default_rng(23)
    n_month = 600
    reference = xr.DataArray(
        rng.gamma(shape=3.0, scale=0.5, size=(n_month, N_LAT, N_LON)),
        coords={"month": np.arange(n_month), "lat": LATS, "lon": LONS},
        dims=("month", "lat", "lon"),
    )

    path = str(tmp_path / "bundle_pr.nc")
    export_timeseries_bundle(
        noise,
        pattern,
        path,
        LOCATIONS,
        variable="pr",
        pr_reference=reference,
        pr_reference_start_year=1850,
        dtype=np.float64,
    )
    bundle = load_timeseries_bundle(path)

    assert "pr_reference" in bundle
    assert bundle.attrs["pr_reference_start_year"] == 1850
    assert bundle["pr_reference"].shape == (len(LOCATIONS), n_month)

    from meteor.geo_data_utils import extract_point, global_mean, regional_mean

    for i, spec in enumerate(LOCATIONS):
        location = parse_location(spec)
        if location["kind"] == "global":
            expected = global_mean(reference).values
        elif location["kind"] == "region":
            expected = regional_mean(reference, region_code=location["region"]).values
        else:
            expected = extract_point(reference, location["lat"], location["lon"]).values
        np.testing.assert_allclose(
            bundle["pr_reference"].values[i], expected, rtol=1e-12
        )


def test_bundle_omits_pr_reference_when_not_supplied(tmp_path):
    """tas bundles carry no precipitation reference and no start year."""
    noise = _make_noise_model()
    pattern = _make_pattern_model()
    path = str(tmp_path / "bundle_tas.nc")
    export_timeseries_bundle(noise, pattern, path, LOCATIONS, variable="tas")
    bundle = load_timeseries_bundle(path)
    assert "pr_reference" not in bundle
    assert bundle.attrs["pr_reference_start_year"] == -1


def test_forced_response_tolerates_zero_step_experiments(tmp_path):
    """The full forcing mapping, including a zero-step 'base', is accepted.

    METEOR's own _predict_combined_experiment_from_forcer_series skips 'base'
    explicitly. The simple climate model still returns an entry for it, with a
    zero step magnitude, and dividing the forcing increments by that yields
    NaN. Passing a hand-picked subset of experiments hides this; passing what
    the model actually produces does not.
    """
    noise = _make_noise_model()
    pattern = _make_pattern_model()
    path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(
        noise, pattern, path, LOCATIONS, variable="tas", dtype=np.float64
    )
    bundle = load_timeseries_bundle(path)
    assert float(bundle["exp_forc"].sel(exp="base").values) == 0.0

    forcing = np.linspace(0.0, 5.0, 120)
    full = {
        "base": np.zeros_like(forcing),
        "co2x4": forcing,
        "sulxanom": -0.3 * forcing,
    }
    subset = {"co2x4": forcing, "sulxanom": -0.3 * forcing}

    with_base = forced_response_from_bundle(bundle, "global", full, year_0=1850)
    without_base = forced_response_from_bundle(bundle, "global", subset, year_0=1850)

    assert np.all(np.isfinite(with_base)), "zero-step experiment produced NaN"
    # 'base' carries no forced response, so including it must change nothing.
    np.testing.assert_allclose(with_base, without_base, rtol=1e-12, atol=1e-12)


def test_bundle_ships_cholesky_of_innovation_covariance(tmp_path):
    """Clients get the factor they need to draw correlated shocks."""
    noise = _make_noise_model()
    pattern = _make_pattern_model()
    path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(
        noise, pattern, path, LOCATIONS, variable="tas", dtype=np.float64
    )
    bundle = load_timeseries_bundle(path)

    assert "varx_residual_chol" in bundle
    chol = bundle["varx_residual_chol"].values
    # Lower triangular, and reconstructs the covariance.
    assert np.allclose(np.triu(chol, k=1), 0.0)
    np.testing.assert_allclose(
        chol @ chol.T, bundle["varx_residual_cov"].values, rtol=1e-10, atol=1e-12
    )


def test_bundle_documents_annual_to_monthly_rule(tmp_path):
    """The expansion rule a client must implement is stated in the artifact."""
    noise = _make_noise_model()
    pattern = _make_pattern_model()
    path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(noise, pattern, path, LOCATIONS, variable="tas")
    bundle = load_timeseries_bundle(path)
    assert "repeated" in bundle.attrs["annual_to_monthly"]


def test_forcing_from_bundle_errors_without_scenarios(tmp_path):
    """A bundle exported without scenarios says so, rather than returning junk."""
    from meteor.timeseries_bundle import forcing_from_bundle

    noise = _make_noise_model()
    pattern = _make_pattern_model()
    path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(noise, pattern, path, LOCATIONS, variable="tas")
    bundle = load_timeseries_bundle(path)

    assert bundle.attrs["forcing_year_start"] == -1
    assert "scenario_forcing" not in bundle
    with pytest.raises(KeyError, match="no scenario forcing"):
        forcing_from_bundle(bundle, "ssp245")


def test_bundle_with_scenario_forcing_closes_the_loop(tmp_path):
    """A bundle carrying forcing needs nothing external to drive a scenario."""
    from meteor.timeseries_bundle import forcing_from_bundle

    noise = _make_noise_model()
    pattern = _make_pattern_model()
    path = str(tmp_path / "bundle_scen.nc")
    export_timeseries_bundle(
        noise,
        pattern,
        path,
        LOCATIONS,
        variable="tas",
        scenarios=["ssp245"],
        dtype=np.float64,
    )
    bundle = load_timeseries_bundle(path)

    assert bundle["scenario_forcing"].shape == (1, len(pattern.exp_list), 351)
    assert bundle.attrs["forcing_year_start"] == 1750

    forcing = forcing_from_bundle(bundle, "ssp245")
    assert set(forcing) <= set(pattern.exp_list)

    # The zero-step 'base' experiment is present in the mapping and must not
    # poison the result.
    series = forced_response_from_bundle(
        bundle, "global", forcing, year_0=bundle.attrs["forcing_year_start"]
    )
    assert series.shape == (351,)
    assert np.all(np.isfinite(series))
    assert np.any(series != 0)

    with pytest.raises(KeyError, match="not in bundle"):
        forcing_from_bundle(bundle, "ssp585")


def test_pr_reference_drops_singleton_ensemble_dim(tmp_path):
    """CMIP6 composites carry an 'ens' axis that must not leak into the bundle.

    _load_transform_reference returns dims ('ens', 'month', 'lat', 'lon') with
    ens of length 1. The spatial reduction preserves it, so the projected
    series is (1, n_month) rather than (n_month,).
    """
    from meteor.timeseries_bundle import forcing_from_bundle  # noqa: F401

    noise = _make_noise_model()
    pattern = _make_pattern_model()
    rng = np.random.default_rng(77)
    n_month = 240
    reference = xr.DataArray(
        rng.gamma(3.0, 0.5, size=(1, n_month, N_LAT, N_LON)),
        coords={
            "ens": [1],
            "month": np.arange(n_month),
            "lat": LATS,
            "lon": LONS,
        },
        dims=("ens", "month", "lat", "lon"),
    )
    path = str(tmp_path / "bundle_pr.nc")
    export_timeseries_bundle(
        noise,
        pattern,
        path,
        LOCATIONS,
        variable="pr",
        pr_reference=reference,
        pr_reference_start_year=1850,
        dtype=np.float64,
    )
    bundle = load_timeseries_bundle(path)
    assert bundle["pr_reference"].shape == (len(LOCATIONS), n_month)
    assert "ens" not in bundle.dims


def test_pr_reference_rejects_real_ensemble(tmp_path):
    """A genuine multi-member reference is refused, not silently collapsed."""
    noise = _make_noise_model()
    pattern = _make_pattern_model()
    rng = np.random.default_rng(78)
    reference = xr.DataArray(
        rng.gamma(3.0, 0.5, size=(3, 60, N_LAT, N_LON)),
        coords={"ens": [1, 2, 3], "month": np.arange(60), "lat": LATS, "lon": LONS},
        dims=("ens", "month", "lat", "lon"),
    )
    with pytest.raises(ValueError, match="expected a single time series"):
        export_timeseries_bundle(
            noise,
            pattern,
            str(tmp_path / "b.nc"),
            LOCATIONS,
            variable="pr",
            pr_reference=reference,
            pr_reference_start_year=1850,
        )


def test_golden_fixture_inherits_bundle_provenance(tmp_path):
    """A fixture is traceable to the same deposit as its bundle."""
    from meteor.timeseries_bundle import export_golden_fixture

    noise = _make_noise_model()
    pattern = _make_pattern_model()
    bundle_path = str(tmp_path / "bundle.nc")
    export_timeseries_bundle(
        noise,
        pattern,
        bundle_path,
        LOCATIONS,
        variable="tas",
        cmip6_model="TEST-ESM",
        source_url="https://example.org/deposit",
        dtype=np.float64,
    )
    fixture_path = str(tmp_path / "golden.nc")
    export_golden_fixture(
        bundle_path, fixture_path, noise, np.linspace(0.3, 2.0, 60), seed=3
    )

    with xr.open_dataset(fixture_path) as ds:
        assert ds.attrs["source_url"] == "https://example.org/deposit"
        assert ds.attrs["cmip6_model"] == "TEST-ESM"
        assert ds.attrs["meteor_version"]
        assert ds.attrs["created"]
        assert "doi" in ds.attrs
