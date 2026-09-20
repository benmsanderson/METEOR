"""
Compact per-location bundles for timeseries-only clients.

A full portable artifact (see :mod:`meteor.portable_artifact`) still carries the
EOF *maps* and the per-gridpoint seasonal coefficients, because it must be able
to reconstruct gridded fields. A client that only ever wants regional or point
time series never touches those maps, and shipping them is the difference
between tens of megabytes and tens of kilobytes.

The reduction rests on two identities, both of which are linear and therefore
exact:

* A region mean is a fixed weighted sum over gridpoints. So the region mean of
  ``X @ coef.T + intercept`` equals ``X @ (coef.T @ w) + intercept @ w`` -- nine
  seasonal coefficients and one intercept per location, not nine per gridpoint.
* The forced response is ``pcs @ v`` for step-response PCs ``pcs`` and spatial
  patterns ``v``, so its region mean is ``pcs @ (v @ w)`` -- one projected
  pattern value per mode per location.

What a bundle therefore contains, per location, is a handful of numbers; the
VARX arrays that drive the stochastic component are shared across all
locations.

Deliberately *not* in a bundle: gridded output, and the machinery to run
arbitrary emissions scenarios through the simple climate model. A bundle ships
the step-response kernel and per-scenario forcing, so a client can convolve any
forcing trajectory it likes -- including rescaled forcers -- but it cannot turn
raw emissions into forcing on its own.

See ``docs/emulator_artifact_schema.md`` for the versioned layout.
"""

import json

import numpy as np
import xarray as xr

from . import pattern_logic_lib
from .geo_data_utils import extract_point, global_mean, regional_mean
from .portable_artifact import (
    SCHEMA_VERSION,
    SEASONAL_FEATURE_NAMES,
    _provenance_attrs,
    _step_response_arrays,
)

#: ``format`` attribute identifying a compact timeseries bundle.
BUNDLE_FORMAT = "meteor-timeseries-bundle"


def parse_location(spec):
    """
    Parse a location specifier into its kind and coordinates.

    Accepts the same grammar as the ``aggregations`` argument of
    :meth:`MeteorInterface.generate_ensemble_outputs`, so a bundle can be built
    for exactly the locations a caller already asks for.

    Parameters
    ----------
    spec : str
        ``'global'``, ``'regional:<AR6 code>'``, or ``'point:<lat>,<lon>'``.

    Returns
    -------
    dict
        ``{'spec', 'kind', 'region', 'lat', 'lon'}``; ``lat``/``lon`` are NaN
        for non-point locations.

    Raises
    ------
    ValueError
        If the specifier is not recognised.

    Examples
    --------
    >>> parse_location("point:59.9,10.8")["kind"]
    'point'
    """
    if spec == "global":
        return {
            "spec": spec,
            "kind": "global",
            "region": "global",
            "lat": np.nan,
            "lon": np.nan,
        }
    if spec.startswith("regional:"):
        parts = spec.split(":")
        if len(parts) != 2 or not parts[1]:
            raise ValueError(
                f"Unsupported location {spec!r}: bundles cover 'global', "
                "'regional:<AR6 code>' and 'point:<lat>,<lon>'"
            )
        return {
            "spec": spec,
            "kind": "region",
            "region": parts[1],
            "lat": np.nan,
            "lon": np.nan,
        }
    if spec.startswith("point:"):
        lat_str, lon_str = spec.split(":", 1)[1].split(",")
        return {
            "spec": spec,
            "kind": "point",
            "region": "",
            "lat": float(lat_str),
            "lon": float(lon_str),
        }
    raise ValueError(
        f"Unsupported location {spec!r}: bundles cover 'global', "
        "'regional:<AR6 code>' and 'point:<lat>,<lon>'"
    )


def _project_field(field, location):
    """
    Reduce a gridded field to a location using the pattern-path weighting.

    Deliberately routed through :mod:`meteor.geo_data_utils` rather than the
    noise generator's own reduction: the pattern and noise paths build their
    AR6 masks differently, and a bundle must reproduce each path's own
    arithmetic to stay faithful to METEOR's output.

    Parameters
    ----------
    field : xr.DataArray
        Field with ``lat``/``lon`` dimensions.
    location : dict
        Parsed location from :func:`parse_location`.

    Returns
    -------
    np.ndarray
        Field reduced over the spatial dimensions.
    """
    if location["kind"] == "global":
        return np.asarray(global_mean(field).values)
    if location["kind"] == "region":
        return np.asarray(regional_mean(field, region_code=location["region"]).values)
    return np.asarray(extract_point(field, location["lat"], location["lon"]).values)


def _noise_weight_vector(noise_model, location):
    """
    Spatial weights for a location, using the noise path's own reduction.

    Parameters
    ----------
    noise_model : MeteorNoiseGenerator
        Fitted noise generator.
    location : dict
        Parsed location from :func:`parse_location`.

    Returns
    -------
    np.ndarray
        Weights over the stacked spatial axis, shape (n_lat * n_lon,).
    """
    if location["kind"] == "point":
        return noise_model.region_weight_vector(
            lat=location["lat"], lon=location["lon"]
        )
    return noise_model.region_weight_vector(region=location["region"])


def export_timeseries_bundle(
    noise_model,
    pattern_model,
    filepath,
    locations,
    variable=None,
    cmip6_model=None,
    training_scenario=None,
    training_config=None,
    pr_reference=None,
    pr_reference_start_year=None,
    dtype=np.float32,
):
    """
    Write a compact per-location bundle for timeseries-only clients.

    Parameters
    ----------
    noise_model : MeteorNoiseGenerator
        Fitted noise generator supplying the VARX, seasonal and EOF terms.
    pattern_model : MeteorPatternScaling
        Fitted pattern-scaling model supplying the forced-response kernel.
    filepath : str
        Destination path (``.nc``).
    locations : list of str
        Location specifiers; see :func:`parse_location`.
    variable : str, optional
        Variable name. Defaults to the noise model's ``variable_name``.
    cmip6_model, training_scenario, training_config : optional
        Provenance, recorded in the artifact.
    pr_reference : xr.DataArray, optional
        Gridded CMIP6 reference field for the precipitation transform. When
        given it is aggregated per location and stored, so a client can fit the
        1D gamma transform without re-fetching CMIP6 data at generation time.
    pr_reference_start_year : int, optional
        First calendar year of ``pr_reference``, needed to index it by year.
    dtype : np.dtype, default np.float32
        Storage precision. float32 is the default here: a bundle is a wire
        format consumed by reimplementations that will not reproduce float64
        arithmetic bit for bit anyway.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If the noise model is unfitted or a location is unrecognised.

    Examples
    --------
    >>> export_timeseries_bundle(  # doctest: +SKIP
    ...     noise, pattern, "bundle.nc", ["global", "regional:NEU"]
    ... )
    """
    if not noise_model.fitted:
        raise ValueError("Noise model must be fitted before exporting")

    parsed = [parse_location(spec) for spec in locations]
    n_loc = len(parsed)
    n_modes = noise_model.n_modes

    eof_projection = np.zeros((n_loc, n_modes))
    seasonal_coef = np.zeros((n_loc, len(noise_model.seasonal_coef[0])))
    seasonal_intercept = np.zeros(n_loc)

    physical_eofs = noise_model.physical_eof_components()
    for i, location in enumerate(parsed):
        weights = _noise_weight_vector(noise_model, location)
        eof_projection[i] = physical_eofs @ weights
        seasonal_coef[i] = noise_model.seasonal_coef.T @ weights
        seasonal_intercept[i] = noise_model.seasonal_intercept @ weights

    variable = variable or noise_model.variable_name or ""
    exps = list(pattern_model.exp_list)
    fld = variable if variable in pattern_model.patternflds else None
    if fld is None:
        fld = next(iter(pattern_model.patternflds))
    n_pattern_modes = int(pattern_model.patternflds[fld])

    pattern_projection = np.full((n_loc, len(exps), n_pattern_modes), np.nan)
    step_coeffs = np.full((len(exps), n_pattern_modes), np.nan)
    step_timescales = np.full((len(exps), n_pattern_modes), np.nan)

    for j, exp in enumerate(exps):
        entry = pattern_model.pattern_dict.get(exp, {}).get(fld)
        if not isinstance(entry, dict) or not isinstance(
            entry.get("pattern_full"), dict
        ):
            continue
        patterns = entry["pattern_full"]["v"]
        step_coeffs[j], step_timescales[j] = _step_response_arrays(
            entry["outp"], n_pattern_modes
        )
        for i, location in enumerate(parsed):
            pattern_projection[i, j] = [
                _project_field(patterns.isel(mode=k), location)
                for k in range(n_pattern_modes)
            ]

    data_vars = {
        "varx_intercept": (("mode",), np.asarray(noise_model.varx_intercept, dtype)),
        "varx_A": (("lag", "mode", "mode_in"), np.asarray(noise_model.varx_A, dtype)),
        "varx_residual_cov": (
            ("mode", "mode_in"),
            np.asarray(noise_model.varx_sigma_u, dtype),
        ),
        "eof_projection": (("location", "mode"), eof_projection.astype(dtype)),
        "seasonal_coef": (("location", "feature"), seasonal_coef.astype(dtype)),
        "seasonal_intercept": (("location",), seasonal_intercept.astype(dtype)),
        "pattern_projection": (
            ("location", "exp", "pattern_mode"),
            pattern_projection.astype(dtype),
        ),
        "step_coeffs": (("exp", "pattern_mode"), step_coeffs.astype(dtype)),
        "step_timescales": (("exp", "pattern_mode"), step_timescales.astype(dtype)),
        "exp_forc": (
            ("exp",),
            np.array(
                [float(pattern_model.exp_forc_dict[e]) for e in exps], dtype=np.float64
            ),
        ),
        "location_kind": (("location",), np.array([p["kind"] for p in parsed], "U8")),
        "location_lat": (
            ("location",),
            np.array([p["lat"] for p in parsed], dtype=np.float64),
        ),
        "location_lon": (
            ("location",),
            np.array([p["lon"] for p in parsed], dtype=np.float64),
        ),
    }
    if noise_model.varx_B is not None:
        data_vars["varx_B"] = (
            ("mode", "exog"),
            np.asarray(noise_model.varx_B, dtype),
        )

    if pr_reference is not None:
        reference = np.stack([_project_field(pr_reference, p) for p in parsed])
        data_vars["pr_reference"] = (
            ("location", "reference_month"),
            reference.astype(dtype),
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "location": ("location", np.array([p["spec"] for p in parsed], "U64")),
            "mode": ("mode", np.arange(n_modes)),
            "mode_in": ("mode_in", np.arange(n_modes)),
            "lag": ("lag", np.arange(1, noise_model.lag_order + 1)),
            "feature": (
                "feature",
                np.array(SEASONAL_FEATURE_NAMES, dtype="U32"),
            ),
            "exp": ("exp", np.array(exps, dtype="U64")),
            "pattern_mode": ("pattern_mode", np.arange(n_pattern_modes)),
        },
    )

    ds.attrs = {
        "format": BUNDLE_FORMAT,
        "variable_name": variable,
        "n_modes": int(n_modes),
        "lag_order": int(noise_model.lag_order),
        "use_exog": str(noise_model.use_exog),
        "pattern_field": fld,
        "n_locations": int(n_loc),
        "gridded_output": 0,
        "custom_emissions": 0,
        "step_response": "u_i(t) = s_i * (1 - exp(-t / tau_i)), t in years from year_0",
        "reconstruction": (
            "series = seasonal_coef @ X(t) + seasonal_intercept "
            "+ pcs(t) @ eof_projection + sum_exp pattern_projection @ "
            "convolve(step_response, dF_exp / exp_forc)"
        ),
        "pr_reference_start_year": (
            -1 if pr_reference_start_year is None else int(pr_reference_start_year)
        ),
        **_provenance_attrs(
            cmip6_model, training_scenario, training_config, created=None
        ),
    }

    ds.to_netcdf(filepath)
    return filepath


def load_timeseries_bundle(filepath):
    """
    Load a compact timeseries bundle.

    Returns the raw dataset rather than a model object: a bundle is a wire
    format, and its consumers are reimplementations that want the arrays.

    Parameters
    ----------
    filepath : str
        Path to a bundle written by :func:`export_timeseries_bundle`.

    Returns
    -------
    xr.Dataset
        The bundle contents, loaded into memory.

    Raises
    ------
    ValueError
        If the file is not a bundle or uses a newer schema version.
    """
    with xr.open_dataset(filepath) as opened:
        ds = opened.load()

    fmt = ds.attrs.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValueError(
            f"{filepath} is not a METEOR timeseries bundle "
            f"(format={fmt!r}, expected {BUNDLE_FORMAT!r})"
        )
    version = int(ds.attrs.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{filepath} uses schema version {version}, but this METEOR release "
            f"understands at most {SCHEMA_VERSION}. Upgrade METEOR to read it."
        )
    return ds


def forced_response_from_bundle(bundle, location, forcing_by_exp, year_0=1850):
    """
    Reproduce a location's forced response from bundle arrays alone.

    This is the reference implementation of the bundle's forced-term contract:
    a port to another language should match it. It uses only arrays present in
    the bundle plus the caller's forcing trajectories.

    Parameters
    ----------
    bundle : xr.Dataset
        Bundle from :func:`load_timeseries_bundle`.
    location : str
        Location specifier present in the bundle's ``location`` coordinate.
    forcing_by_exp : dict
        Mapping of experiment name to an annual forcing trajectory. Accepts the
        full mapping the simple climate model produces, including ``base``:
        experiments that are absent, mapped to None, or carry a zero step
        magnitude contribute nothing, matching how METEOR skips ``base``.
    year_0 : int, default 1850
        First calendar year of the forcing trajectories.

    Returns
    -------
    np.ndarray
        Annual forced response at the location.

    Examples
    --------
    >>> forced_response_from_bundle(  # doctest: +SKIP
    ...     bundle, "global", {"co2x4": forcing}
    ... )
    """
    loc_idx = list(bundle["location"].values).index(location)
    total = None
    for j, exp in enumerate(str(e) for e in bundle["exp"].values):
        forcing = forcing_by_exp.get(exp)
        if forcing is None:
            continue
        # A zero step magnitude means the experiment carries no forced response
        # -- this is the 'base' experiment, which METEOR's own
        # _predict_combined_experiment_from_forcer_series skips explicitly.
        # Dividing the forcing increments by it would yield NaN, so skip it
        # here too rather than poisoning the sum.
        forc_step = float(bundle["exp_forc"].values[j])
        if not np.isfinite(forc_step) or forc_step == 0.0:
            continue
        params = {}
        for k in range(bundle.sizes["pattern_mode"]):
            params[f"t{k}"] = float(bundle["step_timescales"].values[j, k])
            params[f"s{k}"] = float(bundle["step_coeffs"].values[j, k])
        if not np.isfinite(list(params.values())).all():
            continue
        pcs = pattern_logic_lib.imodel_filter(
            params,
            forcing,
            forc_step=forc_step,
            year_0=year_0,
        )
        projection = bundle["pattern_projection"].values[loc_idx, j]
        contribution = np.asarray(pcs.values) @ projection
        total = contribution if total is None else total + contribution
    if total is None:
        return np.zeros(0)
    return total


def bundle_summary(filepath):
    """
    Describe a bundle's contents and on-disk size.

    Parameters
    ----------
    filepath : str
        Path to a bundle.

    Returns
    -------
    dict
        Summary with location count, mode counts and size in bytes.
    """
    import os  # pylint: disable=import-outside-toplevel

    ds = load_timeseries_bundle(filepath)
    return {
        "variable": ds.attrs.get("variable_name", ""),
        "n_locations": int(ds.sizes["location"]),
        "n_modes": int(ds.sizes["mode"]),
        "n_pattern_modes": int(ds.sizes["pattern_mode"]),
        "experiments": [str(e) for e in ds["exp"].values],
        "has_pr_reference": "pr_reference" in ds,
        "size_bytes": os.path.getsize(filepath),
        "schema_version": int(ds.attrs.get("schema_version", 0)),
        "provenance": json.loads(ds.attrs.get("training_config", "{}")),
    }


#: ``format`` attribute identifying a golden validation fixture.
GOLDEN_FORMAT = "meteor-golden-fixture"


def export_golden_fixture(
    bundle_path,
    filepath,
    noise_model,
    t_glob,
    seed=0,
    n_realizations=2,
    forcing_by_exp=None,
    year_0=1850,
):
    """
    Write fixed-seed reference output for validating a reimplementation.

    A port of the bundle contract to another language cannot be checked against
    METEOR by running METEOR. This writes the inputs and the exact expected
    outputs as plain arrays instead, so the port can be validated offline.

    The stochastic draws come from an explicit
    :class:`numpy.random.Generator`, not the global ``numpy.random`` state, so
    the fixture is reproducible across processes. Note that a port will not
    reproduce NumPy's Mersenne/PCG streams: the PC sequence is therefore stored
    *as data*, and a port should feed it back in rather than try to regenerate
    it. The deterministic parts -- seasonal cycle, EOF projection, forced
    response -- are what a port must reproduce from the PCs.

    Parameters
    ----------
    bundle_path : str
        Path to the bundle the fixture belongs to.
    filepath : str
        Destination path (``.nc``).
    noise_model : MeteorNoiseGenerator
        Model used to draw the PC sequence.
    t_glob : array-like
        Monthly global-warming trajectory driving the fixture.
    seed : int, default 0
        Seed for the explicit generator.
    n_realizations : int, default 2
        Number of realizations to store.
    forcing_by_exp : dict, optional
        Forcing trajectories per experiment for the forced term.
    year_0 : int, default 1850
        First year of the forcing trajectories.

    Returns
    -------
    str
        The path written.
    """
    bundle = load_timeseries_bundle(bundle_path)
    t_glob = np.asarray(t_glob, dtype=np.float64)
    n_time = len(t_glob)

    pcs = noise_model.generate_stochastic_pcs(
        t_glob, n_realizations=n_realizations, rng=np.random.default_rng(seed)
    )
    pcs = np.atleast_3d(pcs) if pcs.ndim == 2 else pcs

    time = np.arange(n_time)
    design = np.vstack(
        [
            t_glob,
            np.cos(2 * np.pi * time / 12),
            np.sin(2 * np.pi * time / 12),
            np.cos(4 * np.pi * time / 12),
            np.sin(4 * np.pi * time / 12),
            t_glob * np.cos(2 * np.pi * time / 12),
            t_glob * np.sin(2 * np.pi * time / 12),
            t_glob * np.cos(4 * np.pi * time / 12),
            t_glob * np.sin(4 * np.pi * time / 12),
        ]
    ).T

    specs = [str(s) for s in bundle["location"].values]
    series = np.zeros((len(specs), n_realizations, n_time))
    forced = np.zeros((len(specs), 0))
    if forcing_by_exp:
        forced = np.stack(
            [
                forced_response_from_bundle(bundle, spec, forcing_by_exp, year_0)
                for spec in specs
            ]
        )
    for i, spec in enumerate(specs):
        seasonal = (
            design @ bundle["seasonal_coef"].values[i]
            + bundle["seasonal_intercept"].values[i]
        )
        series[i] = seasonal[None, :] + pcs @ bundle["eof_projection"].values[i]

    data_vars = {
        "t_glob": (("month",), t_glob),
        "stochastic_pcs": (("realization", "month", "mode"), pcs),
        "series": (("location", "realization", "month"), series),
    }
    if forced.size:
        data_vars["forced_response"] = (("location", "year"), forced)

    ds = xr.Dataset(
        data_vars,
        coords={
            "location": ("location", np.array(specs, dtype="U64")),
            "realization": ("realization", np.arange(n_realizations)),
            "month": ("month", time),
            "mode": ("mode", np.arange(bundle.sizes["mode"])),
        },
    )
    ds.attrs = {
        "format": GOLDEN_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "rng": "numpy.random.default_rng(seed) -> PCG64",
        "year_0": int(year_0),
        "variable_name": bundle.attrs.get("variable_name", ""),
        "cmip6_model": bundle.attrs.get("cmip6_model", ""),
        "training_scenario": bundle.attrs.get("training_scenario", ""),
        "usage": (
            "Feed stochastic_pcs and t_glob into the reimplementation; it must "
            "reproduce series (and forced_response when present)."
        ),
    }
    ds.to_netcdf(filepath)
    return filepath
