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
from scipy.stats import gamma as _gamma
from scipy.stats import norm as _norm

from . import __version__, pattern_logic_lib, scm_forcer_engine
from .geo_data_utils import extract_point, global_mean, regional_mean
from .portable_artifact import (
    SCHEMA_VERSION,
    SEASONAL_FEATURE_NAMES,
    _provenance_attrs,
    _step_response_arrays,
)
from .precipitation_transform import (
    _month_of_year_indices,
    fit_distribution_parameters_1d_seasonal,
)
from .scm_input_lib import load_emissions_concentrations_from_name

#: ``format`` attribute identifying a compact timeseries bundle.
BUNDLE_FORMAT = "meteor-timeseries-bundle"

#: On-disk netCDF flavour for the wire formats. Classic netCDF-3 rather than
#: the HDF5-backed NETCDF4, because the intended consumers are browsers: a
#: classic file is readable by a few-kilobyte JavaScript parser, whereas HDF5
#: needs a one-to-two megabyte WebAssembly build of libhdf5 before a single
#: byte can be read. Every numeric array round-trips bit-for-bit, and the
#: files come out smaller.
WIRE_NETCDF_FORMAT = "NETCDF3_64BIT"


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


def compute_scenario_forcing(pattern_model, scenarios):
    """
    Run the simple climate model to get per-experiment forcing per scenario.

    This is the one step a browser client cannot perform for itself: turning
    emissions and concentrations into effective radiative forcing requires
    CICERO-SCM. Running it here, at export time, lets the bundle ship the
    resulting trajectories so the client only has to convolve them with the
    step-response kernel.

    Parameters
    ----------
    pattern_model : MeteorPatternScaling
        Supplies the experiment list the forcing is split across.
    scenarios : list of str or dict
        Either scenario names to load from the shipped inputs
        (e.g. ``['ssp126', 'ssp245', 'ssp585']``), or a mapping of name to
        ``(emissions, concentrations)`` dataframes already in hand.

        The mapping form exists for scenarios METEOR does not ship and cannot:
        CMIP7's ScenarioMIP emissions, for instance, are third-party data that
        a user must obtain themselves and that this package has no right to
        redistribute. Supplying them here means the resulting *forcing* can be
        bundled without the emissions ever being redistributed.

    Returns
    -------
    tuple
        ``(forcing, year_start)`` where ``forcing`` has shape
        ``(n_scenarios, n_exps, n_years)`` with NaN for absent experiments,
        and ``year_start`` is the first calendar year of the trajectories.

    Raises
    ------
    ValueError
        If the scenarios do not share a common start year or length.

    Examples
    --------
    >>> compute_scenario_forcing(pattern, ["ssp245"])  # doctest: +SKIP
    >>> compute_scenario_forcing(  # doctest: +SKIP
    ...     pattern, {"my-scenario": (emissions_df, concentrations_df)}
    ... )
    """
    exps = list(pattern_model.exp_list)
    # A mapping supplies its own inputs; a sequence names inputs we ship.
    supplied = scenarios if isinstance(scenarios, dict) else {}
    names = list(scenarios)
    per_scenario = []
    year_start = None
    n_years = None
    for scenario in names:
        if scenario in supplied:
            emissions, concentrations = supplied[scenario]
        else:
            emissions, concentrations = load_emissions_concentrations_from_name(
                scenario
            )
        start = int(emissions.index[0])
        if year_start is None:
            year_start = start
        elif start != year_start:
            raise ValueError(
                f"scenario {scenario!r} starts at {start}, but {names[0]!r} "
                f"starts at {year_start}; bundles need a common year axis"
            )
        engine = scm_forcer_engine.ScmEngineForPatternScaling(
            {
                "conc_run": False,
                "nystart": start,
                "emstart": start + 100,
                "nyend": 2100,
                "concentrations_data": concentrations,
                "emissions_data": emissions,
            }
        )
        series = engine.run_and_return_per_forcer_results(exps)
        lengths = {len(np.asarray(v)) for v in series.values() if v is not None}
        if len(lengths) != 1:
            raise ValueError(
                f"scenario {scenario!r} returned ragged forcing: {lengths}"
            )
        length = lengths.pop()
        if n_years is None:
            n_years = length
        elif length != n_years:
            raise ValueError(
                f"scenario {scenario!r} has {length} years, expected {n_years}"
            )
        block = np.full((len(exps), length), np.nan)
        for j, exp in enumerate(exps):
            if series.get(exp) is not None:
                block[j] = np.asarray(series[exp], dtype=np.float64)
        per_scenario.append(block)
    return np.stack(per_scenario), year_start


def _gamma_probability_grid(n_body=101, n_tail=50):
    """
    Probability grid for the shared gamma quantile table.

    Uniform through the body, geometrically refined into both tails. The
    refinement is not cosmetic: the gamma quantile function is steep as
    ``p`` approaches 0 or 1, and on a uniform grid linear interpolation there
    is wrong by tens of percent.

    Parameters
    ----------
    n_body : int
        Points spanning the central range.
    n_tail : int
        Points in each tail.

    Returns
    -------
    np.ndarray
        Strictly increasing probabilities in (0, 1).
    """
    body = np.linspace(0.02, 0.98, n_body)
    lower = 0.02 * np.geomspace(1e-4, 1.0, n_tail)
    return np.unique(np.concatenate([lower, body, 1.0 - lower[::-1]]))


def build_gamma_quantile_table(shape_grid=None, probability_grid=None):
    """
    Build the shared, model-independent gamma quantile table.

    A client applying the precipitation transform needs the gamma quantile
    function, which is awkward to implement outside SciPy. This table removes
    that need: with ``loc=0`` the gamma factorises exactly as
    ``ppf(p; a, scale) = scale * ppf(p; a, 1)``, so the table has to be indexed
    by shape alone, never by scale. Storing ``ppf(p; a, 1) / a`` -- normalised
    to unit mean -- keeps the surface smooth in ``log(a)`` and so kind to
    linear interpolation.

    The result depends on nothing but mathematics: one table serves every
    model, variable, location and window.

    Parameters
    ----------
    shape_grid : np.ndarray, optional
        Gamma shape values, geometrically spaced. Defaults to 160 points
        spanning 0.5 to 1e4, which brackets the shapes seen in fitted CMIP6
        precipitation.
    probability_grid : np.ndarray, optional
        Probabilities; defaults to :func:`_gamma_probability_grid`.

    Returns
    -------
    tuple
        ``(shape_grid, probability_grid, table)`` with ``table`` of shape
        ``(len(shape_grid), len(probability_grid))``.
    """
    if shape_grid is None:
        shape_grid = np.geomspace(0.5, 1e4, 160)
    if probability_grid is None:
        probability_grid = _gamma_probability_grid()
    table = np.array(
        [_gamma.ppf(probability_grid, a, loc=0.0, scale=1.0) / a for a in shape_grid]
    )
    return shape_grid, probability_grid, table


def _fit_transform_parameters(reference, parsed, project):
    """
    Fit per-location, per-month-of-year gamma parameters to a reference field.

    Parameters
    ----------
    reference : xr.DataArray
        Gridded reference field, already sliced to the target window.
    parsed : list of dict
        Parsed locations.
    project : callable
        Reduces the field to one location.

    Returns
    -------
    tuple of np.ndarray
        ``(shape, scale, baseline)`` with shapes ``(n_loc, 12)``,
        ``(n_loc, 12)`` and ``(n_loc,)``. The baseline is the mean of the
        first twelve months of the window, which is what the precipitation
        path adds before transforming.
    """
    shape = np.full((len(parsed), 12), np.nan)
    scale = np.full((len(parsed), 12), np.nan)
    baseline = np.full(len(parsed), np.nan)
    for i, location in enumerate(parsed):
        series = np.squeeze(np.asarray(project(reference, location)))
        if series.ndim != 1:
            raise ValueError(
                f"transform reference reduced to shape {series.shape} at "
                f"{location['spec']}; expected a single time series. Reduce any "
                "ensemble dimension before exporting."
            )
        params = fit_distribution_parameters_1d_seasonal(series, "gamma")
        shape[i] = np.asarray(params["shape"])
        scale[i] = np.asarray(params["scale"])
        baseline[i] = float(np.mean(series[:12]))
    return shape, scale, baseline


def export_timeseries_bundle(
    noise_model,
    pattern_model,
    filepath,
    locations,
    variable=None,
    cmip6_model=None,
    training_scenario=None,
    training_config=None,
    transform_reference=None,
    transform_window=None,
    scenarios=None,
    doi=None,
    source_url=None,
    dtype=np.float32,
    netcdf_format=WIRE_NETCDF_FORMAT,
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
    transform_reference : xr.DataArray, optional
        Gridded CMIP6 reference field for a distribution transform, **already
        sliced to the output window** the bundle targets. It is reduced per
        location, fitted per month-of-year, and only the fitted gamma shape and
        scale are stored -- two floats per location per month, rather than the
        reference series itself.
    transform_window : tuple of int, optional
        ``(start_year, end_year)`` the reference was sliced to. Recorded so a
        client can tell which window the fitted parameters are valid for; the
        fit depends on it.
    scenarios : list of str or dict, optional
        Scenarios whose forcing trajectories should be baked in. Without these
        a client has the step-response kernel but no forcing to convolve it
        with, and obtaining forcing means running CICERO-SCM -- which a browser
        cannot do. Costs roughly 4 KB per scenario.

        Either names of the shipped inputs, or a mapping of name to
        ``(emissions, concentrations)`` dataframes for scenarios METEOR does
        not ship. See :func:`compute_scenario_forcing`; the mapping form lets a
        bundle carry forcing derived from third-party emissions without
        redistributing the emissions themselves.
    doi : str, optional
        DOI of the deposit this artifact belongs to, recorded so a downloaded
        copy can be traced back.
    source_url : str, optional
        Where the artifact is published, recorded alongside the DOI.
    netcdf_format : str, default 'NETCDF3_64BIT'
        On-disk netCDF flavour. Classic netCDF-3 by default so browsers can
        read it without a WebAssembly HDF5 build; pass ``'NETCDF4'`` for an
        HDF5-backed file.
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

    # Lower-triangular factor of the innovation covariance. Clients need it to
    # draw correlated shocks; shipping it removes both the cost and the
    # convention ambiguity of factoring a 40x40 matrix in the browser.
    try:
        chol = np.linalg.cholesky(np.asarray(noise_model.varx_sigma_u, np.float64))
        data_vars["varx_residual_chol"] = (("mode", "mode_in"), chol.astype(dtype))
    except np.linalg.LinAlgError:
        # Not positive definite: omit rather than ship something misleading.
        # Clients fall back to factoring varx_residual_cov themselves.
        pass

    forcing_year_start = -1
    if scenarios:
        # Passed through rather than listed: a mapping carries the supplied
        # emissions, and list() would reduce it to bare names.
        forcing, forcing_year_start = compute_scenario_forcing(pattern_model, scenarios)
        data_vars["scenario_forcing"] = (
            ("scenario", "exp", "year"),
            forcing.astype(dtype),
        )

    gamma_grid = None
    if transform_reference is not None:
        shape, scale, baseline = _fit_transform_parameters(
            transform_reference, parsed, _project_field
        )
        gamma_grid = build_gamma_quantile_table()
        gamma_table = gamma_grid[2]
        data_vars["transform_shape"] = (
            ("location", "month_of_year"),
            shape.astype(dtype),
        )
        data_vars["transform_scale"] = (
            ("location", "month_of_year"),
            scale.astype(dtype),
        )
        data_vars["transform_baseline"] = (("location",), baseline.astype(dtype))
        data_vars["gamma_quantile_norm"] = (
            ("gamma_shape", "gamma_probability"),
            gamma_table.astype(dtype),
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
            **(
                {
                    "month_of_year": ("month_of_year", np.arange(1, 13)),
                    "gamma_shape": ("gamma_shape", gamma_grid[0]),
                    "gamma_probability": ("gamma_probability", gamma_grid[1]),
                }
                if transform_reference is not None
                else {}
            ),
            **(
                {
                    "scenario": ("scenario", np.array(list(scenarios), dtype="U32")),
                    "year": (
                        "year",
                        np.arange(
                            forcing_year_start,
                            forcing_year_start
                            + data_vars["scenario_forcing"][1].shape[2],
                        ),
                    ),
                }
                if scenarios
                else {}
            ),
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
        "transform_type": "gamma" if transform_reference is not None else "",
        "transform_window_start": (
            -1 if not transform_window else int(transform_window[0])
        ),
        "transform_window_end": (
            -1 if not transform_window else int(transform_window[1])
        ),
        "forcing_year_start": int(forcing_year_start),
        "annual_to_monthly": "each annual value is repeated for all 12 months",
        **_provenance_attrs(
            cmip6_model, training_scenario, training_config, doi, source_url
        ),
    }

    ds.to_netcdf(filepath, format=netcdf_format)
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
        "has_transform": "transform_shape" in ds,
        "size_bytes": os.path.getsize(filepath),
        "schema_version": int(ds.attrs.get("schema_version", 0)),
        "provenance": json.loads(ds.attrs.get("training_config", "{}")),
    }


#: ``format`` attribute identifying a golden validation fixture.
GOLDEN_FORMAT = "meteor-golden-fixture"


def _transformed_fixture_series(
    bundle,
    specs,
    indices,
    design,
    pcs,
    forced,
    window_start,
    year_0,
):
    """
    Build the fully transformed series for a golden fixture, or ``None``.

    Returns ``None`` unless the bundle carries a transform and there is both a
    forced response and a window to place it in -- the transform's gamma
    parameters are fitted for a specific window, so the series it is applied to
    has to be the one covering that window.

    Runs the complete recipe in the order the schema documents, using the
    **anomaly** seasonal form, because that is what METEOR's own timeseries
    path produces and therefore what a port should be checked against.

    Parameters
    ----------
    bundle : xr.Dataset
        The loaded bundle.
    specs : list of str
        Location specifiers, in fixture order.
    indices : list of int
        Their positions in the bundle's location coordinate.
    design : np.ndarray
        Harmonic design matrix, shape ``(n_time, 9)``.
    pcs : np.ndarray
        Stochastic PCs, shape ``(n_realizations, n_time, n_modes)``.
    forced : np.ndarray
        Annual forced response per location, or an empty array.
    window_start : int or None
        First calendar year of the fixture's months.
    year_0 : int
        First calendar year of the forcing trajectories, and therefore of
        ``forced``.

    Returns
    -------
    np.ndarray or None
        Shape ``(n_location, n_realization, n_time)``.
    """
    if "transform_shape" not in bundle or not forced.size or window_start is None:
        return None

    n_time = design.shape[0]
    n_years = n_time // 12
    offset = int(window_start) - int(year_0)
    if offset < 0 or offset + n_years > forced.shape[1]:
        raise ValueError(
            f"window_start {window_start} with {n_years} years does not fit the "
            f"forcing axis: {forced.shape[1]} years from year_0={year_0}"
        )

    out = np.zeros((len(specs), pcs.shape[0], n_time))
    for i, (spec, idx) in enumerate(zip(specs, indices)):
        # Anomaly seasonal form: no intercept, and no t_glob term.
        seasonal = design[:, 1:] @ bundle["seasonal_coef"].values[idx][1:]
        stochastic = seasonal[None, :] + pcs @ bundle["eof_projection"].values[idx]

        monthly_forced = np.repeat(forced[i, offset : offset + n_years], 12)
        values = (
            stochastic
            + monthly_forced[None, :]
            + float(bundle["transform_baseline"].values[idx])
        )
        out[i] = apply_transform_from_bundle(bundle, spec, values)
    return out


def export_golden_fixture(
    bundle_path,
    filepath,
    noise_model,
    t_glob,
    seed=0,
    n_realizations=2,
    forcing_by_exp=None,
    year_0=1850,
    locations=None,
    window_start=None,
    dtype=np.float32,
    netcdf_format=WIRE_NETCDF_FORMAT,
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

    ``series`` holds the **absolute** seasonal form documented in the schema's
    reconstruction, plus the EOF projection: it is not the output of
    :meth:`MeteorInterface.generate_ensemble_outputs`, which subtracts the
    intercept and the ``t_glob`` term and adds the forced response. See the
    schema's "Two forms of the seasonal cycle".

    For a bundle carrying a distribution transform, and when ``forcing_by_exp``
    and ``window_start`` are supplied, the fixture also stores
    ``series_transformed``: the complete recipe, anomaly seasonal form with the
    forced response, the baseline and the quantile mapping applied. Without it
    a port can pass every other array in this file with its ``pr`` path
    unimplemented, since none of the steps unique to precipitation are
    otherwise exercised.

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
        First calendar year of the forcing trajectories, and therefore of the
        stored ``forced_response``. When the forcing came from
        :func:`forcing_from_bundle` this must be the bundle's
        ``forcing_year_start`` -- which is 1750 for bundles built from the
        shipped scenarios, not the 1850 default. The default is kept for
        backwards compatibility, but passing forcing read out of a bundle and
        leaving it alone mislabels the year axis by a century.
    locations : list of str, optional
        Location specifiers to include. Defaults to every location in the
        bundle, which for a 67-location bundle makes a needlessly large
        fixture: a handful of locations validate the arithmetic just as well.
    window_start : int, optional
        First calendar year of ``t_glob``. Required to add
        ``series_transformed``, because the forced response must be sliced to
        the window the fixture covers before it can be combined with the
        monthly terms.
    netcdf_format : str, default 'NETCDF3_64BIT'
        On-disk netCDF flavour; see :func:`export_timeseries_bundle`.
    dtype : np.dtype, default np.float32
        Storage precision. float32 by default for the same reason bundles use
        it: a fixture validates a reimplementation that reads float32 bundle
        arrays, so it cannot be held to a tighter tolerance than that, and the
        PC array dominates the file size.

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

    all_specs = [str(s) for s in bundle["location"].values]
    if locations is None:
        specs = all_specs
    else:
        specs = [str(s) for s in locations]
        missing = [s for s in specs if s not in all_specs]
        if missing:
            raise ValueError(f"locations not in bundle: {', '.join(missing)}")
    indices = [all_specs.index(s) for s in specs]

    series = np.zeros((len(specs), n_realizations, n_time))
    forced = np.zeros((len(specs), 0))
    if forcing_by_exp:
        forced = np.stack(
            [
                forced_response_from_bundle(bundle, spec, forcing_by_exp, year_0)
                for spec in specs
            ]
        )
    for i, idx in enumerate(indices):
        seasonal = (
            design @ bundle["seasonal_coef"].values[idx]
            + bundle["seasonal_intercept"].values[idx]
        )
        series[i] = seasonal[None, :] + pcs @ bundle["eof_projection"].values[idx]

    data_vars = {
        "t_glob": (("month",), t_glob.astype(dtype)),
        "stochastic_pcs": (("realization", "month", "mode"), pcs.astype(dtype)),
        "series": (("location", "realization", "month"), series.astype(dtype)),
    }
    if forced.size:
        data_vars["forced_response"] = (("location", "year"), forced.astype(dtype))

    transformed = _transformed_fixture_series(
        bundle,
        specs,
        indices,
        design,
        pcs,
        forced,
        window_start,
        year_0,
    )
    if transformed is not None:
        data_vars["series_transformed"] = (
            ("location", "realization", "month"),
            transformed.astype(dtype),
        )

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
        # A fixture is derived from one specific bundle and is meaningless
        # apart from it, so it inherits that bundle's provenance: deposited
        # together, they must be traceable together.
        "meteor_version": bundle.attrs.get("meteor_version", __version__),
        "doi": bundle.attrs.get("doi", ""),
        "source_url": bundle.attrs.get("source_url", ""),
        "created": np.datetime_as_string(np.datetime64("now", "s"), unit="s"),
        "usage": (
            "Feed stochastic_pcs and t_glob into the reimplementation; it must "
            "reproduce series (and forced_response when present). 'series' is "
            "the absolute seasonal form plus the EOF projection, with no "
            "forced response; 'series_transformed', when present, is the "
            "complete recipe in the anomaly seasonal form."
        ),
        "seasonal_form": "absolute",
    }
    ds.to_netcdf(filepath, format=netcdf_format)
    return filepath


def forcing_from_bundle(bundle, scenario):
    """
    Read a scenario's per-experiment forcing out of a bundle.

    Together with :func:`forced_response_from_bundle` this closes the loop: a
    client with only the bundle can produce a named scenario's forced response
    without running the simple climate model.

    Parameters
    ----------
    bundle : xr.Dataset
        Bundle from :func:`load_timeseries_bundle`.
    scenario : str
        Scenario name present in the bundle's ``scenario`` coordinate.

    Returns
    -------
    dict
        Mapping of experiment name to forcing trajectory, ready to pass to
        :func:`forced_response_from_bundle`.

    Raises
    ------
    KeyError
        If the bundle carries no forcing, or not for this scenario.

    Examples
    --------
    >>> forcing = forcing_from_bundle(bundle, "ssp245")  # doctest: +SKIP
    >>> series = forced_response_from_bundle(  # doctest: +SKIP
    ...     bundle, "global", forcing, year_0=bundle.attrs["forcing_year_start"]
    ... )
    """
    if "scenario_forcing" not in bundle:
        raise KeyError(
            "bundle carries no scenario forcing; re-export with scenarios=[...]"
        )
    names = [str(s) for s in bundle["scenario"].values]
    if scenario not in names:
        raise KeyError(f"scenario {scenario!r} not in bundle; has {names}")
    idx = names.index(scenario)
    block = bundle["scenario_forcing"].values[idx]
    out = {}
    for j, exp in enumerate(str(e) for e in bundle["exp"].values):
        series = block[j]
        if np.all(np.isnan(series)):
            continue
        out[exp] = series
    return out


def apply_transform_from_bundle(bundle, location, values):
    """
    Apply a bundle's distribution transform using only bundle arrays.

    This is the reference implementation of the transform contract, and the
    thing a port to another language should match. It needs the normal CDF and
    linear interpolation, and nothing else: no gamma fitting, no gamma quantile
    function, no SciPy equivalent.

    The Gaussian side is fitted here rather than shipped, because it describes
    *the ensemble the caller just generated* and so cannot be precomputed. It
    is only a mean and a standard deviation per month-of-year.

    Parameters
    ----------
    bundle : xr.Dataset
        Bundle carrying ``transform_shape``, ``transform_scale`` and
        ``gamma_quantile_norm``.
    location : str
        Location specifier present in the bundle.
    values : np.ndarray
        Generated values, shape ``(n_realizations, n_months)``, starting in
        January and already including ``transform_baseline``.

    Returns
    -------
    np.ndarray
        Transformed values, same shape as ``values``.

    Raises
    ------
    KeyError
        If the bundle carries no transform.

    Examples
    --------
    >>> out = apply_transform_from_bundle(bundle, "global", ens)  # doctest: +SKIP
    """
    if "transform_shape" not in bundle:
        raise KeyError(
            "bundle carries no distribution transform; re-export with "
            "transform_reference=..."
        )
    idx = list(bundle["location"].values).index(location)
    shapes = np.asarray(bundle["transform_shape"].values[idx], dtype=np.float64)
    scales = np.asarray(bundle["transform_scale"].values[idx], dtype=np.float64)
    log_shape_grid = np.log(np.asarray(bundle["gamma_shape"].values, dtype=np.float64))
    prob_grid = np.asarray(bundle["gamma_probability"].values, dtype=np.float64)
    table = np.asarray(bundle["gamma_quantile_norm"].values, dtype=np.float64)

    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    out = np.empty_like(values)
    for month, columns in enumerate(_month_of_year_indices(values.shape[1])):
        block = values[:, columns]
        mean, std = block.mean(), block.std()
        probabilities = (
            _norm.cdf((block - mean) / std) if std > 0 else np.full(block.shape, 0.5)
        )
        shape, scale = shapes[month], scales[month]
        # Bilinear in (log shape, probability): interpolate the two bracketing
        # shape rows, then between them. Stored quantiles are normalised to
        # unit mean, so the physical value is shape * scale * table value.
        position = np.interp(
            np.log(shape), log_shape_grid, np.arange(log_shape_grid.size)
        )
        low = int(np.clip(np.floor(position), 0, log_shape_grid.size - 2))
        weight = position - low
        curve = (1.0 - weight) * table[low] + weight * table[low + 1]
        out[:, columns] = shape * scale * np.interp(probabilities, prob_grid, curve)
    return out


def scale_to_warming_pathway(forced_response, global_tas_response, pathway):
    """
    Rescale a forced response to follow a prescribed global-warming pathway.

    This is the "pick or draw a warming pathway" mode: instead of running a
    named scenario, the caller prescribes global mean warming and the forced
    response is rescaled to match it. METEOR does this by scaling the anomaly
    about its first year by the ratio of desired to predicted global warming.

    ``global_tas_response`` is a separate argument rather than something
    derived from the bundle because it must always be the **temperature**
    response, even when rescaling precipitation. METEOR takes the denominator
    from the ``tas`` pattern model regardless of the variable being generated,
    so a precipitation client has to load the ``tas`` bundle too and pass its
    ``global`` forced response here. Passing the precipitation response instead
    silently produces wrong numbers, which is why this is a required argument
    with no default.

    Parameters
    ----------
    forced_response : array-like
        Annual forced response at the location of interest, from
        :func:`forced_response_from_bundle`, starting at the bundle's
        ``forcing_year_start``.
    global_tas_response : array-like
        Annual **temperature** forced response at ``global``, over the same
        years, taken from the ``tas`` bundle.
    pathway : array-like
        Desired global mean warming over the same years. Only its anomaly
        relative to the first year matters.

    Returns
    -------
    np.ndarray
        Rescaled annual forced response, same length as the inputs.

    Raises
    ------
    ValueError
        If the three series are not the same length.

    Examples
    --------
    >>> scaled = scale_to_warming_pathway(  # doctest: +SKIP
    ...     forced_pr, forced_global_tas, drawn_pathway
    ... )
    """
    forced = np.asarray(forced_response, dtype=np.float64)
    global_tas = np.asarray(global_tas_response, dtype=np.float64)
    wanted = np.asarray(pathway, dtype=np.float64)
    if not forced.shape == global_tas.shape == wanted.shape:
        raise ValueError(
            "forced_response, global_tas_response and pathway must share a "
            f"year axis; got {forced.shape}, {global_tas.shape}, {wanted.shape}"
        )

    anomaly = forced - forced[0]
    denominator = global_tas - global_tas[0]
    desired = wanted - wanted[0]
    # Where the predicted warming is exactly zero there is nothing to scale --
    # at the base year, and before forcing departs from it -- so leave the
    # anomaly (itself zero there) untouched rather than divide by zero.
    scaling = np.where(
        denominator != 0,
        desired / np.where(denominator == 0, 1.0, denominator),
        1.0,
    )
    return forced[0] + anomaly * scaling
