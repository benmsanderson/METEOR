"""
Portable, non-executable serialization of trained METEOR emulators.

The historical :meth:`MeteorNoiseGenerator.save_model` and
:meth:`MeteorPatternScaling.save_model` paths pickle live ``statsmodels``,
``sklearn`` and ``lmfit`` objects. Such a file only reloads inside a
byte-compatible Python environment, and unpickling a user-supplied file
executes arbitrary code. This module writes the same trained state as plain
arrays in a self-describing netCDF container, so a trained emulator can be
archived, shipped, and re-loaded by any reader -- including a reimplementation
in another language.

The exported artifact carries exactly what *generation* needs. Training-only
state (the fitted library objects themselves, and the gridded CMIP6 anomaly
field ``dacanom``, which is >99% of a pattern-scaling pickle) is deliberately
omitted.

See ``docs/emulator_artifact_schema.md`` for the versioned on-disk layout.
"""

import json
import os
import warnings

import numpy as np
import xarray as xr

from . import __version__
from .noise_generator import MeteorNoiseGenerator
from .pattern_logic_lib import step_response_values

#: Bump on any incompatible change to the on-disk layout.
SCHEMA_VERSION = 1

#: ``format`` attribute identifying a portable noise-model artifact.
NOISE_FORMAT = "meteor-noise-emulator"

#: ``format`` attribute identifying a portable pattern-scaling artifact.
PATTERN_FORMAT = "meteor-pattern-scaling"

#: Feature names of the harmonic design matrix, in column order. Mirrors
#: :meth:`MeteorNoiseGenerator._create_harmonic_features`; recorded in the
#: artifact so a reimplementation can rebuild the design matrix unambiguously.
SEASONAL_FEATURE_NAMES = [
    "t_glob",
    "annual_cos",
    "annual_sin",
    "semiannual_cos",
    "semiannual_sin",
    "t_glob_x_annual_cos",
    "t_glob_x_annual_sin",
    "t_glob_x_semiannual_cos",
    "t_glob_x_semiannual_sin",
]


def _provenance_attrs(
    cmip6_model,
    training_scenario,
    training_config,
    doi=None,
    source_url=None,
    created=None,
):
    """
    Build the common provenance attribute block.

    Artifacts are meant to be published outside this repository -- a deposit
    with a DOI rather than a committed file -- so they carry the identifiers
    needed to trace a downloaded copy back to its source.

    Warns when publication metadata (``doi`` or ``source_url``) is supplied
    while the recorded METEOR version is marked dirty: the artifact was then
    built from an uncommitted working tree and cannot be reproduced from any
    commit. A scratch export from a dirty tree is normal while iterating, so
    the warning is deliberately scoped to the citable case.

    Parameters
    ----------
    cmip6_model : str or None
        Source CMIP6 model name.
    training_scenario : str or None
        Scenario the emulator was trained on.
    training_config : dict or None
        Full training configuration; stored as a JSON string.
    created : str or None
        ISO-8601 creation date. Defaults to today when None.
    doi : str or None
        DOI of the deposit this artifact belongs to, when it has one.
    source_url : str or None
        Where the artifact is published, or the code that produced it.

    Returns
    -------
    dict
        Attributes to attach to the exported dataset.
    """
    if created is None:
        created = np.datetime_as_string(np.datetime64("now", "s"), unit="s")
    # Only warn when publication metadata is supplied. A scratch export from a
    # dirty tree is normal while iterating; a *citable* one is not, and that is
    # what a doi or source_url signals.
    if (doi or source_url) and "dirty" in __version__:
        warnings.warn(
            f"Exporting with METEOR version {__version__!r}: the working tree "
            "has uncommitted changes, so this artifact cannot be reproduced "
            "from any commit. Commit before depositing.",
            UserWarning,
            stacklevel=3,
        )
    return {
        "doi": "" if doi is None else str(doi),
        "source_url": "" if source_url is None else str(source_url),
        "schema_version": SCHEMA_VERSION,
        "meteor_version": __version__,
        "cmip6_model": "" if cmip6_model is None else str(cmip6_model),
        "training_scenario": (
            "" if training_scenario is None else str(training_scenario)
        ),
        "training_config": json.dumps(
            training_config or {}, default=str, sort_keys=True
        ),
        "created": created,
    }


def export_noise_model(
    model,
    filepath,
    cmip6_model=None,
    training_scenario=None,
    training_config=None,
    doi=None,
    source_url=None,
    dtype=np.float64,
):
    """
    Write a fitted noise generator to a portable netCDF artifact.

    Stores the VARX coefficient arrays, the seasonal regression coefficients,
    and the EOF basis as plain arrays. No ``statsmodels`` or ``sklearn`` object
    is written, so the result loads without those libraries and without
    executing code.

    Parameters
    ----------
    model : MeteorNoiseGenerator
        Fitted noise generator.
    filepath : str
        Destination path (``.nc``).
    cmip6_model : str, optional
        Source CMIP6 model name, recorded as provenance.
    training_scenario : str, optional
        Training scenario, recorded as provenance.
    training_config : dict, optional
        Full training configuration, recorded as provenance.
    doi : str, optional
        DOI of the deposit this artifact belongs to, recorded so a downloaded
        copy can be traced back.
    source_url : str, optional
        Where the artifact is published, recorded alongside the DOI.
    doi : str, optional
        DOI of the deposit this artifact belongs to, recorded so a downloaded
        copy can be traced back.
    source_url : str, optional
        Where the artifact is published, recorded alongside the DOI.
    dtype : np.dtype, default np.float64
        Storage precision for the exported float arrays. The default preserves
        generation bit-for-bit; ``np.float32`` roughly halves the file at the
        cost of exact reproducibility (the VARX recursion amplifies the ~1e-7
        relative rounding error over long trajectories).

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If the model is not fitted.

    Examples
    --------
    >>> export_noise_model(noise_model, "NorESM2-MM_tas_noise.nc")  # doctest: +SKIP
    """
    if not model.fitted:
        raise ValueError("Model must be fitted before exporting")

    n_lat = len(model.coords["lat"])
    n_lon = len(model.coords["lon"])

    data_vars = {
        "varx_intercept": (("mode",), np.asarray(model.varx_intercept, dtype=dtype)),
        "varx_A": (("lag", "mode", "mode_in"), np.asarray(model.varx_A, dtype=dtype)),
        "varx_residual_cov": (
            ("mode", "mode_in"),
            np.asarray(model.varx_sigma_u, dtype=dtype),
        ),
        "seasonal_coef": (
            ("space", "feature"),
            np.asarray(model.seasonal_coef, dtype=dtype),
        ),
        "seasonal_intercept": (
            ("space",),
            np.asarray(model.seasonal_intercept, dtype=dtype),
        ),
        "eof_components": (
            ("mode", "space"),
            np.asarray(model.eof_components, dtype=dtype),
        ),
    }
    if model.varx_B is not None:
        data_vars["varx_B"] = (
            ("mode", "exog"),
            np.asarray(model.varx_B, dtype=dtype),
        )
    if model.eof_weights is not None:
        data_vars["eof_weights"] = (
            ("space",),
            np.asarray(model.eof_weights, dtype=dtype).reshape(-1),
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "lat": ("lat", np.asarray(model.coords["lat"], dtype=np.float64)),
            "lon": ("lon", np.asarray(model.coords["lon"], dtype=np.float64)),
            "mode": ("mode", np.arange(model.n_modes)),
            "mode_in": ("mode_in", np.arange(model.n_modes)),
            "lag": ("lag", np.arange(1, model.lag_order + 1)),
            "feature": ("feature", np.array(SEASONAL_FEATURE_NAMES, dtype="U32")),
        },
    )

    ds.attrs = {
        "format": NOISE_FORMAT,
        "variable_name": "" if model.variable_name is None else model.variable_name,
        "n_modes": int(model.n_modes),
        "lag_order": int(model.lag_order),
        "use_exog": str(model.use_exog),
        "weight_eofs": int(bool(model.weight_eofs)),
        "n_lat": int(n_lat),
        "n_lon": int(n_lon),
        "space_ordering": (
            "space index = lat_index * n_lon + lon_index (C order over lat, lon)"
        ),
        **_provenance_attrs(
            cmip6_model, training_scenario, training_config, doi, source_url
        ),
    }

    ds.to_netcdf(filepath)
    return filepath


def load_noise_model(filepath):
    """
    Load a portable noise-model artifact into a usable generator.

    Reconstructs a :class:`MeteorNoiseGenerator` whose generation paths are
    driven entirely by the stored arrays. ``seasonal_model``, ``pca`` and
    ``varx_results`` are left as ``None`` -- nothing in the generation path
    reads them.

    Parameters
    ----------
    filepath : str
        Path to a netCDF artifact written by :func:`export_noise_model`.

    Returns
    -------
    MeteorNoiseGenerator
        Fitted generator ready for ``generate_*`` calls.

    Raises
    ------
    ValueError
        If the file is not a noise-model artifact, or its schema version is
        newer than this release understands.

    Examples
    --------
    >>> model = load_noise_model("NorESM2-MM_tas_noise.nc")  # doctest: +SKIP
    """
    with xr.open_dataset(filepath) as ds:
        ds = ds.load()

    fmt = ds.attrs.get("format")
    if fmt != NOISE_FORMAT:
        raise ValueError(
            f"{filepath} is not a METEOR noise-model artifact "
            f"(format={fmt!r}, expected {NOISE_FORMAT!r})"
        )
    version = int(ds.attrs.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{filepath} uses schema version {version}, but this METEOR release "
            f"understands at most {SCHEMA_VERSION}. Upgrade METEOR to read it."
        )

    model = MeteorNoiseGenerator(
        n_modes=int(ds.attrs["n_modes"]),
        lag_order=int(ds.attrs["lag_order"]),
        use_exog=str(ds.attrs["use_exog"]),
        weight_eofs=bool(int(ds.attrs["weight_eofs"])),
    )

    model.varx_intercept = ds["varx_intercept"].values
    model.varx_A = ds["varx_A"].values
    model.varx_B = ds["varx_B"].values if "varx_B" in ds else None
    model.varx_sigma_u = ds["varx_residual_cov"].values
    model.seasonal_coef = ds["seasonal_coef"].values
    model.seasonal_intercept = ds["seasonal_intercept"].values
    model.eof_components = ds["eof_components"].values
    model.eof_weights = ds["eof_weights"].values if "eof_weights" in ds else None

    model.coords = {
        "lat": ds["lat"].values,
        "lon": ds["lon"].values,
    }
    variable_name = ds.attrs.get("variable_name", "")
    model.variable_name = variable_name or None
    model.fitted = True
    return model


def is_portable_artifact(filepath):
    """
    Report whether a path looks like a portable METEOR artifact.

    Used by the cache layer to route a cached file to the netCDF loader or the
    legacy pickle loader without attempting an unpickle first.

    Parameters
    ----------
    filepath : str
        Candidate path.

    Returns
    -------
    bool
        True when the file exists and carries a METEOR artifact ``format``
        attribute readable by xarray.
    """
    if not os.path.exists(filepath):
        return False
    try:
        with xr.open_dataset(filepath) as ds:
            return ds.attrs.get("format") in (NOISE_FORMAT, PATTERN_FORMAT)
    except (OSError, ValueError):
        return False


def _step_response_arrays(pars, n_modes):
    """
    Split fitted step-response parameters into coefficient and timescale arrays.

    Parameters
    ----------
    pars : lmfit.parameter.Parameters or dict
        Fitted step-response parameters holding ``s<i>``/``t<i>`` entries.
    n_modes : int
        Number of response modes to extract.

    Returns
    -------
    tuple of np.ndarray
        ``(coeffs, timescales)``, each of shape (n_modes,).
    """
    vals = step_response_values(pars)
    coeffs = np.array([vals[f"s{i}"] for i in range(n_modes)], dtype=np.float64)
    timescales = np.array([vals[f"t{i}"] for i in range(n_modes)], dtype=np.float64)
    return coeffs, timescales


def _step_response_mapping(coeffs, timescales):
    """
    Rebuild the ``s<i>``/``t<i>`` mapping consumed by the step-response model.

    Parameters
    ----------
    coeffs : np.ndarray
        Response coefficients, shape (n_modes,).
    timescales : np.ndarray
        Response timescales, shape (n_modes,).

    Returns
    -------
    dict
        Mapping accepted by :func:`pattern_logic_lib.pmodel` in place of an
        ``lmfit.Parameters`` object.
    """
    mapping = {}
    for i, (coeff, tau) in enumerate(zip(coeffs, timescales)):
        mapping[f"t{i}"] = float(tau)
        mapping[f"s{i}"] = float(coeff)
    return mapping


def export_pattern_scaling(
    model,
    filepath,
    cmip6_model=None,
    training_scenario=None,
    training_config=None,
    doi=None,
    source_url=None,
    dtype=np.float64,
):
    """
    Write a fitted pattern-scaling model to a portable netCDF artifact.

    Stores the step-response kernel (one coefficient and one timescale per
    mode, per experiment and field) alongside the spatial patterns. Neither the
    ``lmfit.Parameters`` objects nor the gridded CMIP6 training field
    ``dacanom`` are written: ``dacanom`` is training-only state that no
    prediction path reads, and it accounts for over 99% of a pickled model.

    Parameters
    ----------
    model : MeteorPatternScaling
        Fitted pattern-scaling model.
    filepath : str
        Destination path (``.nc``).
    cmip6_model : str, optional
        Source CMIP6 model name, recorded as provenance.
    training_scenario : str, optional
        Training scenario, recorded as provenance.
    training_config : dict, optional
        Full training configuration, recorded as provenance.
    dtype : np.dtype, default np.float64
        Storage precision for exported float arrays.

    Returns
    -------
    str
        The path written.

    Examples
    --------
    >>> export_pattern_scaling(pattern, "NorESM2-MM_tas_pattern.nc")  # doctest: +SKIP
    """
    exps = list(model.exp_list)
    flds = list(model.patternflds)
    n_modes = max(int(n) for n in model.patternflds.values())

    sample = None
    for exp in exps:
        for fld in flds:
            entry = model.pattern_dict.get(exp, {}).get(fld)
            if isinstance(entry, dict) and isinstance(entry.get("pattern_full"), dict):
                sample = entry["pattern_full"]["v"]
                break
        if sample is not None:
            break
    if sample is None:
        raise ValueError("Pattern scaling model holds no usable spatial patterns")

    n_lat, n_lon = sample.shape[1], sample.shape[2]
    shape = (len(exps), len(flds), n_modes)
    pattern_v = np.full(shape + (n_lat, n_lon), np.nan)
    coeffs = np.full(shape, np.nan)
    timescales = np.full(shape, np.nan)
    has_pattern = np.zeros((len(exps), len(flds)), dtype=np.int8)

    for i, exp in enumerate(exps):
        for j, fld in enumerate(flds):
            entry = model.pattern_dict.get(exp, {}).get(fld)
            if not isinstance(entry, dict):
                continue
            pattern_full = entry.get("pattern_full")
            if not isinstance(pattern_full, dict):
                # Patterns that failed to fit are stored as NaN by
                # _make_pattern_dict; keep them flagged rather than faking data.
                continue
            fld_modes = int(model.patternflds[fld])
            pattern_v[i, j, :fld_modes] = np.asarray(pattern_full["v"])[:fld_modes]
            fld_coeffs, fld_taus = _step_response_arrays(entry["outp"], fld_modes)
            coeffs[i, j, :fld_modes] = fld_coeffs
            timescales[i, j, :fld_modes] = fld_taus
            has_pattern[i, j] = 1

    ds = xr.Dataset(
        {
            "pattern_v": (
                ("exp", "fld", "mode", "lat", "lon"),
                pattern_v.astype(dtype),
            ),
            "step_coeffs": (("exp", "fld", "mode"), coeffs.astype(dtype)),
            "step_timescales": (("exp", "fld", "mode"), timescales.astype(dtype)),
            "has_pattern": (("exp", "fld"), has_pattern),
            "exp_forc": (
                ("exp",),
                np.array(
                    [float(model.exp_forc_dict[exp]) for exp in exps], dtype=np.float64
                ),
            ),
        },
        coords={
            "exp": ("exp", np.array(exps, dtype="U64")),
            "fld": ("fld", np.array(flds, dtype="U32")),
            "mode": ("mode", np.arange(n_modes)),
            "lat": ("lat", np.asarray(sample["lat"].values, dtype=np.float64)),
            "lon": ("lon", np.asarray(sample["lon"].values, dtype=np.float64)),
        },
    )

    ds.attrs = {
        "format": PATTERN_FORMAT,
        "name": str(model.name),
        "patternflds": json.dumps({k: int(v) for k, v in model.patternflds.items()}),
        "anom_timescales": json.dumps(
            {k: int(v) for k, v in getattr(model, "anom_timescales", {}).items()}
        ),
        "dacanom_included": 0,
        "step_response": "u_i(t) = s_i * (1 - exp(-t / tau_i)), t in years from year_0",
        **_provenance_attrs(
            cmip6_model, training_scenario, training_config, doi, source_url
        ),
    }

    ds.to_netcdf(filepath)
    return filepath


def load_pattern_scaling(filepath):
    """
    Load a portable pattern-scaling artifact into a usable model.

    Rebuilds a :class:`MeteorPatternScaling` whose ``predict_*`` methods work
    unchanged. ``dacanom`` is set to None: it is training-only state, so the
    reloaded model supports prediction but not the plotting helpers in
    :mod:`meteor.meteor_plot_utils` that inspect the training field.

    Parameters
    ----------
    filepath : str
        Path to a netCDF artifact written by :func:`export_pattern_scaling`.

    Returns
    -------
    MeteorPatternScaling
        Model ready for prediction.

    Raises
    ------
    ValueError
        If the file is not a pattern-scaling artifact, or its schema version is
        newer than this release understands.
    """
    # Imported here: meteor.meteor imports this module's siblings, and a
    # top-level import would close an import cycle.
    from .meteor import MeteorPatternScaling  # pylint: disable=import-outside-toplevel

    with xr.open_dataset(filepath) as opened:
        ds = opened.load()

    fmt = ds.attrs.get("format")
    if fmt != PATTERN_FORMAT:
        raise ValueError(
            f"{filepath} is not a METEOR pattern-scaling artifact "
            f"(format={fmt!r}, expected {PATTERN_FORMAT!r})"
        )
    version = int(ds.attrs.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{filepath} uses schema version {version}, but this METEOR release "
            f"understands at most {SCHEMA_VERSION}. Upgrade METEOR to read it."
        )

    model = MeteorPatternScaling.__new__(MeteorPatternScaling)
    model.name = ds.attrs["name"]
    model.patternflds = json.loads(ds.attrs["patternflds"])
    anom_timescales = json.loads(ds.attrs.get("anom_timescales", "{}"))
    if anom_timescales:
        model.anom_timescales = anom_timescales
    model.exp_list = [str(e) for e in ds["exp"].values]
    model.exp_forc_dict = {
        exp: float(val) for exp, val in zip(model.exp_list, ds["exp_forc"].values)
    }
    model.dacanom = None

    lat = ds["lat"]
    lon = ds["lon"]
    pattern_dict = {}
    for i, exp in enumerate(model.exp_list):
        pattern_dict[exp] = {}
        for j, fld in enumerate(str(f) for f in ds["fld"].values):
            if not int(ds["has_pattern"].values[i, j]):
                pattern_dict[exp][fld] = {"pattern_full": np.nan, "outp": None}
                continue
            fld_modes = int(model.patternflds[fld])
            values = ds["pattern_v"].values[i, j, :fld_modes]
            pattern_dict[exp][fld] = {
                "pattern_full": {
                    "v": xr.DataArray(
                        values,
                        coords={
                            "mode": np.arange(fld_modes),
                            "lat": lat.values,
                            "lon": lon.values,
                        },
                        dims=("mode", "lat", "lon"),
                    )
                },
                "outp": _step_response_mapping(
                    ds["step_coeffs"].values[i, j, :fld_modes],
                    ds["step_timescales"].values[i, j, :fld_modes],
                ),
            }
    model.pattern_dict = pattern_dict
    return model
