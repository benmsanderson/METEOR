# METEOR portable emulator artifacts — schema v1

Three related on-disk formats, all netCDF, all self-describing, none requiring
code execution to read. Each carries a `format` attribute identifying it and a
`schema_version` attribute; a reader must refuse a `schema_version` higher than
it understands.

| Format string | Written by | Purpose |
|---|---|---|
| `meteor-noise-emulator` | [`portable_artifact.export_noise_model`](../src/meteor/portable_artifact.py) | Full noise generator, gridded-capable |
| `meteor-pattern-scaling` | [`portable_artifact.export_pattern_scaling`](../src/meteor/portable_artifact.py) | Full pattern-scaling model, gridded-capable |
| `meteor-timeseries-bundle` | [`timeseries_bundle.export_timeseries_bundle`](../src/meteor/timeseries_bundle.py) | Compact per-location bundle, timeseries only |
| `meteor-golden-fixture` | [`timeseries_bundle.export_golden_fixture`](../src/meteor/timeseries_bundle.py) | Fixed-seed reference output for validating a port |

## Why these exist

The historical `save_model` paths pickle live `statsmodels`, `sklearn` and
`lmfit` objects. Those files reload only inside a byte-compatible Python
environment, and unpickling an untrusted file executes arbitrary code. The
artifacts below store the same trained state as plain arrays.

They are also much smaller, because a pickled pattern-scaling model carries
`dacanom` — the gridded CMIP6 training field — which no prediction path reads:

| Artifact | Pickle | netCDF (float64) | netCDF (float32) |
|---|---:|---:|---:|
| NorESM2-MM `tas` noise model | 25.5 MB | 21.2 MB | 10.6 MB |
| NorESM2-MM `tas` pattern scaling | 321.5 MB | 3.8 MB | 1.9 MB |
| 61-location `tas` bundle | — | — | 64.6 KB |

## Precision

Full artifacts default to **float64**, which makes a reload generate
bit-for-bit identically to the in-memory model. float32 halves the file but the
VARX recursion amplifies its ~1e-7 relative rounding over long trajectories, so
exact reproduction is lost.

Bundles default to **float32**. They are a wire format for clients that will
not reproduce float64 arithmetic bit for bit anyway; measured error against
METEOR's own output is ~1e-7 relative, i.e. single-precision storage noise.

## Array layout is part of the contract

All arrays are stored **C-contiguous** (row-major). This is load-bearing:
matrix-multiply summation order depends on memory layout, so a model whose
arrays were non-contiguous in memory would otherwise produce last-bit
differences after a round trip. METEOR normalises to C order when it derives
its array state, so a freshly fitted model, a reloaded pickle and a reloaded
artifact all agree exactly.

Spatial fields are flattened as `space_index = lat_index * n_lon + lon_index`.

---

## `meteor-noise-emulator`

### Dimensions
`mode`, `mode_in` (both `n_modes`), `lag` (`lag_order`), `feature` (9),
`space` (`n_lat * n_lon`), `lat`, `lon`, `exog` (only when present).

### Variables

| Name | Dims | Notes |
|---|---|---|
| `varx_intercept` | `(mode,)` | VAR constant term |
| `varx_A` | `(lag, mode, mode_in)` | Lag matrices; `A[i] @ y_{t-i-1}` |
| `varx_B` | `(mode, exog)` | Present only when `use_exog != 'none'` |
| `varx_residual_cov` | `(mode, mode_in)` | Innovation covariance (Σ) |
| `seasonal_coef` | `(space, feature)` | Per-gridpoint harmonic regression |
| `seasonal_intercept` | `(space,)` | |
| `eof_components` | `(mode, space)` | As fitted; in weighted space when `eof_weights` present |
| `eof_weights` | `(space,)` | Optional sqrt(cos-lat) weights |

**EOF convention:** `eof_components` is stored *raw* — in the weighted space the
PCA was fitted in. Physical-units components are `eof_components / eof_weights`
when `eof_weights` is present, and `eof_components` itself otherwise. This is
the `weights-and-raw` option; the alternative (storing pre-divided physical
components) was not taken, so that the stored basis matches what the fit
produced.

### VARX parameter ordering

`varx_A` and `varx_B` are stored already decomposed, because the stacked
`statsmodels` parameter matrix orders its rows
`[const, exog_1..exog_k, L1.y_1..L1.y_m, L2.y_1..L2.y_m, ...]` — exogenous rows
sit **immediately after the intercept, not at the end**. Reading them as
trailing rows silently interleaves exogenous coefficients into the first lag
matrix. Consumers of this schema get the decomposed arrays and need not know
the stacking convention.

### Attributes
`format`, `schema_version`, `meteor_version`, `variable_name`, `n_modes`,
`lag_order`, `use_exog`, `weight_eofs`, `n_lat`, `n_lon`, `space_ordering`,
`cmip6_model`, `training_scenario`, `training_config` (JSON), `created`.

### Reconstruction

```
X       = harmonic design matrix (see feature coordinate for column order)
pcs_t   = varx_intercept + Σ_i varx_A[i] @ pcs_{t-i-1} [+ varx_B @ x_t] + ε_t,
          ε_t ~ N(0, varx_residual_cov),  pcs_t = 0 for t < lag_order
field   = X @ seasonal_coef.T + seasonal_intercept + pcs @ physical_eofs
```

The `feature` coordinate names the nine design-matrix columns in order:
`t_glob`, `annual_cos`, `annual_sin`, `semiannual_cos`, `semiannual_sin`, and
`t_glob` times each of the four harmonics. With `t` a month index,
`annual_* = cos|sin(2πt/12)` and `semiannual_* = cos|sin(4πt/12)`.

---

## `meteor-pattern-scaling`

### Dimensions
`exp`, `fld`, `mode` (`n_pattern_modes`), `lat`, `lon`.

### Variables

| Name | Dims | Notes |
|---|---|---|
| `pattern_v` | `(exp, fld, mode, lat, lon)` | Spatial patterns; NaN where absent |
| `step_coeffs` | `(exp, fld, mode)` | `s_i` |
| `step_timescales` | `(exp, fld, mode)` | `τ_i`, in years |
| `has_pattern` | `(exp, fld)` | 1 where a fit exists |
| `exp_forc` | `(exp,)` | Step-experiment forcing magnitude |

`dacanom` is **not** included (`dacanom_included = 0`). A reloaded model
supports prediction but not the training-data plotting helpers in
`meteor_plot_utils`.

The `u` array from the pickled `pattern_full` is also omitted: both `rmodel`
and `recon_separately` overwrite it with the caller's PC matrix before use.

### Forced response

```
u_i(t)  = s_i * (1 - exp(-t / τ_i))          # t in years from year_0
dF(t)   = diff(forcing) appended with 0, divided by exp_forc
pcs     = convolve(u, dF)[:n_times]          # per mode
field   = pcs @ pattern_v                    # summed over modes
```

Contributions from multiple experiments are summed; the `base` experiment
contributes nothing.

---

## `meteor-timeseries-bundle`

This is what a browser client downloads. It never carries the EOF maps.

### Dimensions
`location`, `mode`, `mode_in`, `lag`, `feature` (9), `exp`, `pattern_mode`,
`exog` (optional), `reference_month` (optional).

### Variables

| Name | Dims | Notes |
|---|---|---|
| `varx_intercept`, `varx_A`, `varx_residual_cov`, `varx_B` | as above | Shared across locations |
| `eof_projection` | `(location, mode)` | EOF basis projected onto the location |
| `seasonal_coef` | `(location, feature)` | Nine coefficients per location |
| `seasonal_intercept` | `(location,)` | |
| `pattern_projection` | `(location, exp, pattern_mode)` | Spatial pattern projected onto the location |
| `step_coeffs`, `step_timescales` | `(exp, pattern_mode)` | Shared kernel |
| `exp_forc` | `(exp,)` | |
| `location_kind` | `(location,)` | `global` / `region` / `point` |
| `location_lat`, `location_lon` | `(location,)` | NaN for non-point locations |
| `pr_reference` | `(location, reference_month)` | Optional; see below |

The `location` coordinate holds specifiers in the same grammar the generation
API uses: `global`, `regional:<AR6 code>`, `point:<lat>,<lon>`.

### Why this is small

A region mean is a fixed, data-independent weighted sum over gridpoints, so it
commutes with the linear model:

```
mean_region(X @ coef.T + intercept) = X @ (coef.T @ w) + intercept @ w
mean_region(pcs @ eofs)             = pcs @ (eofs @ w)
mean_region(pcs @ pattern_v)        = pcs @ (pattern_v @ w)
```

Nine seasonal coefficients, `n_modes` EOF projections and `n_pattern_modes`
pattern projections per location — about 200 bytes each at float32 — instead of
per-gridpoint fields. Point extraction is the same identity with `w` a one-hot
vector at the nearest gridpoint.

Both reductions are verified numerically against METEOR's own output; the
residual is float32 storage noise.

### Reconstruction

```
series = X @ seasonal_coef[loc] + seasonal_intercept[loc]
       + pcs @ eof_projection[loc]
       + Σ_exp convolve(step_response[exp], dF_exp / exp_forc[exp]) @ pattern_projection[loc, exp]
```

`timeseries_bundle.forced_response_from_bundle` is the reference implementation
of the last term.

### What is deliberately absent

* **Gridded output** (`gridded_output = 0`). Reconstructing fields needs the EOF
  maps and per-gridpoint seasonal coefficients — use the full artifacts.
* **Custom emissions scenarios** (`custom_emissions = 0`). A bundle ships the
  step-response kernel, so a client can convolve any *forcing* trajectory it
  supplies, including rescaled forcers. Turning raw *emissions* into forcing
  requires the simple climate model, which is not in the bundle.
* **Arbitrary locations.** A bundle covers the locations it was built for.

### Precipitation reference

`pr` generation normally re-fetches gridded CMIP6 data at generation time to fit
the gamma transform. The timeseries path aggregates that reference per region
and uses the 1D transform, so `pr_reference` stores the already-aggregated 1D
series per location, with `pr_reference_start_year` giving its first calendar
year (`-1` when absent).

Storing the aggregated *series* rather than fitted gamma parameters is
deliberate: the fit depends on the requested output window. The reference is
sliced to `start_year..end_year`, and the precipitation baseline is the mean of
the first twelve months **of that window**. Since per-timestep aggregation
commutes with both the time slice and the twelve-month mean, shipping the series
reproduces current behaviour exactly for any window, whereas pre-fitted
parameters would only be valid for one.

---

## `meteor-golden-fixture`

Fixed-seed reference output for validating a reimplementation offline.

| Name | Dims |
|---|---|
| `t_glob` | `(month,)` |
| `stochastic_pcs` | `(realization, month, mode)` |
| `series` | `(location, realization, month)` |
| `forced_response` | `(location, year)` (optional) |

A port will not reproduce NumPy's PCG64 stream, so the PC sequence is stored
**as data**. Feed `stochastic_pcs` and `t_glob` into the port; it must
reproduce `series`. The deterministic parts — seasonal cycle, EOF projection,
forced response — are what is being validated, not the random number generator.

Fixtures are written with an explicit `numpy.random.Generator`, so they are
reproducible across processes. Note that METEOR's single- and
multi-realization PC paths draw from the generator in different shapes, so
`n_realizations=1` and `n_realizations=n` do not share a common prefix.
