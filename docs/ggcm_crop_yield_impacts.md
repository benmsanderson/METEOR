# GGCMI Phase 2 Crop Yield Impacts

This document describes METEOR's implementation of the GGCMI Phase 2 crop yield
emulators and explains how it maps onto the original published methodology.

**Reference:** Franke, J. A., et al. (2020). The GGCMI Phase 2 emulators: global
gridded crop model yield responses to changes in CO2, temperature, water, and
nitrogen. *Geoscientific Model Development*, 13, 3995–4018.
<https://doi.org/10.5194/gmd-13-3995-2020>

**Original Python implementation:** <https://github.com/iiasa/ggcm_emulator>

---

## Background

Process-based crop models are computationally expensive and hard to embed in
large-ensemble or integrated-assessment workflows.  The GGCMI Phase 2 project
addressed this by running nine globally-gridded crop models across a structured
*parameter sweep* — up to 756 combinations of CO₂ concentration (C), temperature
perturbation (T), water supply (W), and nitrogen application (N), each repeated
under two growing-season adaptation assumptions (A0/A1) — and fitting a simple
polynomial to the climatological-mean yield response at every 0.5° grid cell.

The result is a set of spatially-varying polynomial coefficient tensors that can
reproduce the long-term mean yield of each crop model under arbitrary future
C/T/W/N conditions at negligible computational cost.

---

## The Polynomial (Eq. 1, Franke et al. 2020)

The emulator evaluates a **34-term third-order polynomial** in four transformed
inputs:

| Symbol | Physical meaning | Transformation |
|--------|-----------------|----------------|
| C | CO₂ concentration (ppm) | raw value |
| T | Temperature anomaly (°C) | `Ta − T_AgMERRA` |
| W | Precipitation ratio (–) | `Wa / W_AgMERRA` |
| N | Nitrogen application (kg N ha⁻¹ yr⁻¹) | raw value |

> **Why anomalies?**  The GGCMI Phase 2 simulations apply temperature
> perturbations as *additive mean shifts* and precipitation as *fractional
> multipliers* relative to the historical AgMERRA climatology.  Converting
> absolute inputs to the same anomaly/ratio form before evaluating the polynomial
> is therefore essential for physical consistency.

The 35th term of a full third-order polynomial in four variables would be N³,
but this term is **deliberately omitted** (Sect. 3.1, Franke et al.) because
the training data samples only three nitrogen levels — insufficient to constrain
a cubic in N.  METEOR stores all 35 coefficient slots (K[0]…K[34]) in the
netCDF4 files, with K[34] set to zero for every model/crop/variant.

The complete polynomial as stored in `coefficients.py`:

```
Yield = K[0]
      + K[1]·C   + K[2]·T    + K[3]·W    + K[4]·N
      + K[5]·C²  + K[6]·CT   + K[7]·CW   + K[8]·CN
      + K[9]·T²  + K[10]·TW  + K[11]·TN
      + K[12]·W² + K[13]·WN
      + K[14]·N²
      + K[15]·C³  + K[16]·C²T  + K[17]·C²W  + K[18]·C²N
      + K[19]·CT² + K[20]·CTW  + K[21]·CTN
      + K[22]·CW² + K[23]·CWN
      + K[24]·CN²
      + K[25]·T³  + K[26]·T²W  + K[27]·T²N
      + K[28]·TW² + K[29]·TWN
      + K[30]·TN²
      + K[31]·W³  + K[32]·W²N
      + K[33]·WN²
      (K[34]·N³ omitted — cannot be fitted from three N levels)
```

Yields are clipped to zero from below; negative raw predictions are set to 0.

---

## Input Clamping and Out-of-Bounds Diagnostics

The polynomial is only reliable within the training ranges (Table 2,
Franke et al.):

| Variable | Lower bound | Upper bound |
|----------|-------------|-------------|
| C | 360 ppm | 810 ppm |
| T | T_AgMERRA − 1 °C | T_AgMERRA + 6 °C |
| W | 0.5 × W_AgMERRA | 1.3 × W_AgMERRA |
| N | 10 kg ha⁻¹ yr⁻¹ | 200 kg ha⁻¹ yr⁻¹ |

`get_yields()` clamps all inputs to these ranges before evaluation and returns
two diagnostic fields:

- **`T_oob`** — per-cell temperature excess beyond the training boundary (°C;
  negative = below lower bound)
- **`W_oob`** — per-cell precipitation excess (mm yr⁻¹; same sign convention)

These can be used to flag grid cells where projections are extrapolating.

---

## AgMERRA Baseline

Most of the nine crop models in GGCMI Phase 2 use the AgMIP Modern-Era
Retrospective Analysis for Research and Applications (**AgMERRA**) as their
historical climate driver (Sect. 2.1, Franke et al.).  The 1980–2010
climatological mean temperature and precipitation fields from AgMERRA therefore
define the reference point against which the T and W inputs to the polynomial
are expressed.

METEOR bundles two pre-computed 0.5° AgMERRA climatology files directly with
the package in `src/meteor/impacts/ggcm/data/`:

| File | Variable | Units |
|------|----------|-------|
| `tas-avg-1980-2010-05deg-adjlon.nc4` | T_AgMERRA | °C |
| `pr-avg-avg-1980-2010-05deg-adjlon.nc4` | W_AgMERRA | mm yr⁻¹ |

`load_agmerra_baseline()` in `baseline.py` reads these files and returns
NumPy arrays of shape (360, 720) (global 0.5° grid, 90°N–90°S,
180°W–180°E).  Precipitation values are floored at 1 mm yr⁻¹ to avoid
division-by-zero when computing the W ratio.

---

## Coefficient Files

The fitted K tensors for each crop model / crop / adaptation-variant combination
are stored in netCDF4 files on Zenodo (record 3592453), one file per
combination.  Each file contains a single variable `K_rf` of shape
`(35, 360, 720)` — 35 coefficient planes on the same 0.5° global grid.

### Available combinations

| | CARAIB | EPIC-TAMU | GEPIC | JULES | LPJ-GUESS | LPJmL | pDSSAT | PEPIC | PROMET |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| maize | A0/A1 | A0/A1 | A0/A1 | A0 | A0/A1 | A0/A1 | A0/A1 | A0/A1 | A0/A1 |
| rice | A0/A1 | A0/A1 | A0/A1 | A0 | A0/A1 | A0/A1 | A0/A1 | A0/A1 | A0/A1 |
| soy | A0/A1 | A0/A1 | A0/A1 | A0 | — | A0/A1 | A0/A1 | A0/A1 | A0/A1 |
| spring_wheat | A0/A1 | A0/A1 | A0/A1 | A0 | A0/A1 | A0/A1 | A0/A1 | A0/A1 | A0/A1 |
| winter_wheat | A0/A1 | A0/A1 | A0/A1 | — | A0/A1 | A0/A1 | A0/A1 | A0/A1 | A0/A1 |

JULES contributes A0 scenarios only; its seasonal-length adaptation could not
be represented under the A1 protocol.  LPJ-GUESS did not simulate soybean.

Adaptation scenarios:
- **A0** — no cultivar adaptation; growing seasons shorten in warmer climates
- **A1** — cultivar adaptation retains fixed growing-season length

Files are downloaded on demand by `GgcmDownloader` with resume support; cached
under `<cache_dir>/ggcm/`.

---

## Integration into METEOR

When crop yield impacts are requested via `generate_ensemble_outputs()`, the
`_apply_crop_yields()` method in `meteor_interface.py`:

1. **Generates annual gridded T and P** from pattern scaling over the requested
   year range (using the already-trained METEOR pattern models for both `tas`
   and `pr`).
2. **Reconstructs absolute fields** by adding the piControl climatological mean
   to pattern-scaling anomalies, converting units to °C and mm yr⁻¹.
3. **Regrids** both fields to the GGCM 0.5° grid via bilinear interpolation;
   fills any coastal/polar NaN gaps with the AgMERRA climatological values.
4. **Reads CO₂** for each year from the scenario concentration data.
5. **Evaluates `get_yields()`** per crop with the pre-loaded K tensor, producing
   a (360 × 720) yield field for each year.
6. **Aggregates spatially** using the same keys as the climate time series
   (`"global"`, `"regional:CODE"`, `"point:LAT,LON"`, etc.).

### Usage

```python
ensemble = emulator.generate_ensemble_outputs(
    scenario="ssp245",
    start_year=2020,
    end_year=2100,
    timeseries=["global", "regional:EAS"],
    impacts={
        "crop_yield": {
            "crops": ["maize", "spring_wheat"],
            "crop_model": "LPJmL",
            "variant": "A0",
            "N": 100,               # kg N ha⁻¹ yr⁻¹ (uniform)
        }
    },
)

# Access results
maize_global = ensemble.crop_impacts["maize"]["global"]   # (n_years,) ndarray
```

Results are stored on `EnsembleOutput.crop_impacts` as a nested dict:
`crop_impacts[crop][aggregation_key]` → 1-D NumPy array of annual yields in
t DM ha⁻¹ yr⁻¹.

---

## Caveats and Limitations

- The emulators capture **climatological-mean** yield responses only; year-to-year
  variability is not represented (see Sect. 2.2, Franke et al., for discussion of
  why annual and climatological responses differ).
- Nitrogen is applied **uniformly** across the globe. Country- or region-specific
  fertilisation rates are not currently supported.
- Only **rainfed** yield emulators (`K_rf`) are used.  Irrigated emulators exist
  in the Zenodo archive but are not yet exposed.
- Grid cells outside the current cultivated extent may produce non-physical yield
  estimates; the polynomial was validated primarily over harvested-area masks
  (Sect. 4, Franke et al.).
- Projections exceeding the training bounds (notably T > T_AgMERRA + 6 °C under
  high-end scenarios) are extrapolated; check `T_oob` / `W_oob` if this matters
  for your application.
