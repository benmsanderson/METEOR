# Crop Yield Emulation in METEOR

*Draft section for EU project report — ~2 pages.*

## Overview

METEOR has been extended with a fully-integrated global crop yield emulator
covering the five staple crops — maize, rice, soybean, spring wheat and winter
wheat — that together account for the bulk of global calorific production. The
emulator reproduces the yield response of the nine global gridded crop models
(GGCMs) that took part in the GGCMI Phase 2 inter-comparison
(Franke et al., 2020, *Geosci. Model Dev.* **13**, 3995–4018). It allows
METEOR users to translate any climate scenario produced by the spatial emulator
into a corresponding global, 0.5°-resolution annual yield projection in a few
seconds per simulation year, with no need to run a full process-based crop
model.

## Scientific basis

Process-based crop models are computationally expensive and difficult to embed
in large climate ensembles or in integrated-assessment workflows. GGCMI
Phase 2 addressed this by running each participating crop model over a
structured parameter sweep of up to 756 combinations of carbon dioxide
concentration (C), temperature perturbation (T), water supply (W) and nitrogen
application (N), each repeated under two adaptation assumptions, and fitting a
spatially-varying third-order polynomial to the climatological yield response
at every 0.5° grid cell. The resulting polynomial coefficient tensors
faithfully reproduce the long-term mean yield of each contributing crop model
under arbitrary future C/T/W/N conditions at negligible computational cost,
which is precisely the regime in which METEOR operates.

## Mathematical formulation

For every (crop, GGCM, adaptation-variant) combination, yield Y at each
0.5° grid cell is evaluated as

> Y = Σᵢⱼₖₗ Kᵢⱼₖₗ · Cⁱ · Tʲ · Wᵏ · Nˡ,        i+j+k+l ≤ 3,

with 34 active monomials (Eq. 1 of Franke et al. 2020): the polynomial is
third-order in C, T and W and second-order in N (the N³ term is omitted
because only three nitrogen levels are sampled in the training data). The
coefficient tensor *K* therefore has shape (34, 360, 720), stored as a
single netCDF variable per crop / model / variant.

The four polynomial inputs follow the same conventions as the GGCMI Phase 2
protocol:

| Input | Physical meaning | Transformation                | Valid range                              |
|-------|------------------|-------------------------------|------------------------------------------|
| C     | CO₂ concentration | absolute (ppm)               | 360 – 810 ppm                            |
| T     | Temperature       | anomaly vs. AgMERRA, °C       | −1 °C to +6 °C from AgMERRA              |
| W     | Precipitation     | ratio to AgMERRA              | 0.5× to 1.3× AgMERRA                     |
| N     | Nitrogen          | uniform application (kg/ha/yr)| 10 – 200 kg N ha⁻¹ yr⁻¹                  |

Anomalies and the precipitation ratio are taken relative to the
1980–2010 AgMERRA climatology (Ruane et al., 2015), which is the historical
reference climate used by most of the GGCMI Phase 2 participants. Inputs are
clamped to the valid ranges before evaluation; the per-cell out-of-bounds
offsets for T and W are returned alongside the yield field so that users can
flag grid cells where projections enter the extrapolation regime. Negative
yields are clipped to zero.

## Implementation in METEOR

The implementation lives under `src/meteor/impacts/ggcm/` and consists of four
loosely-coupled modules: a vendored polynomial evaluator (`coefficients.py`)
that reproduces the original GGCMI mathematics in pure NumPy, a baseline
loader (`baseline.py`) that ships the pre-computed AgMERRA 1980–2010
climatology as two 0.5° netCDF files inside the package, a download manager
(`downloader.py`) that fetches the polynomial coefficient files on demand from
the published Zenodo archive (record 3592453) with resume support and
on-disk caching, and a static availability catalogue (`data_catalog.py`)
mapping every supported (crop, model, variant) triple to its filename and
URL. Five crops × nine GGCMs × two adaptation variants give up to 82
coefficient files, totalling ~9 GB, but only those actually requested by the
user are downloaded.

The emulator is fully integrated into the high-level
`MeteorInterface.generate_ensemble_outputs()` workflow via the
`_apply_crop_yields()` method. Per simulated year, the method (i) reads the
pattern-scaling annual gridded fields of `tas` and `pr` produced by METEOR,
(ii) reconstructs absolute temperature and precipitation by adding the
gridded piControl baseline of the host CMIP6 model, (iii) bilinearly regrids
both fields to the 0.5° GGCM target grid, filling any residual NaN gaps
along coastlines with the AgMERRA climatology, (iv) reads the corresponding
global annual mean CO₂ concentration from the scenario forcing data and
(v) evaluates `get_yields()` for each requested crop. The resulting yield
fields are then aggregated using exactly the same regional/point keys used
elsewhere in METEOR (global mean, AR6 regions, custom bounding boxes, named
gridpoints), so that crop-yield outputs are interoperable with the climate
diagnostics produced for the same simulation.

## Outputs and use in the project

A typical METEOR call returns, for every scenario realisation, a nested
dictionary of annual yield arrays in tonnes of dry matter per hectare per
year (`crop_impacts[crop][aggregation_key]`). Across an ensemble of pattern-
scaling realisations this provides both ensemble-mean projections and the
spread arising from the climate emulator, which can in turn be propagated
to econometric, food-security or land-use sub-models in the wider EU
project. Because the polynomial captures the response of nine independent
crop models, switching between them — or producing a multi-model ensemble —
amounts to swapping a single file path.

## Limitations and planned extensions

The emulator targets climatological yield responses; year-to-year variability
around the long-term mean is not represented in the polynomial itself, so
high-frequency variability in METEOR climate fields is carried into yields
only through the annual mean inputs. Nitrogen is applied uniformly across
the globe; spatially-resolved fertilisation maps are a natural next step.
Only the rainfed coefficient tensors (`K_rf`) are currently exposed, although
irrigated variants are available in the Zenodo archive and can be added with
no change to the evaluator. Finally, grid cells outside the currently
cultivated area should be masked when reporting global aggregates; the
polynomial was calibrated primarily over harvested-area masks and produces
non-physical numbers outside them.
