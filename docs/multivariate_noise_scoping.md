# METEOR v2: a multivariate noise model — scoping

*Scoping document, 2026-10-02; revised 2026-10-16 to take IAM-FIRE as a
consumer. Code references are to `base` at `2edad23`; IAM-FIRE references are
to [bc3LC/IAM-FIRE](https://github.com/bc3LC/IAM-FIRE) at `ad04771`.*

## Summary

METEOR v2 should replace independent per-variable noise with a **conditional
chain**: variables are generated in a fixed order, each conditioned on those
before it. This is what lets METEOR represent compound extremes (hot and dry
together), and add variables beyond temperature and precipitation without
refitting the ones already trained.

Today each variable's internal variability is fitted and drawn independently,
so a hot-and-dry summer comes out about as often as the two separate chances
multiplied together. Real climate, and CMIP6, has hot summers that tend to be
dry over land, so METEOR under-counts exactly the events that matter most for
crops and fire.

The chain keeps v1's speed (generation stays linear in the number of
variables), keeps each variable's fit separate, and reduces exactly to v1 when
every coupling is zero. Temperature and precipitation are the first case; an
annual heat extreme (hottest day of the year, TXx) is the proposed third,
because it tests the design's hardest part: mixing monthly and annual time
steps.

v2 also has a first named consumer: **IAM-FIRE**, the burned-area and
fire-emissions emulator of the Basque Centre for Climate Change (BC3). Its
climate predictors fix which variables v2 adds for fire (humidity, a monthly
dry-day count, surface solar radiation) and in what form. They also show that
v2 is not only a noise project. IAM-FIRE depends first on the **forced** change
in each month, which v1 does not cleanly separate, and only then on
variability. So v2 runs as two tracks: a forced-response track, which can start
on v1 and gives IAM-FIRE its mean projections, and the noise chain, which
gives it realistic year-to-year fire.

## Motivation

The impact metrics people ask for mostly depend on more than one variable, or
on weather finer than a monthly mean.

| Metric | Needs | Possible in v1? |
| --- | --- | --- |
| Drought (SPI-3, -6, -12) | Monthly precipitation only | Yes: METEOR already fits a gamma distribution to precipitation, which is what SPI uses |
| Unusually hot, wet or dry seasons | Monthly temperature or precipitation | Yes, one variable at a time |
| Heating, cooling and growing degree days | Monthly temperature | Yes: `meteor.impacts` already has heating and cooling degree days |
| Hot-and-dry seasons, crop stress | Temperature and precipitation **together** | Wrong: their variability is uncorrelated, so joint events are too rare |
| Hottest day, wettest day (TXx, Rx1day) | Daily data | No |
| Burned area and fire emissions (IAM-FIRE) | Monthly temperature, humidity, solar radiation and precipitation, and daily dry-day counts, together | No: see below |
| Heat stress (WBGT) | Temperature and humidity | No |
| Soil-moisture drought | Soil moisture, with long memory | No |

The first three rows can be built downstream (e.g. in meteor-view) on v1.
Everything below them needs either coupled variability or new variables, which
is what v2 is for.

Compound hot-dry events are the most pressing case. Over mid-latitude land in
summer, hot months tend to be dry months, so the joint chance is well above the
product of the separate chances; Zscheischler & Seneviratne (2017) is the usual
reference for how much that dependence matters. METEOR v1 gives a ratio of
about 1 everywhere, by construction. The crop model (`cropmodel` branch) is
driven by exactly these growing-season heat-and-drought combinations.

## A consumer: IAM-FIRE

IAM-FIRE (repository above; inputs at
[doi:10.5281/zenodo.20312037](https://doi.org/10.5281/zenodo.20312037))
projects burned area and fire carbon emissions for IAM scenarios. Today it
gets its climate from a chain of heavy tools: STITCHES (Tebaldi et al. 2022)
recombines one ESM's archived daily output to follow a scenario's global
temperature, then BASD (ISIMIP3BASD, Lange 2019) bias-adjusts and downscales it
to W5E5 at 0.5°, daily. It has been run for two ESMs, CanESM5 and
MPI-ESM1-2-LR, both among METEOR's models. METEOR would replace that chain,
for any of its models, in minutes rather than HPC campaigns.

**What the fire model needs.** Burned area comes from a binomial GLM with a
logit link and 14 coefficients (`code/python/IF_glm_model.py`), fitted to the
**time-mean** of observed predictors over 2002–2019 at 0.5°, and then applied
year by year. Its climate predictors:

| Predictor | How IAM-FIRE computes it | Inputs |
| --- | --- | --- |
| VPD (log) | Annual maximum of monthly vapour-pressure deficit | Monthly `tas`, `hurs` |
| Dry days (log) | Annual maximum of the monthly count of days with ≤ 1 mm | Daily `pr` |
| Dry-day seasonality (log, and its square) | (max − min) / mean of that year's 12 monthly counts | Daily `pr` |
| Precipitation (log) | Mean monthly total | Monthly `pr` |
| GPP (log) | P-model (pyrealm; Stocker et al. 2020), monthly, with fAPAR held at its climatology | Monthly `tas`, VPD, `rsds`, CO₂ |

The rest of the GLM is land use (grassland, cropland), the human development
index, grazing pressure and terrain, which come from the IAM and static
layers, not from METEOR. Fire emissions are burned area times fuel
consumption, which IAM-FIRE scales with annual-mean VPD, so they need nothing
new. Wind is computed but is not in the fitted model.

**What that means for v2.** Five requirements follow, each picked up below.

1. **Forced change month by month.** The annual maximum of monthly VPD and
   the seasonality of dry days depend on how each month changes, not the
   annual mean. This is the largest gap, and it is in the forced response, not
   the noise (see *Forced response* below).
2. **Humidity, coupled to temperature.** VPD is a nonlinear function of
   temperature and humidity together, so humidity has to come with its
   covariance with temperature. That is exactly what the chain provides; with
   it, VPD is computed from generated `tas` and `hurs` rather than emulated
   as its own field, and humidity is reused for heat stress.
3. **A monthly derived index.** The dry-day count needs daily precipitation
   that METEOR will not generate. It is emulated directly, as a monthly
   variable conditioned on the same month's precipitation: without that
   coupling a wet month could come out full of dry days.
4. **Several drivers for humidity and radiation.** VPD is IAM-FIRE's main
   climate driver, and land humidity falls partly because CO₂ closes plant
   stomata, not only because of warming; surface solar radiation is
   aerosol-dominated.
5. **Changes, not levels.** The GLM uses logs of absolute values, fitted on
   observations. The consistent input is the observed predictor plus METEOR's
   change in that predictor (see *Implications* below).

## Current design (v1)

In v1, every variable is a separate model: its own forced pattern, its own
variability, its own random draws.

- **Forced response:** pattern scaling of each variable against global
  temperature from the simple climate model (`scm_forcer_engine.py`,
  `pattern_logic_lib.py`), annual. `to_monthly` (`meteor.py:599`) repeats each
  year's value across its 12 months.
- **Seasonal cycle:** fitted inside the noise model, per grid cell, as annual
  and semiannual harmonics, each also multiplied by global temperature
  (`noise_generator.py:143`). So the noise model carries the *forced change* in
  the seasonal cycle, scaled linearly with global warming: `noise_only=True`
  keeps those temperature-modulated harmonics and removes only the intercept
  and the direct temperature term.
- **Variability:** one `MeteorNoiseGenerator` per variable
  (`noise_generator.py`). It fits 40 area-weighted EOFs to the variable's
  anomalies, then a VAR model (lag 2) on their principal components, with
  global temperature as an exogenous term. It is trained on historical plus
  the training scenario (`ssp245` by default), against the pattern model's
  own warming for that scenario (`meteor_interface.py:500`); piControl, cut to
  its first 150 years (`cmip6_meteor_data_getter.py:679`), serves as the
  baseline for anomalies.
- **Non-Gaussian variables:** a per-month distribution transform (gamma for
  precipitation) from the transform registry (`variable_transforms.py`),
  applied around the noise model. Entries for `hurs` and `sfcWind` are
  sketched there but commented out.
- **Generation:** `generate_ensemble_outputs` loops over variables, each with
  its own random draws (`meteor_interface.py:739`). Temperature and
  precipitation share the forced warming, but their variability is independent.
- **Artifacts:** one portable bundle per variable per model, which meteor-view
  loads and runs in the browser.

This design is fast, simple, and lets each variable be trained and shipped on
its own. v2 should keep all three properties.

## Requirements and constraints

1. **Speed.** Generation cost must stay about linear in the number of
   variables. Heavy computation belongs in training, never at runtime.
2. **Backward compatibility.** With every coupling set to zero, v2 must
   reproduce v1 exactly, so v1 golden fixtures still pass.
3. **Incremental variables.** Adding a variable must not refit the ones already
   trained. With ~40 models, some needing a 32 GB machine, each full retraining
   is a campaign.
4. **Estimable from the data.** The noise model sees about 250 years of
   historical and scenario data per ensemble member (1850–2100), so about
   250 samples per calendar month. Any coupling must be
   estimable from that, or from more data drawn on deliberately.
5. **Lazy loading in the browser.** meteor-view should load only the variables a
   view needs, plus whatever they are conditioned on.
6. **Uneven availability.** Fewer CMIP6 models publish humidity, wind, soil
   moisture or daily output than temperature and precipitation. Each model must
   be able to carry its own set of variables.
7. **Separable forced seasonality.** A consumer must be able to get the forced
   change in each calendar month, without variability, as cheaply as the
   annual forced response. IAM-FIRE's mean projections need nothing else.

## Kinds of variable

The variables worth adding differ in ways a single model has to accommodate.

| Kind | Examples | What the design must allow |
| --- | --- | --- |
| Unbounded or skewed monthly fields | tas, pr, psl, rsds | Already handled in v1: transform to Gaussian, EOFs, VAR model |
| Bounded fields | hurs (0–100%), soil-moisture fraction | Other marginal transforms (logit, beta), added through the transform registry |
| Monthly counts from daily data | Dry days (≤ 1 mm), wet days, frost days | A bounded count, 0 to the days in the month (a binomial-type or logit transform), trained on indices computed from each ESM's daily output, and conditioned on the monthly variable it derives from |
| Long-memory fields | mrso (soil moisture) | Longer or seasonal lags; lag 2 in months misses interannual memory |
| Annual extremes | TXx, Rx1day | An annual time step, extreme-value (GEV) marginals, conditioned on the monthly state of that year |
| Derived indices | Fire Weather Index, WBGT | The index emulated directly, because the daily inputs needed to compute it are not available |

Statistics *across* a year's months, such as IAM-FIRE's annual-maximum VPD,
annual-maximum dry-day count and dry-day seasonality, are not variables:
they are computed from the 12 generated months, so they inherit the monthly
fields' coupling and need no marginal of their own.

The forced side generalizes too. Some variables do not scale with global
temperature alone: surface solar radiation is aerosol-dominated, and CO₂'s
direct effect on plants changes soil moisture, humidity and runoff
independently of warming. Each variable should be able to scale with more than
one driver; how far the current forcing engine already supports this per
forcing agent is an open question.

## Options considered

Only the conditional chain meets all six requirements on variability; the
others either can't be estimated at scale or force full retraining. (The
seventh, forced seasonality, is independent of this choice.)

| Option | How it couples | Captures lagged effects | Adding a variable | Estimable at 6 variables × 40 modes | Verdict |
| --- | --- | --- | --- | --- | --- |
| Joint shock covariance | One covariance across all variables' shocks, per month | No | Refits the whole matrix | No: 240×240 per month from a few hundred samples is ill-conditioned | Fine for tas–pr alone; doesn't scale |
| Joint VAR model | One VAR on all variables' components together | Yes | Refits everything | No: parameters grow with the square of total modes | Overfits |
| Shared EOFs | One EOF basis on all fields stacked | Yes | Refits the basis, changes every bundle | Weighting between variables is arbitrary | Too disruptive |
| Shared latent drivers | Every variable loads on a few common modes (ENSO-like) | Yes, through the drivers | Fits one new set of loadings | Yes | Physical, but hard to fit; fallback |
| **Conditional chain** | Each variable regressed on earlier ones, plus its own noise | Yes, through lagged terms | Fits only the new variable | Yes: one variable's parameters at a time | **Proposed** |

The chain's main cost is that the order sets the direction of influence (see
below).

## Proposed design: the conditional chain

Each variable keeps its v1 machinery (transform, EOFs, VAR model) and gains one
new term: a regression of its shocks on its parents' components.

**The model.** For variable *v* with principal components *u_v* and parents
*P(v)*, in calendar month *m*:

$$
u_{v,t} = \sum_{l=1}^{L_v} A_{v,l}\, u_{v,t-l}
\;+\; \sum_{p \in P(v)} \sum_{l=0}^{K} C_{v,p,l}(m)\, u_{p,t-l}
\;+\; \varepsilon_{v,t}
$$

The first term is v1's VAR model. The second is the coupling: same-month
(*l* = 0) and lagged effects of each parent. With *C* = 0 the model is exactly
v1.

**Ordering.** A proposed starting order:

1. Temperature (root, unchanged from v1)
2. Precipitation, given temperature
3. Dry-day count, given precipitation and temperature
4. Humidity, given temperature and precipitation
5. Surface solar radiation, given temperature, precipitation and humidity
   (cloud links all three)
6. Soil moisture, given temperature and precipitation, with longer own lags
7. Annual indices, given the monthly state of their year: TXx given
   temperature, Rx1day given precipitation

Items 3–5 are IAM-FIRE's. The earlier draft's annual fire-season length,
given temperature, humidity and precipitation, is dropped: IAM-FIRE's
predictors cover fire with monthly variables, and a fire-season index can
be computed from them if it is wanted.

```mermaid
flowchart LR
  subgraph monthly [Monthly variables]
    direction LR
    tas["Temperature<br/><small>root, as in v1</small>"]
    pr["Precipitation<br/><small>given tas</small>"]
    ndd["Dry days<br/><small>given pr, tas</small>"]
    hurs["Humidity<br/><small>given tas, pr</small>"]
    rsds["Solar radiation<br/><small>given tas, pr, hurs</small>"]
    mrso["Soil moisture<br/><small>given tas, pr; long lags</small>"]
    tas --> pr --> ndd
    pr --> hurs --> rsds
    pr --> mrso
  end
  subgraph annual [Annual indices, GEV marginals]
    direction LR
    txx["TXx (hottest day)<br/><small>given summer tas</small>"]
    rx1["Rx1day (wettest day)<br/><small>given pr</small>"]
  end
  tas --> txx
  pr --> rx1
  classDef stage1 stroke-width:3px
  class tas,pr stage1
  classDef fire stroke-dasharray:4 2
  class ndd,hurs,rsds fire
```

Arrows show only the generation order; each box lists all the variables it is
conditioned on. Stage 1 (bold) is temperature and precipitation; dashed boxes
are the variables IAM-FIRE needs.

**Estimation.**

- **Seasonality:** *C* varies with the month through a few smooth harmonics,
  not 12 free matrices. The temperature–precipitation relationship flips sign
  between summer and winter, so a year-round *C* would partly cancel itself,
  but 12 independent fits would overfit.
- **Shrinkage:** ridge or similar on *C*, chosen by cross-validation on
  held-out years.
- **More data:** additional ensemble members, and piControl years beyond
  those used as the baseline.

**Annual indices.** An annual variable is a regression on its parents' monthly
components in the relevant season (e.g. TXx on June–August temperature
components), plus its own residual. That residual gets an extreme-value (GEV)
marginal through the transform registry. This mixes time steps without changing
the monthly model.

**Forced response.** Each variable keeps its own pattern scaling, extended in
two ways that IAM-FIRE needs.

- **Forced seasonality, made explicit.** In v1 the forced change in the
  seasonal cycle lives inside the noise model's harmonics (see *Current
  design*). That works, but a consumer that wants only the forced monthly
  change has to load and run a noise bundle with the noise switched off, and
  v2's couplings act on the stochastic components, not on those harmonics.
  v2 should decide where forced seasonality lives: either kept in the noise
  bundle but documented and exported as a forced component of its own, or
  moved into the pattern model as monthly or harmonic patterns, which would
  also let it scale with more than global temperature. Either way, IAM-FIRE's
  mean projections need only this component plus the annual pattern.
- **Several drivers.** Variables that need it scale with more than one
  driver: aerosol forcing for solar radiation; CO₂ concentration, as well as
  warming, for humidity and soil moisture over land.

**The trade-off.** The order sets the direction of influence. Conditioning soil
moisture on temperature and precipitation is natural, but means drying soil
can't feed back to sustain a heatwave. Temperature's own autocorrelation partly
stands in for that. If validation shows the gap matters, shared latent drivers
are the fallback.

**Cost.** Training adds one regression per variable. Generation adds one small
matrix multiply per parent per month: still linear in the number of variables.

## Validation

Each coupling is accepted only when the emulator reproduces the CMIP6 model's
own joint behaviour, on years held out from training.

- **Correlation maps:** the grid-box correlation between parent and child
  variables in each season, e.g. June–August temperature against
  precipitation, emulator against piControl and historical.
- **Joint exceedance ratio:** the chance of a compound event (e.g. hottest 10%
  for temperature and driest 10% for precipitation in the same season), divided
  by the product of the two separate chances. v1 gives about 1 everywhere; the
  target is the CMIP6 model's own map.
- **Persistence:** autocorrelation and multi-year drought frequency, especially
  for soil moisture and SPI-12.
- **Annual indices:** the distribution of TXx and Rx1day against the model's
  own daily-derived values, including their change with warming.
- **Forced seasonality:** the change in each calendar month's mean, against
  the ESM's own scenario runs, for each variable IAM-FIRE uses.
- **Burned area, end to end:** IAM-FIRE's GLM run on predictors from three
  sources, on held-out years: the ESM's own output, METEOR v1 (independent
  noise) and METEOR v2 (coupled). Because the GLM is nonlinear (logit link,
  logged predictors), the gap between v1 and v2 is a single impact-relevant
  measure of what coupling buys. For CanESM5 and MPI-ESM1-2-LR, IAM-FIRE's
  own STITCHES and BASD run gives a fourth comparison.
- **No regression:** with couplings off, v1 golden fixtures must pass
  unchanged.

## Implications for the artifact format and meteor-view

The chain maps cleanly onto per-variable bundles; the main change is generating
variables together from one random stream.

- **Bundles:** each variable's bundle gains a list of its parents and its
  coupling coefficients. A v1 bundle is a v2 bundle with no parents, so
  existing exports keep working.
- **Format version:** bundles move to `_v2`. Readers accept both, treating v1
  as uncoupled. `docs/emulator_artifact_schema.md` (on the export branch) would
  document the new fields.
- **Generation:** variables are generated in chain order, in one pass, from one
  seeded stream, in Python and in meteor-view's JS kernel alike. A client loads
  only the variables it shows plus their parents.
- **Forced seasonality:** however it is stored, a reader can produce the
  forced monthly change without generating noise.
- **Annual variables:** a new bundle kind with an annual time step and GEV
  parameters.
- **Per-model variable lists:** the model manifest records which variables each
  model has, so clients can hide metrics a model can't support.
- **Testing:** golden fixtures are regenerated for coupled models; v1 fixtures
  still pass with couplings off.
- **New views in meteor-view:** compound-event probabilities ("chance of a
  hot-and-dry summer"), and how often today's 1-in-20 event happens at each
  warming level.

**A fire impact module.** IAM-FIRE's scoring becomes `meteor.impacts.fire`, an
`ImpactCalculator` beside degree days and crops:

- **The GLM, frozen:** its 14 coefficients stored as data, not refitted on
  every run as IAM-FIRE does now. That is small enough for meteor-view.
- **Predictors by the delta method:** each predictor is computed from METEOR
  on the ESM grid, as a change from 2002–2019, regridded to 0.5° and added to
  the observed predictor that the GLM was fitted on. Bias in the ESM's own
  climate then drops out, which BASD otherwise handles, and the observed
  predictor already includes the effect of observed variability.
- **GPP:** the P-model (pyrealm, an optional dependency), monthly and per
  cell, with CO₂ from CICERO-SCM for the same scenario, so temperature and
  CO₂ come from one model run.
- **Fuel consumption:** IAM-FIRE's biome and fuel-load tables, scaled by VPD
  as it does.
- **Non-climate inputs:** land use, human development index, grazing pressure
  and terrain, from the scenario, not from METEOR (see *Open questions*).

## Staging

v2 lands in stages, each with a gate that must pass before the next starts.
Stage 0 is the forced-response track and can run alongside stage 1.

0. **Forced seasonality and IAM-FIRE's forced patterns, on v1.** Settle where
   the forced seasonal cycle lives (see *Forced response*), and fit forced
   patterns for humidity, solar radiation and the dry-day count, each with
   the drivers it needs. Requirement 3 already allows a variable to be added
   with a forced pattern and no noise. Then build `meteor.impacts.fire`
   with the delta method.
   - Gate: each forced monthly change matches the ESM's own scenario runs,
     and burned area from METEOR's forced response tracks burned area from
     the ESM's own output, decade by decade.
   - Delivers: mean burned-area and emission projections from IAM-FIRE for
     every METEOR model with these variables, for any emissions scenario.
1. **The mechanism, with temperature and precipitation.** Build the chain with
   precipitation conditioned on temperature.
   - Gate: joint hot-dry exceedance ratios match CMIP6 for several models, and
     v1 fixtures pass with couplings off.
2. **A third variable: TXx.** Add the annual hottest day, conditioned on summer
   temperature. It tests mixed time steps and GEV marginals, and is the most
   useful single extreme. One pass over each model's daily output extracts
   everything later stages need from it: TXx, Rx1day and the monthly dry-day
   count.
   - Gate: TXx distributions and their change with warming match each model's
     daily-derived values.
3. **meteor-view support.** v2 bundles, chain-order generation in JS, and the
   first compound-event views.
   - Gate: browser output matches METEOR's golden fixtures for coupled models.
4. **Fire weather for IAM-FIRE, then further variables, one at a time.**
   First the dry-day count, humidity and solar radiation, in chain order,
   then IAM-FIRE run on coupled realizations. After that, soil moisture
   (drought with memory) and heat stress, which reuses humidity.
   - Gate per variable: its own validation, without refitting the earlier ones.
   - Gate for IAM-FIRE: the end-to-end burned-area comparison above.

**Relation to the current training run.** v2 is weeks of work, so meteor-view's
remaining large-grid models are still being trained on v1, which completes its
land-masked release. All models are then retrained once on v2 before the
Zenodo deposit.

## Open questions and risks

- [ ] Is precipitation transformed to Gaussian **before** its EOFs, so that
      couplings work on the Gaussian scale and carry through the
      back-transform?
- [ ] Does v1 add residual noise beyond the 40 modes? If so, that residual
      needs coupling too, or local correlations will be too weak.
- [ ] Does the forcing engine already support patterns per forcing agent (CO₂,
      aerosol), or does that need adding for solar radiation, humidity and
      soil moisture?
- [ ] How well do the noise model's temperature-modulated harmonics reproduce
      each ESM's change in the seasonal cycle, and should forced seasonality
      move into the pattern model?
- [ ] How does the crop model (`cropmodel` branch) consume temperature and
      precipitation, and what does it need from v2?
- [ ] Is the ordering right? In particular, should soil moisture come before
      temperature over land, to keep the drying–heat feedback?
- [ ] How many of the CMIP6 models publish each candidate variable: monthly
      `hurs` and `rsds`, and daily `tasmax` and `pr` for TXx, Rx1day and
      dry days? The models with all of IAM-FIRE's set are its ensemble.
- [ ] Are about 250 samples per calendar month enough once couplings are
      added, or should more members be used?
- [ ] Can the IAM-FIRE authors share the GLM's observed predictors and fitted
      coefficients under a licence METEOR can ship, and their STITCHES and BASD
      predictors for CanESM5 and MPI-ESM1-2-LR for validation?
- [ ] Land use for METEOR's scenarios: IAM-FIRE expects ESA CCI land-cover
      fractions downscaled from GCAM by Demeter. The SSP and CMIP7 scenarios
      would come from LUH2 or LUH3, whose classes need mapping onto ESA CCI's.
      The human development index is chosen by SSP, so the CMIP7 markers each
      need an SSP.

Risks:

- **The chain's ordering misses feedbacks.** Mitigation: validate persistence
  and compound frequencies; fall back to shared latent drivers where they
  fail.
- **Annual indices drift from monthly ones.** A TXx inconsistent with the same
  year's summer temperature would confuse users. Mitigation: conditioning on
  that year's components, and checking their joint distribution.
- **Forced seasonality scaled with global temperature alone** mis-scales
  humidity and radiation where CO₂ and aerosols drive them, and so
  mis-scales fire. Mitigation: the multi-driver patterns of stage 0.
- **Retraining cost.** v2 means one more full run across ~40 models, the large
  ones on a 32 GB machine.

## References

- Lange, S. (2019). Trend-preserving bias adjustment and statistical
  downscaling with ISIMIP3BASD (v1.0). *Geoscientific Model Development*, 12,
  3055–3070. <https://doi.org/10.5194/gmd-12-3055-2019>
- Stocker, B. D. et al. (2020). P-model v1.0: an optimality-based light use
  efficiency model for simulating ecosystem gross primary production.
  *Geoscientific Model Development*, 13, 1545–1581.
  <https://doi.org/10.5194/gmd-13-1545-2020>
- Tebaldi, C. et al. (2022). STITCHES: creating new scenarios of climate
  model output by stitching together pieces of existing simulations. *Earth
  System Dynamics*, 13, 1557–1609. <https://doi.org/10.5194/esd-13-1557-2022>
- Zscheischler, J. & Seneviratne, S. I. (2017). Dependence of drivers affects
  risks associated with compound events. *Science Advances*, 3(6), e1700263.
  <https://doi.org/10.1126/sciadv.1700263>
