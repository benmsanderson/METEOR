# METEOR Baseline Performance Profile

Model: `NorESM2-MM`, variables `[tas, pr]`.
Harness: [scripts/profiling/run_profile.py](../scripts/profiling/run_profile.py) — cProfile + tracemalloc + psutil RSS.
Runs: [scripts/profiling/results/](../scripts/profiling/results/).

## Summary

| Workload | Params | Wall (s) | Peak MB | RSS Δ (MB) |
|---|---|---:|---:|---:|
| `train_fresh` | fresh cache, tas+pr | 63.4 | 16,175 | 178 |
| `gen_no_noise` | N=1, 2 aggr. | 23.9 | 7,416 | 233 |
| `gen_global_ts` | N=10 | 23.9 | 7,109 | 430 |
| `gen_global_ts` | N=100 | 51.7 | 7,231 | 534 |
| `gen_global_ts` | N=500 | 130.5 | 7,773 | 847 |
| `gen_global_ts` | N=1000 | 228.0 | 8,450 | 304 |
| `gen_multi_scale` | N=100, 5 aggr. | 52.0 | 7,566 | 235 |
| `gen_impacts` | N=100, HDD/CDD | 63.2 | 7,232 | 291 |
| `gen_gridded` | N=1 | 171.2 | 10,926 | 167 |
| `gen_gridded` | N=5 | 416.0 | 33,259 | 156 |

Each generation workload includes a call to `emulator.train()` with cache hit (~22s fixed cost from importing meteor + loading pkl models + loading CICERO forcing + rebuilding pattern-scaling training tables). This is the same across every generation row.

## Key observations

### 1. Noise generation dominates timeseries workloads

At N=1000 (`gen_global_ts_n1000`, 228s wall):

| Function | Location | cumtime | tottime | ncalls |
|---|---|---:|---:|---:|
| `_generate_stochastic_pcs` | [noise_generator.py:626](../src/meteor/noise_generator.py#L626) | 200.1s | **181.1s** | 2000 |
| `.copy()` (numpy) | (inside AR loop) | — | 18.3s | 8.4M |
| `_get_or_compute_pattern_scaling` | [meteor_interface.py:778](../src/meteor/meteor_interface.py#L778) | 5.4s | — | 2 |
| `global_mean` | [geo_data_utils.py:202](../src/meteor/geo_data_utils.py#L202) | 5.3s | — | 6 |

`_generate_stochastic_pcs` is a Python-level for-loop over ~3012 time steps for the VAR(2) autoregression ([noise_generator.py:681-695](../src/meteor/noise_generator.py#L681)). Called 2× per realization (tas + pr), so 2000 invocations. Per-call cost ≈ 100 ms, all spent in the Python loop; the noise-shock batching optimization already landed but the state loop itself is still per-timestep-per-realization.

**Scaling of `gen_global_ts` is essentially linear**: fixed ~22s (train w/ cache hit + pattern-scaling prediction), then ~0.2s per realization per variable. At N=100 we're already 60% inside `_generate_stochastic_pcs`; at N=1000, 88%.

### 2. Gridded output is dominated by the precipitation gamma transform, not noise

At N=1 (`gen_gridded_n1`, 171s wall):

| Function | Location | cumtime | tottime |
|---|---|---:|---:|
| `fit_distribution_parameters_3d_seasonal` | [precipitation_transform.py:440](../src/meteor/precipitation_transform.py#L440) | 97.3s | — |
| `fit_distribution_parameters_3d` | [precipitation_transform.py:153](../src/meteor/precipitation_transform.py#L153) | 96.6s | — |
| ↳ `fit_distribution_parameters_1d` | [precipitation_transform.py:25](../src/meteor/precipitation_transform.py#L25) | 92.8s | 5.1s (663k calls) |
| ↳↳ `scipy.stats.gamma.fit` | scipy | 83.8s | 29.0s (663k calls) |
| `apply_distribution_transform_seasonal` | [precipitation_transform.py:487](../src/meteor/precipitation_transform.py#L487) | 46.4s | — |
| ↳ `gamma._ppf` | scipy | 43.4s | **43.4s** (24 calls) |
| `generate_realization` (noise) | [noise_generator.py:483](../src/meteor/noise_generator.py#L483) | — | 7.0s |

Two independent problems:

- **Fit is a per-gridpoint Python loop** at [precipitation_transform.py:212-228](../src/meteor/precipitation_transform.py#L212): `for i in range(n_spatial)` calling `scipy.stats.gamma.fit()` per grid cell × 12 months × 2 variables = 663k calls. This cost is **independent of N** — it fits the *reference* distribution — but is not cached across generation calls. Roughly 97s per gridded call.
- **Apply is scipy's `gamma._ppf`** which appears vectorized (only 24 calls) but internally spends 43s in `_brentq` root-finding across all gridpoint × month elements. This scales with N: N=1 → 46s, N=5 → 246s (≈5×).

Memory scales badly with N: N=1 peak 11 GB, N=5 peak 33 GB. Extrapolates to ~60 GB at N=10, mostly from full-resolution transformed ensemble arrays held in memory.

### 3. Degree-days is a serial per-realization loop

`gen_impacts` at N=100 (63s wall):

| Function | Location | cumtime | ncalls |
|---|---|---:|---:|
| `_apply_impacts` | [meteor_interface.py:1985](../src/meteor/meteor_interface.py#L1985) | 20.6s | 1 |
| `degree_days.calculate` | [impacts/degree_days.py:145](../src/meteor/impacts/degree_days.py#L145) | 20.5s | 100 |

At N=100, HDD/CDD takes ~0.2s per realization × 100 realizations. Scales linearly. Trivially parallelizable (embarrassingly parallel across realizations), but currently serial.

### 4. Training cost is roughly 65% noise-model fit, 20% pattern scaling, 15% imports/loading

`train_fresh` (63s wall):

| Component | Location | cumtime |
|---|---|---:|
| `_train_noise_model` (both vars) | [meteor_interface.py:440](../src/meteor/meteor_interface.py#L440) | 40.3s |
| ↳ `train_noise_model_from_cmip6` | [noise_generator.py:1204](../src/meteor/noise_generator.py#L1204) | 34.9s |
| ↳↳ `NoiseGenerator.fit` | [noise_generator.py:208](../src/meteor/noise_generator.py#L208) | 29.1s |
| `_train_pattern_scaling` (both vars) | [meteor_interface.py:388](../src/meteor/meteor_interface.py#L388) | 10.7s |
| ↳ CMIP6 composite training data | [cmip6_meteor_data_getter.py:908](../src/meteor/cmip6_meteor_data_getter.py#L908) | 6.3s |
| ↳ `MeteorPatternScaling.__init__` | [meteor.py:220](../src/meteor/meteor.py#L220) | 9.5s |
| ↳↳ `get_timescales` (lmfit) | [pattern_logic_lib.py:470](../src/meteor/pattern_logic_lib.py#L470) | 4.0s |
| Module imports (meteor + deps) | | 8.9s |

The noise-model `fit` (29s of self-time inside sklearn PCA + statsmodels VAR) is the single largest training component. lmfit pattern scaling is only 4s — not the concern I flagged in the plan.

## Optimization targets, ranked by leverage

Ordered by expected wall-time win per unit implementation effort.

### High leverage

1. **Vectorize `_generate_stochastic_pcs` across realizations** — [noise_generator.py:626](../src/meteor/noise_generator.py#L626). The AR loop is serial in time (unavoidable) but currently also serial across realizations. Restructure to carry state as `(n_realizations, n_modes)` and do batched matmuls. Expected: 5-10× speedup on the 180s hot spot → cuts N=1000 wall by ~150s. Also eliminates the 8.4M `.copy()` calls.

2. **Vectorize `fit_distribution_parameters_3d`** — [precipitation_transform.py:212](../src/meteor/precipitation_transform.py#L212). Replace the per-gridpoint `for i in range(n_spatial)` scipy `gamma.fit` with a vectorized method-of-moments estimator (shape = mean² / var, scale = var / mean, along the time axis). Expected: 90s+ → <1s. Kills nearly all of the fixed cost of gridded output.

3. **Cache fitted gamma parameters** — the fit is a function of the reference precipitation data + noise model, not of the ensemble. Cache alongside the noise model pkl (or once per `MeteorInterface` instance). Even without vectorization, this makes the second gridded call in the same session ~2× faster.

### Medium leverage

4. **Vectorize `_apply_impacts` degree-days loop** — [meteor_interface.py:2089](../src/meteor/meteor_interface.py#L2089). Currently serial `for i in range(n_realizations)`. Either vectorize inside `DegreeDays.calculate` to accept a leading realization axis, or `joblib.Parallel`. Expected: ~10× speedup for N=100+.

5. **Investigate `apply_distribution_transform` `_ppf`** — [precipitation_transform.py:246](../src/meteor/precipitation_transform.py#L246). scipy's `gamma.ppf` uses `_brentq` root-finding when it could use the closed-form via `scipy.special.gammaincinv`. 43s → potentially a few seconds. Or use a lookup-table quantile map.

6. **Kill the training-data re-preparation on cache hit** — noticed in every generation workload log: even when both pattern-scaling pkl files are cache-hits, `train()` still runs "Preparing pattern scaling training data" and loads CICERO forcing data (~6s). Should be short-circuited when the model is loaded from cache. Recovers ~6s from every generation call.

### Low leverage (paper revision context)

7. Noise-model training PCA/VAR fit (29s) — inside sklearn/statsmodels. Meaningful only if re-training becomes common; probably not worth it if models are typically cached.

8. Memory scaling in gridded output — 33 GB at N=5 is problematic. Streaming per-realization or chunking the transformed-ensemble array would help. Only matters if the paper argues for large gridded ensembles.

## Optimization outcomes

_Populated as follow-up PRs land. See CHANGELOG entries under ``[Unreleased]``._

### N=10 gridded ran out of memory

Extrapolated ~65 GB. The n=1 and n=5 baselines above bracket the working range; anything above N≈8 needs the memory optimization (target #8).

## Reproducing

```bash
# One workload
python scripts/profiling/run_profile.py --workload gen_global_ts --n 100

# Everything
for w in train_fresh gen_no_noise gen_multi_scale gen_impacts; do
    python scripts/profiling/run_profile.py --workload $w --tag baseline
done
for n in 10 100 500 1000; do
    python scripts/profiling/run_profile.py --workload gen_global_ts --n $n --tag n$n
done
for n in 1 5 10; do
    python scripts/profiling/run_profile.py --workload gen_gridded --n $n --tag n$n
done

# Analysis
python scripts/profiling/analyze.py                       # summary + leaderboard
python scripts/profiling/analyze.py --run gen_global_ts_n1000   # drill-down
python scripts/profiling/analyze.py --run gen_gridded_n5 --exclude frozen
```
