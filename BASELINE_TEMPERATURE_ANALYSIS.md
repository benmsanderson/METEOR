# METEOR Baseline Temperature Handling: Analysis and Recommendations

## Overview

This document explains how pre-industrial temperatures are removed from METEOR training data and provides guidance on appropriate baseline temperatures for comparing noise model realizations with ESM model output.

## Current Baseline Handling in METEOR

### 1. Pattern Scaling Component

**Location**: `src/meteor/meteor.py`, function `prepare_training_data()`

**Method**: 
```python
ctrl = exp_list.index("base")  # Find piControl experiment
for var in varis:
    dacanom[var] = dac[var].isel(ens=0).drop_vars("ens") - dac[var][
        ctrl, :, :, :, :
    ].mean(dim="year", skipna=True).isel(ens=0).drop_vars("ens")
```

**Baseline**: The mean of the **entire piControl experiment** is subtracted from all experiments. This removes the pre-industrial climatology and creates anomalies relative to pre-industrial conditions.

### 2. Monthly Noise Generation Component

**Location**: `src/meteor/noise_generator.py`, method `fit()`

**Method**:
```python
# Calculate global mean temperature and remove ensemble mean for reference
t_globm = ds[variable_name].mean(dim=["lat", "lon", "ens"])
t_globm = t_globm - t_globm[:500].mean()  # Remove baseline
```

**Baseline**: The mean of the **first 500 months (~42 years)** of the training data is subtracted from the global temperature trajectory used for temperature-dependent seasonal cycles.

## Key Findings

### Different Baseline Approaches

1. **Pattern Scaling**: Uses full piControl climatology as baseline
2. **Noise Generation**: Uses first 42 years of training data as baseline

### Implications for Comparisons

When comparing noise model realizations with ESM output, the appropriate baseline depends on:

1. **What ESM data baseline was used**
2. **Which component of METEOR is being compared**
3. **The time period of interest**

## Baseline Temperature Recommendations

### For Comparing Full METEOR Projections with ESM Output

**Recommended Approach**: Use **pre-industrial climatology** from the same ESM model as baseline for both METEOR output and ESM comparisons.

```python
# Example: Compare METEOR vs ESM using pre-industrial baseline
# 1. Extract pre-industrial climatology from ESM piControl run
pi_control_data = esm_data.sel(experiment="piControl")
pi_baseline = pi_control_data.mean(dim="year")

# 2. Remove this baseline from both METEOR and ESM data
meteor_anomaly = meteor_output - pi_baseline
esm_anomaly = esm_output - pi_baseline

# 3. Compare anomalies
comparison = meteor_anomaly - esm_anomaly
```

### For Comparing Noise-Only Realizations

**Recommended Approach**: Use the **same baseline as the noise training data**.

Since the noise generator removes the first 42 years as baseline:
```python
# If training data spans 1850-2100, baseline is 1850-1892 average
training_start_year = 1850
baseline_period = slice(training_start_year, training_start_year + 42)
noise_baseline = training_data.sel(year=baseline_period).mean(dim="year")
```

### For Temperature-Dependent Seasonal Analysis

**Recommended Approach**: Account for the **global temperature baseline** used in noise training.

The temperature-dependent features use:
```python
t_glob = t_globm - t_globm[:500].mean()  # Relative to first 42 years
```

So temperature trajectories should be **relative to the same 42-year baseline** when generating realizations.

## Practical Guidelines

### 1. Using Custom Global Temperature

When providing `custom_global_temp` to noise training, ensure it uses the **same baseline** as you want for comparisons:

```python
# Option A: Use ESM piControl baseline for consistency with pattern scaling
custom_temp = your_temperature - picontrol_mean_temp

# Option B: Use early period baseline for consistency with default noise training  
custom_temp = your_temperature - your_temperature[:42*12].mean()  # First 42 years
```

### 2. Interpreting Noise Realizations

When `noise_only=True`, the generated realizations represent:
- **Seasonal anomalies** relative to the temperature-dependent seasonal cycle
- **Stochastic variability** with temperature-modulated patterns
- **No absolute temperature information** (intercept and direct temp effect removed)

These should be **added to** your pattern-scaled projections that already include the appropriate baseline.

### 3. Full Workflow Baseline Consistency

For a complete METEOR projection workflow:

```python
# 1. Pattern scaling: automatically uses piControl baseline
patterns = meteor.train_patterns(data_getter, ["base", "co2x4"], model)

# 2. Noise training: use consistent baseline
noise_model = data_getter.train_noise_model(
    experiments=["historical", "ssp245"],
    model=model,
    variable_name="tas",
    custom_global_temp=temp_trajectory_relative_to_picontrol
)

# 3. Combined projection: baselines are already consistent
monthly_projection = patterns.predict_monthly_with_noise(
    emissions, concentrations, 
    noise_models={"tas": noise_model},
    n_realizations=5
)
```

## Current Limitations and Recommendations

### Limitation 1: Inconsistent Baselines
Pattern scaling and noise generation use different baseline periods, which can cause inconsistencies in combined projections.

**Recommendation**: Consider harmonizing baselines by:
- Using piControl climatology for both components, OR
- Making the noise baseline period configurable

### Limitation 2: Hardcoded Baseline Period
The 500-month (42-year) baseline period is hardcoded in the noise generator.

**Recommendation**: Make baseline period configurable:
```python
def fit(self, monthly_data, variable_name, custom_global_temp=None, baseline_years=42):
    # Allow user to specify baseline period
    baseline_months = baseline_years * 12
    t_globm = t_globm - t_globm[:baseline_months].mean()
```

### Limitation 3: Unclear Documentation
The baseline handling is not well documented for users.

**Recommendation**: Add clear documentation about:
- What baselines are used in each component
- How to ensure consistency when combining components
- What baseline to use for comparisons with ESM output

## Summary

**For ESM Comparisons**: Use pre-industrial (piControl) climatology as baseline for both METEOR output and ESM data.

**For Noise-Only Analysis**: Use the same 42-year baseline period as the noise training data.

**For Custom Temperature**: Ensure your temperature trajectory uses the same baseline as intended for your comparison analysis.

**For Combined Projections**: The current METEOR workflow handles baselines automatically, but users should be aware of potential inconsistencies between pattern scaling (piControl baseline) and noise generation (early period baseline).