# METEOR Quick Start Guide

## Basic Monthly Climate Projection Workflow

### 1. Setup and Data Access
```python
from meteor import Cmip6MeteorDataGetter, MeteorPatternScaling
import numpy as np

# Initialize CMIP6 data access
data_getter = Cmip6MeteorDataGetter(
    exps=["piControl", "abrupt-4xCO2", "historical", "ssp245"],
    flds=["tas", "pr"]
)
model = "CanESM5"
```

### 2. Train Pattern Scaling Model
```python
# Create pattern scaling model with aerosol residuals
pattern_model = MeteorPatternScaling(
    "monthly-demo",
    {"tas": 2, "pr": 2},
    lambda key: data_getter.make_meteor_training_data_composite(key, model)
        if isinstance(key, list) else data_getter.make_meteor_training_data(key, model),
    exp_list=["base", "co2x4", ["historical", "ssp245"]]
)
```

### 3. Train Monthly Noise Model
```python
# Train noise model for monthly variability
noise_model = data_getter.train_noise_model(
    experiments=["historical", "ssp245"],
    model=model,
    variable_name="tas",
    n_modes=8,
    cache_dir="./cache"
)
```

### 4. Generate Projections
```python
from ciceroscm import input_handler

# Load scenario data
emissions = input_handler.read_emissions("ssp245_em_RCMIP.txt")
concentrations = input_handler.read_inputfile("ssp245_conc_RCMIP.txt")

# Generate annual projection
annual_pred = pattern_model.predict_from_combined_experiment(
    emissions, concentrations, ["tas"]
)["tas"]

# Convert to monthly
monthly_annual = pattern_model.to_monthly(annual_pred, start_year=1850)

# Generate noise realizations
global_temp = annual_pred.mean(dim=["lat", "lon"]).values
monthly_temp = np.repeat(global_temp, 12)

noise = noise_model.generate_realization(
    monthly_temp, 
    noise_only=True, 
    random_seed=42
)

# Combine for full monthly projection
monthly_climate = monthly_annual + noise
```

### 5. Quick Analysis
```python
import matplotlib.pyplot as plt

# Plot global mean time series
global_mean = monthly_climate.mean(dim=["lat", "lon"])
years = 1850 + np.arange(len(global_mean))/12

plt.figure(figsize=(12, 6))
plt.plot(years, global_mean)
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (K)")
plt.title("METEOR Monthly Climate Projection")
plt.show()
```

## Key Methods Reference

### MeteorPatternScaling
- `predict_from_combined_experiment()`: Generate annual climate projections
- `to_monthly()`: Convert annual to monthly resolution

### MeteorNoiseGenerator  
- `generate_realization(noise_only=True)`: Create monthly climate noise
- `fit()`: Train on monthly CMIP6 data

### Cmip6MeteorDataGetter
- `train_noise_model()`: Integrated noise model training
- `make_meteor_training_data()`: Pattern scaling training data

## Common Parameters

### Pattern Scaling
- `n_patterns`: Number of response patterns per variable (typically 2)
- `exp_list`: Training experiments ["base", "co2x4", "composite"]

### Noise Generation
- `n_modes`: PCA modes to retain (8-12 typical)
- `lag_order`: VARX temporal lags (1-3 typical)
- `noise_only=True`: Generate additive noise component

### Ensemble Generation
- Multiple `random_seed` values for different realizations
- Combine same annual pattern with different noise realizations

This workflow provides monthly climate projections that combine long-term climate trends from pattern scaling with realistic short-term variability from stochastic modeling.