# METEOR Monthly Noise Generation

## Overview
a monthly noise generation capability for METEOR that integrates temperature-dependent seasonal cycle and PCA/VARX methodology into the broader METEOR workflow.

### 1. Core Noise Generation Class (`MeteorNoiseGenerator`)
**Location**: `src/meteor/noise_generator.py`

**Key Features**:
- Temperature-dependent seasonal cycle modeling using modulated harmonic regression
- PCA-based spatial decomposition of climate anomalies
- VARX modeling of principal components with external drivers
- Stochastic simulation of new climate realizations
- Model persistence (save/load trained models)

**Key Methods**:
- `fit(monthly_data, variable_name)` - Train on monthly composite data
- `generate_realization(global_temp_trajectory, n_realizations=1)` - Generate stochastic realizations
- `save_model(filepath)` / `load_model(filepath)` - Model persistence

### 2. Extended Data Getter (`Cmip6MeteorDataGetter`)
**Location**: `src/meteor/cmip6_meteor_data_getter.py`

**New Methods**:
- `get_single_var_mod_data_monthly()` - Get monthly data with proper formatting
- `train_noise_model()` - Train noise model for single model/variable combination
- `train_all_noise_models()` - Batch training for multiple models/variables

**Enhanced Methods**:
- `make_meteor_training_data(monthly=True)` - Now supports monthly output
- `make_meteor_training_data_composite(monthly=True)` - Monthly composite data with proper concatenation

### 3. Extended Pattern Scaling (`MeteorPatternScaling`)
**Location**: `src/meteor/meteor.py`

**New Methods**:
- `predict_monthly_with_noise()` - Generate monthly predictions with stochastic noise
- `train_and_predict_monthly()` - Convenience method for complete workflow

### 4. Integration and Utilities
**Location**: `src/meteor/__init__.py`

- Exposed new classes in main METEOR namespace
- Added convenience function `train_noise_model_from_composite()`


## Functionality

### Noise Model Training
```python
# Train from historical+scenario composite
noise_model = data_getter.train_noise_model(
    experiments=["historical", "ssp245"],
    model="CanESM5", 
    variable_name="tas",
    n_modes=10,
    lag_order=2,
    cache_dir="./models"
)
```

### Stochastic Realization Generation
```python
# Generate multiple realizations
realizations = noise_model.generate_realization(
    temperature_trajectory,
    n_realizations=5,
    random_seed=42
)
```

### Integrated METEOR Workflow
```python
# Full monthly prediction workflow
monthly_predictions = pattern_scaling.predict_monthly_with_noise(
    emissions_data, concentrations_data, 
    fields=["tas", "pr"],
    noise_models=trained_models,
    n_realizations=3
)
```


## Usage Examples

### Basic Training
```python
from meteor import Cmip6MeteorDataGetter, MeteorNoiseGenerator

# Setup and train
data_getter = Cmip6MeteorDataGetter(...)
noise_model = data_getter.train_noise_model(
    ["historical", "ssp245"], "CanESM5", "tas"
)
```

### Realization Generation
```python
# Generate monthly climate data
monthly_realization = noise_model.generate_realization(
    temperature_trajectory=temp_series,
    random_seed=42
)
```

### Full Workflow
```python
# Complete prediction with noise
monthly_climate = pattern_scaling.train_and_predict_monthly(
    data_getter, ["historical", "ssp245"], "CanESM5",
    emissions_data, concentrations_data, ["tas", "pr"]
)
```
