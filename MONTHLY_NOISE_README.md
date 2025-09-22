# METEOR: Multivariate Emulation of Time-Evolving and Overlapping Responses

## Overview

METEOR is a fast climate emulator that combines pattern scaling for long-term climate response with stochastic noise modeling for short-term variability. It enables rapid generation of ensemble climate projections at both annual and monthly resolution by learning from CMIP6 climate model data.

### Key Capabilities

- **Pattern Scaling**: Rapid climate projection based on global temperature trajectories
- **Monthly Noise Generation**: Stochastic monthly climate variability using PCA/VARX modeling  
- **Multi-forcing Support**: Separate treatment of greenhouse gas and aerosol forcings
- **Ensemble Generation**: Multiple realizations for uncertainty quantification
- **CMIP6 Integration**: Direct access to cloud-based CMIP6 data archives

## Model Description

### Core Methodology

METEOR operates on two complementary modeling approaches:

#### 1. Annual Pattern Scaling
- **Training**: Learn spatial climate response patterns from CMIP6 experiments (piControl, abrupt-4xCO2, historical, SSPs)
- **Forcing**: Use simple climate model (CICERO-SCM) to convert emissions to temperature trajectories
- **Prediction**: Scale learned patterns by global temperature to generate annual climate fields

#### 2. Monthly Noise Generation  
- **Seasonal Modeling**: Temperature-dependent seasonal cycles using modulated harmonic regression
- **Spatial Decomposition**: PCA-based representation of climate anomalies
- **Temporal Modeling**: Vector autoregression (VARX) of principal components
- **Stochastic Simulation**: Generate new realizations preserving spatial-temporal covariance

### Mathematical Framework

**Pattern Scaling Equation:**
```
ΔC(x,t) = Σᵢ αᵢ(t) × Pᵢ(x)
```
Where:
- `ΔC(x,t)`: Climate change at location x, time t
- `αᵢ(t)`: Time-varying forcing amplitude for pattern i  
- `Pᵢ(x)`: Spatial response pattern i

**Monthly Noise Model:**
```
C_monthly(x,t) = S(x,t,T) + Σⱼ PCⱼ(t) × EOFⱼ(x)
```
Where:
- `S(x,t,T)`: Temperature-dependent seasonal cycle
- `PCⱼ(t)`: Principal component time series (from VARX model)
- `EOFⱼ(x)`: Empirical orthogonal function j

## Complete Workflow

### Phase 1: Data Preparation

#### 1.1 Initialize CMIP6 Data Access
```python
from meteor import Cmip6MeteorDataGetter

# Define experiments and variables
experiments = ["piControl", "abrupt-4xCO2", "historical", "ssp245", "ssp585"]
variables = ["tas", "pr"]  # temperature, precipitation

# Create data getter
data_getter = Cmip6MeteorDataGetter(
    exps=experiments, 
    flds=variables,
    dbe=['CMIP', 'CMIP', 'CMIP', 'ScenarioMIP', 'ScenarioMIP']
)

# Select models (example: CanESM5)
models = ['CanESM5']
```

#### 1.2 Prepare Training Data
```python
# Create training datasets for pattern scaling
training_data = {}
for model in models:
    training_data[model] = {
        "base": data_getter.make_meteor_training_data("base", model),      # Control
        "co2x4": data_getter.make_meteor_training_data("co2x4", model),   # CO2 response
        "ssp245": data_getter.make_meteor_training_data_composite(        # Historical+future
            ["historical", "ssp245"], model
        )
    }
```

#### 1.3 Prepare Forcing Data
```python
from ciceroscm import input_handler

# Load emissions and concentrations for scenarios
scenarios = ["ssp126", "ssp245", "ssp585"]
emissions_data = []
concentrations_data = []

for scenario in scenarios:
    emissions_data.append(
        input_handler.read_emissions(f"{scenario}_em_RCMIP.txt")
    )
    concentrations_data.append(
        input_handler.read_inputfile(f"{scenario}_conc_RCMIP.txt")
    )
```

### Phase 2: Pattern Training

#### 2.1 Train GHG-only Patterns
```python
from meteor import MeteorPatternScaling

# Pure CO2 response patterns
ghg_pattern = MeteorPatternScaling(
    "cmip6-CanESM5-ghg",
    {"tas": 2, "pr": 2},  # 2 response patterns per variable
    lambda key: training_data["CanESM5"][key],
    exp_list=["base", "co2x4"]
)
```

#### 2.2 Train Multi-forcing Patterns
```python
# Include aerosol effects from historical/future residuals
multiforcing_pattern = MeteorPatternScaling(
    "cmip6-CanESM5-multiforcing", 
    {"tas": 2, "pr": 2},
    lambda key: training_data["CanESM5"][key],
    exp_list=["base", "co2x4", "ssp245"]  # Includes aerosol residuals
)
```

### Phase 3: Monthly Noise Training

#### 3.1 Train Noise Models
```python
# Train separate noise models for each variable
tas_noise_model = data_getter.train_noise_model(
    experiments=["historical", "ssp245"],
    model="CanESM5",
    variable_name="tas",
    n_modes=8,        # Number of PCA modes
    lag_order=2,      # VARX lag order
    cache_dir="./noise_cache"
)

pr_noise_model = data_getter.train_noise_model(
    experiments=["historical", "ssp245"],
    model="CanESM5", 
    variable_name="pr",
    n_modes=8,
    lag_order=2,
    cache_dir="./noise_cache"
)
```

### Phase 4: Climate Projections

#### 4.1 Generate Annual Projections
```python
# Create annual climate projections for scenarios
annual_predictions = {}
for i, scenario in enumerate(scenarios):
    annual_predictions[scenario] = multiforcing_pattern.predict_from_combined_experiment(
        emissions_data[i],
        concentrations_data[i], 
        ["tas", "pr"]
    )
```

#### 4.2 Convert Annual to Monthly Resolution
```python
# Convert annual predictions to monthly intervals
monthly_annual = {}
for scenario in scenarios:
    monthly_annual[scenario] = {}
    for variable in ["tas", "pr"]:
        monthly_annual[scenario][variable] = multiforcing_pattern.to_monthly(
            annual_predictions[scenario][variable],
            start_year=1850
        )
```

#### 4.3 Generate Monthly Noise
```python
# Create global temperature trajectory for noise generation
global_temp_trajectory = annual_predictions["ssp245"]["tas"].mean(dim=["lat", "lon"]).values
monthly_temp_trajectory = np.repeat(global_temp_trajectory, 12)

# Generate monthly noise realizations
n_realizations = 10
noise_realizations = {}

for variable in ["tas", "pr"]:
    noise_model = tas_noise_model if variable == "tas" else pr_noise_model
    noise_realizations[variable] = []
    
    for i in range(n_realizations):
        noise = noise_model.generate_realization(
            monthly_temp_trajectory,
            noise_only=True,  # Generate noise component only
            random_seed=42 + i
        )
        noise_realizations[variable].append(noise)
```

#### 4.4 Combine for Complete Monthly Projections
```python
# Combine annual patterns with monthly noise for full ensemble
monthly_ensemble = {}

for scenario in scenarios:
    monthly_ensemble[scenario] = {}
    for variable in ["tas", "pr"]:
        monthly_ensemble[scenario][variable] = []
        
        for i in range(n_realizations):
            # Ensure dimensions match
            annual_subset = monthly_annual[scenario][variable]
            noise_subset = noise_realizations[variable][i]
            
            # Combine annual trend + monthly variability
            combined = annual_subset + noise_subset
            monthly_ensemble[scenario][variable].append(combined)
```

### Phase 5: Analysis and Validation

#### 5.1 Global Mean Time Series
```python
import matplotlib.pyplot as plt

# Plot ensemble of global mean trajectories
scenario = "ssp245"
variable = "tas"

plt.figure(figsize=(12, 6))
for i, realization in enumerate(monthly_ensemble[scenario][variable]):
    global_mean = realization.mean(dim=["lat", "lon"])
    years = 1850 + np.arange(len(global_mean))/12
    
    if i == 0:
        plt.plot(years, global_mean, alpha=0.7, label="METEOR Monthly Ensemble")
    else:
        plt.plot(years, global_mean, alpha=0.7)

# Add annual pattern for comparison  
annual_global = annual_predictions[scenario][variable].mean(dim=["lat", "lon"])
annual_years = 1850 + np.arange(len(annual_global))
plt.plot(annual_years, annual_global, 'k--', linewidth=2, label="METEOR Annual")

plt.xlabel("Year")
plt.ylabel("Global Mean Temperature Anomaly (K)")
plt.title(f"METEOR Climate Projections - {scenario.upper()}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### 5.2 Seasonal Analysis
```python
# Analyze seasonal patterns
realization = monthly_ensemble["ssp245"]["tas"][0]  # First realization
seasonal_cycle = realization.groupby("month").mean()

# Plot seasonal climatology
fig, ax = plt.subplots(figsize=(10, 6))
global_seasonal = seasonal_cycle.mean(dim=["lat", "lon"])
ax.plot(range(1, 13), global_seasonal, 'o-', linewidth=2)
ax.set_xlabel("Month")
ax.set_ylabel("Temperature (K)")
ax.set_title("Global Mean Seasonal Cycle - METEOR Monthly")
ax.grid(True, alpha=0.3)
plt.show()
```

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
