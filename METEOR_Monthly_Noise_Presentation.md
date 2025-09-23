---
marp: true
theme: default
paginate: true
math: mathjax
---

# METEOR Monthly Noise Generation
## From Annual Pattern Scaling to Monthly Climate Variability

**METEOR**: Multivariate Emulation of Time-Evolving and Overlapping Responses

*A fast climate emulator combining pattern scaling with stochastic noise modeling*

---

## Monthly Noise Generation

### Scientific Foundation
Real climate exhibits variability at multiple timescales:
- **Seasonal cycles**: Temperature-dependent patterns
- **Interannual variability**: El Niño, volcanic eruptions
- **Spatial covariance**: Coherent patterns of variability


---

## Noise Model Mathematical Framework

### 1. Temperature-Dependent Seasonal Cycle
$$S(x,t,T) = \beta_0(x) + \beta_T(x) \cdot T(t) + \sum_{k=1}^{2} \left[ \beta_{k,c}(x) + \beta_{k,T}(x) \cdot T(t) \right] \cos(2\pi k t/12)$$
$$+ \sum_{k=1}^{2} \left[ \beta_{k,s}(x) + \beta_{k,sT}(x) \cdot T(t) \right] \sin(2\pi k t/12)$$

---

## Noise Model Mathematical Framework
### 2. Spatial Decomposition (PCA)
$$A(x,t) = \sum_{j=1}^{n} PC_j(t) \times EOF_j(x)$$

### 3. Temporal Modeling (VARX)
$$\mathbf{PC}(t) = \sum_{l=1}^{p} \mathbf{A}_l \mathbf{PC}(t-l) + \mathbf{B} \mathbf{X}(t) + \boldsymbol{\epsilon}(t)$$

---

## How VARX Works: Conceptual Understanding

### Vector Autoregression with eXogenous variables (VARX)





#### **Vector (V)**
*"Multiple variables evolve together"*

- All PC modes predicted simultaneously
- **Cross-correlations**: Mode 1 affects Mode 2
- **Coherent patterns**: Realistic spatial relationships

---
#### **Autoregression (AR)**
*"Current state depends on past states"*

- PC values at time $t$ depend on previous months
- **Memory effect**: La Niña follows El Niño patterns
- **Persistence**: Anomalies don't disappear instantly

---


#### **eXogenous variables (X)**
*"External forcing drives the system"*

- Global temperature trajectory as external input
- **Climate change signal**: Warming modifies patterns
- **Non-stationary**: Variability changes over time

---



#### **Noise (ε)**
*"Unpredictable random component"*

- Represents chaotic weather variability
- **Stochastic element**: Each realization is different
- **Calibrated variance**: Matches observed variability



---

## Complete Monthly Climate Model

$$C_{monthly}(x,t) = S(x,t,T) + \sum_{j=1}^{n} PC_j(t) \times EOF_j(x)$$




---

## Usage: Basic Training

### Automatic Consistent Baseline (Recommended)
```python
from meteor import Cmip6MeteorDataGetter

# Initialize data getter
data_getter = Cmip6MeteorDataGetter(flds=['tas', 'pr'])

# Train noise model - automatically uses piControl baseline
noise_model = data_getter.train_noise_model(
    experiments=['historical', 'ssp245'],
    model='CanESM5',
    variable_name='tas',
    n_modes=10,
    lag_order=2
)
```
---
**Output:**
```
Fetched piControl baseline for CanESM5 tas: 287.456
Using piControl baseline: 287.456
✅ Noise generator fitted successfully.
```

---

## Usage: Generating Realizations

### Full Monthly Climate
```python
# Generate complete monthly climate realization
realization = noise_model.generate_realization(
    global_temp_trajectory,  # Your temperature scenario
    n_realizations=5,
    random_seed=42
)
```
---
### Noise-Only Mode
```python
# Generate just the variability component
noise_component = noise_model.generate_realization(
    global_temp_trajectory,
    noise_only=True,  # Only stochastic component
    n_realizations=5
)

# Add to METEOR annual projection
monthly_projection = annual_meteor + noise_component
```

---

## Integration with Pattern Scaling

### Complete Workflow
```python
# 1. Train pattern scaling (existing)
patterns = MeteorPatternScaling(
    "model-name", {"tas": 2, "pr": 2},
    training_data_function,
    exp_list=["base", "co2x4", "historical"]
)

# 2. Train noise models (NEW)
noise_models = data_getter.train_all_noise_models(
    experiments=['historical', 'ssp245'],
    models=['CanESM5', 'CESM2'],
    variables=['tas', 'pr']
)
```
