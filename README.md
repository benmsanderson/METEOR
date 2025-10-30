# METEOR: Multivariate Emulation of Time-Evolving and Overlapping Responses

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15732955.svg)](https://doi.org/10.5281/zenodo.15732955)

METEOR is a spatial climate emulator that rapidly generates ensemble climate projections by combining:
- **Pattern Scaling**: Fast long-term climate response modeling
- **Monthly Variability**: Realistic short-term climate noise and seasonality  
- **Impact Assessment**: Climate impact calculations and ensemble analysis

## Quick Start

### Installation
```bash
git clone https://github.com/benmsanderson/METEOR.git
cd METEOR
make first-venv
make clean  
make virtual-environment
source venv/bin/activate
```

### Basic Usage

#### 1. Simple Annual Climate Projection
```python
from meteor import Cmip6MeteorDataGetter, MeteorPatternScaling
from ciceroscm import input_handler

# Setup data access
data_getter = Cmip6MeteorDataGetter()
model = "CanESM5"

# Create pattern model
pattern_model = MeteorPatternScaling(
    "demo",
    {"tas": 2, "pr": 2},  # 2 patterns per variable
    lambda key: data_getter.make_meteor_training_data(key, model),
    exp_list=["base", "co2x4"]
)

# Load scenario and predict
emissions = input_handler.read_emissions("ssp245_em_RCMIP.txt")
concentrations = input_handler.read_inputfile("ssp245_conc_RCMIP.txt")
prediction = pattern_model.predict_from_combined_experiment(
    emissions, concentrations, ["tas", "pr"]
)
```

#### 2. Monthly Climate with Variability
```python
import numpy as np

# Train monthly noise model
noise_model = data_getter.train_noise_model(
    experiments=["historical", "ssp245"],
    model=model,
    variable_name="tas",
    n_modes=8
)

# Generate monthly climate with noise
annual_pred = prediction["tas"]  
monthly_base = pattern_model.to_monthly(annual_pred, start_year=1850)
global_temp = annual_pred.mean(dim=["lat", "lon"]).values
monthly_temp = np.repeat(global_temp, 12)

noise = noise_model.generate_realization(monthly_temp, noise_only=True)
monthly_climate = monthly_base + noise
```

#### 3. Climate Impact Assessment
```python
from meteor.impacts import DegreeDaysCalculator, create_impact_ensemble

# Calculate cooling degree days
cdd_calc = DegreeDaysCalculator(base_temperature=18.0, mode="cooling")
cdd_result = cdd_calc.calculate(monthly_climate)

# Create ensemble with multiple noise realizations
ensemble = create_impact_ensemble(
    monthly_base, cdd_calc, noise_model, global_temp, n_realizations=10
)
print(f"Mean CDD: {ensemble.mean():.1f}")
print(f"Ensemble spread: {ensemble.std():.1f}")
```

See `QUICK_START.md` for detailed workflows and `MONTHLY_NOISE_README.md` for advanced monthly modeling.

## Architecture Overview

### Core Components

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| **Pattern Scaling** | Annual climate projections from forcings | `MeteorPatternScaling` |
| **Monthly Generation** | Convert annual to monthly + seasonality | `MeteorNoiseGenerator` |
| **Data Access** | CMIP6 cloud data integration | `Cmip6MeteorDataGetter` |
| **Impact Assessment** | Climate impact calculations | `DegreeDaysCalculator`, `ImpactEnsemble` |
| **Plotting** | Visualization utilities | `meteor_plot_utils` |

### Workflow Patterns

```
Emissions/Concentrations → Pattern Scaling → Annual Climate
                                              ↓
Monthly Base Climate ← to_monthly()          Annual Climate
        ↓                                       ↓
Monthly Climate ← + Noise ← generate_realization(global_temp)
        ↓
Climate Impacts ← calculate() ← Impact Calculator
```

## Documentation & Examples

### Notebooks (`notebooks/`)
- **`METEOR_single_model_pattern_example.ipynb`**: Single model pattern creation
- **`CMIP6_demo_with_residual.ipynb`**: Aerosol forcing from residuals  
- **`CMIP6_mmdemo_with_archiving.ipynb`**: Multi-model workflow with monthly noise ⭐
- **`Climate_Bench_METEOR.ipynb`**: ClimateBench metrics calculation
- **Paper Figures**: `METEOR_paper_figures*.ipynb` (research reproducibility)

### Scripts (`scripts/`)
- **`test_install.py`**: Quick installation test
- **`test_cmip6_read_indata.py`**: CMIP6 data access demonstration
- **`make_pattern_4xco2_plots.py`**: 4×CO₂ pattern generation example

### Additional Documentation
- **`QUICK_START.md`**: Detailed workflow examples
- **`MONTHLY_NOISE_README.md`**: Complete monthly modeling guide
- **`streamlit/README.md`**: Web app deployment guide

## API Reference

### Primary Classes

#### `MeteorPatternScaling`
```python
# Annual climate projection engine
model = MeteorPatternScaling(name, n_patterns_dict, data_loader, exp_list)
prediction = model.predict_from_combined_experiment(emissions, concentrations, variables)
monthly_data = model.to_monthly(annual_data, start_year=1850)
```

#### `MeteorNoiseGenerator`
```python
# Monthly climate variability modeling  
noise_model = MeteorNoiseGenerator(n_modes=8, lag_order=1)
noise_model.fit(monthly_climate_data, global_temp_trajectory)
noise = noise_model.generate_realization(temp_trajectory, noise_only=True)
```

#### `Cmip6MeteorDataGetter`
```python
# CMIP6 data access and preprocessing
data_getter = Cmip6MeteorDataGetter(exps=["piControl", "abrupt-4xCO2"], flds=["tas", "pr"])
training_data = data_getter.make_meteor_training_data("base", "CanESM5")
noise_model = data_getter.train_noise_model(experiments, model, variable, n_modes)
```

#### Impact Assessment
```python
# Climate impact calculations
from meteor.impacts import DegreeDaysCalculator, create_impact_ensemble

calc = DegreeDaysCalculator(base_temperature=18.0, mode="cooling")
result = calc.calculate(climate_data)
ensemble = create_impact_ensemble(base_climate, calc, noise_model, temp, n_realizations=10)
```

### Key Parameters

| Parameter | Typical Values | Purpose |
|-----------|---------------|----------|
| `n_patterns` | 2-3 | Response patterns per climate variable |
| `n_modes` | 8-12 | PCA modes for monthly variability |
| `lag_order` | 1-3 | Temporal dependencies in VARX model |
| `base_temperature` | 18°C (cooling), 15°C (heating) | Degree day calculation threshold |
| `noise_only=True` | Boolean | Generate additive noise vs. full signal |

### Common Workflows

#### Multi-realization Ensemble
```python
# Generate multiple noise realizations for uncertainty quantification
realizations = []
for seed in range(10):
    noise = noise_model.generate_realization(temp_trajectory, noise_only=True, random_seed=seed)
    realizations.append(monthly_base + noise)
ensemble = xr.concat(realizations, dim='realization')
```

#### Multi-variable Processing
```python
# Process temperature and precipitation together
variables = ["tas", "pr"]
predictions = pattern_model.predict_from_combined_experiment(emissions, concentrations, variables)
for var in variables:
    monthly_var = pattern_model.to_monthly(predictions[var], start_year=1850)
    # Apply variable-specific noise models...
```

## Development & Testing

### Setting Up Development Environment
```bash
git clone git@github.com:benmsanderson/meteor.git
cd METEOR
git checkout -b your-feature-branch
make first-venv
make virtual-environment
source venv/bin/activate
```

### Running Tests
```bash
make test          # Run test suite
make checks        # Tests + formatting checks  
make format-checks # Formatting only
```

**Test Coverage**: 91%+ comprehensive test suite including:
- Unit tests for individual methods (`tests/unit/`)
- Integration tests for workflows (`tests/integration/`)  
- Optimized with coarsened real PDRMIP data for fast execution
- Notebook testing for key examples

### Code Organization

| Module | Purpose |
|--------|---------|
| `meteor.py` | Main `MeteorPatternScaling` class for annual projections |
| `noise_generator.py` | `MeteorNoiseGenerator` for monthly variability |
| `cmip6_meteor_data_getter.py` | CMIP6 cloud data access and preprocessing |
| `impacts/` | Climate impact assessment framework |
| `prpatt.py` | Pattern scaling algorithms and PDRMIP processing |
| `scm_forcer_engine.py` | Simple climate model integration |
| `meteor_plot_utils.py` | Visualization utilities |

### Contributing
1. Create feature branch from `main`
2. Add tests for new functionality  
3. Ensure all tests pass (`make checks`)
4. Update documentation as needed
5. Submit pull request

---

## Citation

If you use METEOR in your research, please cite:

```bibtex
@Article{sandstad_meteor,
AUTHOR = {Sandstad, M. and Steinert, N. J. and Baur, S. and Sanderson, B. M.},
TITLE = {METEORv1.0.1: A novel framework for emulating multi-timescale regional climate responses},
JOURNAL = {EGUsphere},
VOLUME = {2025},
YEAR = {2025},
PAGES = {1--49},
URL = {https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1038/},
DOI = {10.5194/egusphere-2025-1038}
}
```

## Support

- **Documentation**: See `QUICK_START.md` and `MONTHLY_NOISE_README.md`
- **Examples**: Explore `notebooks/` directory
- **Issues**: [GitHub Issues](https://github.com/benmsanderson/METEOR/issues)
- **Updates**: `git pull && make virtual-environment`

**Note**: Some research notebooks (`METEOR_paper_figures_*.ipynb`) require large datasets and are excluded from automated testing.