# METEOR: Multivariate Emulation of Time-Evolving and Overlapping Responses

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17523936.svg)](https://doi.org/10.5281/zenodo.17523936)

METEOR is a fast spatial climate emulator that generates large ensembles of monthly climate projections with realistic variability. Perfect for impact assessment, uncertainty quantification, and exploring climate scenarios.

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

### Getting Started with MeteorInterface

The `MeteorInterface` provides the simplest way to generate climate projections. It handles all the complexity of pattern scaling, noise modeling, and data management automatically.

#### Example 1: Your First Climate Ensemble

Generate 100 realizations of global temperature and precipitation for SSP2-4.5:

```python
from meteor import MeteorInterface

# Create and train emulator (uses cached models if available)
emulator = MeteorInterface.from_cmip6(
    model='NorESM2-MM',
    variables=['tas', 'pr'],
    cache_dir='./cache'
)
emulator.train(auto=True, verbose=True)

# Generate ensemble
ensemble = emulator.generate(
    scenario='ssp245',
    start_year=1850,
    end_year=2100,
    n_realizations=100,
    timeseries=['global']  # Global mean
)

# Access results
global_temp = ensemble['tas'].timeseries['global']  # Shape: (100, 3012 months)
print(f"Ensemble shape: {global_temp.shape}")
print(f"2100 warming: {global_temp[:, -1].mean():.2f} ± {global_temp[:, -1].std():.2f} K")
```

#### Example 2: Regional and City-Scale Projections

Add spatial detail with regional means and specific locations:

```python
# Generate multiple spatial scales
ensemble = emulator.generate(
    scenario='ssp245',
    start_year=1850,
    end_year=2100,
    n_realizations=100,
    timeseries=[
        'global',
        'regional:NEU',  # Northern Europe (AR6 region)
        'regional:EAS',  # East Asia
        'point:59.9,10.8',  # Oslo coordinates (lat, lon)
        'point:28.6,77.2'   # Delhi
    ]
)

# Access different scales
neu_temp = ensemble['tas'].timeseries['regional:NEU']
oslo_temp = ensemble['tas'].timeseries['point:59.9,10.8']
```

#### Example 3: Climate Impact Metrics

Calculate heating and cooling degree days automatically:

```python
ensemble = emulator.generate(
    scenario='ssp245',
    start_year=1850,
    end_year=2100,
    n_realizations=100,
    timeseries=['point:59.9,10.8'],  # Oslo
    impacts={
        'tas': {
            'degree_days': {
                'hdd_base': 18.0,  # Heating degree days
                'cdd_base': 18.0   # Cooling degree days
            }
        }
    }
)

# Access impact metrics (annual values)
hdd = ensemble['tas'].impacts['hdd']['point:59.9,10.8']  # Shape: (100, 251 years)
cdd = ensemble['tas'].impacts['cdd']['point:59.9,10.8']

print(f"2020 HDD: {hdd[:, 170].mean():.0f} degree-days")
print(f"2100 HDD: {hdd[:, -1].mean():.0f} degree-days")
print(f"HDD change: {((hdd[:, -1].mean() - hdd[:, 170].mean()) / hdd[:, 170].mean() * 100):.1f}%")
```

#### Example 4: Multi-Scenario Comparison

Compare different emission scenarios:

```python
scenarios = ['ssp126', 'ssp245', 'ssp585']
ensembles = {}

for scenario in scenarios:
    ensembles[scenario] = emulator.generate(
        scenario=scenario,
        start_year=2015,
        end_year=2100,
        n_realizations=50,
        timeseries=['global']
    )

# Compare 2100 warming
for scenario in scenarios:
    temp_2100 = ensembles[scenario]['tas'].timeseries['global'][:, -1]
    print(f"{scenario}: {temp_2100.mean():.2f} ± {temp_2100.std():.2f} K")
```

### What MeteorInterface Does For You

- ✅ **Automatic Training**: Loads CMIP6 data and trains both pattern scaling and noise models
- ✅ **Smart Caching**: Reuses trained models to save time on subsequent runs
- ✅ **Variable Transforms**: Handles precipitation's gamma distribution automatically
- ✅ **Monthly Resolution**: Generates realistic monthly variability with proper seasonality
- ✅ **Spatial Aggregation**: Computes global, regional, and point timeseries on-the-fly
- ✅ **Impact Metrics**: Calculates degree days and other climate impacts
- ✅ **Ensemble Management**: Organizes multiple realizations with clean data structures

See the **`notebooks/METEOR_Interface_Demo.ipynb`** for a complete tutorial with visualization examples.

## Customizing MeteorInterface

For advanced users who want control over model parameters while keeping the convenience of the high-level interface.

### Custom Model Configuration

Override default parameters using the `train()` method with `variable_configs`:

```python
from meteor import MeteorInterface

# Create emulator
emulator = MeteorInterface.from_cmip6(
    model='NorESM2-MM',
    variables=['tas', 'pr'],
    cache_dir='./cache'
)

# Train with custom configuration per variable
emulator.train(
    auto=True,                    # Use smart defaults as baseline
    training_scenario='ssp245',   # Default training scenario
    variable_configs={
        'tas': {
            'n_modes_pattern': 3,        # Number of pattern scaling modes (default: 3)
            'n_modes_noise': 40,         # Number of noise PCA modes (default: 40)
            'lag_order': 2,              # Temporal memory in noise model (default: 2)
            'use_exog': 'all',           # Use exogenous variables ('all', 'temp_only', 'none')
            'training_scenario': 'ssp245'  # Override scenario for this variable
        },
        'pr': {
            'n_modes_noise': 50,         # More modes for precipitation
            'training_scenario': 'ssp370',  # Different scenario for pr
            'transform': True            # Apply gamma transform (default: True for pr)
        }
    },
    verbose=True
)
```

### Training Options

Control the training process:

```python
# Automatic training with smart defaults (recommended)
emulator.train(auto=True, verbose=True)

# Manual training with specific settings
emulator.train(
    auto=False,
    training_scenario='ssp370',
    variable_configs={
        'tas': {'n_modes_noise': 30, 'lag_order': 1}
    },
    verbose=True
)
```

### Custom Data Sources

Customize the CMIP6 data source during initialization:

```python
from meteor import MeteorInterface

# Custom experiments and data sources
emulator = MeteorInterface.from_cmip6(
    model='NorESM2-MM',
    variables=['tas', 'pr'],
    cache_dir='./cache',
    exps=['piControl', 'abrupt-4xCO2', 'historical', 'ssp245', 'ssp370'],
    dbe=['CMIP', 'CMIP', 'CMIP', 'ScenarioMIP', 'ScenarioMIP']
)
```
from meteor import MeteorInterface, Cmip6MeteorDataGetter

# Custom data getter with specific experiments
data_getter = Cmip6MeteorDataGetter(
    exps=['piControl', 'abrupt-4xCO2', 'historical', 'ssp245'],
    flds=['tas', 'pr'],
    enable_cache=True,
    cache_dir='/custom/cache/path'
)

# Create emulator with custom data source
emulator = MeteorInterface(
    model='NorESM2-MM',
    variables=['tas', 'pr'],
    data_getter=data_getter
)
```

### Advanced Generation Options

Customize ensemble generation behavior:

```python
ensemble = emulator.generate(
    scenario='ssp245',
    start_year=1850,
    end_year=2100,
    n_realizations=100,
    timeseries=['global', 'regional:NEU', 'point:59.9,10.8'],
    
    # Impact calculations
    impacts={
        'tas': {
            'degree_days': {
                'hdd_base': 15.0,      # Custom base temperature for heating
                'cdd_base': 22.0       # Custom base temperature for cooling
            }
        }
    },
    
    verbose=True
)
```


## Advanced Usage: Lower-Level API

For users who need fine-grained control over the emulation process, METEOR provides direct access to the underlying components.

### Pattern Scaling Engine

Build custom pattern scaling models with explicit control:

```python
from meteor import Cmip6MeteorDataGetter, MeteorPatternScaling
from ciceroscm import input_handler

# Setup data access
data_getter = Cmip6MeteorDataGetter()
model = "CanESM5"

# Create pattern model with custom configuration
pattern_model = MeteorPatternScaling(
    "demo",
    {"tas": 2, "pr": 2},  # 2 patterns per climate variable
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

### Monthly Noise Generation

Explicit control over variability modeling:

```python
import numpy as np

# Train monthly noise model with custom parameters
noise_model = data_getter.train_noise_model(
    experiments=["historical", "ssp245"],
    model=model,
    variable_name="tas",
    n_modes=8,  # Number of PCA modes
    lag_order=1  # Temporal dependencies
)

# Generate monthly climate with noise
annual_pred = prediction["tas"]  
monthly_base = pattern_model.to_monthly(annual_pred, start_year=1850)
global_temp = annual_pred.mean(dim=["lat", "lon"]).values
monthly_temp = np.repeat(global_temp, 12)

noise = noise_model.generate_realization(monthly_temp, noise_only=True)
monthly_climate = monthly_base + noise
```

### Custom Impact Calculations

Build your own impact assessment pipeline:

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
| **High-Level Interface** | Simple ensemble generation | `MeteorInterface` |
| **Pattern Scaling** | Annual climate projections from forcings | `MeteorPatternScaling` |
| **Monthly Generation** | Convert annual to monthly + seasonality | `MeteorNoiseGenerator` |
| **Data Access** | CMIP6 cloud data integration | `Cmip6MeteorDataGetter` |
| **Impact Assessment** | Climate impact calculations | `DegreeDaysCalculator`, `ImpactEnsemble` |
| **Plotting** | Visualization utilities | `meteor_plot_utils` |

### Workflow Patterns

```
High-Level (MeteorInterface):
Scenario → .generate() → Ensemble (with timeseries, impacts, etc.)

Low-Level (Component-by-Component):
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
- **`METEOR_Interface_Demo.ipynb`**: Complete MeteorInterface tutorial ⭐ **START HERE**
- **`METEOR_single_model_pattern_example.ipynb`**: Single model pattern creation
- **`CMIP6_demo_with_residual.ipynb`**: Aerosol forcing from residuals  
- **`CMIP6_noise_model_examples.ipynb`**: Multi-model workflow with monthly noise
- **`Climate_Bench_METEOR.ipynb`**: ClimateBench metrics calculation
- **Paper Figures**: `METEOR_paper_figures*.ipynb` (research reproducibility)

### Scripts (`scripts/`)
- **`test_install.py`**: Quick installation test
- **`test_cmip6_read_indata.py`**: CMIP6 data access demonstration
- **`make_pattern_4xco2_plots.py`**: 4×CO₂ pattern generation example

### Additional Documentation
- **`QUICK_START.md`**: Detailed workflow examples (lower-level API)
- **`MONTHLY_NOISE_README.md`**: Complete monthly modeling guide
- **`streamlit/README.md`**: Web app deployment guide


## API Reference

### MeteorInterface (Recommended)

The high-level interface for easy ensemble generation:

```python
from meteor import MeteorInterface

# Create emulator from CMIP6 model
emulator = MeteorInterface.from_cmip6(
    model='NorESM2-MM',           # CMIP6 model name
    variables=['tas', 'pr'],      # Climate variables
    cache_dir='./cache'           # Cache location
)

# Train models (auto mode recommended)
emulator.train(auto=True, verbose=True)

# Generate ensemble
ensemble = emulator.generate(
    scenario='ssp245',            # Emission scenario
    start_year=1850,              # Start year
    end_year=2100,                # End year
    n_realizations=100,           # Number of ensemble members
    timeseries=[                  # Spatial aggregations
        'global',
        'regional:NEU',           # AR6 region code
        'point:59.9,10.8'         # lat,lon
    ],
    impacts={                     # Optional impact metrics
        'tas': {
            'degree_days': {
                'hdd_base': 18.0,
                'cdd_base': 18.0
            }
        }
    }
)

# Access results
ensemble['tas'].timeseries['global']        # Temperature timeseries
ensemble['pr'].timeseries['regional:NEU']   # Precipitation regional mean
ensemble['tas'].impacts['hdd']['point:59.9,10.8']  # Heating degree days
ensemble.list_variables()                   # List available variables
ensemble['tas'].list_impacts()              # List available impact metrics
```

### Lower-Level API

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

## Configuration

### Local Data Caching

METEOR accesses CMIP6 climate model data from Google Cloud Storage. By default, data is downloaded on-demand **without local caching**.

#### Optional Caching (Recommended for Repeated Use)

For improved performance when repeatedly accessing the same data, you can enable local caching:

- **Default cache location**: `~/.meteor/cmip6_cache/`
- **Purpose**: Stores downloaded CMIP6 data locally to avoid repeated downloads
- **Benefits**: Significantly faster repeated data access, reduced bandwidth usage
- **Storage**: Compressed NetCDF files with automatic metadata tagging

#### Cache Configuration
```python
# Enable caching (recommended for repeated use)
data_getter = Cmip6MeteorDataGetter(enable_cache=True)

# Custom cache directory
data_getter = Cmip6MeteorDataGetter(enable_cache=True, cache_dir="/custom/path/to/cache")

# Default: No caching (data downloaded each time)
data_getter = Cmip6MeteorDataGetter()  # or enable_cache=False

# Clear cache when needed (if caching was enabled)
data_getter.clear_cache()
```

**Storage Considerations:**
- Each model/scenario/variable: ~10-50 MB (compressed)
- Full multi-model ensemble: Can reach several GB
- Enable caching only if you have sufficient disk space and plan to reuse data

**Note**: If you enable caching and directory creation fails due to permissions, caching will be automatically disabled but METEOR will continue to function normally.

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