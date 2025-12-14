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

### Simple Example

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

### 📓 More Examples

For comprehensive examples including regional projections, impact metrics, gridded output, multi-scenario comparisons, and advanced customization, see:

**[METEOR Interface Examples Notebook](notebooks/METEOR_Interface_Examples.ipynb)**

This interactive notebook covers:
- Regional and city-scale projections
- Climate impact metrics (heating/cooling degree days)
- Full gridded output generation
- Climatology without noise
- Multi-scenario comparisons
- Custom model configurations

## What METEOR Does

- ✅ **Fast Ensemble Generation**: Create 100+ realizations faster than running a single GCM simulation
- ✅ **Monthly Resolution**: Realistic monthly variability with proper seasonality and persistence
- ✅ **Spatial Detail**: Global, regional (AR6 regions), and point-based projections
- ✅ **Multiple Variables**: Temperature, precipitation, and more with variable-specific treatment
- ✅ **Impact Metrics**: Built-in calculation of degree days and custom impact assessments
- ✅ **Smart Caching**: Automatic caching of trained models and downloaded data
- ✅ **CMIP6 Integration**: Direct access to cloud-based CMIP6 data

## Architecture Overview

### Core Components

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| **High-Level Interface** | Simple ensemble generation | `MeteorInterface` |
| **Pattern Scaling** | Annual climate projections from forcings | `MeteorPatternScaling` |
| **Monthly Generation** | Convert annual to monthly + seasonality | `MeteorNoiseGenerator` |
| **Data Access** | CMIP6 cloud data integration | `Cmip6MeteorDataGetter` |
| **Impact Assessment** | Climate impact calculations | Degree days, custom impacts |
| **Variable Transforms** | Variable-specific data processing | Gamma transform for precipitation |
| **Ensemble Output** | Structured output containers | `EnsembleOutput`, `VariableOutput` |

### Workflow

```
Emissions/Concentrations → Pattern Scaling → Annual Climate → Monthly Base
                                                                      ↓
                                                          + Stochastic Noise
                                                                      ↓
                                                          Monthly Ensemble
                                                                      ↓
                                                          Impact Calculations
```

## Additional Resources

### Documentation
- **[METEOR Interface Examples](notebooks/METEOR_Interface_Examples.ipynb)**: Complete tutorial with visualizations

### Other Notebooks
- `Climate_Bench_METEOR.ipynb`: ClimateBench metrics

### Module Reference

| Module | Purpose |
|--------|---------|
| `meteor_interface.py` | High-level `MeteorInterface` API |
| `meteor.py` | `MeteorPatternScaling` for annual projections |
| `noise_generator.py` | `MeteorNoiseGenerator` for monthly variability |
| `cmip6_meteor_data_getter.py` | CMIP6 cloud data access |
| `ensemble_output.py` | Output container classes |
| `variable_transforms.py` | Variable-specific transformations |
| `impacts/` | Climate impact assessment |
| `prpatt.py` | Pattern scaling algorithms |
| `scm_forcer_engine.py` | Simple climate model integration |

## Development & Testing

Run tests and checks:

```bash
# Run all tests
make checks

# Run specific test suite
pytest tests/unit
pytest tests/integration

#other make options
make format-checks        # run all the checks
make format               # re-format files
make format-notebooks     # format the notebooks
make black                # apply black formatter to source and tests
make isort                # format the code
make docs                 # build the docs
make test                 # run the full testsuite
make test-pypi-install    # test whether installing from PyPI works
make test-install         # test installing works
make virtual-environment  # update venv, create a new venv if it doesn't exist make
make first-venv           # create a new virtual environment for the very first repo 
```

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

- **Examples**: Explore `notebooks/` directory
- **Issues**: [GitHub Issues](https://github.com/benmsanderson/METEOR/issues)
- **Updates**: `git pull && make virtual-environment`

**Note**: Some research notebooks (`METEOR_paper_figures_*.ipynb`) require large datasets and are excluded from automated testing.
