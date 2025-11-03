# CHANGELOG - Impacts Refactor PR (METEOR 1.5 candidate)

## Overview
Addition of monthly climate variability modeling and climate impact assessment framework to METEOR, with data caching capabilities and revised tests

---

## Monthly Climate Variability (Noise Model)

### Core Implementation
- **Added**: `src/meteor/noise_generator.py` - PCA/VARX-based monthly climate variability modeling
- Enhanced `src/meteor/meteor.py` - Added `to_monthly()` method for converting annual to monthly climate data
- enhanced: `src/meteor/prpatt.py` - Improved pattern scaling algorithms for monthly data support

### Functionality
- Temperature-dependent seasonality that evolves with global warming
- Ensemble generation for uncertainty quantification
- Training from CMIP6 historical and scenario data
- Integration with existing METEOR pattern scaling workflow

### Examples & Documentation
- Added: `notebooks/CMIP6_noise_model_examples.ipynb` - Multi-model workflow demonstration
- Added: `notebooks/METEOR_Impacts_Clean_Demo.ipynb` - Combined impacts and noise modeling

---

## Climate Impact Assessment Framework

### Core Framework
- **Added**: `src/meteor/impacts/impacts_core.py` - Base classes for impact calculators and ensemble analysis
- **Added**: `src/meteor/impacts/calculators/degree_days.py` - Heating and cooling degree days calculator (example impact parameterisation)
- **Added**: `src/meteor/impacts/ensemble.py` - Ensemble statistics and analysis tools
- **Added**: `src/meteor/impacts/utils.py` - Utility functions for climate data processing

### Functionality  
- Extensible `ImpactCalculator` class for custom impact metrics
- `ImpactEnsemble` class for statistical analysis across multiple realizations
- Built-in degree days calculator with customizable parameters
- General workflows for ensemble creation and impact calculation

### Testing
- Added test suite in `tests/unit/impacts/` covering all framework components
- Added `tests/integration/impacts/test_workflow.py` - End-to-end impacts workflow validation

---

## Enhanced Data Access & Caching

### CMIP6 Data Integration
- **Enhanced**: `src/meteor/cmip6_meteor_data_getter.py` - Added noise model training capabilities
- **Added**: Automatic local caching system for CMIP6 data with configurable cache directory
- **Added**: Error handling for cache setup 

### Training Data Infrastructure
- **Added**: `tests/test-data/create_small_test_data.py` - Optimised test data generation
- **Added**: Small test datasets for faster testing

---

## Other Technical Developments

### Documentation
-  `README.md` - added quick start guide, API reference, and development setup
- All modules with numpy-style docstrings and comprehensive inline documentation
