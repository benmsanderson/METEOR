# METEOR Scripts Directory

This directory contains various scripts for testing, demonstration, and analysis of METEOR functionality.

## Demo Scripts

### `demo_impacts.py`
Demonstrates the METEOR-impacts layer for calculating climate impact metrics from temperature data. Shows:
- Single climate realization processing
- Ensemble processing and uncertainty quantification  
- Base temperature sensitivity analysis
- Integration with METEOR workflow

Run with: `python demo_impacts.py`

### `demo_noise_only.py`
Demonstrates the METEOR-noise module for generating monthly climate variability from annual climatologies.

### `demo_unified_api.py`
Demonstrates the unified METEOR API for generating climate projections.

## Test Scripts

### `test_install.py`
Tests basic METEOR installation and functionality.

### `test_cmip6_read_in_data.py`
Tests reading and processing CMIP6 climate model data.

## Analysis Scripts

### `make_cmip6_test_plots.py`
Generates test plots from CMIP6 data for validation.

### `make_noresm_test.py`
Specific tests for NorESM climate model output.

### `make_pattern_4xco2_plots.py`
Creates plots of climate patterns under 4x CO2 scenarios.

## Notebooks

### `METEOR_multi_model_pattern_example.ipynb`
Jupyter notebook demonstrating multi-model pattern analysis with METEOR.

## Usage

Most scripts can be run directly from this directory:

```bash
cd scripts
python demo_impacts.py
python test_install.py
# etc.
```

Some scripts may require specific data files or configuration. Check the script headers for requirements.