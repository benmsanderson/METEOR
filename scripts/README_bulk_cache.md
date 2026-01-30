# METEOR Bulk Data Caching Tool

This interactive script helps you pre-cache CMIP6 data for METEOR analysis, ensuring data is available offline or for batch processing workflows.

## Features

- **Interactive Mode**: Prompts for user preferences with sensible defaults
- **Non-interactive Mode**: Use command line arguments for automated workflows
- **Progress Tracking**: Real-time progress bar with ETA estimates
- **Comprehensive Defaults**: Pre-configured for complete climate analysis workflows
- **Cache Location Control**: Specify custom cache directories or use METEOR defaults

## Usage

### Interactive Mode (Recommended for first-time users)

```bash
python scripts/bulk_cache_data.py
```

The script will guide you through:
1. Selecting scenarios to cache (with defaults for comprehensive analysis)
2. Choosing models to cache (includes 20 major CMIP6 models)
3. Selecting variables to cache (precipitation and temperature by default)
4. Setting cache location (or using METEOR's default: `~/.meteor/cmip6_cache`)

### Non-interactive Mode (For automation)

```bash
# Use all defaults
python scripts/bulk_cache_data.py --non-interactive

# Custom selection
python scripts/bulk_cache_data.py --non-interactive \
    --scenarios historical ssp245 ssp585 \
    --models CanESM5 CESM2 GFDL-ESM4 \
    --variables tas pr

# Single test case
python scripts/bulk_cache_data.py --non-interactive \
    --scenarios historical \
    --models CanESM5 \
    --variables tas
```

## Default Configurations

### Scenarios (7 total)
- `piControl` - Pre-industrial control runs
- `abrupt-4xCO2` - Idealized 4×CO2 experiments
- `historical` - Historical simulations (1850-2014)
- `ssp126` - Low emission scenario
- `ssp245` - Intermediate emission scenario
- `ssp370` - High emission scenario
- `ssp585` - Very high emission scenario

### Models (20 total)
ACCESS-ESM1-5, AWI-CM-1-1-MR, BCC-CSM2-MR, CAMS-CSM1-0, CAS-ESM2-0, CESM2, CMCC-ESM2, CNRM-ESM2-1, CanESM5, EC-Earth3-Veg, FGOALS-f3-L, GFDL-ESM4, GISS-E2-1-H, INM-CM5-0, KACE-1-0-G, MCM-UA-1-0, MIROC-ES2L, MPI-ESM1-2-HR, NorESM2-MM, UKESM1-0-LL

### Variables (2 total)
- `tas` - Near-surface air temperature
- `pr` - Precipitation

## Command Line Options

```
--non-interactive     Use all defaults without prompting
--scenarios [LIST]    Space-separated list of scenarios to cache
--models [LIST]       Space-separated list of models to cache  
--variables [LIST]    Space-separated list of variables to cache
--help               Show help message
```

## Output

The script provides:
- Real-time progress with visual progress bar
- ETA estimates based on processing speed
- Success/failure indicators for each combination
- Final summary with success rates
- Cache location information

## Notes

- Some model/scenario/variable combinations may not be available in CMIP6
- Failed combinations are normal and expected
- Data is stored in METEOR's standard cache format
- Cached data significantly speeds up subsequent METEOR analyses
- Internet connection required for initial data download

## Examples

```bash
# Quick test with single model
python scripts/bulk_cache_data.py --non-interactive \
    --scenarios historical --models CanESM5 --variables tas

# Cache key scenarios for ensemble analysis
python scripts/bulk_cache_data.py --non-interactive \
    --scenarios historical ssp245 ssp585

# Cache specific models only
python scripts/bulk_cache_data.py --non-interactive \
    --models CanESM5 CESM2 GFDL-ESM4 UKESM1-0-LL
```