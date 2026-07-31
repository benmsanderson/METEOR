Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

The changes listed in this file are categorised as follows:

    - Added: new features
    - Changed: changes in existing functionality
    - Deprecated: soon-to-be removed features
    - Removed: now removed features
    - Fixed: any bug fixes
    - Security: in case of vulnerabilities.

[Unreleased]
---------------------

### Added

- Performance profiling harness (``scripts/profiling/run_profile.py``, ``scripts/profiling/analyze.py``) with parameterized workloads covering training and generation. Baseline numbers and optimization targets documented in ``docs/profiling_baseline.md``.

### Changed

- Now possibly to send variable length temperature scaling timeseries, fixed noise generator for wrong ordering of base data dimensions

### Fixed

- ``Cmip6MeteorDataGetter.validate_pattern_scaling_cache`` now accepts an optional ``variable`` argument matching the per-variable suffix ``MeteorPatternScaling`` saves under (``cmip6-{model}-aer-{variable}``). Without it, a cache that ``MeteorPatternScaling.__init__`` loads successfully by filename was reported as "invalid" by the outer validator, causing ``MeteorInterface`` to run ``prepare_pattern_scaling_training_data`` (which fetches and assembles CMIP6 data over the network) on every generation call and immediately discard the result. Threading ``variable=`` through ``_train_pattern_scaling`` fixes the mismatch and eliminates the redundant work (~13–18 s per generation call in the NorESM2-MM profiling workloads).
- Variable length timeseries now works also when don't have "year" as time dimension.
- Fixes to generate annual and monthly gridded ensembles with unified noise and preserving more of the variance.


[Version 1.6.0]
-----------------------------

### Added

- **High-level `MeteorInterface` API** in ``src/meteor/meteor_interface.py`` for simplified ensemble generation
    - Single entry point for training and generation with automatic caching
    - Support for multiple variables, scenarios, and aggregation types (global, regional, point-based)
    - Built-in handling of variable-specific transforms (e.g., gamma distribution for precipitation)
    - Automatic training state tracking and validation
- **Bulk data caching system** via ``scripts/bulk_cache_data.py``
    - Interactive and non-interactive modes for batch downloading CMIP6 data
    - Checkpoint/resume capability for interrupted downloads
    - Progress tracking and comprehensive error handling
- **Rationalized cache location handling** in ``src/meteor/cache_handling.py``
    - ``CacheHandler`` class with automatic cache location detection
    - ``find_suitable_cache_location()`` intelligently chooses cache directory:
        - Development environments (git clones): ``.cache/`` in repository root
        - Pip-installed packages: ``~/.meteor/cache/`` in user home directory
    - Organized cache structure with separate subdirectories for CMIP6 data, pattern scaling models, and noise models
    - Graceful fallback if cache directory creation fails
    - Validation and automatic cleanup of corrupted cache files
- **Variable-specific transformation framework** in ``src/meteor/variable_transforms.py``
    - ``VariableTransformConfig`` class for managing variable-specific data processing
    - Transform registry system with support for gamma, Weibull, lognormal, and generalized gamma distributions
    - Separate 1D and 3D fitting functions for flexible data handling
    - Empirical quantile mapping support
- **SCM input handling module** in ``src/meteor/scm_input_lib.py``
    - ``parse_scenario_input()`` for flexible scenario specification (string or dictionary)
    - ``load_emissions_concentrations()`` for loading emission and concentration data from files or DataFrames
    - ``load_emissions_concentrations_from_name()`` for loading default scenario data
    - Support for custom scenario definitions with user-provided emissions/concentrations
- **Structured ensemble output containers** in ``src/meteor/ensemble_output.py``
    - ``EnsembleOutput`` class for organizing multi-variable ensemble results
    - ``VariableOutput`` class for storing timeseries, gridded output, and impact metrics
    - Comprehensive metadata tracking (model, scenario, year range, realizations)
- **netCDF4 compression support** in ``Cmip6MeteorDataGetter``
    - Configurable compression levels (1-9) with zlib compression
    - ~45% storage reduction for typical CMIP6 data
    - Enabled by default with ``compression_level=4``
    - Added ``models`` and ``tabids`` parameters to ``Cmip6MeteorDataGetter`` to allow filtering by specific climate models and querying different CMIP6 table IDs (defaults to Amon but no longer hardcoded)
    - Added type hints to function signatures in ``cmip6_meteor_data_getter.py`` and ``meteor_interface.py``
    - Added comprehensive test coverage for new functionality and some old missing tests in ``cmip6_meteor_data_getter.py``
    
- **Enhanced noise model capabilities**
    - Regional mean generation with AR6 region support via ``generate_regional_mean_realizations()``
    - Point-based extraction for city-scale projections
    - Separate stochastic PC generation via ``generate_stochastic_pcs()``
    - Exogenous variable support in VARX models via ``use_exog`` parameter
    - Diagnostic output saving with ``save_diagnostics=True``
    - Added accessible variance decomposition metrics for model evaluation
- **Comprehensive test coverage**
    - Unit tests for ``MeteorInterface`` (``tests/unit/test_meteor_interface.py``)
    - Unit tests for variable transforms (``tests/unit/test_variable_transforms.py``)
    - Unit tests for precipitation transforms (``tests/unit/test_precipitation_transform.py``)
    - Unit tests for SCM input handling (``tests/unit/test_scm_input_lib.py``)
    - Unit tests for pattern logic library (``tests/unit/test_pattern_logic_lib.py``)
    - Enhanced noise generator tests with 4D ensemble support
    - Test coverage increased to 90%+
- **Global temperature timeseries scaling**
    - Support for scaling underlaying annual patterns by user-provided global temperature trajectories

### Changed

- **Refactored `prpatt.py` into modular components**
    - Split into ``geo_data_utils.py`` for geographic data operations (area weighting, global means)
    - Split into ``pattern_logic_lib.py`` for pattern scaling mathematical functions
    - Improved code organization and maintainability
- **Enhanced `MeteorNoiseGenerator` with new features**
    - Support for 4D data (n_ensemble, n_time, n_lat, n_lon) in addition to 3D
    - Improved fitting with custom global temperature trajectories
    - Better handling of piControl baseline data
    - Added ``use_picontrol_baseline`` parameter for baseline control
    - Minimum time series length validation (default 60 months)
- **Improved `Cmip6MeteorDataGetter` architecture**
    - Separate cache validation for pattern scaling and noise models
    - Enhanced cache key generation with compression settings
    - Better error handling for cache operations
    - Support for composite scenario training data
    - sorting for ensemble members in correct numerical order
    - More effective building of initial datasets avoiding fragmented pandas
    - Improved docstrings and error messages throughout ``cmip6_meteor_data_getter.py`` for better clarity and detailed information about missing data
    - Refactored ``_set_fld_exps_dbe`` to ``_set_models_flds_tabids_exps_dbe`` to support new model and table ID filtering capabilities

- **Updated `ScmEngineForPatternScaling` configuration**
    - New ``ScmEngineConfigurations`` dataclass for managing SCM inputs
    - ``_validate_and_set_defaults()`` method for configuration validation
    - Better error messages for configuration issues
    - Support for custom natural CH4 and N2O emissions
- **Enhanced precipitation handling**
    - Gamma distribution fitting with improved error handling for common scipy errors
    - Support for 3D and 4D spatial data
    - Multiple distribution options (gamma, Weibull, lognormal, generalized gamma)
    - Better validation of positive-only data requirements
- **Improved CI/CD and testing infrastructure**
    - Updated Python version support: 3.10, 3.11, 3.12 (dropped 3.9)
    - Minimum coverage requirement raised from 85% to 90%
    - Updated notebook tests to use new interface examples
    - Removed outdated workflow files
- **Updated documentation and examples**
    - Comprehensive README rewrite with quick start guide
    - New ``METEOR_Interface_Examples.ipynb`` notebook with full workflow examples
    - New ``METEOR_Interface_Paper_plots.ipynb`` notebook
    - Updated ``GCAM_predict.ipynb`` notebook
    - Removed outdated example notebooks
- **Updated dependencies, and overall infrastructure improvements**
    - Removed  setup.py, setup.cfg and  ``requirements.txt`` and moved to ``pyproject.toml`` for modern packaging
    - Improved Makefile and ci-cd workflows to fit with new infrastructure
    - Updating to require newer ciceroscm version with various improvements

### Fixed

- Precipitation data now correctly processed as absolute values, not anomalies
- Temperature anomaly calculation properly uses piControl baseline
- Noise model fitting with short time series (added minimum length check)
- Error handling in gamma distribution fitting for edge cases
- Configuration validation in SCM engine (nystart/emstart/nyend relationships)
- Ensemble saving to netCDF can handle annual and monthly gridded data in same Dataset (#68)
- Fixed default behavior for noise modelling - changed ``use_exog`` from ``"temp_only"`` to ``"none"`` for non-temperature/precipitation variables to avoid unintended use of temperature as an exogenous variable in noise models for other variables

### Storage Optimization

- Cache only monthly variable-specific files
- Compute annual and training data on-demand from monthly cache
- netCDF4/zlib compression reduces file sizes by ~45%
- Configurable compression levels for storage/speed tradeoffs


[Version 1.5.0]
-----------------------------

### Added

- Monthly climate variability modeling via ``src/meteor/noise_generator.py`` - PCA/VARX-based monthly climate variability modeling
- ``to_monthly()`` method in ``src/meteor/meteor.py`` for converting annual to monthly climate data
- Climate impact assessment framework in ``src/meteor/impacts/`` module:
    - ``impacts_core.py`` - Base classes for impact calculators and ensemble analysis
    - ``calculators/degree_days.py`` - Heating and cooling degree days calculator (example impact parameterisation)
    - ``ensemble.py`` - Ensemble statistics and analysis tools
    - ``utils.py`` - Utility functions for climate data processing
- Automatic local caching system for CMIP6 data with configurable cache directory
- Error handling for cache setup with graceful fallback
- Noise model training capabilities in ``src/meteor/cmip6_meteor_data_getter.py``
- Test suite in ``tests/unit/impacts/`` covering all framework components
- End-to-end impacts workflow validation in ``tests/integration/impacts/test_workflow.py``
- Small test datasets for faster testing in ``tests/test-data/``
- ``notebooks/CMIP6_noise_model_examples.ipynb`` - Multi-model workflow demonstration
- ``notebooks/METEOR_Impacts_Clean_Demo.ipynb`` - Combined impacts and noise modeling

### Changed

- Enhanced ``src/meteor/prpatt.py`` with improved pattern scaling algorithms for monthly data support
- Updated ``README.md`` with quick start guide, API reference, and development setup
- All modules with numpy-style docstrings and comprehensive inline documentation

### Functionality Features

- Temperature-dependent seasonality that evolves with global warming
- Ensemble generation for uncertainty quantification
- Training from CMIP6 historical and scenario data
- Integration with existing METEOR pattern scaling workflow
- Extensible ``ImpactCalculator`` class for custom impact metrics
- ``ImpactEnsemble`` class for statistical analysis across multiple realizations
- Built-in degree days calculator with customizable parameters
- General workflows for ensemble creation and impact calculation

[Version 1.0.2]
---------------

### Added

- Additional notebooks for revised article version

[Version 1.0.1]
---------------

### Added

- Version for description paper, including figure reproduction
- Preprint available: https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1038/

[Version 1.0.0]
---------------

### Added

- Initial release with core METEOR functionality

### Fixed

- Known issues with modelling negative emissions
