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
