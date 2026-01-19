"""
High-Level METEOR Interface

This module provides a simplified, user-friendly interface to METEOR's
pattern scaling and noise generation capabilities.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
from ciceroscm import input_handler

from . import cache_utils
from .cmip6_meteor_data_getter import Cmip6MeteorDataGetter
from .ensemble_output import EnsembleOutput, VariableOutput
from .geo_data_utils import (
    create_region_mask,
    extract_point,
    global_mean,
    regional_mean,
)
from .meteor import MeteorPatternScaling
from .noise_generator import train_noise_model_from_cmip6
from .variable_transforms import get_variable_transform_config


class MeteorInterface:
    """
    High-level interface for training and generating METEOR emulators.

    This class simplifies the workflow of:
    1. Loading CMIP6 data
    2. Training pattern scaling models
    3. Training noise models
    4. Applying variable-specific transforms (e.g., precipitation positivity)
    5. Generating ensemble outputs with flexible spatial aggregations
    6. Computing impact metrics

    Parameters
    ----------
    model : str
        CMIP6 model name (e.g., 'NorESM2-MM', 'CESM2')
    variables : str or list of str
        Climate variable(s) to emulate ('tas', 'pr', or ['tas', 'pr'])
    cache_dir : str, optional
        Directory for caching trained models and data
    data_getter_kwargs : dict, optional
        Additional arguments for Cmip6MeteorDataGetter

    Examples
    --------
    >>> # Simple single-variable case
    >>> emulator = MeteorInterface(
    ...     model='NorESM2-MM',
    ...     variables='pr',
    ...     cache_dir='./cache'
    ... )
    >>> emulator.train(auto=True)
    >>> ensemble = emulator.generate_ensemble_outputs(
    ...     scenario='ssp245',
    ...     start_year=2020,
    ...     end_year=2100,
    ...     n_realizations=100,
    ...     timeseries=['global', 'regional:EAS']
    ... )
    >>>
    >>> # Multi-variable with gridded output
    >>> emulator = MeteorInterface(
    ...     model='CESM2',
    ...     variables=['tas', 'pr'],
    ...     cache_dir='./cache'
    ... )
    >>> emulator.train(auto=True)
    >>> ensemble = emulator.generate_ensemble_outputs(
    ...     scenario='ssp245',
    ...     start_year=2020,
    ...     end_year=2100,
    ...     n_realizations=100,
    ...     timeseries=['global'],
    ...     gridded={'annual': [2030, 2050, 2100]}
    ... )
    >>>
    >>> # Custom emissions scenario
    >>> ensemble = emulator.generate_ensemble_outputs(
    ...     scenario={'emissions': 'path/to/custom_emissions.txt'},
    ...     start_year=2020,
    ...     end_year=2100,
    ...     n_realizations=100,
    ...     timeseries=['global']
    ... )
    """

    def __init__(self, model, variables, cache_dir=None, data_getter_kwargs=None):
        """
        Initialize MeteorInterface.

        Parameters
        ----------
        model : str
            CMIP6 model name
        variables : str or list of str
            Climate variable(s) to emulate
        cache_dir : str, optional
            Directory for caching models and data
        data_getter_kwargs : dict, optional
            Additional arguments for Cmip6MeteorDataGetter
        """
        # Normalize variables to list
        if isinstance(variables, str):
            self.variables = [variables]
        else:
            self.variables = list(variables)

        self.model = model
        self.cache_dir = cache_dir or "./cache"

        # Initialize data getter
        data_getter_kwargs = data_getter_kwargs or {}
        default_exps = ["piControl", "historical", "ssp245", "abrupt-4xCO2"]
        default_dbe = ["CMIP", "CMIP", "ScenarioMIP", "CMIP"]

        # Note: We explicitly enable caching for the high-level interface to provide
        # good performance by default. Users of the low-level Cmip6MeteorDataGetter
        # can control caching behavior directly.
        self.data_getter = Cmip6MeteorDataGetter(
            exps=data_getter_kwargs.get("exps", default_exps),
            flds=self.variables,
            dbe=data_getter_kwargs.get("dbe", default_dbe),
            enable_cache=True,
        )

        # Storage for trained models
        self.pattern_models = {}
        self.noise_models = {}

        # Set up variable-specific transforms
        self.transforms = {}
        for var in self.variables:
            self.transforms[var] = get_variable_transform_config(var)

        # Training configuration
        self._training_config = {}
        self._is_trained = {var: False for var in self.variables}

    def train(
        self, auto=True, training_scenario="ssp245", variable_configs=None, verbose=True
    ):
        """
        Train pattern scaling and noise models for all variables.

        Parameters
        ----------
        auto : bool, optional
            Use automatic smart defaults (default True)
        training_scenario : str, optional
            Default scenario for training noise models (default 'ssp245').
            Can be overridden per-variable in variable_configs.
        variable_configs : dict, optional
            Custom configuration per variable. Keys are variable names,
            values are dicts with configuration options including:
            - n_modes_pattern : int
            - n_modes_noise : int
            - lag_order : int
            - use_exog : str ('all', 'temp_only', 'none')
            - training_scenario : str (overrides method parameter)
            - transform : bool
            - transform_type : str
        verbose : bool, optional
            Print progress messages (default True)

        Examples
        --------
        >>> # Automatic training with smart defaults
        >>> emulator.train(auto=True)
        >>>
        >>> # Custom training scenario for all variables
        >>> emulator.train(training_scenario='ssp370')
        >>>
        >>> # Custom configuration with per-variable scenarios
        >>> emulator.train(
        ...     training_scenario='ssp245',  # default for most variables
        ...     variable_configs={
        ...         'tas': {'n_modes_noise': 40, 'use_exog': 'all'},
        ...         'pr': {'n_modes_noise': 40, 'training_scenario': 'ssp370'}  # override for pr
        ...     }
        ... )
        """
        if verbose:
            print("=" * 60)
            print(f"Training METEOR emulator for {self.model}")
            print(f"Variables: {', '.join(self.variables)}")
            print("=" * 60)

        for variable in self.variables:
            if verbose:
                print(f"\n🔧 Training {variable.upper()}...")

            # Get configuration with hybrid precedence
            if auto:
                config = self._get_default_config(variable)
                # Override with method parameter if different from default
                if training_scenario != "ssp245":
                    config["training_scenario"] = training_scenario
                # Override with variable-specific config if provided
                if variable_configs and variable in variable_configs:
                    config.update(variable_configs[variable])
            else:
                config = variable_configs.get(variable, {}) if variable_configs else {}
                # Apply method parameter as default if not in config
                if "training_scenario" not in config:
                    config["training_scenario"] = training_scenario

            self._training_config[variable] = config

            # Train pattern scaling
            if verbose:
                print("   → Training pattern scaling model...")
            self._train_pattern_scaling(variable, config, verbose=verbose)

            # Train noise model
            if verbose:
                print("   → Training noise model...")
            self._train_noise_model(variable, config, verbose=verbose)

            # Fit transforms if needed
            transform_config = get_variable_transform_config(variable)
            if transform_config.transform_type and config.get("transform", True):
                if verbose:
                    print(
                        f"   → Fitting {transform_config.transform_type} transform..."
                    )
                    print(f"      Reason: {transform_config.reason}")
                self._fit_transform(variable, transform_config, config, verbose=verbose)

            self._is_trained[variable] = True

            if verbose:
                print(f"   ✅ {variable.upper()} training complete")

        if verbose:
            print("\n" + "=" * 60)
            print("✅ All variables trained successfully")
            print("=" * 60)

    def _get_default_config(self, variable):
        """Get smart default configuration for a variable."""
        config = {
            "n_modes_pattern": 3,
            "n_modes_noise": 40,
            "lag_order": 2,
            "use_picontrol_baseline": True,
            "training_scenario": "ssp245",  # ✅ Add default training scenario
        }

        # Variable-specific defaults
        if variable == "tas":
            config["use_exog"] = "all"
            config["transform"] = False
        elif variable == "pr":
            config["use_exog"] = "none"
            config["transform"] = True
            config["transform_type"] = "gamma"
        else:
            # Generic defaults for other variables
            config["use_exog"] = "temp_only"
            config["transform"] = False

        return config

    def _train_pattern_scaling(self, variable, config, verbose=True):
        """
        Train pattern scaling model for a variable.

        Creates a MeteorPatternScaling model that maps global temperature
        trajectories to spatial climate patterns. Uses cached model if available.

        Parameters
        ----------
        variable : str
            Climate variable to train ('tas', 'pr', etc.)
        config : dict
            Configuration with 'n_modes_pattern' specifying number of patterns
        verbose : bool
            Print training progress messages
        """
        cache_dir = os.path.join(self.cache_dir, "pattern_scaling")
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = cache_utils.get_pattern_scaling_cache_path(
            self.model, cache_dir, variable=variable
        )

        # Check cache
        is_valid, cached_model, info = cache_utils.validate_pattern_scaling_cache(
            cache_file, self.model, expected_fields=[variable]
        )

        if is_valid:
            if verbose:
                print("      ✓ Using cached pattern scaling model")
            training_data = None
            ssp_config = None
        else:
            if verbose:
                print(f"      ⚠️  Cache miss: {info.get('message', 'No cache found')}")

            # Prepare training data
            training_data = self.data_getter.prepare_pattern_scaling_training_data(
                self.model, "ssp245"
            )
            ssp_config = self.data_getter.load_ssp_config("ssp245")

        # Create pattern scaling model
        self.pattern_models[variable] = MeteorPatternScaling(
            f"cmip6-{self.model}-aer-{variable}",
            {variable: config["n_modes_pattern"]},
            None if training_data is None else lambda key: training_data[key],
            ssp_input=ssp_config,
            from_file=False,
            exp_list=None if training_data is None else ["base", "co2x4", "sulxanom"],
            anom_timescales={variable: config["n_modes_pattern"]},
            cache_dir=cache_dir,
        )

    def _train_noise_model(self, variable, config, verbose=True):
        """
        Train monthly noise model for a variable.

        Creates a MeteorNoiseGenerator model that adds realistic monthly
        variability to annual pattern scaling predictions. Uses cached model
        if available and configuration matches.

        Parameters
        ----------
        variable : str
            Climate variable to train ('tas', 'pr', etc.)
        config : dict
            Configuration including:
            - 'n_modes_noise': Number of PCA modes for noise representation
            - 'lag_order': Temporal memory in VARX model
            - 'use_exog': Exogenous variable usage ('all', 'temp_only', 'none')
            - 'training_scenario': Scenario for temperature trajectory (default: 'ssp245')
        verbose : bool
            Print training progress messages
        """
        cache_dir = os.path.join(self.cache_dir, "noise_models")
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = cache_utils.get_noise_model_cache_path(
            self.model, variable, cache_dir
        )

        # Check cache
        is_valid, cached_model, info = cache_utils.validate_noise_model_cache(
            cache_file,
            variable,
            n_modes=config["n_modes_noise"],
            lag_order=config["lag_order"],
        )

        if is_valid:
            if verbose:
                print("      ✓ Using cached noise model")
            self.noise_models[variable] = cached_model
        else:
            if verbose:
                print("      ⚠️  Training new noise model...")

            # ✅ Get training scenario from config
            training_scenario = config.get("training_scenario", "ssp245")

            # ✅ Generate pattern scaling prediction for custom_global_temp
            # This ensures noise model training uses same temperature trajectory as generation

            cscm_data_dir = os.path.join(os.path.dirname(__file__), "default_scm_data")
            conc_file = os.path.join(
                cscm_data_dir, f"{training_scenario}_conc_RCMIP.txt"
            )
            em_file = os.path.join(cscm_data_dir, f"{training_scenario}_em_RCMIP.txt")

            ih = input_handler.InputHandler({})
            conc_data = input_handler.read_inputfile(conc_file)
            em_data = ih.read_emissions(em_file)

            # Generate pattern prediction
            pattern_model = self.pattern_models[variable]
            climate_prediction = pattern_model.predict_from_combined_experiment(
                em_data, conc_data, [variable]
            )
            annual_prediction = climate_prediction[variable]

            # Convert to monthly
            monthly_prediction = pattern_model.to_monthly(
                annual_prediction, start_year=0
            )
            monthly_warming = global_mean(monthly_prediction).values

            # Trim first 100 years (spin-up) to match working notebook
            monthly_warming_trimmed = monthly_warming[1200:]  # 100 years * 12 months

            if verbose:
                print(
                    f"      → Using {training_scenario} pattern prediction for training"
                )
                print(
                    f"      → Temperature trajectory: {len(monthly_warming_trimmed) // 12} years"
                )

            # Train noise model using data getter interface
            self.noise_models[variable] = train_noise_model_from_cmip6(
                self.data_getter,
                experiments=["historical", training_scenario],  # ✅ Use config scenario
                model_name=self.model,
                variable_name=variable,
                n_modes=config["n_modes_noise"],
                lag_order=config["lag_order"],
                use_exog=config["use_exog"],
                custom_global_temp=monthly_warming_trimmed,  # ✅ Pass pattern prediction
                cache_dir=cache_dir,
            )

    def _fit_transform(self, variable, transform_config, config, verbose=True):
        """
        Prepare variable-specific transform configuration.

        Stores transform settings for use during generation. Actual fitting of
        distribution parameters is deferred until generation time when we have
        the source data to fit.

        Parameters
        ----------
        variable : str
            Climate variable
        transform_config : VariableTransformConfig
            Transform configuration from registry
        config : dict
            Training configuration
        verbose : bool
            Print status messages
        """
        # Store transform configuration for later use during generation
        # We fit parameters on-demand during generation because we need
        # the generated data to fit the source distribution
        self.transforms[variable] = {
            "config": transform_config,
            "fitted_params_1d": {},  # Will be populated per aggregation during generation
            "fitted_params_3d": None,  # Will be fitted if gridded output requested
        }

    def generate_ensemble_outputs(
        self,
        scenario,
        start_year,
        end_year,
        n_realizations,
        timeseries=None,
        gridded=None,
        impacts=None,
        include_noise=True,
        save_to=None,
        custom_regions=None,
        verbose=True,
    ):
        """
        Generate ensemble outputs for all variables.

        Parameters
        ----------
        scenario : str or dict
            Either:
            - String: SSP scenario ('ssp126', 'ssp245', 'ssp370', 'ssp585')
            - Dict with custom emissions/concentrations:
              * {'emissions': path_or_DataFrame} - Uses ssp245 concentrations by default
              * {'emissions': path_or_DataFrame, 'base_scenario': 'ssp370'} - Custom base
              * {'emissions': path_or_DataFrame, 'concentrations': path_or_DataFrame} - Full custom
              * {'emissions': path_or_DataFrame, 'name': 'my_scenario'} - Optional label
        start_year : int
            First year of output
        end_year : int
            Last year of output (inclusive)
        n_realizations : int
            Number of ensemble members to generate. Ignored if include_noise=False.
        timeseries : list of str, optional
            Spatial aggregations for time series output. Options:
            - 'global' : Global mean
            - 'regional:CODE' : AR6 region (e.g., 'regional:EAS')
            - 'regional:custom:NAME' : Custom region (requires custom_regions dict)
            - 'point:LAT,LON' : Specific location (e.g., 'point:19.0,72.8')
        gridded : dict, optional
            Gridded output specification. Keys:
            - 'annual' : list of years for annual means
            - 'monthly' : True for all monthly fields (memory intensive!)
            - 'climatology' : {period: season} for multi-year seasonal means
        impacts : dict, optional
            Impact models to apply. Format: {variable: {impact_name: config}}
        include_noise : bool, optional
            If True, generate ensemble with stochastic variability (default).
            If False, return climatology only (n_realizations forced to 1).
        custom_regions : dict, optional
            Custom region definitions for 'regional:custom:NAME' aggregations.
            Format: {'name': {'lat': (min, max), 'lon': (min, max)}}
        save_to : str, optional
            Path to save outputs to netCDF
        verbose : bool, optional
            Print progress messages (default True)

        Returns
        -------
        EnsembleOutput
            Container with all generated outputs

        Examples
        --------
        >>> # Generate ensemble with noise
        >>> ensemble = emulator.generate_ensemble_outputs(
        ...     scenario='ssp245',
        ...     start_year=2020,
        ...     end_year=2100,
        ...     n_realizations=100,
        ...     timeseries=['global', 'regional:EAS', 'point:59.9,10.8'],
        ...     gridded={'annual': [2030, 2050, 2100]},
        ...     impacts={'tas': {'degree_days': {'hdd_base': 18.0, 'cdd_base': 18.0}}}
        ... )
        >>>
        >>> # Generate climatology only (no noise)
        >>> climatology = emulator.generate_ensemble_outputs(
        ...     scenario='ssp245',
        ...     start_year=2020,
        ...     end_year=2100,
        ...     n_realizations=1,  # Ignored, forced to 1
        ...     timeseries=['global'],
        ...     include_noise=False
        ... )
        """
        # Handle climatology-only mode
        if not include_noise:
            if verbose and n_realizations > 1:
                print(
                    "Note: include_noise=False, forcing n_realizations=1 (climatology only)"
                )
            n_realizations = 1

        # Parse scenario to get name for display
        scenario_info = self._parse_scenario_input(scenario)
        scenario_name = scenario_info["name"]

        # Check all variables are trained
        for var in self.variables:
            if not self._is_trained[var]:
                raise RuntimeError(f"Variable '{var}' not trained. Call train() first.")

        if verbose:
            print("=" * 60)
            print(f"Generating ensemble for {scenario_name}")
            print(f"  Years: {start_year}-{end_year}")
            print(f"  Realizations: {n_realizations}")
            print("=" * 60)

        results = {}

        for variable in self.variables:
            if verbose:
                print(f"\n📊 Generating {variable.upper()}...")

            var_output = VariableOutput(variable)

            # Generate timeseries if requested
            if timeseries:
                if verbose:
                    print(f"   → Time series: {len(timeseries)} aggregations")
                var_output.timeseries = self._generate_timeseries(
                    variable,
                    scenario,
                    start_year,
                    end_year,
                    n_realizations,
                    timeseries,
                    custom_regions=custom_regions,
                    include_noise=include_noise,
                    verbose=verbose,
                )

            # Generate gridded if requested
            if gridded:
                if verbose:
                    print("   → Gridded outputs...")
                var_output.gridded = self._generate_gridded(
                    variable,
                    scenario,
                    start_year,
                    end_year,
                    n_realizations,
                    gridded,
                    include_noise=include_noise,
                    verbose=verbose,
                )

            # Apply impacts if requested
            if impacts and variable in impacts:
                if verbose:
                    print("   → Computing impact metrics...")
                var_output.impacts = self._apply_impacts(
                    var_output,
                    variable,
                    impacts[variable],
                    custom_regions=custom_regions,
                    verbose=verbose,
                )

            results[variable] = var_output

            if verbose:
                print(f"   ✅ {variable.upper()} complete")

        # Create ensemble output
        ensemble = EnsembleOutput(
            results,
            metadata={
                "model": self.model,
                "scenario": scenario_name,
                "year_range": f"{start_year}-{end_year}",
                "n_realizations": n_realizations,
                "variables": self.variables,
            },
        )

        # Save if requested
        if save_to:
            ensemble.to_netcdf(save_to)

        if verbose:
            print("\n" + "=" * 60)
            print("✅ Generation complete")
            print("=" * 60)

        return ensemble

    def generate(self, *args, **kwargs):
        """
        Generate ensemble outputs for all variables.

        .. deprecated::
            Use :meth:`generate_ensemble_outputs` instead.
            This method will be removed in a future version.

        See Also
        --------
        generate_ensemble_outputs : Replacement method with same signature
        """
        import warnings

        warnings.warn(
            "generate() is deprecated and will be removed in a future version. "
            "Use generate_ensemble_outputs() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.generate_ensemble_outputs(*args, **kwargs)

    def _parse_scenario_input(self, scenario):
        """
        Parse scenario input and return emissions/concentrations info.

        Parameters
        ----------
        scenario : str or dict
            Scenario specification

        Returns
        -------
        dict
            Dictionary with keys:
            - 'type': 'ssp' or 'custom'
            - 'name': scenario name
            - 'emissions': emissions file path or DataFrame (if custom)
            - 'concentrations': concentrations file path or DataFrame (if custom)
        """
        if isinstance(scenario, str):
            # Standard SSP scenario
            return {
                "type": "ssp",
                "name": scenario,
                "emissions": None,
                "concentrations": None,
            }
        elif isinstance(scenario, dict):
            # Custom scenario
            if "emissions" not in scenario:
                raise ValueError("Custom scenario dict must include 'emissions' key")

            # Get emissions (path or DataFrame)
            emissions = scenario["emissions"]

            # Get concentrations (path, DataFrame, or use base_scenario)
            if "concentrations" in scenario:
                concentrations = scenario["concentrations"]
            else:
                # Use base_scenario concentrations (default: ssp245)
                base_scenario = scenario.get("base_scenario", "ssp245")
                cscm_data_dir = os.path.join(
                    os.path.dirname(__file__), "default_scm_data"
                )
                concentrations = os.path.join(
                    cscm_data_dir, f"{base_scenario}_conc_RCMIP.txt"
                )

            # Get scenario name for labeling
            scenario_name = scenario.get("name", "custom")

            return {
                "type": "custom",
                "name": scenario_name,
                "emissions": emissions,
                "concentrations": concentrations,
            }
        else:
            raise TypeError(f"scenario must be str or dict, got {type(scenario)}")

    def _load_emissions_concentrations(
        self, emissions_spec, concentrations_spec, verbose=False
    ):
        """
        Load emissions and concentrations from files or DataFrames.

        Parameters
        ----------
        emissions_spec : str or pd.DataFrame
            Path to emissions file or DataFrame
        concentrations_spec : str or pd.DataFrame
            Path to concentrations file or DataFrame
        verbose : bool
            Print loading messages

        Returns
        -------
        tuple
            (emissions_data, concentrations_data) as DataFrames
        """
        # Load emissions
        if isinstance(emissions_spec, pd.DataFrame):
            em_data = emissions_spec
            if verbose:
                print("      → Using provided emissions DataFrame")
        elif isinstance(emissions_spec, str):
            ih = input_handler.InputHandler({})
            em_data = ih.read_emissions(emissions_spec)
            if verbose:
                print(
                    f"      → Loaded emissions from {os.path.basename(emissions_spec)}"
                )
        else:
            raise TypeError(
                f"emissions must be str path or DataFrame, got {type(emissions_spec)}"
            )

        # Load concentrations
        if isinstance(concentrations_spec, pd.DataFrame):
            conc_data = concentrations_spec
            if verbose:
                print("      → Using provided concentrations DataFrame")
        elif isinstance(concentrations_spec, str):
            conc_data = input_handler.read_inputfile(concentrations_spec)
            if verbose:
                print(
                    f"      → Loaded concentrations from {os.path.basename(concentrations_spec)}"
                )
        else:
            raise TypeError(
                f"concentrations must be str path or DataFrame, got {type(concentrations_spec)}"
            )

        return em_data, conc_data

    def _prepare_scm_forcing_data(self, scenario, start_year, end_year, verbose=True):
        """
        Prepare emissions and concentrations data for SCM pattern scaling.

        Handles both standard SSP scenarios and custom emissions/concentrations,
        including data validation, range extension, and clipping warnings.

        Parameters
        ----------
        scenario : str or dict
            Scenario specification (SSP name or custom dict)
        start_year : int
            Requested start year
        end_year : int
            Requested end year
        verbose : bool
            Print status messages

        Returns
        -------
        tuple
            (em_data, conc_data, actual_data_end, effective_start_year, effective_end_year)
            where actual_data_end is None for SSP scenarios
        """
        # Parse scenario input
        scenario_info = self._parse_scenario_input(scenario)
        scenario_name = scenario_info["name"]

        # Load forcing data
        if scenario_info["type"] == "ssp":
            # Standard SSP scenario
            cscm_data_dir = os.path.join(os.path.dirname(__file__), "default_scm_data")
            conc_file = os.path.join(cscm_data_dir, f"{scenario_name}_conc_RCMIP.txt")
            em_file = os.path.join(cscm_data_dir, f"{scenario_name}_em_RCMIP.txt")

            ih = input_handler.InputHandler({})
            conc_data = input_handler.read_inputfile(conc_file)
            em_data = ih.read_emissions(em_file)
            actual_data_end = None
            effective_start_year = start_year
            effective_end_year = end_year
        else:
            # Custom scenario
            em_data, conc_data = self._load_emissions_concentrations(
                scenario_info["emissions"],
                scenario_info["concentrations"],
                verbose=verbose,
            )

            # Check if emissions data covers requested time range
            em_start_year = em_data.index[0]
            em_end_year = em_data.index[-1]

            effective_start_year = start_year
            effective_end_year = end_year

            if end_year > em_end_year:
                if verbose:
                    print(
                        f"      ⚠️  Warning: Emissions data ends at {em_end_year}, requested {end_year}"
                    )
                    print(f"      → Clipping output to {em_start_year}-{em_end_year}")
                effective_end_year = em_end_year

            if start_year < em_start_year:
                if verbose:
                    print(
                        f"      ⚠️  Warning: Emissions data starts at {em_start_year}, requested {start_year}"
                    )
                    print(f"      → Clipping output to {em_start_year}-{em_end_year}")
                effective_start_year = em_start_year

            # Extend data to 2100 if needed (SCM default nyend)
            data_end = min(em_data.index[-1], conc_data.index[-1])

            if data_end < 2100:
                if verbose:
                    print(
                        f"      → Emissions data ends at {data_end}, extending to 2100 (holding final values)"
                    )

                # Extend by repeating last year's values
                years_to_add = list(range(data_end + 1, 2101))
                for year in years_to_add:
                    em_data.loc[year] = em_data.loc[data_end]
                    conc_data.loc[year] = conc_data.loc[data_end]

                # Sort index to maintain chronological order
                em_data = em_data.sort_index()
                conc_data = conc_data.sort_index()

            actual_data_end = data_end

        return (
            em_data,
            conc_data,
            actual_data_end,
            effective_start_year,
            effective_end_year,
        )

    def _get_or_compute_pattern_scaling(
        self, variable, scenario, start_year, end_year, verbose=True
    ):
        """
        Get pattern scaling results from cache or compute if not cached.

        Caches the FULL scenario trajectory (e.g., 1750-2300) to maximize reuse
        across different time ranges and output types. Returns sliced results
        for the requested time range.

        Parameters
        ----------
        variable : str
            Climate variable ('tas', 'pr')
        scenario : str or dict
            Scenario specification (SSP name or custom dict)
        start_year : int
            Start year for slicing
        end_year : int
            End year for slicing (inclusive)
        verbose : bool
            Print cache status messages

        Returns
        -------
        tuple
            (monthly_prediction_sliced, monthly_warming_sliced, em_data, conc_data)
        """
        scenario_info = self._parse_scenario_input(scenario)
        scenario_name = scenario_info["name"]

        if verbose:
            if scenario_info["type"] == "ssp":
                print(
                    f"      → Computing pattern scaling for {variable}, {scenario_name}..."
                )
            else:
                print(
                    f"      → Computing pattern scaling for {variable}, custom scenario '{scenario_name}'..."
                )

        # Prepare emissions and concentrations data
        em_data, conc_data, actual_data_end, eff_start_year, eff_end_year = (
            self._prepare_scm_forcing_data(scenario, start_year, end_year, verbose)
        )

        # Generate FULL pattern scaling prediction (annual)
        pattern_model = self.pattern_models[variable]
        climate_prediction = pattern_model.predict_from_combined_experiment(
            em_data, conc_data, [variable]
        )
        annual_prediction = climate_prediction[variable]

        # Convert to monthly
        full_monthly_prediction = pattern_model.to_monthly(
            annual_prediction, start_year=0
        )

        # Determine base year
        if hasattr(full_monthly_prediction, "year"):
            base_year = int(full_monthly_prediction.year[0])
        else:
            base_year = 1750  # Default assumption

        # Compute global mean for noise model
        full_monthly_warming = global_mean(full_monthly_prediction).values

        # Slice to requested time range (using effective years from data preparation)
        if actual_data_end is not None and eff_end_year < end_year and verbose:
            print(
                f"      → Clipping output to {eff_start_year}-{eff_end_year} (data availability)"
            )

        start_month_idx = (eff_start_year - base_year) * 12
        end_month_idx = (eff_end_year - base_year + 1) * 12  # +1 to include end_year

        monthly_prediction_sliced = full_monthly_prediction.isel(
            month=slice(start_month_idx, end_month_idx)
        )
        monthly_warming_sliced = full_monthly_warming[start_month_idx:end_month_idx]

        return monthly_prediction_sliced, monthly_warming_sliced, em_data, conc_data

    def _generate_timeseries(
        self,
        variable,
        scenario,
        start_year,
        end_year,
        n_realizations,
        aggregations,
        custom_regions=None,
        include_noise=True,
        verbose=True,
    ):
        """
        Generate time series outputs with spatial aggregations.

        Parameters
        ----------
        variable : str
            Climate variable to generate
        scenario : str
            Emission scenario ('ssp245', 'ssp585', etc.)
        start_year : int
            Start year
        end_year : int
            End year (inclusive)
        n_realizations : int
            Number of ensemble members
        aggregations : list of str
            Spatial aggregations to compute:
            - 'global': Global mean
            - 'regional:CODE': AR6 region mean (e.g., 'regional:NEU')
            - 'regional:custom:NAME': Custom region (requires custom_regions dict)
            - 'point:LAT,LON': Single grid point (e.g., 'point:59.9,10.8')
        custom_regions : dict, optional
            Custom region definitions: {'name': {'lat': (min, max), 'lon': (min, max)}}
        include_noise : bool
            If True, add stochastic noise; if False, return forced response only
        verbose : bool
            Print progress messages

        Returns
        -------
        dict
            Dictionary mapping aggregation names to xarray DataArrays
            with shape (n_realizations, n_months)
        """
        # Parse scenario to get the actual name (handle both string and dict)
        scenario_info = self._parse_scenario_input(scenario)
        scenario_name = scenario_info["name"]

        # For custom scenarios, use ssp245 as the training scenario
        # (we need CMIP6 data for transform fitting, not custom emissions)
        training_scenario = (
            "ssp245" if scenario_info["type"] == "custom" else scenario_name
        )

        # Get pattern scaling results
        monthly_prediction, monthly_warming, em_data, conc_data = (
            self._get_or_compute_pattern_scaling(
                variable, scenario, start_year, end_year, verbose
            )
        )

        # Get CMIP6 data for transform fitting
        if verbose:
            print(f"      → Loading CMIP6 training data for {training_scenario}...")
        ssp_data = self.data_getter.make_meteor_training_data_composite(
            ["historical", training_scenario], self.model, monthly=True
        )[variable]

        # Convert tas to anomalies from piControl baseline for comparison with emulated output
        if variable == "tas":
            if verbose:
                print("      → Converting tas to anomalies from piControl baseline...")
            # Load piControl data for baseline
            picontrol_data = self.data_getter.make_meteor_training_data_composite(
                ["piControl"], self.model, monthly=True
            )[variable]
            # Compute piControl climatology (mean across all time)
            picontrol_mean = picontrol_data.mean(dim="month")
            # Convert to anomalies
            ssp_data = ssp_data - picontrol_mean

        # ✅ Generate stochastic PCs (or skip if climatology only)
        noise_model = self.noise_models[variable]
        stochastic_pcs = None

        if include_noise:
            # CRITICAL: Generate stochastic PCs ONCE for all aggregations
            # This ensures all spatial scales share the same underlying variability
            if verbose:
                print(
                    f"      → Generating {n_realizations} stochastic PC realizations..."
                )

            stochastic_pcs = noise_model.generate_stochastic_pcs(
                monthly_warming,
                n_realizations=n_realizations,
                random_seed=None,  # Can expose this as parameter if needed
            )
        else:
            if verbose:
                print("      → Climatology only (no stochastic variability)")

        # Generate outputs for each aggregation
        results = {}
        transform_info = self.transforms.get(variable, None)

        # Handle both dict (fitted) and VariableTransformConfig (not fitted) cases
        if isinstance(transform_info, dict):
            transform_config = transform_info.get("config")
        else:
            transform_config = transform_info  # It's a VariableTransformConfig object

        for agg in aggregations:
            if verbose:
                print(f"      • {agg}")

            # Parse aggregation type
            if agg == "global":
                # Global mean
                pattern_agg = global_mean(monthly_prediction).values
                if include_noise:
                    raw_ensemble = noise_model.generate_regional_mean_realizations(
                        monthly_warming,
                        region="global",
                        n_realizations=n_realizations,
                        stochastic_pcs=stochastic_pcs,
                        noise_only=True,
                        add_base=pattern_agg,
                        return_numpy=False,
                    )
                else:
                    # Climatology only: return pattern scaling with shape (1, time)
                    raw_ensemble = pattern_agg[np.newaxis, :]
                cmip6_agg = global_mean(ssp_data)

            elif agg.startswith("regional:"):
                # Check if it's a custom region
                parts = agg.split(":")
                if len(parts) == 3 and parts[1] == "custom":
                    # Custom region: regional:custom:NAME
                    region_name = parts[2]
                    if custom_regions is None or region_name not in custom_regions:
                        raise ValueError(
                            f"Custom region '{region_name}' not found in custom_regions. "
                            f"Available: {list(custom_regions.keys()) if custom_regions else 'None'}"
                        )

                    # Create mask from bounding box
                    region_bbox = custom_regions[region_name]
                    region_mask = create_region_mask(
                        monthly_prediction, bbox=region_bbox
                    )

                    # Apply to pattern and CMIP6 data
                    pattern_agg = regional_mean(
                        monthly_prediction, region_mask=region_mask
                    ).values
                    cmip6_agg = regional_mean(ssp_data, region_mask=region_mask)

                    # For noise, use global since we don't have EOFs for custom regions
                    if include_noise:
                        raw_ensemble = noise_model.generate_regional_mean_realizations(
                            monthly_warming,
                            region="global",  # Use global noise as approximation
                            n_realizations=n_realizations,
                            stochastic_pcs=stochastic_pcs,
                            noise_only=True,
                            add_base=pattern_agg,
                            return_numpy=False,
                        )
                    else:
                        raw_ensemble = pattern_agg[np.newaxis, :]
                else:
                    # AR6 region
                    region_code = parts[1]
                    pattern_agg = regional_mean(
                        monthly_prediction, region_code=region_code
                    ).values
                    if include_noise:
                        raw_ensemble = noise_model.generate_regional_mean_realizations(
                            monthly_warming,
                            region=region_code,
                            n_realizations=n_realizations,
                            stochastic_pcs=stochastic_pcs,
                            noise_only=True,
                            add_base=pattern_agg,
                            return_numpy=False,
                        )
                    else:
                        # Climatology only: return pattern scaling with shape (1, time)
                        raw_ensemble = pattern_agg[np.newaxis, :]
                    cmip6_agg = regional_mean(ssp_data, region_code=region_code)

            elif agg.startswith("point:"):
                # Point extraction
                coords = agg.split(":")[1]
                lat_str, lon_str = coords.split(",")
                lat = float(lat_str)
                lon = float(lon_str)

                pattern_agg = extract_point(monthly_prediction, lat, lon).values
                if include_noise:
                    raw_ensemble = noise_model.generate_regional_mean_realizations(
                        monthly_warming,
                        lat=lat,
                        lon=lon,
                        n_realizations=n_realizations,
                        stochastic_pcs=stochastic_pcs,
                        noise_only=True,
                        add_base=pattern_agg,
                        return_numpy=False,
                    )
                else:
                    # Climatology only: return pattern scaling with shape (1, time)
                    raw_ensemble = pattern_agg[np.newaxis, :]
                cmip6_agg = extract_point(ssp_data, lat, lon)
            else:
                raise ValueError(f"Unknown aggregation type: {agg}")

            # Apply transform if needed
            if transform_config and transform_config.transform_type:
                # Ensure raw_ensemble is 2D for transforms
                # (noise model returns 1D for n_realizations=1, but transform expects 2D)
                if isinstance(raw_ensemble, np.ndarray):
                    ensemble_for_transform = (
                        raw_ensemble
                        if raw_ensemble.ndim == 2
                        else raw_ensemble[np.newaxis, :]
                    )
                else:
                    # xarray DataArray
                    ensemble_for_transform = (
                        raw_ensemble.values
                        if raw_ensemble.ndim == 2
                        else raw_ensemble.values[np.newaxis, :]
                    )

                # Fit Gaussian to generated data
                gaussian_params = transform_config.fit_1d_func(
                    ensemble_for_transform, "gaussian"
                )

                # Fit target distribution to CMIP6 data
                target_params = transform_config.fit_1d_func(
                    cmip6_agg, transform_config.transform_type
                )

                # Apply transform
                transformed_ensemble = transform_config.apply_func(
                    ensemble_for_transform,
                    gaussian_params,
                    target_params,
                    target_dist=transform_config.transform_type,
                )

                results[agg] = transformed_ensemble
            else:
                # Convert to numpy if needed and ensure 2D
                if isinstance(raw_ensemble, np.ndarray):
                    results[agg] = (
                        raw_ensemble
                        if raw_ensemble.ndim == 2
                        else raw_ensemble[np.newaxis, :]
                    )
                else:
                    # xarray DataArray
                    results[agg] = (
                        raw_ensemble.values
                        if raw_ensemble.ndim == 2
                        else raw_ensemble.values[np.newaxis, :]
                    )

        return results

    def _generate_gridded(
        self,
        variable,
        scenario,
        start_year,
        end_year,
        n_realizations,
        gridded_spec,
        include_noise=True,
        verbose=True,
    ):
        """
        Generate gridded (spatial) outputs.

        Parameters
        ----------
        gridded_spec : dict
            Dictionary specifying gridded outputs:
            - 'annual': list of years for annual mean fields
            - 'monthly': list of years for monthly fields (all 12 months)
            - 'climatology': list of [start, end] year pairs for climatological means

        Returns
        -------
        dict
            Dictionary with keys 'annual', 'monthly', 'climatology' containing
            xarray DataArrays with gridded fields
        """
        import xarray as xr

        # Get pattern scaling results (from cache or compute)
        monthly_prediction, monthly_warming, em_data, conc_data = (
            self._get_or_compute_pattern_scaling(
                variable, scenario, start_year, end_year, verbose
            )
        )

        # Get noise model
        noise_model = self.noise_models[variable]

        # Generate stochastic PCs (or skip if climatology only)
        if include_noise:
            if verbose:
                print(f"      → Generating {n_realizations} gridded realizations")
        else:
            if verbose:
                print("      → Generating gridded climatology (no noise)")
            n_realizations = 1  # Force to 1 for climatology

        # Extract requested time slices
        results = {}
        n_months = len(monthly_warming)

        # Helper to convert year to month index
        def year_to_month_idx(year):
            return (year - start_year) * 12

        # Annual means
        if "annual" in gridded_spec:
            if verbose:
                print(
                    f"      → Extracting annual means for {len(gridded_spec['annual'])} years"
                )
            annual_fields = {}
            for year in gridded_spec["annual"]:
                start_idx = year_to_month_idx(year)
                end_idx = start_idx + 12
                if start_idx >= 0 and end_idx <= n_months:
                    # Generate realizations for this year
                    year_realizations = []
                    for i in range(n_realizations):
                        if include_noise:
                            # Generate full field with noise
                            realization = noise_model.generate_realization(
                                monthly_warming[start_idx:end_idx],
                                n_realizations=1,
                                noise_only=True,
                                add_base=monthly_prediction.isel(
                                    month=slice(start_idx, end_idx)
                                ),
                            )
                        else:
                            # Just use pattern scaling
                            realization = monthly_prediction.isel(
                                month=slice(start_idx, end_idx)
                            )

                        # Average over 12 months
                        annual_mean = realization.mean(dim="month")
                        year_realizations.append(annual_mean)

                    # Stack realizations
                    if len(year_realizations) > 1:
                        annual_fields[year] = xr.concat(
                            year_realizations, dim="realization"
                        )
                    else:
                        annual_fields[year] = year_realizations[0].expand_dims(
                            realization=[0]
                        )
                else:
                    if verbose:
                        print(
                            f"        ⚠️  Year {year} outside range {start_year}-{end_year}"
                        )
            results["annual"] = annual_fields

        # Monthly fields
        if "monthly" in gridded_spec:
            if verbose:
                print(
                    f"      → Extracting monthly fields for {len(gridded_spec['monthly'])} years"
                )
            monthly_fields = {}
            for year in gridded_spec["monthly"]:
                start_idx = year_to_month_idx(year)
                end_idx = start_idx + 12
                if start_idx >= 0 and end_idx <= n_months:
                    # Generate realizations for this year
                    year_realizations = []
                    for i in range(n_realizations):
                        if include_noise:
                            # Generate full field with noise
                            realization = noise_model.generate_realization(
                                monthly_warming[start_idx:end_idx],
                                n_realizations=1,
                                noise_only=True,
                                add_base=monthly_prediction.isel(
                                    month=slice(start_idx, end_idx)
                                ),
                            )
                        else:
                            # Just use pattern scaling
                            realization = monthly_prediction.isel(
                                month=slice(start_idx, end_idx)
                            )

                        year_realizations.append(realization)

                    # Stack realizations (shape: realizations, month, lat, lon)
                    if len(year_realizations) > 1:
                        year_months = xr.concat(year_realizations, dim="realization")
                    else:
                        year_months = year_realizations[0].expand_dims(realization=[0])

                    monthly_fields[year] = year_months
                else:
                    if verbose:
                        print(
                            f"        ⚠️  Year {year} outside range {start_year}-{end_year}"
                        )
            results["monthly"] = monthly_fields

        # Climatologies (multi-year means)
        if "climatology" in gridded_spec:
            if verbose:
                print(
                    f"      → Computing {len(gridded_spec['climatology'])} climatological means"
                )
            climatology_fields = {}
            for period in gridded_spec["climatology"]:
                if isinstance(period, (list, tuple)) and len(period) == 2:
                    clim_start, clim_end = period
                    start_idx = year_to_month_idx(clim_start)
                    end_idx = year_to_month_idx(clim_end + 1)  # +1 to include end year
                    if start_idx >= 0 and end_idx <= n_months:
                        # Generate realizations for this period
                        clim_realizations = []
                        for i in range(n_realizations):
                            if include_noise:
                                # Generate full field with noise
                                realization = noise_model.generate_realization(
                                    monthly_warming[start_idx:end_idx],
                                    n_realizations=1,
                                    noise_only=True,
                                    add_base=monthly_prediction.isel(
                                        month=slice(start_idx, end_idx)
                                    ),
                                )
                            else:
                                # Just use pattern scaling
                                realization = monthly_prediction.isel(
                                    month=slice(start_idx, end_idx)
                                )

                            # Average over all months in period
                            clim_mean = realization.mean(dim="month")
                            clim_realizations.append(clim_mean)

                        # Stack realizations
                        if len(clim_realizations) > 1:
                            climatology_fields[f"{clim_start}-{clim_end}"] = xr.concat(
                                clim_realizations, dim="realization"
                            )
                        else:
                            climatology_fields[f"{clim_start}-{clim_end}"] = (
                                clim_realizations[0].expand_dims(realization=[0])
                            )
                    else:
                        if verbose:
                            print(
                                f"        ⚠️  Period {clim_start}-{clim_end} outside range"
                            )
                else:
                    if verbose:
                        print(f"        ⚠️  Invalid climatology period: {period}")
            results["climatology"] = climatology_fields

        return results

    def _apply_impacts(
        self, var_output, variable, impact_configs, custom_regions=None, verbose=True
    ):
        """
        Calculate climate impact metrics from generated data.

        Computes derived impact metrics like heating/cooling degree days from
        the generated climate timeseries. Converts anomaly data to absolute
        temperatures using piControl baseline.

        Parameters
        ----------
        var_output : VariableOutput
            Generated output container with timeseries data
        variable : str
            Climate variable ('tas', 'pr', etc.)
        impact_configs : dict
            Impact calculation specifications:
            - 'degree_days': dict with 'hdd_base' and/or 'cdd_base' temperatures
        custom_regions : dict, optional
            Custom region definitions for baseline calculation
        verbose : bool
            Print calculation progress

        Returns
        -------
        None
            Modifies var_output.impacts in place with calculated metrics
        """
        impacts = {}

        # Check if degree days are requested
        if "degree_days" in impact_configs:
            try:
                from meteor import global_mean
                from meteor.impacts import DegreeDaysCalculator

                dd_config = impact_configs["degree_days"]

                # Determine base temperature - use hdd_base if provided, otherwise cdd_base
                # Both HDD and CDD are calculated from the same base temperature
                if "hdd_base" in dd_config:
                    base_temp = dd_config["hdd_base"]
                elif "cdd_base" in dd_config:
                    base_temp = dd_config["cdd_base"]
                else:
                    raise ValueError(
                        "degree_days config must include either 'hdd_base' or 'cdd_base'"
                    )

                # Get piControl baseline for absolute temperature calculation
                # The timeseries data contains anomalies, we need to add baseline
                picontrol_data = self.data_getter.make_meteor_training_data(
                    "piControl", self.model, monthly=True
                )[variable]

                # Create single calculator instance
                dd_model = DegreeDaysCalculator(base_temperature=base_temp)

                # Initialize impact dictionaries for both HDD and CDD
                impacts["hdd"] = {}
                impacts["cdd"] = {}

                # Apply to all timeseries outputs
                for key, ts_data in var_output.timeseries.items():
                    # Calculate appropriate baseline for this aggregation
                    if key == "global":
                        baseline_k = float(global_mean(picontrol_data).mean())
                    elif key.startswith("regional:custom:"):
                        # Custom region - need to calculate baseline from bbox
                        from meteor import create_region_mask

                        region_name = key.split(":")[2]
                        if custom_regions and region_name in custom_regions:
                            bbox = custom_regions[region_name]
                            mask = create_region_mask(picontrol_data, bbox=bbox)
                            masked_data = picontrol_data.where(mask)
                            baseline_k = float(masked_data.mean())
                        else:
                            raise ValueError(
                                f"Custom region '{region_name}' not found in custom_regions"
                            )
                    elif key.startswith("regional:"):
                        from meteor import regional_mean

                        region = key.split(":")[1]
                        baseline_k = float(regional_mean(picontrol_data, region).mean())
                    elif key.startswith("point:"):
                        from meteor import extract_point

                        coords = key.split(":")[1]
                        lat, lon = map(float, coords.split(","))
                        baseline_k = float(
                            extract_point(picontrol_data, lat, lon).mean()
                        )
                    else:
                        raise ValueError(f"Unknown aggregation type: {key}")

                    # Convert from anomaly (K) to absolute temperature (°C)
                    # ts_data is anomaly in K, baseline_k is absolute temperature in K
                    n_realizations, n_months = ts_data.shape

                    # Create xarray with month dimension (required by calculator)
                    # Absolute temperature in Celsius = (anomaly_K + baseline_K) - 273.15
                    temp_celsius = xr.DataArray(
                        ts_data + baseline_k - 273.15,
                        dims=["realization", "month"],
                        coords={"month": np.arange(n_months)},
                    )

                    # Calculate degree days for each realization
                    # Both HDD and CDD are calculated in the same call
                    hdd_results = []
                    cdd_results = []
                    for i in range(n_realizations):
                        result = dd_model.calculate(temp_celsius[i])
                        hdd_results.append(result.data["annual_hdd"].values)
                        cdd_results.append(result.data["annual_cdd"].values)

                    # Stack back into arrays (n_realizations, n_years)
                    impacts["hdd"][key] = np.array(hdd_results)
                    impacts["cdd"][key] = np.array(cdd_results)

                    if verbose:
                        print(f"         • HDD for {key}")
                        print(f"         • CDD for {key}")

            except ImportError as e:
                if verbose:
                    print(
                        f"      ⚠️  meteor.impacts.DegreeDaysCalculator not available: {e}"
                    )
            except Exception as e:
                if verbose:
                    print(f"      ⚠️  Error calculating degree days: {e}")

        return impacts

    def __repr__(self):
        """Return string representation of MeteorInterface."""
        status = "trained" if all(self._is_trained.values()) else "not trained"
        return (
            f"MeteorInterface(model='{self.model}', "
            f"variables={self.variables}, status='{status}')"
        )
