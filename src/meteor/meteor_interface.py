"""
High-Level METEOR Interface

This module provides a simplified, user-friendly interface to METEOR's
pattern scaling and noise generation capabilities.
"""

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from .cache_handling import CacheHandler
from .cmip6_meteor_data_getter import Cmip6MeteorDataGetter
from .ensemble_output import EnsembleOutput, VariableOutput
from .geo_data_utils import (
    create_region_mask,
    extend_temeperature_anomaly_timeseries_for_scaling,
    extract_point,
    get_time_name,
    global_mean,
    regional_mean,
)
from .impacts import DegreeDaysCalculator
from .meteor import MeteorPatternScaling
from .noise_generator import train_noise_model_from_cmip6, validate_noise_model_cache
from .scm_input_lib import (
    load_emissions_concentrations,
    load_emissions_concentrations_from_name,
    load_ssp_config,
    parse_scenario_input,
)
from .variable_transforms import get_variable_transform_config


def _get_default_config(variable):
    """Get smart default configuration for a variable."""
    config = {
        "n_modes_pattern": 3,
        "n_modes_noise": 40,
        "lag_order": 2,
        "use_picontrol_baseline": True,
        "training_scenario": "ssp245",
    }

    # Variable-specific defaults
    if variable == "tas":
        # 'none' (pure VAR): using t_glob as a VAR-X exogenous regressor absorbs
        # the persistent low-frequency global variability into the deterministic
        # forced term, so it is lost at generation (the prescribed trajectory has
        # no internal variability) -- the noise becomes white and annual/decadal
        # global variance collapses ~2.5x. The temperature-dependent mean/seasonal
        # response is already captured by the seasonal model, so the exog is
        # redundant here. See MeteorNoiseGenerator.use_exog.
        config["use_exog"] = "none"
        config["transform"] = False
    elif variable == "pr":
        config["use_exog"] = "none"
        config["transform"] = True
        config["transform_type"] = "gamma"
    else:
        # Generic defaults for other variables
        config["use_exog"] = "none"
        config["transform"] = False

    return config


@dataclass
class PatternScalingResult:
    """
    Result of a pattern scaling computation.

    Holds both the time-sliced pattern scaling output used directly by the
    generators and the full-trajectory warming plus slice bookkeeping needed to
    generate spun-up stochastic PCs over the full trajectory and slice them to
    the requested output window.

    Attributes
    ----------
    monthly_prediction : xr.DataArray
        Monthly pattern prediction sliced to the requested output window.
    monthly_warming : np.ndarray
        Global-mean monthly warming sliced to the requested output window.
    em_data : Any
        Emissions data used to drive the pattern model.
    conc_data : Any
        Concentration data used to drive the pattern model.
    full_monthly_warming : np.ndarray
        Global-mean monthly warming over the full (un-sliced) trajectory. Used to
        generate stochastic PCs so the autoregressive spin-up transient is parked
        at the trajectory start rather than inside the output window.
    base_year : int
        First year of the full monthly trajectory (origin for all month indexing).
    start_month_idx : int
        Month index (relative to ``base_year``) of the first output month.
    end_month_idx : int
        Month index (relative to ``base_year``) one past the last output month.
    """

    monthly_prediction: xr.DataArray
    monthly_warming: np.ndarray
    em_data: Any
    conc_data: Any
    full_monthly_warming: np.ndarray
    base_year: int
    start_month_idx: int
    end_month_idx: int


@dataclass
class GenerationInputs:
    """
    Shared inputs prepared once per variable and consumed by both generators.

    Produced by :meth:`MeteorInterface._prepare_generation` and passed to both
    ``_generate_timeseries`` and ``_generate_gridded`` so the two paths share the
    same pattern scaling and the same spun-up stochastic PC realisations.

    Attributes
    ----------
    pattern : PatternScalingResult
        Pattern scaling result (sliced prediction/warming + slice bookkeeping).
    stochastic_pcs : np.ndarray or None
        Stochastic PCs generated once over the full trajectory and sliced to the
        output window. Shape ``(n_realizations, n_months, n_modes)``. ``None`` when
        ``include_noise`` is False.
    """

    pattern: PatternScalingResult
    stochastic_pcs: Any


def _stack_realizations(realizations):
    """
    Stack a list of per-realization DataArrays along a ``realization`` dimension.

    Parameters
    ----------
    realizations : list of xr.DataArray
        One DataArray per ensemble member.

    Returns
    -------
    xr.DataArray
        Concatenated array with a leading ``realization`` dimension.
    """
    if len(realizations) > 1:
        return xr.concat(realizations, dim="realization")
    return realizations[0].expand_dims(realization=[0])


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
    >>> emulator.train()
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
    >>> emulator.train()
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
        self.variables = [variables] if isinstance(variables, str) else list(variables)
        self.model = model
        self.cache_handler = CacheHandler(
            purpose="general",
            cache_dir=cache_dir,
        )

        # Initialize data getter
        data_getter_kwargs = data_getter_kwargs or {}
        default_exps = ["piControl", "historical", "ssp245", "abrupt-4xCO2"]
        default_dbe = ["CMIP", "CMIP", "ScenarioMIP", "CMIP"]

        # Always request 'tas' (needed for noise model exog)
        flds_to_request = self.variables.copy()
        needs_tas = "tas" not in flds_to_request
        if needs_tas:
            flds_to_request.append("tas")

        # Match tabids length to flds if provided
        tabids_for_flds_to_request = data_getter_kwargs.get("tabids")
        if tabids_for_flds_to_request is not None:
            tabids_for_flds_to_request = (
                [tabids_for_flds_to_request]
                if isinstance(tabids_for_flds_to_request, str)
                else list(tabids_for_flds_to_request)
            )
            if needs_tas:
                tabids_for_flds_to_request.append("Amon")

        # Note: We explicitly enable caching for the high-level interface to provide
        # good performance by default. Users of the low-level Cmip6MeteorDataGetter
        # can control caching behavior directly.
        self.data_getter = Cmip6MeteorDataGetter(
            models=[self.model],
            exps=data_getter_kwargs.get("exps", default_exps),
            tabids=tabids_for_flds_to_request,
            flds=flds_to_request,
            dbe=data_getter_kwargs.get("dbe", default_dbe),
            enable_cache=True,
            cache_handler=self.cache_handler,
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

    def train(self, training_scenario="ssp245", variable_configs=None, verbose=True):
        """
        Train pattern scaling and noise models for all variables.

        Parameters
        ----------
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
        >>> emulator.train()
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
        if verbose:  # pragma: no cover
            print("=" * 60)
            print(f"Training METEOR emulator for {self.model}")
            print(f"Variables: {', '.join(self.variables)}")
            print("=" * 60)

        if "tas" not in self.variables:
            if verbose:  # pragma: no cover
                print("\n🔧 Training tas pattern scaling only")
            config = _get_default_config("tas")
            self._train_pattern_scaling("tas", config, verbose=verbose)

        for variable in self.variables:
            if verbose:  # pragma: no cover
                print(f"\n🔧 Training {variable.upper()}...")

            # Get configuration with hybrid precedence

            config = _get_default_config(variable)
            # Override with method parameter if different from default
            if training_scenario != "ssp245":
                config["training_scenario"] = training_scenario
            # Override with variable-specific config if provided
            if variable_configs and variable in variable_configs:
                config.update(variable_configs[variable])
            self._training_config[variable] = config

            # Train pattern scaling
            if verbose:  # pragma: no cover
                print("   → Training pattern scaling model...")
            self._train_pattern_scaling(variable, config, verbose=verbose)

            # Train noise model
            if verbose:  # pragma: no cover
                print("   → Training noise model...")
            self._train_noise_model(variable, config, verbose=verbose)

            # Fit transforms if needed
            transform_config = get_variable_transform_config(variable)
            if transform_config.transform_type and config.get("transform", True):
                if verbose:  # pragma: no cover
                    print(
                        f"   → Fitting {transform_config.transform_type} transform..."
                    )
                    print(f"      Reason: {transform_config.reason}")
                self._fit_transform(variable, transform_config)

            self._is_trained[variable] = True

            if verbose:  # pragma: no cover
                print(f"   ✅ {variable.upper()} training complete")

        if verbose:  # pragma: no cover
            print("\n" + "=" * 60)
            print("✅ All variables trained successfully")
            print("=" * 60)

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
        cache_file = self.cache_handler.get_pattern_scaling_cache_path(
            self.model, variable=variable
        )

        # Check cache
        is_valid, cached_model, info = (  # pylint: disable=unused-variable
            self.data_getter.validate_pattern_scaling_cache(cache_file, self.model)
        )

        if is_valid:
            if verbose:  # pragma: no cover
                print("      ✓ Using cached pattern scaling model")
            training_data = None
            ssp_config = None
        else:
            if verbose:  # pragma: no cover
                print(f"      ⚠️  Cache miss: {info.get('message', 'No cache found')}")

            # Prepare training data
            training_data = self.data_getter.prepare_pattern_scaling_training_data(
                self.model, "ssp245"
            )
            ssp_config = load_ssp_config("ssp245")

        # Create pattern scaling model
        self.pattern_models[variable] = MeteorPatternScaling(
            f"cmip6-{self.model}-aer-{variable}",
            {variable: config["n_modes_pattern"]},
            None if training_data is None else lambda key: training_data[key],
            ssp_input=ssp_config,
            from_file=False,
            exp_list=None if training_data is None else ["base", "co2x4", "sulxanom"],
            anom_timescales={variable: config["n_modes_pattern"]},
            cache_dir=os.path.join(self.cache_handler.cache_dir, "pattern_scaling"),
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
        cache_file = self.cache_handler.get_noise_model_cache_path(self.model, variable)

        # Check cache
        is_valid, cached_model, info = (  # pylint: disable=unused-variable
            validate_noise_model_cache(
                cache_file,
                variable,
                n_modes=config["n_modes_noise"],
                lag_order=config["lag_order"],
            )
        )

        if is_valid:
            if verbose:  # pragma: no cover
                print("      ✓ Using cached noise model")
            self.noise_models[variable] = cached_model
        else:
            if verbose:  # pragma: no cover
                print("      ⚠️  Training new noise model...")

            # ✅ Get training scenario from config
            training_scenario = config.get("training_scenario", "ssp245")

            # ✅ Generate pattern scaling prediction for custom_global_temp
            # This ensures noise model training uses same temperature trajectory as generation

            em_data, conc_data = load_emissions_concentrations_from_name(
                training_scenario
            )

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

            if verbose:  # pragma: no cover
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
                cache_dir=os.path.join(self.cache_handler.cache_dir, "noise_models"),
                verbose=verbose,
            )

    def _fit_transform(self, variable, transform_config):
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
        scenario: str | dict,
        start_year: int,
        end_year: int,
        n_realizations: int = 1,
        timeseries: str | list[str] | None = None,
        gridded: dict[str, int | list[int]] | None = None,
        impacts=None,
        include_noise: bool = True,
        save_to: str | None = None,
        custom_regions: dict | None = None,
        temp_scaling_ts: xr.DataArray | None = None,
        verbose: bool = True,
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
        n_realizations : int, optional
            Number of ensemble members to generate. Ignored if include_noise=False.
            Default is 1.
        timeseries : str or list of str, optional
            Spatial aggregations for time series output. Options:
            - 'global' : Global mean
            - 'regional:CODE' : AR6 region (e.g., 'regional:EAS')
            - 'regional:custom:NAME' : Custom region (requires custom_regions)
            - 'point:LAT,LON' : Specific location (e.g., 'point:19.0,72.8')
        gridded : dict[str, int | list[int]], optional
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
        custom_regions : dict, optional
            Custom region definitions for 'regional:custom:NAME' aggregations.
            Format: {'name': {'lat': (min, max), 'lon': (min, max)}}
        temp_scaling_ts: xr.DataArray, optional
            Should be one-dimensional xr.DataArray with dimension year, giving a
            timeseries of global mean temperatures to scale to. The timeseries length
            needs to match the scenario length of the scenario that is being generated
            at generation (default is 1750-2100) and needs to include the base_year (default 1750).
            This is used to scale the annual pattern from teh MeteorPatternScaling prediction.
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
            if verbose and n_realizations > 1:  # pragma: no cover
                print(
                    "Note: include_noise=False, forcing n_realizations=1 (climatology only)"
                )
            n_realizations = 1

        # Parse scenario to get name for display
        scenario_info = parse_scenario_input(scenario)
        scenario_name = scenario_info["name"]

        # Check all variables are trained
        for var in self.variables:
            if not self._is_trained[var]:
                raise RuntimeError(f"Variable '{var}' not trained. Call train() first.")

        if verbose:  # pragma: no cover
            print("=" * 60)
            print(f"Generating ensemble for {scenario_name}")
            print(f"  Years: {start_year}-{end_year}")
            print(f"  Realizations: {n_realizations}")
            print("=" * 60)

        results = {}

        for variable in self.variables:
            if verbose:  # pragma: no cover
                print(f"\n📊 Generating {variable.upper()}...")

            var_output = VariableOutput(variable)

            # Prepare shared generation inputs ONCE when both output types are
            # requested, so the time series and gridded paths are driven by the
            # same pattern scaling and the same spun-up stochastic PCs (mutually
            # consistent noise). When only one type is requested there is nothing
            # to be consistent with, so that generator builds its own inputs.
            gen_inputs = None
            if timeseries and gridded:
                gen_inputs = self._prepare_generation(
                    variable,
                    scenario,
                    start_year,
                    end_year,
                    n_realizations,
                    include_noise=include_noise,
                    temp_scaling_ts=temp_scaling_ts,
                    verbose=verbose,
                )

            # Generate timeseries if requested
            if timeseries:
                if verbose:  # pragma: no cover
                    print(f"   → Time series: {len(timeseries)} aggregations")
                var_output.timeseries = self._generate_timeseries(
                    variable,
                    scenario,
                    start_year,
                    end_year,
                    n_realizations,
                    timeseries,
                    gen_inputs=gen_inputs,
                    custom_regions=custom_regions,
                    include_noise=include_noise,
                    temp_scaling_ts=temp_scaling_ts,
                    verbose=verbose,
                )

            # Generate gridded if requested
            if gridded:
                if verbose:  # pragma: no cover
                    print("   → Gridded outputs...")
                var_output.gridded = self._generate_gridded(
                    variable,
                    scenario,
                    start_year,
                    end_year,
                    n_realizations,
                    gridded,
                    gen_inputs=gen_inputs,
                    include_noise=include_noise,
                    temp_scaling_ts=temp_scaling_ts,
                    verbose=verbose,
                )

            # Apply impacts if requested
            if impacts and variable in impacts:
                if verbose:  # pragma: no cover
                    print("   → Computing impact metrics...")
                var_output.impacts = self._apply_impacts(
                    var_output,
                    variable,
                    impacts[variable],
                    custom_regions=custom_regions,
                    verbose=verbose,
                )

            results[variable] = var_output

            if verbose:  # pragma: no cover
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

        if verbose:  # pragma: no cover
            print("\n" + "=" * 60)
            print("✅ Generation complete")
            print("=" * 60)

        return ensemble

    def _get_or_compute_pattern_scaling(
        self,
        variable,
        scenario,
        start_year,
        end_year,
        verbose=True,
        temp_scaling_ts=None,
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
        scenario : str
            Scenario name ('ssp245', 'ssp585', etc.)
        start_year : int
            Start year for slicing
        end_year : int
            End year for slicing (inclusive)
        verbose : bool
            Print cache status messages

        Returns
        -------
        PatternScalingResult
            Sliced monthly prediction/warming plus the full-trajectory warming and
            slice bookkeeping (``base_year``, ``start_month_idx``, ``end_month_idx``)
            needed to generate spun-up stochastic PCs aligned to the output window.
        """
        # Parse scenario input
        scenario_info = parse_scenario_input(scenario)
        scenario_name = scenario_info["name"]

        if verbose:  # pragma: no cover
            if scenario_info["type"] == "ssp":
                print(
                    f"      → Computing pattern scaling for {variable}, {scenario_name}..."
                )
            else:
                print(  # pragma: no cover
                    f"      → Computing pattern scaling for {variable}, custom scenario '{scenario_name}'..."
                )

        # Load forcing data
        if scenario_info["type"] == "ssp":
            # Standard SSP scenario
            em_data, conc_data = load_emissions_concentrations_from_name(scenario_name)
        else:
            # Custom scenario
            em_data, conc_data = load_emissions_concentrations(
                scenario_info["emissions"],
                scenario_info["concentrations"],
                verbose=verbose,
            )

            # Check if emissions data covers requested time range
            em_start_year = em_data.index[0]
            em_end_year = em_data.index[-1]

            if end_year > em_end_year:
                if verbose:  # pragma: no cover
                    print(
                        f"      ⚠️  Warning: Emissions data ends at {em_end_year}, requested {end_year}"
                    )
                    print(f"      → Clipping output to {em_start_year}-{em_end_year}")
                # Update end_year to match data availability
                end_year = em_end_year

            if start_year < em_start_year:
                if verbose:  # pragma: no cover
                    print(
                        f"      ⚠️  Warning: Emissions data starts at {em_start_year}, requested {start_year}"
                    )
                    print(f"      → Clipping output to {em_start_year}-{em_end_year}")
                start_year = em_start_year

        # Handle custom emissions data range
        # The pattern model SCM runs to 2100 by default, so we need to extend data if needed
        if scenario_info["type"] == "custom":
            # data_start = max(em_data.index[0], conc_data.index[0])
            data_end = min(em_data.index[-1], conc_data.index[-1])

            # Extend data to 2100 if needed (hold last value constant)
            # This allows the SCM to run to its default nyend=2100
            if data_end < 2100:
                if verbose:  # pragma: no cover
                    print(
                        f"      → Emissions data ends at {data_end}, extending to 2100 (holding final values)"
                    )

                # Extend emissions - create new rows by repeating last year's values
                years_to_add = list(range(data_end + 1, 2101))
                for year in years_to_add:
                    em_data.loc[year] = em_data.loc[data_end]
                    conc_data.loc[year] = conc_data.loc[data_end]

                # Sort index to maintain chronological order
                em_data = em_data.sort_index()
                conc_data = conc_data.sort_index()

            # Store the actual data end year for later clipping
            actual_data_end = data_end
        else:
            actual_data_end = None

        # Generate FULL pattern scaling prediction (annual)
        pattern_model = self.pattern_models[variable]
        climate_prediction = pattern_model.predict_from_combined_experiment(
            em_data, conc_data, [variable]
        )
        annual_prediction = climate_prediction[variable]

        # Determine base year
        if hasattr(annual_prediction, "year"):
            base_year = int(annual_prediction.year[0])
        else:
            base_year = 1750  # Default assumption

        if temp_scaling_ts is not None:
            print(
                f" Ts scaleing before calculating annual pred {temp_scaling_ts['year']}"
            )
            annual_prediction = self._compute_timeseries_scaling(
                variable,
                annual_prediction,
                base_year,
                em_data,
                conc_data,
                temp_scaling_ts,
            )

        # Convert to monthly
        full_monthly_prediction = pattern_model.to_monthly(
            annual_prediction, start_year=0
        )

        # Compute global mean for noise model
        full_monthly_warming = global_mean(full_monthly_prediction).values

        # Slice to requested time range (or actual data range for custom scenarios)
        if scenario_info["type"] == "custom" and actual_data_end is not None:
            # Clip to actual data availability
            effective_end_year = min(end_year, actual_data_end)
            if effective_end_year < end_year and verbose:  # pragma: no cover
                print(
                    f"      → Clipping output to {start_year}-{effective_end_year} (data availability)"
                )
        else:
            effective_end_year = end_year

        start_month_idx = (start_year - base_year) * 12
        end_month_idx = (
            effective_end_year - base_year + 1
        ) * 12  # +1 to include end_year

        monthly_prediction_sliced = full_monthly_prediction.isel(
            month=slice(start_month_idx, end_month_idx)
        )
        monthly_warming_sliced = full_monthly_warming[start_month_idx:end_month_idx]

        return PatternScalingResult(
            monthly_prediction=monthly_prediction_sliced,
            monthly_warming=monthly_warming_sliced,
            em_data=em_data,
            conc_data=conc_data,
            full_monthly_warming=full_monthly_warming,
            base_year=base_year,
            start_month_idx=start_month_idx,
            end_month_idx=end_month_idx,
        )

    # TODO - possibly add verbosity?
    def _compute_timeseries_scaling(
        self,
        variable,
        annual_prediction,
        base_year,
        em_data,
        conc_data,
        temp_scaling_ts,
        verbose=False,
    ):
        """
        Compute scaling factor for time series outputs based on pattern scaling.

        This is used to adjust the noise variability to match the forced response
        of the pattern scaling prediction for the specific scenario and time range.

        Parameters
        ----------
        variable : str
            Climate variable ('tas', 'pr')
        annual_prediction : xr.DataArray
            Annual prediction from the MeteorPatternScaling to be scaled
        base_year : int
            Base year for scaling, to make sure only anomalies are scaled
        em_data : pd.DataFrame
            Emissions input data to drive MetorPatternScaling, to be used
            to generate temperature predictions for the scaling if the variable
            is not tas
        conc_data : pd.DataFrame
            Concentrations input data to drive MetorPatternScaling, to be used
            to generate temperature predictions for the scaling if the variable
            is not tas
        temp_scaling_ts : xr.DataArray
            Time series of global mean temperature from pattern scaling prediction
            Should be one-dimensional xr.DataArray with dimension year, giving a
            timeseries of global mean temperatures to scale to. The timeseries length
            needs to match the scenario length of the scenario that is being generated
            at generation (default is 1750-2100) and needs to include the base_year (default 1750).
            This is used to scale the annual pattern from teh MeteorPatternScaling prediction.
        verbose : bool
            Print status messages
        Returns
        -------
        np.ndarray
            Scaling factor to apply to noise variability
        """
        if verbose:  # pragma: no cover
            print("      → Computing time series scaling factor...")
        if not isinstance(temp_scaling_ts, xr.DataArray):
            raise ValueError("temp_scaling_ts must be an xarray DataArray")
        if not hasattr(temp_scaling_ts, "year"):
            raise ValueError("temp_scaling_ts must have a 'year' coordinate")
        if "year" not in temp_scaling_ts.coords:
            raise ValueError("temp_scaling_ts must have a 'year' coordinate")

        # TODO do some cutting to correct values to match the time range of the temp_scaling_ts if needed
        if hasattr(annual_prediction, "year"):
            annual_prediction_base = annual_prediction.sel(year=base_year)
        else:
            annual_prediction_base = annual_prediction[
                0
            ]  # Assuming first value corresponds to base_year
        annual_prediction_anomaly = annual_prediction - annual_prediction_base
        if base_year not in temp_scaling_ts.year:
            temperature_input_base = global_mean(annual_prediction_base)
        else:
            temperature_input_base = temp_scaling_ts.sel(year=base_year)
        if verbose and temperature_input_base != 0:  # pragma: no cover
            print(
                f"The baseline scaling temperature is non-zero ({temperature_input_base})"
            )
            print("This value will be subtracted from the timeseries when scaling")
        if variable == "tas":
            annual_temp_prediction_gm_anomaly = global_mean(annual_prediction_anomaly)
        else:
            annual_temp_prediction = self.pattern_models[
                "tas"
            ].predict_from_combined_experiment(em_data, conc_data, ["tas"])["tas"]
            if hasattr(annual_temp_prediction, "year"):
                annual_temp_prediction_base = annual_temp_prediction.sel(year=base_year)
            else:
                annual_temp_prediction_base = annual_temp_prediction[
                    0
                ]  # Assuming first value corresponds to start_year
            annual_temp_prediction_gm_anomaly = global_mean(
                annual_temp_prediction - annual_temp_prediction_base
            )
        temperature_input_anomaly = temp_scaling_ts - temperature_input_base.values
        if len(temperature_input_anomaly) != len(annual_temp_prediction_gm_anomaly):
            temperature_input_anomaly = (
                extend_temeperature_anomaly_timeseries_for_scaling(
                    annual_temp_prediction_gm_anomaly,
                    temperature_input_anomaly,
                )
            )
        temp_scaling = np.where(
            annual_temp_prediction_gm_anomaly.values != 0,
            temperature_input_anomaly.values / annual_temp_prediction_gm_anomaly.values,
            1.0,
        )
        temp_scaling = xr.DataArray(
            temp_scaling,
            dims=annual_prediction_anomaly[
                get_time_name(annual_prediction_anomaly)
            ].dims,
        )
        return annual_prediction_base + annual_prediction_anomaly * temp_scaling

    def _prepare_generation(
        self,
        variable,
        scenario,
        start_year,
        end_year,
        n_realizations,
        include_noise=True,
        temp_scaling_ts=None,
        verbose=True,
    ):
        """
        Prepare the shared inputs consumed by both output generators.

        Computes pattern scaling once and generates the stochastic PCs once over
        the FULL trajectory (so the autoregressive spin-up transient is parked at
        the trajectory start, not inside the output window), then slices the PCs to
        the requested output window. The resulting :class:`GenerationInputs` is
        passed to both ``_generate_timeseries`` and ``_generate_gridded`` so the two
        paths are driven by identical pattern scaling and identical noise draws.

        Parameters
        ----------
        variable : str
            Climate variable ('tas', 'pr').
        scenario : str or dict
            Scenario specification (see :meth:`generate_ensemble_outputs`).
        start_year, end_year : int
            Output window (inclusive).
        n_realizations : int
            Number of ensemble members.
        include_noise : bool
            If False, no PCs are generated (climatology only).
        temp_scaling_ts : xr.DataArray, optional
            Optional global-mean temperature trajectory to scale the pattern to.
        verbose : bool
            Print progress messages.

        Returns
        -------
        GenerationInputs
            Shared pattern scaling result and (window-sliced) stochastic PCs.
        """
        pattern = self._get_or_compute_pattern_scaling(
            variable,
            scenario,
            start_year,
            end_year,
            temp_scaling_ts=temp_scaling_ts,
            verbose=verbose,
        )

        stochastic_pcs = None
        if include_noise:
            noise_model = self.noise_models[variable]
            if verbose:  # pragma: no cover
                print(
                    f"      → Generating {n_realizations} stochastic PC realizations "
                    "(full trajectory, spun-up)..."
                )
            # Generate over the FULL trajectory so spin-up is resolved before the
            # output window, then slice to the window on a January boundary
            # (start_month_idx is always a multiple of 12).
            full_pcs = noise_model.generate_stochastic_pcs(
                pattern.full_monthly_warming,
                n_realizations=n_realizations,
                random_seed=None,
            )
            if full_pcs.ndim == 2:
                # Single realization -> add leading realization axis
                full_pcs = full_pcs[np.newaxis, ...]
            stochastic_pcs = full_pcs[
                :, pattern.start_month_idx : pattern.end_month_idx, :
            ]

        return GenerationInputs(pattern=pattern, stochastic_pcs=stochastic_pcs)

    def _get_transform_config(self, variable):
        """
        Return the resolved transform config for a variable, or None.

        Handles both the fitted case (stored as a dict with a ``'config'`` entry)
        and the not-yet-fitted case (stored as a ``VariableTransformConfig``).
        """
        transform_info = self.transforms.get(variable, None)
        if isinstance(transform_info, dict):
            return transform_info.get("config")
        return transform_info

    def _load_transform_reference(self, variable, start_year, end_year, verbose=True):
        """
        Load and prepare CMIP6 reference data for distribution-transform fitting.

        Only needed for variables that have a distribution transform (e.g. ``pr``).
        Returns the gridded CMIP6 reference field sliced to the output window plus,
        for precipitation, the gridded first-year baseline field. The timeseries
        path aggregates these per requested region; the gridded path uses them
        directly with the per-gridpoint (3D) transform.

        Parameters
        ----------
        variable : str
            Climate variable.
        start_year, end_year : int
            Output window (inclusive).
        verbose : bool
            Print progress messages.

        Returns
        -------
        tuple
            ``(ssp_data, pr_first_year_mean)`` where ``ssp_data`` is the gridded
            reference field (anomalies for ``tas``, absolute for ``pr``) and
            ``pr_first_year_mean`` is the gridded first-year baseline field for
            ``pr`` (``None`` otherwise).
        """
        transform_training_scenario = self._training_config.get(variable, {}).get(
            "training_scenario", "ssp245"
        )

        if verbose:  # pragma: no cover
            print(
                f"      → Loading CMIP6 training data for {transform_training_scenario}..."
            )
        ssp_data = self.data_getter.make_meteor_training_data_composite(
            ["historical", transform_training_scenario], self.model, monthly=True
        )[variable]

        if verbose:  # pragma: no cover
            print(f"      → Loading piControl baseline for {variable}...")
        picontrol_data = self.data_getter.make_meteor_training_data_composite(
            ["piControl"], self.model, monthly=True
        )[variable]
        picontrol_mean = picontrol_data.mean(dim="month")

        pr_first_year_mean = None

        # For precipitation: use first-year baseline instead of piControl. CMIP6
        # scenarios already include ~1°C of historical warming effects on
        # precipitation, so piControl would create a ~2-4% bias. Use the first 12
        # months of the prediction period (start_year) as the baseline. ssp_data is
        # a composite starting from historical (~1850), so find the index for
        # start_year.
        if variable == "pr":
            n_months = len(ssp_data.month)

            if hasattr(ssp_data, "start_year"):
                composite_start_year = int(ssp_data.start_year)
            else:
                # Default assumption: historical+scenario composite starts at 1850
                expected_months_from_1850 = (end_year - 1850 + 1) * 12
                if n_months < expected_months_from_1850:
                    composite_start_year = end_year - (n_months // 12) + 1
                else:
                    composite_start_year = 1850

            start_year_idx = (start_year - composite_start_year) * 12
            end_year_idx = (end_year - composite_start_year + 1) * 12  # inclusive

            composite_end_year = composite_start_year + n_months // 12 - 1

            if start_year_idx < 0:
                raise ValueError(
                    f"start_year {start_year} is before the composite data start year {composite_start_year}. "
                    f"Valid range: {composite_start_year}-{composite_end_year}"
                )
            if end_year_idx > n_months:
                raise ValueError(
                    f"end_year {end_year} is beyond the composite data end year {composite_end_year}. "
                    f"Valid range: {composite_start_year}-{composite_end_year}"
                )
            if start_year_idx + 12 > n_months:
                raise ValueError(
                    f"start_year {start_year} does not have 12 months of data in the composite. "
                    f"Valid range: {composite_start_year}-{composite_end_year}"
                )

            # Use first year of prediction period as the baseline
            pr_first_year_mean = ssp_data.isel(
                month=slice(start_year_idx, start_year_idx + 12)
            ).mean(dim="month")

            # CRITICAL: Slice ssp_data to only the prediction period for Gamma
            # transform fitting. Using the full historical+scenario composite would
            # result in a lower mean distribution, causing negative bias.
            ssp_data = ssp_data.isel(month=slice(start_year_idx, end_year_idx))

            if verbose:  # pragma: no cover
                print(
                    f"      → Using {start_year} baseline for PR instead of piControl"
                )

        # For temperature: convert CMIP6 to anomalies (pattern scaling outputs
        # anomalies). For precipitation keep absolute values for Gamma fitting.
        if variable == "tas":
            if verbose:  # pragma: no cover
                print(
                    f"      → Converting {variable} to anomalies from piControl baseline..."
                )
            ssp_data = ssp_data - picontrol_mean

        return ssp_data, pr_first_year_mean

    def _generate_timeseries(
        self,
        variable,
        scenario,
        start_year,
        end_year,
        n_realizations,
        aggregations,
        gen_inputs=None,
        custom_regions=None,
        include_noise=True,
        temp_scaling_ts=None,
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
        # Build shared generation inputs if not provided by the caller.
        if gen_inputs is None:
            gen_inputs = self._prepare_generation(
                variable,
                scenario,
                start_year,
                end_year,
                n_realizations,
                include_noise=include_noise,
                temp_scaling_ts=temp_scaling_ts,
                verbose=verbose,
            )

        monthly_prediction = gen_inputs.pattern.monthly_prediction
        monthly_warming = gen_inputs.pattern.monthly_warming
        stochastic_pcs = gen_inputs.stochastic_pcs
        noise_model = self.noise_models[variable]

        # Resolve transform and (only if needed) load CMIP6 reference data.
        # tas has no transform, so its reference data is never loaded.
        transform_config = self._get_transform_config(variable)
        ssp_data = None
        pr_first_year_mean = None
        if transform_config and transform_config.transform_type:
            ssp_data, pr_first_year_mean = self._load_transform_reference(
                variable, start_year, end_year, verbose=verbose
            )

        if verbose:  # pragma: no cover
            if include_noise:
                print("      → Using shared stochastic PC realizations")
            else:
                print("      → Climatology only (no stochastic variability)")

        # Generate outputs for each aggregation
        results = {}

        for agg in aggregations:
            if verbose:  # pragma: no cover
                print(f"      • {agg}")

            # Parse aggregation type
            # For precipitation, compute first-year baseline for this aggregation
            # Pattern scaling outputs anomalies, but Gamma transform needs absolute values
            # We use first-year (2015) baseline instead of piControl to match CMIP6 starting point
            pr_baseline_agg = None
            # CMIP6 reference aggregation, only needed/available when a transform exists
            cmip6_agg = None

            if agg == "global":
                # Global mean
                pattern_agg = global_mean(monthly_prediction).values
                if variable == "pr":
                    pr_baseline_agg = float(
                        global_mean(pr_first_year_mean).values.item()
                    )
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
                if ssp_data is not None:
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
                    if ssp_data is not None:
                        cmip6_agg = regional_mean(ssp_data, region_mask=region_mask)
                    if variable == "pr":
                        pr_baseline_agg = float(
                            regional_mean(
                                pr_first_year_mean, region_mask=region_mask
                            ).values.item()
                        )

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
                    if variable == "pr":
                        pr_baseline_agg = float(
                            regional_mean(
                                pr_first_year_mean, region_code=region_code
                            ).values.item()
                        )
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
                    if ssp_data is not None:
                        cmip6_agg = regional_mean(ssp_data, region_code=region_code)

            elif agg.startswith("point:"):
                # Point extraction
                coords = agg.split(":")[1]
                lat_str, lon_str = coords.split(",")
                lat = float(lat_str)
                lon = float(lon_str)

                pattern_agg = extract_point(monthly_prediction, lat, lon).values
                if variable == "pr":
                    pr_baseline_agg = float(
                        extract_point(pr_first_year_mean, lat, lon).values.item()
                    )
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
                if ssp_data is not None:
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

                # For precipitation: add first-year baseline to convert anomalies to
                # absolute values before fitting/applying the Gamma transform.
                # The pattern scaling outputs anomalies, but Gamma requires positive values.
                # We use first-year (2015) baseline to match CMIP6 starting point.
                if variable == "pr" and pr_baseline_agg is not None:
                    ensemble_for_transform = ensemble_for_transform + pr_baseline_agg

                # Fit per-month-of-year (seasonal) when the transform exposes it
                # — otherwise σ_gaussian is dominated by the seasonal cycle and
                # the quantile map squashes the inter-realization noise band.
                use_seasonal = (
                    transform_config.fit_1d_seasonal_func is not None
                    and transform_config.apply_seasonal_func is not None
                )

                if use_seasonal:
                    gaussian_params = transform_config.fit_1d_seasonal_func(
                        ensemble_for_transform, "gaussian"
                    )
                    target_params = transform_config.fit_1d_seasonal_func(
                        cmip6_agg, transform_config.transform_type
                    )
                    transformed_ensemble = transform_config.apply_seasonal_func(
                        ensemble_for_transform,
                        gaussian_params,
                        target_params,
                        target_dist=transform_config.transform_type,
                    )
                else:
                    gaussian_params = transform_config.fit_1d_func(
                        ensemble_for_transform, "gaussian"
                    )
                    target_params = transform_config.fit_1d_func(
                        cmip6_agg, transform_config.transform_type
                    )
                    transformed_ensemble = transform_config.apply_func(
                        ensemble_for_transform,
                        gaussian_params,
                        target_params,
                        target_dist=transform_config.transform_type,
                    )

                results[agg] = transformed_ensemble
            else:
                # No transform - convert to numpy if needed and ensure 2D
                if isinstance(raw_ensemble, np.ndarray):
                    result_array = (
                        raw_ensemble
                        if raw_ensemble.ndim == 2
                        else raw_ensemble[np.newaxis, :]
                    )
                else:
                    # xarray DataArray
                    result_array = (
                        raw_ensemble.values
                        if raw_ensemble.ndim == 2
                        else raw_ensemble.values[np.newaxis, :]
                    )

                # For precipitation without transform: still add first-year baseline
                # to convert from anomalies to absolute values
                if variable == "pr" and pr_baseline_agg is not None:
                    result_array = result_array + pr_baseline_agg

                results[agg] = result_array

        return results

    def _generate_gridded(
        self,
        variable,
        scenario,
        start_year,
        end_year,
        n_realizations,
        gridded_spec,
        gen_inputs=None,
        include_noise=True,
        temp_scaling_ts=None,
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
        # Build shared generation inputs if not provided by the caller. The PCs
        # are generated once over the full trajectory (spun-up) and sliced to the
        # output window, then sliced again per output field below.
        if gen_inputs is None:
            gen_inputs = self._prepare_generation(
                variable,
                scenario,
                start_year,
                end_year,
                n_realizations,
                include_noise=include_noise,
                temp_scaling_ts=temp_scaling_ts,
                verbose=verbose,
            )

        monthly_prediction = gen_inputs.pattern.monthly_prediction
        monthly_warming = gen_inputs.pattern.monthly_warming
        stochastic_pcs = gen_inputs.stochastic_pcs
        noise_model = self.noise_models[variable]

        if not include_noise:
            if verbose:  # pragma: no cover
                print("      → Generating gridded climatology (no noise)")
        elif verbose:  # pragma: no cover
            print("      → Using shared stochastic PC realizations (gridded)")

        # Resolve transform. For variables with a distribution transform (e.g.
        # pr) we generate the full prediction window in one shot and apply the
        # seasonal (per-month-of-year, per-gridpoint) transform once on the
        # whole window. Doing it per year-slice (the old path) re-fitted the
        # Gaussian on just 12 months at each gridpoint, which normalised every
        # year to its own local mean and wiped the climate-change trend.
        transform_config = self._get_transform_config(variable)
        pre_transformed_ensemble = None
        target_params = None
        pr_baseline_field = None
        if transform_config and transform_config.transform_type:
            pre_transformed_ensemble = self._build_full_window_transformed_ensemble(
                variable,
                monthly_prediction,
                monthly_warming,
                noise_model,
                stochastic_pcs,
                start_year,
                end_year,
                transform_config,
                include_noise,
                verbose=verbose,
            )

        # Extract requested time slices
        results = {}
        n_months = len(monthly_warming)

        # Window-relative month index. monthly_prediction, monthly_warming and the
        # sliced stochastic PCs all share the same origin (start_year), which is
        # derived from base_year inside _get_or_compute_pattern_scaling, so all
        # three stay aligned.
        def year_to_month_idx(year):
            return (year - start_year) * 12

        def _get_ensemble_slice(s_idx, e_idx, reduce_time):
            """Slice the pre-transformed ensemble, or generate+transform per slice."""
            if pre_transformed_ensemble is not None:
                sliced = pre_transformed_ensemble.isel(month=slice(s_idx, e_idx))
                if reduce_time:
                    sliced = sliced.mean(dim="month")
                return sliced
            return self._generate_gridded_slice(
                monthly_prediction,
                monthly_warming,
                noise_model,
                stochastic_pcs,
                s_idx,
                e_idx,
                include_noise,
                reduce_time=reduce_time,
                transform_config=transform_config,
                target_params=target_params,
                pr_baseline_field=pr_baseline_field,
            )

        # Annual means (12-month average of each year)
        if "annual" in gridded_spec:
            if verbose:  # pragma: no cover
                print(
                    f"      → Extracting annual means for {len(gridded_spec['annual'])} years"
                )
            annual_fields = {}
            for year in gridded_spec["annual"]:
                start_idx = year_to_month_idx(year)
                end_idx = start_idx + 12
                if start_idx >= 0 and end_idx <= n_months:
                    annual_fields[year] = _get_ensemble_slice(
                        start_idx, end_idx, reduce_time=True
                    )
                else:
                    if verbose:  # pragma: no cover
                        print(
                            f"        ⚠️  Year {year} outside range {start_year}-{end_year}"
                        )
            results["annual"] = annual_fields

        # Monthly fields (all 12 months retained)
        if "monthly" in gridded_spec:
            if verbose:  # pragma: no cover
                print(
                    f"      → Extracting monthly fields for {len(gridded_spec['monthly'])} years"
                )
            monthly_fields = {}
            for year in gridded_spec["monthly"]:
                start_idx = year_to_month_idx(year)
                end_idx = start_idx + 12
                if start_idx >= 0 and end_idx <= n_months:
                    monthly_fields[year] = _get_ensemble_slice(
                        start_idx, end_idx, reduce_time=False
                    )
                else:
                    if verbose:  # pragma: no cover
                        print(
                            f"        ⚠️  Year {year} outside range {start_year}-{end_year}"
                        )
            results["monthly"] = monthly_fields

        # Climatologies (multi-year means)
        if "climatology" in gridded_spec:
            if verbose:  # pragma: no cover
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
                        climatology_fields[f"{clim_start}-{clim_end}"] = (
                            _get_ensemble_slice(start_idx, end_idx, reduce_time=True)
                        )
                    else:
                        if verbose:  # pragma: no cover
                            print(
                                f"        ⚠️  Period {clim_start}-{clim_end} outside range"
                            )
                else:
                    if verbose:  # pragma: no cover
                        print(f"        ⚠️  Invalid climatology period: {period}")
            results["climatology"] = climatology_fields

        return results

    def _generate_gridded_slice(
        self,
        monthly_prediction,
        monthly_warming,
        noise_model,
        stochastic_pcs,
        start_idx,
        end_idx,
        include_noise,
        reduce_time,
        transform_config=None,
        target_params=None,
        pr_baseline_field=None,
    ):
        """
        Generate one gridded output slice (annual / monthly / climatology).

        Unifies the three previously-duplicated gridded loops. The only behavioural
        differences between output types are the month range (``start_idx`` /
        ``end_idx``, window-relative) and whether the time axis is averaged away
        (``reduce_time``).

        Noise is taken from the shared spun-up PCs (sliced to this window) so that
        gridded fields are consistent with the time series outputs and carry full
        stationary variability. Any distribution transform (e.g. precipitation
        Gamma) is applied to the MONTHLY field *before* time-averaging, since the
        transform is a per-gridpoint quantile map over the sample axis.

        Parameters
        ----------
        monthly_prediction : xr.DataArray
            Window-sliced monthly pattern prediction (month, lat, lon).
        monthly_warming : np.ndarray
            Window-sliced global-mean monthly warming.
        noise_model : object
            Fitted noise model for this variable.
        stochastic_pcs : np.ndarray or None
            Window-sliced shared PCs (n_realizations, n_months, n_modes), or None
            when ``include_noise`` is False.
        start_idx, end_idx : int
            Window-relative month indices bounding this slice.
        include_noise : bool
            Whether to add stochastic noise.
        reduce_time : bool
            If True, average over the month axis (annual / climatology); if False,
            keep all months (monthly fields).
        transform_config : VariableTransformConfig, optional
            Distribution transform configuration (e.g. for ``pr``).
        target_params : dict, optional
            Pre-fitted per-gridpoint target distribution parameters.
        pr_baseline_field : xr.DataArray, optional
            Gridded first-year baseline (lat, lon) added to anomalies before the
            transform to obtain absolute precipitation.

        Returns
        -------
        xr.DataArray
            Stacked realizations with a leading ``realization`` dimension.
        """
        base_slice = monthly_prediction.isel(month=slice(start_idx, end_idx))

        if include_noise:
            pcs_slice = stochastic_pcs[:, start_idx:end_idx, :]
            realizations = noise_model.generate_realization(
                monthly_warming[start_idx:end_idx],
                noise_only=True,
                add_base=base_slice,
                stochastic_pcs=pcs_slice,
            )
            if not isinstance(realizations, list):
                realizations = [realizations]
        else:
            realizations = [base_slice]

        ensemble = _stack_realizations(realizations)

        # Apply the distribution transform on the monthly field, before averaging.
        if transform_config and transform_config.transform_type:
            ensemble = self._apply_gridded_transform(
                ensemble, transform_config, target_params, pr_baseline_field
            )

        if reduce_time:
            ensemble = ensemble.mean(dim="month")

        return ensemble

    def _apply_gridded_transform(
        self, ensemble, transform_config, target_params, pr_baseline_field
    ):
        """
        Apply a per-gridpoint distribution transform to a gridded ensemble.

        Mirrors the time series transform but uses the 3D (per-gridpoint) fitting
        and application path. The input must still retain its month axis so that
        each gridpoint has a sample distribution to map.

        Parameters
        ----------
        ensemble : xr.DataArray
            Generated ensemble (realization, month, lat, lon).
        transform_config : VariableTransformConfig
            Transform configuration providing ``fit_3d_func`` / ``apply_func``.
        target_params : dict
            Pre-fitted per-gridpoint target distribution parameters.
        pr_baseline_field : xr.DataArray or None
            Gridded first-year baseline added to convert anomalies to absolute
            values before the transform (precipitation).

        Returns
        -------
        xr.DataArray
            Transformed ensemble with the same coords/dims as the input.
        """
        data = ensemble
        if pr_baseline_field is not None:
            # The CMIP6 reference field carries a singleton "ens" dimension (added
            # by the data getter via expand_dims). Reduce the baseline to its
            # spatial (lat, lon) grid so it broadcasts cleanly over the ensemble's
            # (realization, month, lat, lon) dims instead of appending a spurious
            # trailing axis.
            extra_dims = [d for d in pr_baseline_field.dims if d not in ("lat", "lon")]
            if extra_dims:
                pr_baseline_field = pr_baseline_field.isel(
                    {d: 0 for d in extra_dims}, drop=True
                )
            # Broadcast (lat, lon) baseline over realization and month
            data = data + pr_baseline_field

        # Fit Gaussian per gridpoint to the generated ensemble, then map to target.
        gaussian_params = transform_config.fit_3d_func(data.values, "gaussian")
        transformed = transform_config.apply_func(
            data.values,
            gaussian_params,
            target_params,
            target_dist=transform_config.transform_type,
        )
        return xr.DataArray(transformed, coords=data.coords, dims=data.dims)

    def _build_full_window_transformed_ensemble(
        self,
        variable,
        monthly_prediction,
        monthly_warming,
        noise_model,
        stochastic_pcs,
        start_year,
        end_year,
        transform_config,
        include_noise,
        verbose=True,
    ):
        """
        Generate the gridded ensemble over the FULL prediction window and apply
        the seasonal (per-month-of-year, per-gridpoint) distribution transform
        once.

        Fitting the Gaussian half of the quantile map over the full window — and
        per month-of-year rather than across all months at once — keeps the
        long-term trend in the variance the map carries through (so the gridded
        trend is preserved) while removing the seasonal cycle from σ (so the
        inter-realization noise band is preserved at every gridpoint).

        Returns
        -------
        xr.DataArray
            Transformed monthly ensemble (realization, month, lat, lon) spanning
            ``start_year..end_year`` inclusive. Callers slice this once per
            requested annual / monthly / climatology field.
        """
        if verbose:  # pragma: no cover
            print(
                "      → Generating full-window gridded ensemble for "
                f"{transform_config.transform_type} transform"
            )

        ssp_data, pr_baseline_field = self._load_transform_reference(
            variable, start_year, end_year, verbose=verbose
        )

        if verbose:  # pragma: no cover
            print(
                f"      → Fitting per-month-of-year per-gridpoint "
                f"{transform_config.transform_type} target distribution..."
            )
        target_params = transform_config.fit_3d_seasonal_func(
            ssp_data, transform_config.transform_type
        )

        if include_noise:
            realizations = noise_model.generate_realization(
                monthly_warming,
                noise_only=True,
                add_base=monthly_prediction,
                stochastic_pcs=stochastic_pcs,
            )
            if not isinstance(realizations, list):
                realizations = [realizations]
        else:
            realizations = [monthly_prediction]
        ensemble = _stack_realizations(realizations)

        data = ensemble
        if pr_baseline_field is not None:
            extra_dims = [d for d in pr_baseline_field.dims if d not in ("lat", "lon")]
            if extra_dims:
                pr_baseline_field = pr_baseline_field.isel(
                    {d: 0 for d in extra_dims}, drop=True
                )
            data = data + pr_baseline_field

        if verbose:  # pragma: no cover
            print(
                "      → Fitting per-month-of-year per-gridpoint Gaussian on "
                "full-window generated ensemble..."
            )
        gaussian_params = transform_config.fit_3d_seasonal_func(data.values, "gaussian")

        transformed = transform_config.apply_seasonal_func(
            data.values,
            gaussian_params,
            target_params,
            target_dist=transform_config.transform_type,
        )
        return xr.DataArray(transformed, coords=data.coords, dims=data.dims)

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
                    region = key.split(":")[1]
                    baseline_k = float(regional_mean(picontrol_data, region).mean())
                elif key.startswith("point:"):
                    coords = key.split(":")[1]
                    lat, lon = map(float, coords.split(","))
                    baseline_k = float(extract_point(picontrol_data, lat, lon).mean())
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

                # DegreeDaysCalculator is internally vectorized over the
                # 'month' dimension, so we pass the whole (realization, month)
                # array in a single call rather than looping per realization.
                result = dd_model.calculate(temp_celsius)
                impacts["hdd"][key] = result.data["annual_hdd"].values
                impacts["cdd"][key] = result.data["annual_cdd"].values

                if verbose:  # pragma: no cover
                    print(f"         • HDD for {key}")
                    print(f"         • CDD for {key}")

        return impacts

    def __repr__(self):
        """Return string representation of MeteorInterface."""
        status = "trained" if all(self._is_trained.values()) else "not trained"
        return (
            f"MeteorInterface(model='{self.model}', "
            f"variables={self.variables}, status='{status}')"
        )
