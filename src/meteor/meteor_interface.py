"""
High-Level METEOR Interface

This module provides a simplified, user-friendly interface to METEOR's
pattern scaling and noise generation capabilities.
"""

import os
import numpy as np
import xarray as xr

from meteor.cmip6_meteor_data_getter import Cmip6MeteorDataGetter
from meteor import MeteorPatternScaling
from meteor.noise_generator import train_noise_model_from_cmip6
from meteor.ensemble_output import EnsembleOutput, VariableOutput
from meteor.variable_transforms import get_variable_transform_config


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
    >>> emulator = MeteorInterface.from_cmip6(
    ...     model='NorESM2-MM',
    ...     variable='pr',
    ...     cache_dir='./cache'
    ... )
    >>> emulator.train(auto=True)
    >>> ensemble = emulator.generate(
    ...     scenario='ssp245',
    ...     start_year=2020,
    ...     end_year=2100,
    ...     n_realizations=100,
    ...     timeseries=['global', 'regional:EAS']
    ... )
    >>> 
    >>> # Multi-variable with gridded output
    >>> emulator = MeteorInterface.from_cmip6(
    ...     model='CESM2',
    ...     variables=['tas', 'pr'],
    ...     cache_dir='./cache'
    ... )
    >>> emulator.train(auto=True)
    >>> ensemble = emulator.generate(
    ...     scenario='ssp245',
    ...     start_year=2020,
    ...     end_year=2100,
    ...     n_realizations=100,
    ...     timeseries=['global'],
    ...     gridded={'annual': [2030, 2050, 2100]}
    ... )
    """
    
    def __init__(self, model, variables, cache_dir=None, data_getter_kwargs=None):
        # Normalize variables to list
        if isinstance(variables, str):
            self.variables = [variables]
        else:
            self.variables = list(variables)
        
        self.model = model
        self.cache_dir = cache_dir or './cache'
        
        # Initialize data getter
        data_getter_kwargs = data_getter_kwargs or {}
        default_exps = ["piControl", "historical", "ssp245", "abrupt-4xCO2"]
        default_dbe = ['CMIP', 'CMIP', 'ScenarioMIP', 'CMIP']
        
        self.data_getter = Cmip6MeteorDataGetter(
            exps=data_getter_kwargs.get('exps', default_exps),
            flds=self.variables,
            dbe=data_getter_kwargs.get('dbe', default_dbe),
            enable_cache=True
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
    
    @classmethod
    def from_cmip6(cls, model, variable=None, variables=None, cache_dir=None, **kwargs):
        """
        Create MeteorInterface from CMIP6 model.
        
        Parameters
        ----------
        model : str
            CMIP6 model name
        variable : str, optional
            Single variable (use this or variables, not both)
        variables : list of str, optional
            Multiple variables (use this or variable, not both)
        cache_dir : str, optional
            Cache directory path
        **kwargs
            Additional arguments for data getter
        
        Returns
        -------
        MeteorInterface
            Configured emulator instance
        """
        if variable is not None and variables is not None:
            raise ValueError("Specify either 'variable' or 'variables', not both")
        
        vars_to_use = variables if variables is not None else variable
        if vars_to_use is None:
            raise ValueError("Must specify either 'variable' or 'variables'")
        
        return cls(model, vars_to_use, cache_dir, data_getter_kwargs=kwargs)
    
    def train(self, auto=True, training_scenario='ssp245', variable_configs=None, verbose=True):
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
                if training_scenario != 'ssp245':
                    config['training_scenario'] = training_scenario
                # Override with variable-specific config if provided
                if variable_configs and variable in variable_configs:
                    config.update(variable_configs[variable])
            else:
                config = variable_configs.get(variable, {}) if variable_configs else {}
                # Apply method parameter as default if not in config
                if 'training_scenario' not in config:
                    config['training_scenario'] = training_scenario
            
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
            if transform_config.transform_type and config.get('transform', True):
                if verbose:
                    print(f"   → Fitting {transform_config.transform_type} transform...")
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
            'n_modes_pattern': 3,
            'n_modes_noise': 40,
            'lag_order': 2,
            'use_picontrol_baseline': True,
            'training_scenario': 'ssp245',  # ✅ Add default training scenario
        }
        
        # Variable-specific defaults
        if variable == 'tas':
            config['use_exog'] = 'all'
            config['transform'] = False
        elif variable == 'pr':
            config['use_exog'] = 'none'
            config['transform'] = True
            config['transform_type'] = 'gamma'
        else:
            # Generic defaults for other variables
            config['use_exog'] = 'temp_only'
            config['transform'] = False
        
        return config
    
    def _train_pattern_scaling(self, variable, config, verbose=True):
        """Train pattern scaling model for a variable."""
        cache_dir = os.path.join(self.cache_dir, 'pattern_scaling')
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = self.data_getter.get_pattern_scaling_cache_path(
            self.model, cache_dir
        )
        
        # Check cache
        is_valid, cached_model, info = self.data_getter.validate_pattern_scaling_cache(
            cache_file, self.model
        )
        
        if is_valid and verbose:
            print("      ✓ Using cached pattern scaling model")
            training_data = None
            ssp_config = None
        else:
            if verbose and not is_valid:
                print(f"      ⚠️  Cache miss: {info.get('message', 'No cache found')}")
            
            # Prepare training data
            training_data = self.data_getter.prepare_pattern_scaling_training_data(
                self.model, "ssp245"
            )
            ssp_config = self.data_getter.load_ssp_config("ssp245")
        
        # Create pattern scaling model
        self.pattern_models[variable] = MeteorPatternScaling(
            f"cmip6-{self.model}-aer-{variable}",
            {variable: config['n_modes_pattern']},
            None if training_data is None else lambda key: training_data[key],
            ssp_input=ssp_config,
            from_file=False,
            exp_list=None if training_data is None else ["base", "co2x4", "sulxanom"],
            anom_timescales={variable: config['n_modes_pattern']},
            cache_dir=cache_dir
        )
    
    def _train_noise_model(self, variable, config, verbose=True):
        """Train noise model for a variable."""
        cache_dir = os.path.join(self.cache_dir, 'noise_models')
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = self.data_getter.get_noise_model_cache_path(
            self.model, variable, cache_dir
        )
        
        # Check cache
        is_valid, cached_model, info = self.data_getter.validate_noise_model_cache(
            cache_file, variable, 
            n_modes=config['n_modes_noise'],
            lag_order=config['lag_order']
        )
        
        if is_valid:
            if verbose:
                print("      ✓ Using cached noise model")
            self.noise_models[variable] = cached_model
        else:
            if verbose:
                print("      ⚠️  Training new noise model...")
            
            # ✅ Get training scenario from config
            training_scenario = config.get('training_scenario', 'ssp245')
            
            # ✅ Generate pattern scaling prediction for custom_global_temp
            # This ensures noise model training uses same temperature trajectory as generation
            from ciceroscm import input_handler
            from meteor import global_mean
            
            cscm_data_dir = os.path.join(
                os.path.dirname(__file__), "default_scm_data"
            )
            conc_file = os.path.join(cscm_data_dir, f"{training_scenario}_conc_RCMIP.txt")
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
                print(f"      → Using {training_scenario} pattern prediction for training")
                print(f"      → Temperature trajectory: {len(monthly_warming_trimmed)//12} years")
            
            # Train noise model using data getter interface
            self.noise_models[variable] = train_noise_model_from_cmip6(
                self.data_getter,
                experiments=["historical", training_scenario],  # ✅ Use config scenario
                model_name=self.model,
                variable_name=variable,
                n_modes=config['n_modes_noise'],
                lag_order=config['lag_order'],
                use_exog=config['use_exog'],
                custom_global_temp=monthly_warming_trimmed,  # ✅ Pass pattern prediction
                cache_dir=cache_dir
            )
    
    def _fit_transform(self, variable, transform_config, config, verbose=True):
        """Fit variable-specific transform during training."""
        # Store transform configuration for later use during generation
        # We fit parameters on-demand during generation because we need
        # the generated data to fit the source distribution
        self.transforms[variable] = {
            'config': transform_config,
            'fitted_params_1d': {},  # Will be populated per aggregation during generation
            'fitted_params_3d': None  # Will be fitted if gridded output requested
        }
    
    def generate(self, scenario, start_year, end_year, n_realizations,
                 timeseries=None, gridded=None, impacts=None,
                 include_noise=True, save_to=None, verbose=True):
        """
        Generate ensemble outputs for all variables.
        
        Parameters
        ----------
        scenario : str
            SSP scenario ('ssp126', 'ssp245', 'ssp370', 'ssp585')
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
        >>> ensemble = emulator.generate(
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
        >>> climatology = emulator.generate(
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
                print("Note: include_noise=False, forcing n_realizations=1 (climatology only)")
            n_realizations = 1
        
        # Check all variables are trained
        for var in self.variables:
            if not self._is_trained[var]:
                raise RuntimeError(
                    f"Variable '{var}' not trained. Call train() first."
                )
        
        if verbose:
            print("=" * 60)
            print(f"Generating ensemble for {scenario}")
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
                    variable, scenario, start_year, end_year,
                    n_realizations, timeseries, include_noise=include_noise, verbose=verbose
                )
            
            # Generate gridded if requested
            if gridded:
                if verbose:
                    print("   → Gridded outputs...")
                var_output.gridded = self._generate_gridded(
                    variable, scenario, start_year, end_year,
                    n_realizations, gridded, include_noise=include_noise, verbose=verbose
                )
            
            # Apply impacts if requested
            if impacts and variable in impacts:
                if verbose:
                    print("   → Computing impact metrics...")
                var_output.impacts = self._apply_impacts(
                    var_output, variable, impacts[variable], verbose=verbose
                )
            
            results[variable] = var_output
            
            if verbose:
                print(f"   ✅ {variable.upper()} complete")
        
        # Create ensemble output
        ensemble = EnsembleOutput(
            results,
            metadata={
                'model': self.model,
                'scenario': scenario,
                'year_range': f"{start_year}-{end_year}",
                'n_realizations': n_realizations,
                'variables': self.variables
            }
        )
        
        # Save if requested
        if save_to:
            ensemble.to_netcdf(save_to)
        
        if verbose:
            print("\n" + "=" * 60)
            print("✅ Generation complete")
            print("=" * 60)
        
        return ensemble
    
    def _generate_timeseries(self, variable, scenario, start_year, end_year,
                            n_realizations, aggregations, include_noise=True, verbose=True):
        """Generate time series outputs with aggregations."""
        from meteor import global_mean, regional_mean, extract_point
        from ciceroscm import input_handler
        
        # Load forcing data for the scenario
        cscm_data_dir = os.path.join(
            os.path.dirname(__file__), "default_scm_data"
        )
        conc_file = os.path.join(cscm_data_dir, f"{scenario}_conc_RCMIP.txt")
        em_file = os.path.join(cscm_data_dir, f"{scenario}_em_RCMIP.txt")
        
        ih = input_handler.InputHandler({})
        conc_data = input_handler.read_inputfile(conc_file)
        em_data = ih.read_emissions(em_file)
        
        # Generate pattern scaling prediction (annual)
        pattern_model = self.pattern_models[variable]
        climate_prediction = pattern_model.predict_from_combined_experiment(
            em_data, conc_data, [variable]
        )
        annual_prediction = climate_prediction[variable]
        
        # Convert to monthly
        monthly_prediction = pattern_model.to_monthly(
            annual_prediction, start_year=0
        )
        
        # ✅ SLICE TO REQUESTED TIME RANGE
        # Pattern model starts at year 1750 (or model-specific base year)
        # Get the base year from the pattern model
        if hasattr(monthly_prediction, 'year'):
            base_year = int(monthly_prediction.year[0])
        else:
            base_year = 1750  # Default assumption
        
        # Calculate month indices for slicing
        start_month_idx = (start_year - base_year) * 12
        end_month_idx = (end_year - base_year + 1) * 12  # +1 to include end_year
        
        # Slice the monthly prediction to requested range
        monthly_prediction = monthly_prediction.isel(month=slice(start_month_idx, end_month_idx))
        
        # Get global mean temperature trajectory for noise model
        monthly_warming = global_mean(monthly_prediction).values
        
        # Get CMIP6 data for transform fitting
        ssp_data = self.data_getter.make_meteor_training_data_composite(
            ["historical", scenario], self.model, monthly=True
        )[variable]
        
        # ✅ Generate stochastic PCs (or skip if climatology only)
        noise_model = self.noise_models[variable]
        stochastic_pcs = None
        
        if include_noise:
            # CRITICAL: Generate stochastic PCs ONCE for all aggregations
            # This ensures all spatial scales share the same underlying variability
            if verbose:
                print(f"      → Generating {n_realizations} stochastic PC realizations...")
            
            stochastic_pcs = noise_model.generate_stochastic_pcs(
                monthly_warming,
                n_realizations=n_realizations,
                random_seed=None  # Can expose this as parameter if needed
            )
        else:
            if verbose:
                print("      → Climatology only (no stochastic variability)")
        
        # Generate outputs for each aggregation
        results = {}
        transform_info = self.transforms.get(variable, None)
        
        # Handle both dict (fitted) and VariableTransformConfig (not fitted) cases
        if isinstance(transform_info, dict):
            transform_config = transform_info.get('config')
        else:
            transform_config = transform_info  # It's a VariableTransformConfig object
        
        for agg in aggregations:
            if verbose:
                print(f"      • {agg}")
            
            # Parse aggregation type
            if agg == 'global':
                # Global mean
                pattern_agg = global_mean(monthly_prediction).values
                if include_noise:
                    raw_ensemble = noise_model.generate_regional_mean_realizations(
                        monthly_warming,
                        region='global',
                        n_realizations=n_realizations,
                        stochastic_pcs=stochastic_pcs,
                        noise_only=True,
                        add_base=pattern_agg,
                        return_numpy=False
                    )
                else:
                    # Climatology only: return pattern scaling with shape (1, time)
                    raw_ensemble = pattern_agg[np.newaxis, :]
                cmip6_agg = global_mean(ssp_data)
                
            elif agg.startswith('regional:'):
                # Regional mean
                region_code = agg.split(':')[1]
                pattern_agg = regional_mean(monthly_prediction, region_code).values
                if include_noise:
                    raw_ensemble = noise_model.generate_regional_mean_realizations(
                        monthly_warming,
                        region=region_code,
                        n_realizations=n_realizations,
                        stochastic_pcs=stochastic_pcs,
                        noise_only=True,
                        add_base=pattern_agg,
                        return_numpy=False
                    )
                else:
                    # Climatology only: return pattern scaling with shape (1, time)
                    raw_ensemble = pattern_agg[np.newaxis, :]
                cmip6_agg = regional_mean(ssp_data, region_code)
                
            elif agg.startswith('point:'):
                # Point extraction
                coords = agg.split(':')[1]
                lat_str, lon_str = coords.split(',')
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
                        return_numpy=False
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
                    ensemble_for_transform = raw_ensemble if raw_ensemble.ndim == 2 else raw_ensemble[np.newaxis, :]
                else:
                    # xarray DataArray
                    ensemble_for_transform = raw_ensemble.values if raw_ensemble.ndim == 2 else raw_ensemble.values[np.newaxis, :]
                
                # Fit Gaussian to generated data
                gaussian_params = transform_config.fit_1d_func(
                    ensemble_for_transform, 'gaussian'
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
                    target_dist=transform_config.transform_type
                )
                
                results[agg] = transformed_ensemble
            else:
                # Convert to numpy if needed and ensure 2D
                if isinstance(raw_ensemble, np.ndarray):
                    results[agg] = raw_ensemble if raw_ensemble.ndim == 2 else raw_ensemble[np.newaxis, :]
                else:
                    # xarray DataArray
                    results[agg] = raw_ensemble.values if raw_ensemble.ndim == 2 else raw_ensemble.values[np.newaxis, :]
        
        return results
    
    def _generate_gridded(self, variable, scenario, start_year, end_year,
                         n_realizations, gridded_spec, include_noise=True, verbose=True):
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
        from ciceroscm import input_handler
        import xarray as xr
        
        # Load forcing data for the scenario
        cscm_data_dir = os.path.join(
            os.path.dirname(__file__), "default_scm_data"
        )
        conc_file = os.path.join(cscm_data_dir, f"{scenario}_conc_RCMIP.txt")
        em_file = os.path.join(cscm_data_dir, f"{scenario}_em_RCMIP.txt")
        
        conc_data = input_handler.read_inputfile(conc_file)
        em_data = input_handler.read_inputfile(em_file)
        
        # Get trained components
        pattern_model = self.pattern_scaling_models[variable]
        noise_model = self.noise_models[variable]
        
        # Run SCM to get forcing
        scm_results = pattern_model.scm_runner.run_with_cmip_data(
            conc_data, em_data, start_year, end_year
        )
        monthly_warming = pattern_model.scm_runner.get_monthly_temp_anomaly(
            scm_results, start_year, end_year
        )
        
        # Generate pattern scaling prediction (3D fields)
        monthly_prediction = pattern_model.predict(monthly_warming)
        
        # Generate stochastic PCs (or skip if climatology only)
        if include_noise:
            # Generate PCs for consistent variability (though not used directly here)
            _ = noise_model.generate_stochastic_pcs(
                monthly_warming, n_realizations=n_realizations, random_seed=None
            )
            if verbose:
                print(f"      → Generating {n_realizations} gridded realizations")
        else:
            if verbose:
                print("      → Generating gridded climatology (no noise)")
        
        # Generate full 3D realizations
        if include_noise:
            # With noise: use noise model to generate full fields
            realizations = []
            for i in range(n_realizations):
                # Generate single realization with this PC timeseries
                realization = noise_model.generate_realization(
                    monthly_warming,
                    n_realizations=1,
                    noise_only=True,
                    add_base=monthly_prediction
                )
                realizations.append(realization[0] if isinstance(realization, list) else realization)
            
            # Stack into single array: (n_realizations, month, lat, lon)
            full_fields = xr.concat(realizations, dim='realization')
        else:
            # Climatology only: just use pattern scaling
            # Add realization dimension for consistency
            full_fields = monthly_prediction.expand_dims(realization=[0])
        
        # Extract requested time slices
        results = {}
        n_months = len(monthly_warming)
        
        # Helper to convert year to month index
        def year_to_month_idx(year):
            return (year - start_year) * 12
        
        # Annual means
        if 'annual' in gridded_spec:
            if verbose:
                print(f"      → Extracting annual means for {len(gridded_spec['annual'])} years")
            annual_fields = {}
            for year in gridded_spec['annual']:
                start_idx = year_to_month_idx(year)
                end_idx = start_idx + 12
                if start_idx >= 0 and end_idx <= n_months:
                    # Average over 12 months for this year
                    annual_mean = full_fields.isel(month=slice(start_idx, end_idx)).mean(dim='month')
                    annual_fields[year] = annual_mean
                else:
                    if verbose:
                        print(f"        ⚠️  Year {year} outside range {start_year}-{end_year}")
            results['annual'] = annual_fields
        
        # Monthly fields
        if 'monthly' in gridded_spec:
            if verbose:
                print(f"      → Extracting monthly fields for {len(gridded_spec['monthly'])} years")
            monthly_fields = {}
            for year in gridded_spec['monthly']:
                start_idx = year_to_month_idx(year)
                end_idx = start_idx + 12
                if start_idx >= 0 and end_idx <= n_months:
                    # Extract all 12 months for this year
                    year_months = full_fields.isel(month=slice(start_idx, end_idx))
                    # Add month-of-year coordinate
                    year_months = year_months.assign_coords(month_of_year=('month', np.arange(1, 13)))
                    monthly_fields[year] = year_months
                else:
                    if verbose:
                        print(f"        ⚠️  Year {year} outside range {start_year}-{end_year}")
            results['monthly'] = monthly_fields
        
        # Climatologies (multi-year means)
        if 'climatology' in gridded_spec:
            if verbose:
                print(f"      → Computing {len(gridded_spec['climatology'])} climatological means")
            climatology_fields = {}
            for period in gridded_spec['climatology']:
                if isinstance(period, (list, tuple)) and len(period) == 2:
                    clim_start, clim_end = period
                    start_idx = year_to_month_idx(clim_start)
                    end_idx = year_to_month_idx(clim_end + 1)  # +1 to include end year
                    if start_idx >= 0 and end_idx <= n_months:
                        clim_mean = full_fields.isel(month=slice(start_idx, end_idx)).mean(dim='month')
                        climatology_fields[f"{clim_start}-{clim_end}"] = clim_mean
                    else:
                        if verbose:
                            print(f"        ⚠️  Period {clim_start}-{clim_end} outside range")
                else:
                    if verbose:
                        print(f"        ⚠️  Invalid climatology period: {period}")
            results['climatology'] = climatology_fields
        
        return results
    
    def _apply_impacts(self, var_output, variable, impact_configs, verbose=True):
        """Apply impact models to generated data."""
        impacts = {}
        
        # Check if degree days are requested
        if 'degree_days' in impact_configs:
            try:
                from meteor.impacts import DegreeDaysCalculator
                from meteor import global_mean
                
                dd_config = impact_configs['degree_days']
                
                # Get piControl baseline for absolute temperature calculation
                # The timeseries data contains anomalies, we need to add baseline
                picontrol_data = self.data_getter.make_meteor_training_data(
                    "piControl", self.model, monthly=True
                )[variable]
                
                # Apply to all timeseries outputs
                if 'hdd_base' in dd_config:
                    dd_model = DegreeDaysCalculator(base_temperature=dd_config['hdd_base'])
                    impacts['hdd'] = {}
                    for key, ts_data in var_output.timeseries.items():
                        # Calculate appropriate baseline for this aggregation
                        if key == 'global':
                            from meteor import global_mean
                            baseline_k = float(global_mean(picontrol_data).mean())
                        elif key.startswith('regional:'):
                            from meteor import regional_mean
                            region = key.split(':')[1]
                            baseline_k = float(regional_mean(picontrol_data, region).mean())
                        elif key.startswith('point:'):
                            from meteor import extract_point
                            coords = key.split(':')[1]
                            lat, lon = map(float, coords.split(','))
                            baseline_k = float(extract_point(picontrol_data, lat, lon).mean())
                        else:
                            baseline_k = 287.15  # Fallback
                        
                        # Convert from anomaly (K) to absolute temperature (°C)
                        # ts_data is anomaly in K, baseline_k is absolute temperature in K
                        n_realizations, n_months = ts_data.shape
                        
                        # Create xarray with month dimension (required by calculator)
                        # Absolute temperature in Celsius = (anomaly_K + baseline_K) - 273.15
                        temp_celsius = xr.DataArray(
                            ts_data + baseline_k - 273.15,
                            dims=['realization', 'month'],
                            coords={'month': np.arange(n_months)}
                        )
                        
                        # Calculate degree days for each realization
                        hdd_results = []
                        for i in range(n_realizations):
                            result = dd_model.calculate(temp_celsius[i])
                            hdd_results.append(result.data['annual_hdd'].values)
                        
                        # Stack back into array (n_realizations, n_years)
                        impacts['hdd'][key] = np.array(hdd_results)
                        
                        if verbose:
                            print(f"         • HDD for {key}")
                
                if 'cdd_base' in dd_config:
                    dd_model = DegreeDaysCalculator(base_temperature=dd_config['cdd_base'])
                    impacts['cdd'] = {}
                    for key, ts_data in var_output.timeseries.items():
                        # Calculate appropriate baseline for this aggregation
                        if key == 'global':
                            from meteor import global_mean
                            baseline_k = float(global_mean(picontrol_data).mean())
                        elif key.startswith('regional:'):
                            from meteor import regional_mean
                            region = key.split(':')[1]
                            baseline_k = float(regional_mean(picontrol_data, region).mean())
                        elif key.startswith('point:'):
                            from meteor import extract_point
                            coords = key.split(':')[1]
                            lat, lon = map(float, coords.split(','))
                            baseline_k = float(extract_point(picontrol_data, lat, lon).mean())
                        else:
                            baseline_k = 287.15  # Fallback
                        
                        # Convert from anomaly (K) to absolute temperature (°C)
                        n_realizations, n_months = ts_data.shape
                        
                        temp_celsius = xr.DataArray(
                            ts_data + baseline_k - 273.15,
                            dims=['realization', 'month'],
                            coords={'month': np.arange(n_months)}
                        )
                        
                        # Calculate degree days for each realization
                        cdd_results = []
                        for i in range(n_realizations):
                            result = dd_model.calculate(temp_celsius[i])
                            cdd_results.append(result.data['annual_cdd'].values)
                        
                        impacts['cdd'][key] = np.array(cdd_results)
                        
                        if verbose:
                            print(f"         • CDD for {key}")
                            
            except ImportError as e:
                if verbose:
                    print(f"      ⚠️  meteor.impacts.DegreeDaysCalculator not available: {e}")
            except Exception as e:
                if verbose:
                    print(f"      ⚠️  Error calculating degree days: {e}")
        
        return impacts
    
    def __repr__(self):
        status = "trained" if all(self._is_trained.values()) else "not trained"
        return (f"MeteorInterface(model='{self.model}', "
                f"variables={self.variables}, status='{status}')")
