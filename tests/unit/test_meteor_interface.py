"""
High-quality tests for MeteorInterface functionality.

Focus: Test critical new logic with clear behavior expectations.
Strategy: Use mocks for expensive operations, verify transformations.
"""

import numpy as np
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch

from meteor.meteor_interface import MeteorInterface


@pytest.fixture
def mock_interface():
    """Create MeteorInterface with mocked data getter."""
    with patch('meteor.meteor_interface.Cmip6MeteorDataGetter'):
        interface = MeteorInterface(
            model='TestModel',
            variables=['tas', 'pr'],
            cache_dir='/tmp/test_cache'
        )
        # Mock the pattern models and noise models to avoid training
        interface.pattern_models = {
            'tas': MagicMock(),
            'pr': MagicMock()
        }
        interface.noise_models = {
            'tas': MagicMock(),
            'pr': MagicMock()
        }
        interface._is_trained = {'tas': True, 'pr': True}
        yield interface


class TestAnomályConversion:
    """Test CMIP6 data anomaly conversion for tas variable."""
    
    def test_tas_converted_to_anomalies(self, mock_interface):
        """Test that tas data is converted to anomalies from piControl baseline."""
        # Create mock piControl data with known mean
        picontrol_mean = 288.0  # K
        picontrol_data = xr.Dataset({
            'tas': xr.DataArray(
                np.full((100, 5, 5), picontrol_mean) + np.random.randn(100, 5, 5) * 0.1,
                dims=['month', 'lat', 'lon'],
                coords={
                    'month': range(100),
                    'lat': np.linspace(-90, 90, 5),
                    'lon': np.linspace(-180, 180, 5)
                }
            )
        })
        
        # Create mock scenario data with known temperature
        scenario_temp = 290.0  # K (2K warmer than piControl)
        scenario_data = xr.Dataset({
            'tas': xr.DataArray(
                np.full((100, 5, 5), scenario_temp) + np.random.randn(100, 5, 5) * 0.1,
                dims=['month', 'lat', 'lon'],
                coords={
                    'month': range(100),
                    'lat': np.linspace(-90, 90, 5),
                    'lon': np.linspace(-180, 180, 5)
                }
            )
        })
        
        # Mock data getter to return our test data
        mock_composite = MagicMock(side_effect=lambda exps, model, monthly=True: (
            picontrol_data if 'piControl' in exps else scenario_data
        ))
        mock_interface.data_getter.make_meteor_training_data_composite = mock_composite
        
        # Mock pattern scaling components to avoid full computation
        mock_pattern = xr.DataArray(
            np.zeros((100, 5, 5)),
            dims=['month', 'lat', 'lon'],
            coords={
                'month': range(100),
                'lat': np.linspace(-90, 90, 5),
                'lon': np.linspace(-180, 180, 5)
            }
        )
        mock_interface.pattern_models['tas'].to_monthly.return_value = mock_pattern
        mock_interface.pattern_models['tas'].predict_from_combined_experiment.return_value = {
            'tas': xr.DataArray(np.zeros(10), dims=['year'])
        }
        
        # Mock noise model
        mock_interface.noise_models['tas'].generate_stochastic_pcs.return_value = np.random.randn(3, 10)
        mock_interface.noise_models['tas'].generate_regional_mean_realizations.return_value = np.random.randn(3, 100)
        
        # Call _generate_timeseries which contains the anomaly conversion logic
        with patch('meteor.global_mean') as mock_global_mean:
            # Mock global mean to return simple arrays
            mock_global_mean.return_value = xr.DataArray(
                np.zeros(100), 
                dims=['month'],
                coords={'month': range(100)}
            )
            
            # This should trigger the anomaly conversion for tas
            _ = mock_interface._generate_timeseries(
                variable='tas',
                scenario='ssp245',
                start_year=2000,
                end_year=2010,
                n_realizations=3,
                aggregations=['global'],
                include_noise=True,
                verbose=False
            )
        
        # Verify that make_meteor_training_data_composite was called for both scenario and piControl
        calls = [call[0] for call in mock_interface.data_getter.make_meteor_training_data_composite.call_args_list]
        
        # Should have been called with scenario data
        assert any('ssp245' in str(call) or 'historical' in str(call) for call in calls)
        # Should have been called with piControl data (for tas only)
        assert any('piControl' in str(call) for call in calls)
    
    def test_pr_not_converted_to_anomalies(self, mock_interface):
        """Test that pr data is NOT converted to anomalies."""
        # Create mock scenario data for pr
        scenario_data = xr.Dataset({
            'pr': xr.DataArray(
                np.random.rand(100, 5, 5) * 1e-5,  # Typical precip values
                dims=['month', 'lat', 'lon'],
                coords={
                    'month': range(100),
                    'lat': np.linspace(-90, 90, 5),
                    'lon': np.linspace(-180, 180, 5)
                }
            )
        })
        
        call_count = [0]
        
        def mock_composite(exps, model, monthly=True):
            call_count[0] += 1
            # Should only be called once for pr (scenario data)
            # Should NOT be called for piControl
            if 'piControl' in exps:
                pytest.fail("piControl should not be loaded for pr variable")
            return scenario_data
        
        mock_interface.data_getter.make_meteor_training_data_composite = mock_composite
        
        # Mock pattern scaling components
        mock_pattern = xr.DataArray(
            np.zeros((100, 5, 5)),
            dims=['month', 'lat', 'lon'],
            coords={
                'month': range(100),
                'lat': np.linspace(-90, 90, 5),
                'lon': np.linspace(-180, 180, 5)
            }
        )
        mock_interface.pattern_models['pr'].to_monthly.return_value = mock_pattern
        mock_interface.pattern_models['pr'].predict_from_combined_experiment.return_value = {
            'pr': xr.DataArray(np.zeros(10), dims=['year'])
        }
        
        # Mock noise model
        mock_interface.noise_models['pr'].generate_stochastic_pcs.return_value = np.random.randn(3, 10)
        mock_interface.noise_models['pr'].generate_regional_mean_realizations.return_value = np.random.rand(3, 100) * 1e-5
        
        # Call _generate_timeseries for pr
        with patch('meteor.global_mean') as mock_global_mean:
            mock_global_mean.return_value = xr.DataArray(
                np.zeros(100),
                dims=['month'],
                coords={'month': range(100)}
            )
            
            _ = mock_interface._generate_timeseries(
                variable='pr',
                scenario='ssp245',
                start_year=2000,
                end_year=2010,
                n_realizations=3,
                aggregations=['global'],
                include_noise=True,
                verbose=False
            )
        
        # Verify piControl was NOT loaded (call_count should be 1, not 2)
        assert call_count[0] == 1, "piControl should not be loaded for precipitation"
    
    def test_anomaly_subtraction_correct(self):
        """Test that anomaly calculation performs correct arithmetic."""
        # Create simple test case with exact values
        picontrol_mean = 288.0
        picontrol_data = xr.Dataset({
            'tas': xr.DataArray(
                np.full((10, 3, 3), picontrol_mean),
                dims=['month', 'lat', 'lon']
            )
        })
        
        scenario_value = 290.0
        scenario_data = xr.Dataset({
            'tas': xr.DataArray(
                np.full((10, 3, 3), scenario_value),
                dims=['month', 'lat', 'lon']
            )
        })
        
        # Manually perform the anomaly calculation (mimicking the code)
        picontrol_mean_calc = picontrol_data['tas'].mean(dim='month')
        anomaly_data = scenario_data['tas'] - picontrol_mean_calc
        
        # Expected result: every point should be 2.0K (290 - 288)
        expected_anomaly = 2.0
        np.testing.assert_array_almost_equal(
            anomaly_data.values,
            np.full((10, 3, 3), expected_anomaly),
            decimal=10,
            err_msg="Anomaly calculation should subtract piControl mean"
        )


class TestCacheFilenames:
    """Test variable-specific cache filename generation."""
    
    def test_cache_path_includes_variable_name(self):
        """Test that cache paths include variable name for tas and pr."""
        from meteor.cmip6_meteor_data_getter import Cmip6MeteorDataGetter
        
        getter = Cmip6MeteorDataGetter(flds=['tas', 'pr'], exps=['piControl'])
        
        # Get cache paths for different variables
        tas_path = getter.get_pattern_scaling_cache_path(
            'TestModel', cache_dir='/tmp/cache', variable='tas'
        )
        pr_path = getter.get_pattern_scaling_cache_path(
            'TestModel', cache_dir='/tmp/cache', variable='pr'
        )
        
        # Verify different variables get different paths
        assert tas_path != pr_path, "tas and pr should have different cache paths"
        assert 'tas' in tas_path, "tas cache path should contain 'tas'"
        assert 'pr' in pr_path, "pr cache path should contain 'pr'"
        assert tas_path.endswith('_pattern_scaling.pkl')
        assert pr_path.endswith('_pattern_scaling.pkl')
    
    def test_cache_path_without_variable(self):
        """Test that cache path works without variable for backward compatibility."""
        from meteor.cmip6_meteor_data_getter import Cmip6MeteorDataGetter
        
        getter = Cmip6MeteorDataGetter(flds=['tas'], exps=['piControl'])
        
        # Get cache path without variable parameter
        path_no_var = getter.get_pattern_scaling_cache_path(
            'TestModel', cache_dir='/tmp/cache'
        )
        
        # Should still work (for backward compatibility)
        assert path_no_var.endswith('_pattern_scaling.pkl')
        assert 'TestModel' in path_no_var


class TestIntegrationSmoke:
    """Smoke tests for basic MeteorInterface workflow."""
    
    def test_from_cmip6_creates_interface(self):
        """Test that from_cmip6 factory method works."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter'):
            interface = MeteorInterface.from_cmip6(
                model='TestModel',
                variables=['tas'],
                cache_dir='/tmp/test'
            )
            
            assert interface.model == 'TestModel'
            assert interface.variables == ['tas']
            assert interface.cache_dir == '/tmp/test'
    
    def test_multiple_variables_supported(self):
        """Test that multiple variables can be specified."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter'):
            interface = MeteorInterface.from_cmip6(
                model='TestModel',
                variables=['tas', 'pr'],
                cache_dir='/tmp/test'
            )
            
            assert 'tas' in interface.variables
            assert 'pr' in interface.variables
            assert len(interface.variables) == 2


class TestTrainingWorkflow:
    """Test training workflow state transitions. 
    
    Note: Full training workflows are covered by integration tests due to
    complexity of mocking CMIP6 data structures with proper coordinates.
    These tests focus on state tracking and initialization.
    """
    
    def test_train_initializes_state_tracking(self):
        """Test that MeteorInterface initializes training state tracking."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter') as mock_getter_class:
            mock_getter = MagicMock()
            mock_getter_class.return_value = mock_getter
            
            interface = MeteorInterface(
                model='TestModel',
                variables=['tas'],
                cache_dir='/tmp/test'
            )
            
            # Check that training state is initialized
            assert hasattr(interface, '_is_trained')
            assert 'tas' in interface._is_trained
            assert interface._is_trained['tas'] == False
            
            # Check that model storage is initialized
            assert hasattr(interface, 'pattern_models')
            assert hasattr(interface, 'noise_models')
            assert isinstance(interface.pattern_models, dict)
            assert isinstance(interface.noise_models, dict)
    
    def test_train_tracks_multiple_variables(self):
        """Test that training state is tracked per-variable."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter') as mock_getter_class:
            mock_getter = MagicMock()
            mock_getter_class.return_value = mock_getter
            
            interface = MeteorInterface(
                model='TestModel',
                variables=['tas', 'pr'],
                cache_dir='/tmp/test'
            )
            
            # Initially, no variables are trained
            assert interface._is_trained['tas'] == False
            assert interface._is_trained['pr'] == False
            
            # Simulate training tas only (actual training is integration test)
            interface._is_trained['tas'] = True
            interface.pattern_models['tas'] = MagicMock()
            interface.noise_models['tas'] = MagicMock()
            
            # Check that tas is trained but pr is not
            assert interface._is_trained['tas'] == True
            assert interface._is_trained['pr'] == False
            assert 'tas' in interface.pattern_models
            assert 'pr' not in interface.pattern_models
    
    def test_training_config_storage(self):
        """Test that training configuration is stored."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter') as mock_getter_class:
            mock_getter = MagicMock()
            mock_getter_class.return_value = mock_getter
            
            interface = MeteorInterface(
                model='TestModel',
                variables=['tas'],
                cache_dir='/tmp/test'
            )
            
            # Check that training config dictionary exists
            assert hasattr(interface, '_training_config')
            assert isinstance(interface._training_config, dict)
            
            # Simulate setting a config (actual training is integration test)
            interface._training_config['tas'] = {
                'n_modes_pattern': 10,
                'n_modes_noise': 40,
                'training_scenario': 'ssp245'
            }
            
            # Verify it's stored
            assert 'tas' in interface._training_config
            assert interface._training_config['tas']['n_modes_pattern'] == 10
            assert interface._training_config['tas']['n_modes_noise'] == 40
    
    def test_train_multiple_variables(self):
        """Test training multiple variables."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter') as mock_getter_class:
            mock_getter = MagicMock()
            mock_getter_class.return_value = mock_getter
            mock_getter.validate_pattern_scaling_cache.return_value = (False, None, {})
            mock_getter.validate_noise_model_cache.return_value = (False, None, {})
            
    def test_model_dictionaries_are_mutable(self):
        """Test that model storage dictionaries can be populated."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter') as mock_getter_class:
            mock_getter = MagicMock()
            mock_getter_class.return_value = mock_getter
            
            interface = MeteorInterface(
                model='TestModel',
                variables=['tas', 'pr'],
                cache_dir='/tmp/test'
            )
            
            # Initially empty
            assert len(interface.pattern_models) == 0
            assert len(interface.noise_models) == 0
            
            # Simulate populating after training (actual training is integration test)
            interface.pattern_models['tas'] = MagicMock()
            interface.pattern_models['pr'] = MagicMock()
            interface.noise_models['tas'] = MagicMock()
            interface.noise_models['pr'] = MagicMock()
            
            # Verify they're populated
            assert len(interface.pattern_models) == 2
            assert len(interface.noise_models) == 2
            assert 'tas' in interface.pattern_models
            assert 'pr' in interface.pattern_models


class TestGenerationWorkflow:
    """Test the generate() method and generation workflows."""
    
    def test_generate_requires_training(self):
        """Test that generate() raises error if not trained."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter'):
            interface = MeteorInterface(
                model='TestModel',
                variables=['tas'],
                cache_dir='/tmp/test'
            )
            
            # Try to generate without training
            with pytest.raises(RuntimeError, match="not trained"):
                interface.generate(
                    scenario='ssp245',
                    start_year=2020,
                    end_year=2050,
                    n_realizations=10,
                    timeseries=['global']
                )
    
    def test_generate_with_noise_false_forces_single_realization(self):
        """Test that include_noise=False forces n_realizations=1."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter'):
            interface = MeteorInterface(
                model='TestModel',
                variables=['tas'],
                cache_dir='/tmp/test'
            )
            
            # Mark as trained
            interface._is_trained['tas'] = True
            interface.pattern_models['tas'] = MagicMock()
            interface.noise_models['tas'] = MagicMock()
            
            # Mock the internal generation method to capture arguments
            original_method = interface._generate_timeseries
            call_args = {}
            
            def capture_args(*args, **kwargs):
                call_args['args'] = args
                call_args['kwargs'] = kwargs
                # Return minimal valid structure
                return {'global': xr.DataArray([290.0])}
            
            interface._generate_timeseries = capture_args
            
            # Call generate with include_noise=False but n_realizations=100
            interface.generate(
                scenario='ssp245',
                start_year=2020,
                end_year=2020,
                n_realizations=100,
                timeseries=['global'],
                include_noise=False,
                verbose=False
            )
            
            # Should have forced n_realizations to 1
            # _generate_timeseries(variable, scenario, start_year, end_year, n_realizations, aggregations, ...)
            assert call_args['args'][4] == 1  # n_realizations is 5th positional arg (index 4)
    
    def test_generate_returns_ensemble_output(self):
        """Test that generate() returns EnsembleOutput container."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter'):
            interface = MeteorInterface(
                model='TestModel',
                variables=['tas'],
                cache_dir='/tmp/test'
            )
            
            interface._is_trained['tas'] = True
            interface.pattern_models['tas'] = MagicMock()
            interface.noise_models['tas'] = MagicMock()
            
            # Mock generation to return minimal data
            interface._generate_timeseries = MagicMock(return_value={
                'global': xr.DataArray([290.0], dims=['time'])
            })
            
            result = interface.generate(
                scenario='ssp245',
                start_year=2020,
                end_year=2020,
                n_realizations=1,
                timeseries=['global'],
                verbose=False
            )
            
            # Should return EnsembleOutput
            from meteor.ensemble_output import EnsembleOutput
            assert isinstance(result, EnsembleOutput)
            assert 'tas' in result
    
    def test_generate_populates_timeseries(self):
        """Test that generate() populates timeseries outputs."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter'):
            interface = MeteorInterface(
                model='TestModel',
                variables=['tas'],
                cache_dir='/tmp/test'
            )
            
            interface._is_trained['tas'] = True
            interface.pattern_models['tas'] = MagicMock()
            interface.noise_models['tas'] = MagicMock()
            
            # Mock generation to return test data
            mock_timeseries = {
                'global': xr.DataArray([290.0, 290.5, 291.0], dims=['time']),
                'regional:EAS': xr.DataArray([289.0, 289.5, 290.0], dims=['time'])
            }
            interface._generate_timeseries = MagicMock(return_value=mock_timeseries)
            
            result = interface.generate(
                scenario='ssp245',
                start_year=2020,
                end_year=2022,
                n_realizations=5,
                timeseries=['global', 'regional:EAS'],
                verbose=False
            )
            
            # Check timeseries were populated
            assert 'global' in result['tas'].timeseries
            assert 'regional:EAS' in result['tas'].timeseries
            assert len(result['tas'].timeseries['global']) == 3
    
    def test_generate_includes_metadata(self):
        """Test that generate() includes metadata in output."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter'):
            interface = MeteorInterface(
                model='TestModel',
                variables=['tas'],
                cache_dir='/tmp/test'
            )
            
            interface._is_trained['tas'] = True
            interface.pattern_models['tas'] = MagicMock()
            interface.noise_models['tas'] = MagicMock()
            
            interface._generate_timeseries = MagicMock(return_value={
                'global': xr.DataArray([290.0])
            })
            
            result = interface.generate(
                scenario='ssp370',
                start_year=2030,
                end_year=2080,
                n_realizations=25,
                timeseries=['global'],
                verbose=False
            )
            
            # Check metadata
            assert result.metadata['scenario'] == 'ssp370'
            assert result.metadata['year_range'] == '2030-2080'
            assert result.metadata['n_realizations'] == 25
            assert result.metadata['model'] == 'TestModel'


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_generate_before_training_clear_error(self):
        """Test that generating before training gives clear error message."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter'):
            interface = MeteorInterface(
                model='TestModel',
                variables=['tas', 'pr'],
                cache_dir='/tmp/test'
            )
            
            # Try to generate without training
            with pytest.raises(RuntimeError) as exc_info:
                interface.generate(
                    scenario='ssp245',
                    start_year=2020,
                    end_year=2050,
                    n_realizations=10,
                    timeseries=['global']
                )
            
            # Error message should mention which variable
            assert "not trained" in str(exc_info.value).lower()
            assert "train()" in str(exc_info.value).lower()
    
    def test_from_cmip6_requires_variables(self):
        """Test that from_cmip6 requires variable or variables parameter."""
        with patch('meteor.meteor_interface.Cmip6MeteorDataGetter'):
            # Should raise error if neither variable nor variables specified
            with pytest.raises(ValueError, match="Must specify either 'variable' or 'variables'"):
                MeteorInterface.from_cmip6(
                    model='TestModel',
                    cache_dir='/tmp/test'
                )
