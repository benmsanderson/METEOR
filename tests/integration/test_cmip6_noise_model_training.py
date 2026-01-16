import os
import tempfile

import numpy as np
import pytest

from meteor import cmip6_meteor_data_getter
from meteor.noise_generator import (
    train_multiple_noise_models_from_cmip6,
    train_noise_model_from_cmip6,
    validate_noise_model_cache,
)


def test_comprehensive_noise_model_training(test_data_dir):
    """Comprehensive test of noise model training functionality including caching, custom temp, and piControl baseline."""
    cache_path = os.path.join(test_data_dir, "light_mock_cache")
    # Test 1: Basic noise model training with caching
    print("Testing basic noise model training with caching...")
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        exps=["historical", "ssp245"],
        dbe=["CMIP", "ScenarioMIP"],
        cache_dir=cache_path,
        enable_cache=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # First call - should create cache and train model
        noise_model1 = train_noise_model_from_cmip6(
            data_getter,
            experiments=["historical", "ssp245"],
            model_name="CanESM5",
            variable_name="tas",
            n_modes=5,  # Use 5 modes for basic test
            lag_order=1,
            cache_dir=tmpdir,
        )

        # Verify basic model training worked
        assert hasattr(noise_model1, "fitted")
        assert noise_model1.fitted is True
        assert noise_model1.n_modes == 5
        assert noise_model1.lag_order == 1

        # Test 2: Cache functionality - second call should load from cache
        print("Testing cache loading...")
        cache_files = os.listdir(tmpdir)
        assert len(cache_files) > 0, "Cache files should have been created"

        # Second call with same parameters - should load from cache
        noise_model2 = train_noise_model_from_cmip6(
            data_getter,
            experiments=["historical", "ssp245"],
            model_name="CanESM5",
            variable_name="tas",
            n_modes=5,
            lag_order=1,
            cache_dir=tmpdir,
        )

        # Both should be fitted
        assert noise_model2.fitted is True

        # Test 3: Custom temperature trajectory
        print("Testing custom temperature trajectory...")
        # The noise model expects monthly data, so we need 251 years * 12 months = 3012 months
        # Get the actual monthly data length to be sure
        composite_data = data_getter.make_meteor_training_data_composite(
            ["historical", "ssp245"], model="CanESM5"
        )
        yearly_length = len(composite_data["year"])
        monthly_length = yearly_length * 12  # Convert years to months
        custom_temp = np.random.normal(15, 2, monthly_length)

        noise_model3 = train_noise_model_from_cmip6(
            data_getter,
            experiments=["historical", "ssp245"],
            model_name="CanESM5",
            variable_name="tas",
            n_modes=3,  # Use fewer modes for efficiency
            lag_order=1,
            custom_global_temp=custom_temp,
            cache_dir=tmpdir,  # Use same cache dir but different params
        )

        # If it works, verify we got a fitted noise model
        assert noise_model3 is not None
        assert noise_model3.fitted is True
        assert noise_model3.n_modes == 3

    # Test 4: piControl baseline (separate data getter needed)
    print("Testing piControl baseline...")
    data_getter_picontrol = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        exps=["piControl", "historical", "ssp245"],
        dbe=["CMIP", "CMIP", "ScenarioMIP"],
        cache_dir=cache_path,
        enable_cache=True,
    )

    noise_model4 = train_noise_model_from_cmip6(
        data_getter_picontrol,
        experiments=["historical", "ssp245"],
        model_name="CanESM5",
        variable_name="tas",
        n_modes=3,  # Use fewer modes for efficiency
        lag_order=1,
        use_picontrol_baseline=True,
    )

    assert noise_model4.fitted is True
    assert noise_model4.n_modes == 3

    print("All noise model training tests completed successfully!")


def test_noise_model_error_handling(test_data_dir):
    """Test error handling in noise model training."""
    cache_path = os.path.join(test_data_dir, "light_mock_cache")
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        exps=["historical", "ssp245"],
        dbe=["CMIP", "ScenarioMIP"],
        cache_dir=cache_path,
        enable_cache=True,
    )
    assert data_getter.cache_handler.cache_functioning

    # Test with invalid model
    with pytest.raises(Exception):  # Should raise some kind of error for invalid model
        train_noise_model_from_cmip6(
            data_getter,
            experiments=["historical", "ssp245"],
            model_name="InvalidModel123",
            variable_name="tas",
            n_modes=3,
            lag_order=1,
        )

    # Test that the function is callable (basic smoke test)
    assert callable(train_noise_model_from_cmip6)


def test_noise_model_validation(test_data_dir):
    """Test the noise model validation function."""
    cache_path = os.path.join(test_data_dir, "light_mock_cache")
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        exps=["historical", "ssp245"],
        dbe=["CMIP", "ScenarioMIP"],
        cache_dir=cache_path,
        enable_cache=True,
    )
    # Test with non-existent file
    validation_results = validate_noise_model_cache("not_a_file", "tas")
    assert validation_results[0] is False
    assert validation_results[1] is None
    assert set(validation_results[2].keys()) == {"expected", "found", "message"}

    invalid_cache_path = os.path.join(
        test_data_dir, "light_mock_cache", "cmip6", "CanESM5_ssp370_pr_yearly.nc"
    )
    validation_results = validate_noise_model_cache(
        invalid_cache_path, "tas", n_modes=4, lag_order=1
    )
    assert validation_results[0] is False
    assert validation_results[1] is None
    assert validation_results[2]["message"].startswith("Error loading cache")

    with tempfile.TemporaryDirectory() as tmpdir:
        noise_models = train_multiple_noise_models_from_cmip6(
            data_getter,
            experiments=["historical", "ssp245"],
            models=["CanESM5"],
            variables=["tas"],
            n_modes=4,
            lag_order=1,
            cache_dir=tmpdir,
        )
        assert set(noise_models.keys()) == set(["CanESM5"])
        cache_file = os.path.join(tmpdir, os.listdir(tmpdir)[0])
        # Valid case
        validation_results = validate_noise_model_cache(
            cache_file, "tas", n_modes=4, lag_order=1
        )
        assert validation_results[0] is True
        assert validation_results[1] is not None
        assert validation_results[2]["found"]["variable_name"] == "tas"
        assert validation_results[2]["found"]["n_modes"] == 4
        assert validation_results[2]["found"]["lag_order"] == 1

        # Invalid variable name
        validation_results = validate_noise_model_cache(
            cache_file, "pr", n_modes=4, lag_order=1
        )
        assert validation_results[0] is False
        assert validation_results[1] is None
        assert validation_results[2]["found"]["variable_name"] == "tas"
        assert validation_results[2]["found"]["n_modes"] == 4
        assert validation_results[2]["found"]["lag_order"] == 1
        assert validation_results[2]["message"].startswith(
            "variable_name mismatch: expected 'pr'"
        )

        # Invalid lag order
        validation_results = validate_noise_model_cache(
            cache_file, "tas", n_modes=4, lag_order=3
        )
        assert validation_results[0] is False
        assert validation_results[1] is None
        assert validation_results[2]["found"]["variable_name"] == "tas"
        assert validation_results[2]["found"]["n_modes"] == 4
        assert validation_results[2]["found"]["lag_order"] == 1
        assert validation_results[2]["message"].startswith(
            "lag_order mismatch: expected 3"
        )

        # Invalid n_modes
        validation_results = validate_noise_model_cache(
            cache_file, "tas", n_modes=6, lag_order=3
        )
        assert validation_results[0] is False
        assert validation_results[1] is None
        assert validation_results[2]["found"]["variable_name"] == "tas"
        assert validation_results[2]["found"]["n_modes"] == 4
        assert validation_results[2]["found"]["lag_order"] == 1
        assert validation_results[2]["message"].startswith(
            "n_modes mismatch: expected 6"
        )
