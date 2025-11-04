import os
import tempfile

import numpy as np
import pytest

from meteor import cmip6_meteor_data_getter
from meteor.noise_generator import train_noise_model_from_cmip6


def test_comprehensive_noise_model_training():
    """Comprehensive test of noise model training functionality including caching, custom temp, and piControl baseline."""

    # Test 1: Basic noise model training with caching
    print("Testing basic noise model training with caching...")
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        exps=["historical", "ssp245"], dbe=["CMIP", "ScenarioMIP"]
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

        try:
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

        except Exception as e:
            # If the noise model training fails, at least verify the function exists
            pytest.fail(f"Custom temperature training failed: {e}")

    # Test 4: piControl baseline (separate data getter needed)
    print("Testing piControl baseline...")
    data_getter_picontrol = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        exps=["piControl", "historical", "ssp245"], dbe=["CMIP", "CMIP", "ScenarioMIP"]
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


def test_noise_model_error_handling():
    """Test error handling in noise model training."""
    data_getter = cmip6_meteor_data_getter.Cmip6MeteorDataGetter(
        exps=["historical", "ssp245"], dbe=["CMIP", "ScenarioMIP"]
    )

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
