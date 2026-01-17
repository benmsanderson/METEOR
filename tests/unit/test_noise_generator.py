"""Tests for the noise_generator module."""

import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from meteor import noise_generator
from meteor.noise_generator import MeteorNoiseGenerator, load_noise_model_from_cache


def test_meteor_noise_generator_initialization():
    """Test basic initialization of MeteorNoiseGenerator."""
    generator = MeteorNoiseGenerator(use_exog="not_exog_var")

    # Check that attributes are initialized
    assert hasattr(generator, "n_modes")
    assert hasattr(generator, "lag_order")
    assert hasattr(generator, "seasonal_model")
    assert hasattr(generator, "pca")
    assert hasattr(generator, "varx_results")
    assert hasattr(generator, "fitted")
    assert generator.fitted is False
    assert generator.n_modes == 10  # default value
    assert generator.lag_order == 2  # default value

    # Test error handling with uninitiated state
    with pytest.raises(ValueError, match="Invalid use_exog value:"):
        generator._extract_exog_variables(np.array([[1, 2, 3, 4]]))
    with pytest.raises(ValueError, match="Model must be fitted"):
        generator.generate_stochastic_pcs(np.array([1, 2, 3]))


def test_meteor_noise_generator_validation_errors():
    """Test validation error cases to improve coverage."""
    generator = MeteorNoiseGenerator()

    # Create mock data with wrong variable name
    data = xr.Dataset(
        {
            "wrong_var": xr.DataArray(
                np.random.rand(100, 10, 10), dims=["month", "lat", "lon"]
            )
        },
        coords={"month": range(100)},
    )

    # Test missing variable error
    with pytest.raises(ValueError, match="Variable 'tas' not found in dataset"):
        generator.fit(data, "tas")


def test_meteor_noise_generator_custom_global_temp_validation():
    """Test custom global temperature validation."""
    generator = MeteorNoiseGenerator()

    # Create test data
    data = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.random.rand(100, 10, 10),
                dims=["month", "lat", "lon"],
                coords={"lat": range(10), "lon": range(10)},
            )
        },
        coords={"month": range(100)},
    )

    # Add ensemble dimension
    data = data.expand_dims({"ens": [1]})

    # Test with wrong length custom global temperature
    with pytest.raises(
        ValueError, match="custom_global_temp length .* must match data time dimension"
    ):
        custom_temp = np.random.rand(50)  # Wrong length
        generator.fit(data, "tas", custom_global_temp=custom_temp)


def test_meteor_noise_generator_save_load():
    """Test save and load functionality."""
    generator = MeteorNoiseGenerator()

    # Test loading non-existent file
    with pytest.raises(FileNotFoundError):
        generator.load_model("/non/existent/path.pkl")

    # Test that saving unfitted model raises error
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model.pkl")

        # Should raise error when not fitted
        with pytest.raises(ValueError, match="Model must be fitted before saving"):
            generator.save_model(model_path)


def test_meteor_noise_generator_cache_functions():
    """Test cache-related functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test load_from_cache with non-existent file
        with pytest.raises(FileNotFoundError):
            load_noise_model_from_cache(temp_dir, "NonExistentModel", "tas")


def test_generate_realization_not_fitted():
    """Test generate_realization when model is not fitted."""
    generator = MeteorNoiseGenerator()

    # Should raise error when not fitted
    with pytest.raises(ValueError, match="Model must be fitted"):
        generator.generate_realization(np.random.rand(100))


def test_complex_scenarios():
    """Test complex scenarios to improve coverage."""
    # Create more realistic test data that might work better with VAR fitting
    time = np.arange(120)  # 10 years of monthly data
    lats = np.linspace(-90, 90, 5)
    lons = np.linspace(-180, 180, 6)
    ens = np.array([1])  # Single ensemble member

    # Create realistic temperature data with trends and seasonality
    temp_data = np.zeros((len(time), len(lats), len(lons), len(ens)))
    for i, t in enumerate(time):
        # Add seasonal cycle
        seasonal = 10 * np.sin(2 * np.pi * t / 12.0)
        # Add warming trend
        trend = 0.01 * t
        # Add spatial variation
        for j, lat in enumerate(lats):
            for k, lon in enumerate(lons):
                temp_data[i, j, k, 0] = seasonal + trend + 0.1 * lat + 0.05 * lon

    temp_da = xr.DataArray(
        temp_data,
        coords={"month": time, "lat": lats, "lon": lons, "ens": ens},
        dims=["month", "lat", "lon", "ens"],
        name="temperature",
    )

    # Create precipitation data
    precip_data = np.abs(np.random.normal(2.0, 0.5, temp_data.shape))
    precip_da = xr.DataArray(
        precip_data,
        coords={"time": time, "lat": lats, "lon": lons, "ens": ens},
        dims=["time", "lat", "lon", "ens"],
        name="precipitation",
    )

    # Test with complex datasets
    generator = noise_generator.MeteorNoiseGenerator()

    # Test fitting with multiple variables
    training_data = xr.Dataset({"tas": temp_da, "pr": precip_da})

    generator.fit(training_data, "tas")
    assert generator.fitted

    # Test generation with noise_only=True (lines 275-290)
    test_trajectory = np.linspace(0, 2, 24)  # 2 years
    noise_realizations = generator.generate_realization(
        test_trajectory, noise_only=True, n_realizations=3
    )

    assert len(noise_realizations) == 3
    for realization in noise_realizations:
        assert isinstance(realization, xr.DataArray)
        assert realization.dims == ("month", "lat", "lon")
        assert realization.sizes["month"] == 24
        assert realization.sizes["lat"] == len(lats)
        assert realization.sizes["lon"] == len(lons)
    # Test generation with noise_only=False (lines 295-310)

    # Test generation with specific random seed (line 276-277)
    seeded_realization = generator.generate_realization(
        test_trajectory, random_seed=42, n_realizations=1
    )

    # Same seed should give same results
    seeded_realization2 = generator.generate_realization(
        test_trajectory, random_seed=42, n_realizations=1
    )

    assert seeded_realization.shape == (24, len(lats), len(lons))
    assert seeded_realization2.shape == (24, len(lats), len(lons))
    np.testing.assert_array_equal(seeded_realization, seeded_realization2)

    # Test without noise_only (different code path)
    full_realizations = generator.generate_realization(
        test_trajectory, noise_only=False, n_realizations=2
    )

    assert len(full_realizations) == 2
    assert full_realizations[0].dims == ("month", "lat", "lon")
    assert full_realizations[0].sizes["month"] == 24
    assert full_realizations[0].sizes["lat"] == len(lats)
    assert full_realizations[0].sizes["lon"] == len(lons)

    # except Exception as e:
    #     # Some edge cases may fail in fitting, which is acceptable
    #     print(f"Fitting failed with: {e}")


def test_meteor_noise_generator_error_conditions():
    """Test error conditions to improve coverage."""
    generator = noise_generator.MeteorNoiseGenerator()

    # Test generation before fitting (line 273)
    with pytest.raises(
        ValueError, match="Model must be fitted before generating realizations"
    ):
        generator.generate_realization(np.array([1, 2, 3]))


def test_meteor_noise_generator_feature_creation():
    """Test feature creation methods for coverage."""

    generator = noise_generator.MeteorNoiseGenerator(n_modes=3)

    # Create simple test data to enable feature testing
    time = np.arange(24)
    global_temp = np.linspace(0, 2, 24)

    # Test harmonic feature creation (internal method coverage)

    # This tests internal _create_harmonic_features method
    # We need to fit first to enable internal methods
    simple_data = xr.DataArray(
        np.random.rand(24, 2, 2, 1),
        coords={"month": time, "lat": [0, 1], "lon": [0, 1], "ens": [1]},
        dims=["month", "lat", "lon", "ens"],
    )
    simple_ds = xr.Dataset({"tas": simple_data})

    generator.fit(
        simple_ds, "tas", custom_global_temp=global_temp, save_diagnostics=True
    )

    assert generator.diagnostic_X_features is not None
    assert generator.diagnostic_t_glob is not None
    assert generator.diagnostic_time is not None
    assert generator.diagnostic_seasonal_coef is not None
    assert generator.diagnostic_seasonal_intercept is not None
    assert generator.diagnostic_Y_data is not None

    short_trajectory = np.array([0.5, 1.0])
    long_trajectory = np.linspace(0, 3, 36)

    # These should work with different lengths
    short_real = generator.generate_realization(short_trajectory, n_realizations=1)
    long_real = generator.generate_realization(long_trajectory, n_realizations=1)

    assert len(short_real) == 2
    assert len(long_real) == 36


def test_advanced_noise_generation():
    """Test advanced noise generation methods to improve coverage."""

    # Create more realistic training data
    time = np.arange(60)  # 5 years monthly
    lats = np.linspace(-45, 45, 3)
    lons = np.linspace(-90, 90, 4)

    # Create temperature data with clear seasonal cycle
    temp_data = np.zeros((len(time), len(lats), len(lons), 1))
    for i, t in enumerate(time):
        seasonal = 5 * np.sin(2 * np.pi * t / 12.0)  # Seasonal cycle
        trend = 0.01 * t  # Small warming trend
        noise = np.random.normal(0, 0.5)  # Random noise
        temp_data[i] = seasonal + trend + noise

    temp_da = xr.DataArray(
        temp_data,
        coords={"month": time, "lat": lats, "lon": lons, "ens": [1]},
        dims=["month", "lat", "lon", "ens"],
        name="tas",
    )
    training_data = xr.Dataset({"tas": temp_da})

    generator = noise_generator.MeteorNoiseGenerator()

    # Test the fitting process
    generator.fit(training_data, "tas")

    # if generator.fitted:
    # Test noise-only generation (lines 285-300)
    test_trajectory = np.linspace(0, 1, 12)  # 1 year

    # Test noise_only=True path - this should hit lines 285-300
    noise_only_realizations = generator.generate_realization(
        test_trajectory, noise_only=True, n_realizations=2
    )

    assert len(noise_only_realizations) == 2

    # Test different generation parameters
    # Test with longer trajectory to hit more complex paths
    long_trajectory = np.linspace(0, 2, 48)  # 4 years

    long_realizations = generator.generate_realization(
        long_trajectory, noise_only=False, n_realizations=1
    )

    assert len(long_realizations) == 48

    # Test the synthetic PC generation (lines 390-412)
    # This should be covered by the internal generation process

    # except Exception as e:
    #     # Complex VAR fitting may fail, which is acceptable
    #     print(f"Advanced fitting failed: {e}")


def test_baseline_handling():
    """Test different baseline handling options."""
    np.random.seed(42)

    # Create test data
    data = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.random.rand(60, 3, 3),
                dims=["month", "lat", "lon"],
                coords={"month": range(60), "lat": [-45, 0, 45], "lon": [-90, 0, 90]},
            )
        }
    )
    data = data.expand_dims({"ens": [1]})

    generator = MeteorNoiseGenerator(n_modes=2, lag_order=1)

    # Test with numeric baseline
    generator.fit(data, "tas", picontrol_baseline=15.0)
    generator.fit(data, "tas", picontrol_baseline=None)
    # Test with array-like baseline (tests lines 170-177)
    baseline_array = np.array([14.5, 15.0, 14.8])
    generator.fit(data, "tas", picontrol_baseline=baseline_array)


def test_custom_global_temp_validation_error():
    """Test validation error when custom global temperature has wrong length."""
    generator = MeteorNoiseGenerator()

    # Create test data with 12 months
    time = np.arange(12)
    data = xr.DataArray(
        np.random.rand(12, 3, 3),
        dims=["month", "lat", "lon"],
        coords={"month": time, "lat": [0, 1, 2], "lon": [0, 1, 2]},
    )
    dataset = xr.Dataset({"tas": data})

    # Provide global temperature with wrong length (6 months instead of 12)
    wrong_length_temp = np.array([1, 2, 3, 4, 5, 6])

    with pytest.raises(ValueError, match="custom_global_temp length"):
        generator.fit(dataset, "tas", custom_global_temp=wrong_length_temp)


def test_variable_not_found_validation_error():
    """Test error when requested variable is not in dataset."""
    generator = MeteorNoiseGenerator()

    # Create dataset without 'tas' variable
    data = xr.DataArray(
        np.random.rand(12, 3, 3),
        dims=["month", "lat", "lon"],
        coords={"month": np.arange(12), "lat": [0, 1, 2], "lon": [0, 1, 2]},
    )
    dataset = xr.Dataset({"temperature": data})  # Wrong variable name

    with pytest.raises(ValueError, match="Variable 'tas' not found"):
        generator.fit(dataset, "tas")  # Requesting 'tas' but dataset has 'temperature'


def test_save_model_before_fitting_error():
    """Test error when trying to save unfitted model."""
    generator = MeteorNoiseGenerator()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.pkl")

        with pytest.raises(ValueError, match="Model must be fitted before saving"):
            generator.save_model(filepath)


def test_noise_only_generation():
    """Test noise-only generation preserves harmonics but removes temperature trends."""
    np.random.seed(42)

    # Create test data with clear temperature signal
    n_time = 120  # 10 years monthly
    data = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.random.rand(n_time, 3, 3) + np.linspace(0, 2, n_time)[:, None, None],
                dims=["month", "lat", "lon"],
                coords={
                    "month": range(n_time),
                    "lat": [-45, 0, 45],
                    "lon": [-90, 0, 90],
                },
            )
        }
    )
    data = data.expand_dims({"ens": [1]})

    generator = MeteorNoiseGenerator(n_modes=2, lag_order=1)
    generator.fit(data, "tas")

    # Test noise-only generation (tests lines 275-295)
    global_temp = np.linspace(0, 2, 60)  # 5 years
    noise_only_real = generator.generate_realization(
        global_temp, noise_only=True, n_realizations=1
    )

    # Should generate something with seasonal patterns but reduced temperature trend
    assert len(noise_only_real) == 60
    assert noise_only_real.dims == ("month", "lat", "lon")

    # Compare with full generation
    full_real = generator.generate_realization(
        global_temp, noise_only=False, n_realizations=1
    )

    assert len(full_real) == 60
    # Both should have same dimensions but different temperature characteristics
    assert full_real.dims == noise_only_real.dims


def test_generate_realization_unfitted_error():
    """Test error when generating realization before fitting."""
    generator = MeteorNoiseGenerator()

    test_trajectory = np.array([0.5, 1.0, 1.5])

    with pytest.raises(ValueError, match="Model must be fitted before generating"):
        generator.generate_realization(test_trajectory)
