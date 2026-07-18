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
    assert generator.n_modes == 40  # default value
    assert generator.lag_order == 2  # default value
    assert generator.weight_eofs is True  # area-weighting on by default
    # pure VAR by default: t_glob exog whitens the noise and collapses
    # low-frequency global variability
    assert MeteorNoiseGenerator().use_exog == "none"

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


def test_meteor_noise_generator_save_load_errors():
    """Test save and load functionality."""
    generator = MeteorNoiseGenerator()
    generator.coords = {"lat": np.array([0, 1]), "lon": "Hello"}
    with pytest.raises(
        ValueError,
        match="Longitude coordinates must be NumPy arrays or xarray.DataArray",
    ):
        generator._fix_coords_to_np()
    generator.coords = {"lat": "Hello", "lon": np.array([10, 20])}
    with pytest.raises(
        ValueError,
        match="Latitude coordinates must be NumPy arrays or xarray.DataArray",
    ):
        generator._fix_coords_to_np()

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

    # Test with complex datasets (small 5x6 grid -> keep n_modes <= n_gridcells)
    generator = noise_generator.MeteorNoiseGenerator(n_modes=3)

    # Test fitting with multiple variables
    training_data = xr.Dataset({"tas": temp_da, "pr": precip_da})

    generator.fit(training_data, "tas")
    assert generator.fitted

    # Test generation with noise_only=True (lines 275-290)
    test_trajectory = np.linspace(0, 2, 24)  # 2 years

    with pytest.raises(ValueError, match="AR6 region 'XYZ' not found"):
        generator.generate_regional_mean_realizations(
            test_trajectory, n_realizations=1, region="XYZ"
        )
    with pytest.raises(
        ValueError,
        match="Both lat and lon must be provided together for point extraction",
    ):
        generator.generate_regional_mean_realizations(
            test_trajectory, n_realizations=1, lat=10
        )
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

    stoch_real = generator.generate_stochastic_pcs(
        test_trajectory, n_realizations=1, random_seed=47
    )
    assert stoch_real.shape == (len(test_trajectory), generator.n_modes)
    stoch_real2 = generator.generate_stochastic_pcs(
        test_trajectory, n_realizations=1, random_seed=47
    )
    np.testing.assert_array_equal(stoch_real, stoch_real2)
    stoch_real3 = generator.generate_stochastic_pcs(
        test_trajectory, n_realizations=1, random_seed=48
    )
    assert not np.array_equal(stoch_real, stoch_real3)
    # except Exception as e:
    #     # Some edge cases may fail in fitting, which is acceptable
    #     print(f"Fitting failed with: {e}")
    point_real = generator.generate_regional_mean_realizations(
        test_trajectory, lat=60, lon=30, n_realizations=1
    )
    assert point_real.shape == (len(test_trajectory),)


def _build_small_fitted_generator(n_modes=3, n_years=25, n_lat=5, n_lon=6, seed=0):
    """Build a MeteorNoiseGenerator fitted on tiny synthetic data.

    Returns the fitted generator plus a temperature trajectory of the same
    length as the fitted time dimension so tests can call `generate_*`
    methods without shape mismatches.
    """
    rng = np.random.default_rng(seed)
    n_time = n_years * 12
    lats = np.linspace(-80, 80, n_lat)
    lons = np.linspace(-170, 170, n_lon)
    ens = np.array([1])

    time = np.arange(n_time)
    seasonal = 5.0 * np.sin(2 * np.pi * time / 12.0)
    trend = 0.005 * time
    global_signal = seasonal + trend  # shape (n_time,)
    spatial = 0.1 * lats[None, :, None] + 0.05 * lons[None, None, :]
    tas_data = (
        global_signal[:, None, None]
        + spatial
        + 0.5 * rng.standard_normal((n_time, n_lat, n_lon))
    )[..., None]  # add ens axis

    tas_da = xr.DataArray(
        tas_data,
        coords={"month": time, "lat": lats, "lon": lons, "ens": ens},
        dims=["month", "lat", "lon", "ens"],
        name="tas",
    )
    ds = xr.Dataset({"tas": tas_da})

    generator = noise_generator.MeteorNoiseGenerator(n_modes=n_modes)
    generator.fit(ds, "tas")
    trajectory = global_signal
    return generator, trajectory


def test_generate_stochastic_pcs_batched_matches_sequential():
    """Batched multi-realization VAR simulation must be statistically
    equivalent to running the single-realization loop N times.

    Because the batched path draws (N, T, n_modes) standard normals in one
    call while the sequential path draws (T, n_modes) per realization, the
    two paths consume identical amounts of random state in the same order;
    outputs are therefore expected to agree to machine precision, not merely
    in statistical moments.
    """
    generator, trajectory = _build_small_fitted_generator(n_modes=3, n_years=25)
    n_realizations = 20

    # Sequential: repeat the private single-realization path with a fresh
    # seed matched to the batched call below.
    np.random.seed(1234)
    time = np.arange(len(trajectory))
    X = generator._create_harmonic_features(time, trajectory)
    X_exog = generator._extract_exog_variables(X)
    n_time = len(trajectory)
    sequential = np.stack(
        [generator._generate_stochastic_pcs(X_exog, n_time) for _ in range(n_realizations)]
    )

    # Batched
    np.random.seed(1234)
    batched = generator._generate_stochastic_pcs_batched(
        X_exog, n_time, n_realizations
    )

    assert batched.shape == (n_realizations, n_time, generator.n_modes)
    assert sequential.shape == batched.shape

    # Machine-precision agreement across realizations
    np.testing.assert_allclose(
        batched, sequential, rtol=0, atol=1e-10,
        err_msg="Batched VAR sim diverged from sequential path.",
    )


def test_generate_stochastic_pcs_public_batched_route():
    """The public `generate_stochastic_pcs` should route n_realizations>1
    through the batched path and return (n_realizations, n_time, n_modes).
    """
    generator, trajectory = _build_small_fitted_generator(n_modes=3, n_years=25)
    out = generator.generate_stochastic_pcs(
        trajectory, n_realizations=5, random_seed=7
    )
    assert out.shape == (5, len(trajectory), generator.n_modes)

    # Single-realization path unchanged: 2-D output, reproducible under seed.
    a = generator.generate_stochastic_pcs(trajectory, n_realizations=1, random_seed=7)
    b = generator.generate_stochastic_pcs(trajectory, n_realizations=1, random_seed=7)
    assert a.shape == (len(trajectory), generator.n_modes)
    np.testing.assert_array_equal(a, b)


def test_meteor_noise_generator_error_conditions():
    """Test error conditions to improve coverage."""
    generator = noise_generator.MeteorNoiseGenerator()

    # Test generation before fitting (line 273)
    with pytest.raises(
        ValueError, match="Model must be fitted before generating realizations"
    ):
        generator.generate_realization(np.array([1, 2, 3]))
    with pytest.raises(
        ValueError, match="Model must be fitted before generating realizations"
    ):
        generator.generate_regional_mean_realizations(np.array([1, 2, 3]))


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

    assert generator.diagnostics["X_features"] is not None
    assert generator.diagnostics["t_glob"] is not None
    assert generator.diagnostics["time"] is not None
    assert generator.diagnostics["seasonal_coef"] is not None
    assert generator.diagnostics["seasonal_intercept"] is not None
    assert generator.diagnostics["Y_data"] is not None

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

    # Small 3x4 grid -> keep n_modes <= n_gridcells
    generator = noise_generator.MeteorNoiseGenerator(n_modes=3)

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
    assert generator.diagnostics["X_features"] is None
    assert generator.diagnostics["t_glob"] is None
    assert generator.diagnostics["time"] is None
    assert generator.diagnostics["seasonal_coef"] is None
    assert generator.diagnostics["seasonal_intercept"] is None
    assert generator.diagnostics["Y_data"] is None
    assert generator.diagnostics["seasonal_r2"] is None
    assert generator.diagnostics["total_variance_explained"] is None

    # Test with numeric baseline
    generator.fit(
        data,
        "tas",
        picontrol_baseline=15.0,
    )
    assert generator.diagnostics["X_features"] is None
    assert generator.diagnostics["t_glob"] is None
    assert generator.diagnostics["time"] is None
    assert generator.diagnostics["seasonal_coef"] is None
    assert generator.diagnostics["seasonal_intercept"] is None
    assert generator.diagnostics["Y_data"] is None
    assert generator.diagnostics["seasonal_r2"] is not None
    assert generator.diagnostics["total_variance_explained"] is not None
    s_r2_1 = generator.diagnostics["seasonal_r2"]
    var_exp_1 = generator.diagnostics["total_variance_explained"]
    assert s_r2_1 is not None
    assert var_exp_1 is not None
    assert s_r2_1 > 0  # Should have some seasonal skill
    assert var_exp_1 > 0  # Should explain some variance
    assert var_exp_1 <= 1.0  # Variance explained should be between 0 and 1
    generator.fit(data, "tas", picontrol_baseline=None)
    s_r2_2 = generator.diagnostics["seasonal_r2"]
    var_exp_2 = generator.diagnostics["total_variance_explained"]
    assert np.allclose(s_r2_1, s_r2_2)  # Should be baseline-independent
    assert np.allclose(
        var_exp_1, var_exp_2
    )  # Should be different with different baselines
    # Test with array-like baseline (tests lines 170-177)
    baseline_array = np.array([14.5, 15.0, 14.8])
    generator.fit(data, "tas", picontrol_baseline=baseline_array)
    assert np.allclose(generator.diagnostics["seasonal_r2"], s_r2_1)
    assert np.allclose(generator.diagnostics["total_variance_explained"], var_exp_1)


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


def _make_training_dataset(nt=120, nla=8, nlo=12):
    """Small monthly dataset spanning the full pole-to-pole latitude range."""
    time = np.arange(nt)
    lats = np.linspace(-90, 90, nla)  # includes exact poles (cos lat == 0)
    lons = np.linspace(0, 360, nlo, endpoint=False)
    rng = np.random.default_rng(0)
    data = (
        rng.standard_normal((nt, nla, nlo, 1)) * 0.5
        + np.cos(np.deg2rad(lats))[None, :, None, None]
        * np.sin(time / 12.0)[:, None, None, None]
    )
    da = xr.DataArray(
        data,
        coords={"month": time, "lat": lats, "lon": lons, "ens": [0]},
        dims=["month", "lat", "lon", "ens"],
        name="tas",
    )
    return xr.Dataset({"tas": da})


def test_eof_area_weighting_default_and_physical_components():
    """Area weighting is on by default, sets weights, and reconstructs finite
    fields even at the poles (where cos(lat) == 0)."""
    ds = _make_training_dataset()
    gen = MeteorNoiseGenerator(n_modes=4, lag_order=1)  # weight_eofs defaults True
    assert gen.weight_eofs is True
    gen.fit(ds, "tas")

    n_space = ds.sizes["lat"] * ds.sizes["lon"]
    assert gen.eof_weights is not None
    assert gen.eof_weights.shape == (n_space,)
    assert np.all(gen.eof_weights > 0)  # floored, so poles are nonzero

    # physical components are the stored components un-weighted, and differ
    # from the raw (weighted-space) components
    phys = gen._physical_components()
    assert phys.shape == gen.pca.components_.shape
    assert not np.allclose(phys, gen.pca.components_)

    traj = np.linspace(0, 2, ds.sizes["month"])
    grid = gen.generate_realization(traj, n_realizations=1)
    assert np.isfinite(grid.values).all()  # no inf/nan from pole division
    glob = gen.generate_regional_mean_realizations(
        traj, region="global", n_realizations=2, return_numpy=True
    )
    assert np.isfinite(glob).all()
    pole = gen.generate_regional_mean_realizations(
        traj, lat=90.0, lon=0.0, n_realizations=1, return_numpy=True
    )
    assert np.isfinite(np.asarray(pole)).all()


def test_eof_weighting_disabled_matches_legacy():
    """weight_eofs=False keeps the basis in raw gridcell space (legacy behavior)."""
    ds = _make_training_dataset()
    gen = MeteorNoiseGenerator(n_modes=4, lag_order=1, weight_eofs=False)
    gen.fit(ds, "tas")
    assert gen.eof_weights is None
    assert np.allclose(gen._physical_components(), gen.pca.components_)


def test_eof_weights_survive_save_load(tmp_path):
    """Saving and loading preserves weighting state and reproduces output."""
    ds = _make_training_dataset()
    gen = MeteorNoiseGenerator(n_modes=4, lag_order=1)
    gen.fit(ds, "tas")

    fp = str(tmp_path / "model.pkl")
    gen.save_model(fp)
    loaded = MeteorNoiseGenerator()
    loaded.load_model(fp)
    assert loaded.weight_eofs is True
    assert np.allclose(loaded.eof_weights, gen.eof_weights)

    traj = np.linspace(0, 2, ds.sizes["month"])
    a = gen.generate_regional_mean_realizations(
        traj, region="global", n_realizations=1, random_seed=7, return_numpy=True
    )
    b = loaded.generate_regional_mean_realizations(
        traj, region="global", n_realizations=1, random_seed=7, return_numpy=True
    )
    assert np.allclose(a, b)
