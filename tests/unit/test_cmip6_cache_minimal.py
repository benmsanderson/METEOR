"""
Minimal high-quality tests for CMIP6 METEOR data getter caching functionality.

Focus: Test real caching behavior with minimal overhead.
Strategy: Use mocks and fixtures, avoid unnecessary data getter initialization.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import xarray as xr

from meteor.cmip6_meteor_data_getter import Cmip6MeteorDataGetter


@pytest.fixture
def temp_cache_dir():
    """Provide temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def cache_getter(temp_cache_dir):
    """Provide data getter with caching enabled."""
    return Cmip6MeteorDataGetter(
        flds=["tas"],
        exps=["piControl"],
        cache_dir=temp_cache_dir,
        enable_cache=True,
        enable_compression=True,
    )


class TestCacheCore:
    """Core caching functionality tests."""

    def test_cache_key_generation(self, cache_getter):
        """Test cache key generation for common scenarios."""
        # Test multiple key types in one test
        keys = {
            "raw": cache_getter._generate_cache_key(
                "get_single_var_mod_data", "piControl", "tas", "CanESM5"
            ),
            "monthly": cache_getter._generate_cache_key(
                "get_single_var_mod_data_monthly", "piControl", "tas", "CanESM5"
            ),
            "yearly": cache_getter._generate_cache_key(
                "get_single_var_mod_data_yearmean", "piControl", "tas", "CanESM5"
            ),
        }

        assert keys["raw"] == "CanESM5_piControl_tas_raw"
        assert keys["monthly"] == "CanESM5_piControl_tas_monthly"
        assert keys["yearly"] == "CanESM5_piControl_tas_yearly"

        # Ensure all keys are filesystem-safe
        for key in keys.values():
            assert all(c.isalnum() or c in "_-" for c in key)

    def test_cache_lifecycle(self, cache_getter, temp_cache_dir):
        """Test complete cache lifecycle: save, validate, load, clear."""
        # Create test data
        test_data = xr.Dataset(
            {
                "tas": xr.DataArray(
                    np.random.rand(50, 10, 10),
                    dims=["time", "lat", "lon"],
                    coords={"time": range(50)},
                )
            }
        )

        cache_key = "lifecycle_test"

        # 1. Save to cache
        cache_getter._save_to_cache(cache_key, test_data)
        cache_path = cache_getter._get_cache_path(cache_key)
        assert os.path.exists(cache_path)

        # 2. Validate
        cache_getter.is_cached(
            "get_single_var_mod_data_monthly", "piControl", "tas", "TestModel"
        )
        # Won't find this specific model, but validates the mechanism works

        # 3. Load
        loaded = cache_getter._load_from_cache(cache_key)
        assert loaded is not None
        assert "tas" in loaded.data_vars
        np.testing.assert_array_almost_equal(
            loaded["tas"].values, test_data["tas"].values
        )

        # 4. Clear cache
        cache_getter.clear_cache()
        assert not os.path.exists(cache_path)
        assert len(os.listdir(temp_cache_dir)) == 0

    def test_validation_rejects_corruption(self, cache_getter):
        """Test validation rejects various corruption types."""
        # Test multiple corruption scenarios efficiently
        scenarios = [
            ("empty", xr.Dataset(), False),
            (
                "wrong_var",
                xr.Dataset({"pr": xr.DataArray([1, 2, 3], dims=["time"])}),
                False,
            ),
            ("no_dims", xr.Dataset({"tas": xr.DataArray(42.0)}), False),
            (
                "valid",
                xr.Dataset(
                    {
                        "tas": xr.DataArray(
                            [1, 2, 3], dims=["time"], coords={"time": [0, 1, 2]}
                        )
                    }
                ),
                True,
            ),
        ]

        for name, ds, expected in scenarios:
            result = cache_getter._validate_cached_data(ds, expected_variable="tas")
            assert result == expected, f"Validation failed for scenario: {name}"

    def test_compression_effectiveness(self, cache_getter, temp_cache_dir):
        """Test that compression reduces file size."""
        # Create highly compressible data (repeated patterns)
        compressible_data = np.tile(np.arange(10), (200, 20, 1)).astype(np.float64)
        test_ds = xr.Dataset(
            {"tas": xr.DataArray(compressible_data, dims=["time", "lat", "lon"])}
        )

        cache_getter._save_to_cache("compression_test", test_ds)
        cache_path = cache_getter._get_cache_path("compression_test")
        compressed_size = os.path.getsize(cache_path)

        # Raw size: 200*20*10*8 = 320,000 bytes
        # With zlib compression on patterned data, should achieve >70% compression
        assert (
            compressed_size < 100_000
        ), "Compression should significantly reduce file size"


class TestCacheErrorHandling:
    """Test error handling and edge cases."""

    def test_corrupted_file_cleanup(self, cache_getter, temp_cache_dir):
        """Test that corrupted files are auto-cleaned."""
        cache_key = "corrupted"
        cache_path = cache_getter._get_cache_path(cache_key)

        # Create corrupted file
        with open(cache_path, "w") as f:
            f.write("not a netCDF file")

        # Load should return None and remove file
        loaded = cache_getter._load_from_cache(cache_key)
        assert loaded is None
        assert not os.path.exists(cache_path)

    def test_wrong_variable_rejection(self, cache_getter, temp_cache_dir):
        """Test that files with wrong variables are rejected."""
        # Create file with wrong variable
        wrong_ds = xr.Dataset(
            {
                "pr": xr.DataArray(
                    np.random.rand(10, 5, 5),
                    dims=["time", "lat", "lon"],
                    coords={"time": range(10)},
                )
            }
        )
        wrong_ds.attrs["cached_by_meteor"] = "true"

        cache_key = cache_getter._generate_cache_key(
            "get_single_var_mod_data_monthly", "piControl", "tas", "TestModel"
        )
        cache_path = cache_getter._get_cache_path(cache_key)
        wrong_ds.to_netcdf(cache_path)

        # is_cached should reject and clean up
        is_cached = cache_getter.is_cached(
            "get_single_var_mod_data_monthly", "piControl", "tas", "TestModel"
        )
        assert not is_cached
        assert not os.path.exists(cache_path)

    def test_cache_disabled_no_operations(self):
        """Test that disabled cache skips all operations."""
        getter = Cmip6MeteorDataGetter(
            flds=["tas"], exps=["piControl"], enable_cache=False
        )

        # All cache operations should return False/None immediately
        assert not getter.is_cached(
            "get_single_var_mod_data_monthly", "piControl", "tas", "Model"
        )
        assert getter._load_from_cache("any_key") is None


class TestCacheFeatures:
    """Test specific cache features."""

    def test_time_bnds_removal(self, cache_getter):
        """Test that problematic time_bnds variable is removed on save."""
        test_ds = xr.Dataset(
            {
                "tas": xr.DataArray(
                    np.random.rand(10, 5, 5),
                    dims=["time", "lat", "lon"],
                    coords={"time": range(10)},
                ),
                "time_bnds": xr.DataArray(np.random.rand(10, 2), dims=["time", "bnds"]),
            }
        )

        cache_getter._save_to_cache("time_bnds_test", test_ds)
        loaded = cache_getter._load_from_cache("time_bnds_test")

        assert "tas" in loaded.data_vars
        assert "time_bnds" not in loaded.variables

    def test_dataarray_conversion(self, cache_getter):
        """Test that DataArrays are properly saved and marked."""
        test_da = xr.DataArray(
            np.random.rand(10, 5, 5),
            dims=["time", "lat", "lon"],
            coords={"time": range(10)},
            name="tas",
        )

        cache_getter._save_to_cache("dataarray_test", test_da)
        cache_path = cache_getter._get_cache_path("dataarray_test")

        # Load raw to check metadata
        ds = xr.open_dataset(cache_path)
        assert ds.attrs["original_type"] == "DataArray"
        assert "tas" in ds.data_vars
        ds.close()
