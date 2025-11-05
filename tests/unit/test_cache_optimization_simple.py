"""
Focused cache optimization tests for CMIP6 METEOR data getter.

These tests focus on actual functional behaviors rather than basic Python logic:
- Cache corruption detection and recovery
- Cache validation with realistic scenarios
- Integration workflow testing
"""

import os
import tempfile
import numpy as np
import pytest
import xarray as xr

from meteor.cmip6_meteor_data_getter import Cmip6MeteorDataGetter


class TestCacheOptimizationFunctional:
    """Tests for actual functional behavior of cache optimization."""

    def setup_method(self):
        """Set up test environment with temporary cache directory."""
        self.temp_cache_dir = tempfile.mkdtemp()

        # Create a minimal data getter for testing
        self.data_getter = Cmip6MeteorDataGetter(
            exps=["piControl"], flds=["tas"], cache_dir=self.temp_cache_dir
        )

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_cache_dir, ignore_errors=True)

    def test_cache_corruption_detection_and_cleanup(self):
        """Test that the data getter detects and removes corrupted cache files."""

        # Create a corrupted cache file (empty dataset)
        cache_file = os.path.join(
            self.temp_cache_dir, "CanESM5_piControl_tas_monthly.nc"
        )
        corrupted_data = xr.Dataset()  # Empty dataset - realistic corruption scenario
        corrupted_data.to_netcdf(cache_file)

        # Verify file exists but is corrupted
        assert os.path.exists(cache_file)

        # The key test: does the data getter detect corruption and clean it up?
        is_cached = self.data_getter.is_cached(
            "get_single_var_mod_data_monthly", "piControl", "tas", "CanESM5"
        )

        # Should detect corruption and remove file
        assert not is_cached
        assert not os.path.exists(cache_file), "Corrupted cache file should be removed"

    def test_valid_cache_file_acceptance(self):
        """Test that the data getter accepts valid cache files."""

        # Create a valid cache file with realistic structure
        valid_data = xr.Dataset(
            {
                "tas": xr.DataArray(
                    np.random.rand(1, 600, 10, 10),  # Realistic monthly data shape
                    dims=["ens", "month", "lat", "lon"],
                    coords={
                        "ens": [1],
                        "month": range(600),
                        "lat": range(10),
                        "lon": range(10),
                    },
                )
            }
        )
        valid_data.attrs["original_type"] = "DataArray"
        valid_data.attrs["cached_by_meteor"] = "true"

        cache_file = os.path.join(
            self.temp_cache_dir, "CanESM5_piControl_tas_monthly.nc"
        )
        valid_data.to_netcdf(cache_file)

        # The key test: does the data getter accept valid files?
        is_cached = self.data_getter.is_cached(
            "get_single_var_mod_data_monthly", "piControl", "tas", "CanESM5"
        )

        assert is_cached
        assert os.path.exists(cache_file), "Valid cache file should remain"

    def test_cache_validation_rejects_wrong_variable(self):
        """Test that cache validation rejects files with wrong variable data."""

        # Create cache file with wrong variable (pr instead of tas)
        wrong_var_data = xr.Dataset(
            {
                "pr": xr.DataArray(
                    np.random.rand(1, 600, 10, 10),
                    dims=["ens", "month", "lat", "lon"],
                    coords={
                        "ens": [1],
                        "month": range(600),
                        "lat": range(10),
                        "lon": range(10),
                    },
                )
            }
        )
        wrong_var_data.attrs["original_type"] = "DataArray"
        wrong_var_data.attrs["cached_by_meteor"] = "true"

        cache_file = os.path.join(
            self.temp_cache_dir, "CanESM5_piControl_tas_monthly.nc"
        )
        wrong_var_data.to_netcdf(cache_file)

        # Should reject file with wrong variable
        is_cached = self.data_getter.is_cached(
            "get_single_var_mod_data_monthly", "piControl", "tas", "CanESM5"
        )

        assert not is_cached


def test_integrated_cache_workflow():
    """Integration test for complete cache workflow."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create data getter with temporary cache
        data_getter = Cmip6MeteorDataGetter(
            exps=["piControl"], flds=["tas"], cache_dir=temp_dir
        )

        # Create a realistic mock cache file
        mock_monthly_data = xr.Dataset(
            {
                "tas": xr.DataArray(
                    np.random.rand(1, 600, 8, 8),  # 50 years of monthly data
                    dims=["ens", "month", "lat", "lon"],
                    coords={
                        "ens": [1],
                        "month": range(600),
                        "lat": range(8),
                        "lon": range(8),
                    },
                )
            }
        )
        mock_monthly_data.attrs["original_type"] = "DataArray"
        mock_monthly_data.attrs["cached_by_meteor"] = "true"

        cache_file = os.path.join(temp_dir, "CanESM5_piControl_tas_monthly.nc")
        mock_monthly_data.to_netcdf(cache_file)

        # Test cache detection
        is_cached = data_getter.is_cached(
            "get_single_var_mod_data_monthly", "piControl", "tas", "CanESM5"
        )
        assert is_cached

        # Test cache loading
        cached_data = data_getter._load_from_cache(
            "CanESM5_piControl_tas_monthly",
            expected_type="DataArray",
            expected_variable="tas",
        )
        assert cached_data is not None
        assert isinstance(cached_data, xr.DataArray)
        assert cached_data.name == "tas"

        # Verify optimization: only one cache file exists (monthly only)
        cache_files = [f for f in os.listdir(temp_dir) if f.endswith(".nc")]
        assert len(cache_files) == 1
        assert "monthly" in cache_files[0]


def test_compression_option():
    """Test that compression option works correctly."""

    # Create test data (doesn't need to be optimally compressible for this test)
    test_data = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.random.rand(1, 12, 2, 2),  # Small but realistic data
                dims=["ens", "month", "lat", "lon"],
                coords={
                    "ens": [1],
                    "month": range(12),
                    "lat": [45.0, 46.0],
                    "lon": [0.0, 1.0],
                },
            )
        }
    )
    test_data.attrs["original_type"] = "DataArray"
    test_data.attrs["cached_by_meteor"] = "true"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with compression enabled
        data_getter_compressed = Cmip6MeteorDataGetter(
            exps=["piControl"],
            flds=["tas"],
            cache_dir=temp_dir,
            enable_compression=True,
        )

        compressed_file = os.path.join(temp_dir, "test_compressed.nc")
        data_getter_compressed._save_to_cache("test_compressed", test_data["tas"])

        # Test with compression disabled
        data_getter_uncompressed = Cmip6MeteorDataGetter(
            exps=["piControl"],
            flds=["tas"],
            cache_dir=temp_dir,
            enable_compression=False,
        )

        uncompressed_file = os.path.join(temp_dir, "test_uncompressed.nc")
        data_getter_uncompressed._save_to_cache("test_uncompressed", test_data["tas"])

        # Verify both files exist and contain identical data
        assert os.path.exists(compressed_file)
        assert os.path.exists(uncompressed_file)

        compressed_data = xr.open_dataset(compressed_file)
        uncompressed_data = xr.open_dataset(uncompressed_file)

        assert np.allclose(
            compressed_data["tas"].values, uncompressed_data["tas"].values
        )

        # Verify compression setting is stored correctly
        assert data_getter_compressed.enable_compression == True
        assert data_getter_uncompressed.enable_compression == False


def test_compression_level_parameter():
    """Test that compression level parameter is properly stored and clamped."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test valid compression level
        data_getter = Cmip6MeteorDataGetter(
            exps=["piControl"], flds=["tas"], cache_dir=temp_dir, compression_level=3
        )
        assert data_getter.compression_level == 3

        # Test compression level clamping (too low)
        data_getter_low = Cmip6MeteorDataGetter(
            exps=["piControl"], flds=["tas"], cache_dir=temp_dir, compression_level=0
        )
        assert data_getter_low.compression_level == 1  # Should be clamped to minimum

        # Test compression level clamping (too high)
        data_getter_high = Cmip6MeteorDataGetter(
            exps=["piControl"], flds=["tas"], cache_dir=temp_dir, compression_level=15
        )
        assert data_getter_high.compression_level == 9  # Should be clamped to maximum

        # Test default compression level
        data_getter_default = Cmip6MeteorDataGetter(
            exps=["piControl"], flds=["tas"], cache_dir=temp_dir
        )
        assert data_getter_default.compression_level == 6  # Default should be 6


def test_storage_optimization_only_monthly_cached():
    """Test that only monthly files are cached, not redundant _raw/_yearly/_training files."""

    with tempfile.TemporaryDirectory() as temp_dir:
        data_getter = Cmip6MeteorDataGetter(
            exps=["piControl"], flds=["tas"], cache_dir=temp_dir
        )

        # Generate cache keys for different data types
        monthly_key = data_getter._generate_cache_key(
            "get_single_var_mod_data_monthly", "piControl", "tas", "CanESM5"
        )

        # The key test: ensure only monthly data is being cached
        assert "monthly" in monthly_key

        # Verify no legacy patterns exist
        for legacy_pattern in ["_raw", "_yearly", "_training"]:
            assert legacy_pattern not in monthly_key
