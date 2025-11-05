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
import xarray as xr

from meteor.cmip6_meteor_data_getter import Cmip6MeteorDataGetter


class TestCacheOptimizationFunctional:
    """Tests for actual functional behavior of cache optimization."""

    def setup_method(self):
        """Set up test environment with temporary cache directory."""
        self.temp_cache_dir = tempfile.mkdtemp()

        # Create a minimal data getter for testing
        self.data_getter = Cmip6MeteorDataGetter(
            exps=["piControl"],
            flds=["tas"],
            cache_dir=self.temp_cache_dir,
            enable_cache=True,  # Enable cache for testing
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
            exps=["piControl"],
            flds=["tas"],
            cache_dir=temp_dir,
            enable_cache=True,  # Enable cache for testing
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
