"""
Common test utilities for impact tests
=====================================

Shared test utilities and mock classes for impact testing.
"""

import xarray as xr

from meteor.impacts.impacts_core import ImpactCalculator, ImpactResult


class MockCalculator(ImpactCalculator):
    """Mock calculator for testing with configurable behavior."""

    def __init__(self, name="MockCalculator", track_calls=False):
        super().__init__(name)
        self.track_calls = track_calls
        self.call_count = 0

    def calculate(self, climate_data):
        if self.track_calls:
            self.call_count += 1

        # Provide multiple output variables for different test needs
        result_data = {
            "output": climate_data * 2,  # Simple transformation
            "squared": climate_data**2,  # For test_base.py compatibility
            "doubled": climate_data * 2,  # For test_base.py compatibility
        }

        if self.track_calls:
            result_data["constant"] = xr.full_like(climate_data, self.call_count)

        metadata = {"mock": True}
        if self.track_calls:
            metadata["call_number"] = self.call_count

        return ImpactResult(result_data, metadata, self.name)

    def validate_input(self, climate_data):
        if not isinstance(climate_data, xr.DataArray):
            raise ValueError("Input must be DataArray")
