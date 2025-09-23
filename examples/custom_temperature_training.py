#!/usr/bin/env python3
"""
Example demonstrating how to use custom global temperature timeseries for noise model training.

This example shows how to provide your own smoothed global mean temperature trajectory
instead of relying on automatic computation from the variable data.
"""

import numpy as np
import xarray as xr
from meteor.cmip6_meteor_data_getter import Cmip6MeteorDataGetter
from meteor.noise_generator import MeteorNoiseGenerator


def create_custom_temperature_series():
    """
    Create a custom smoothed global mean temperature timeseries.

    In practice, this could come from:
    - Different smoothing approaches (e.g., 20-year running mean vs 10-year)
    - External temperature datasets (observations, reanalysis)
    - Pre-processed temperature trajectories
    - Temperature series with specific climate sensitivity assumptions
    """
    # Example: Create a synthetic temperature trajectory
    # In reality, you would load this from your own data
    time = xr.cftime_range("1850-01", "2100-12", freq="YE", calendar="noleap")

    # Synthetic temperature: gradual warming with some variability
    years = np.arange(len(time))
    temp_trend = 0.01 * years  # Linear warming trend
    temp_variability = 0.2 * np.sin(2 * np.pi * years / 20)  # Multi-decadal oscillation
    temp_noise = 0.1 * np.random.randn(len(years))  # Random year-to-year variability

    temperature = (
        14.0 + temp_trend + temp_variability + temp_noise
    )  # Base temp + changes

    # Create xarray DataArray with proper time coordinate
    custom_temp = xr.DataArray(
        temperature,
        coords={"time": time},
        dims=["time"],
        name="global_mean_temperature",
        attrs={
            "units": "degC",
            "long_name": "Custom Global Mean Temperature",
            "description": "User-provided smoothed global mean temperature timeseries",
        },
    )

    return custom_temp


def example_with_data_getter():
    """Example using CMIP6MeteorDataGetter with custom temperature."""
    print("Example 1: Using CMIP6MeteorDataGetter with custom temperature")

    # Create custom temperature series
    custom_temp = create_custom_temperature_series()
    print(
        f"Custom temperature range: {custom_temp.min().values:.2f} to {custom_temp.max().values:.2f} °C"
    )

    # Initialize data getter (you would use real data source here)
    # data_getter = Cmip6MeteorDataGetter(data_source="your_data_path")

    # Train noise model with custom temperature
    # noise_model = data_getter.train_noise_model(
    #     experiments=["historical", "ssp245"],
    #     model="CESM2",
    #     variable_name="tas",
    #     custom_global_temp=custom_temp
    # )

    print("✓ Noise model would be trained using the custom temperature trajectory")
    print(
        "✓ The custom temperature will be used for temperature-dependent seasonal cycles"
    )


def example_direct_training():
    """Example using MeteorNoiseGenerator directly with custom temperature."""
    print("\nExample 2: Using MeteorNoiseGenerator directly with custom temperature")

    # Create custom temperature series
    custom_temp = create_custom_temperature_series()

    # You would load your actual climate data here
    # climate_data = xr.open_dataset("your_climate_data.nc")

    # Create noise generator and fit with custom temperature
    # noise_gen = MeteorNoiseGenerator()
    # noise_gen.fit(
    #     climate_data.tas,  # Your climate variable data
    #     custom_global_temp=custom_temp,
    #     n_modes=10,
    #     lag_order=2
    # )

    print("✓ Noise generator would be fitted using the custom temperature")
    print(
        "✓ Custom temperature enables different temperature-dependence than auto-computed"
    )


def example_use_cases():
    """Document different use cases for custom temperature."""
    print("\nCommon use cases for custom global temperature:")
    print("1. Different smoothing: Use 20-year vs 10-year running means")
    print("2. External datasets: Use observational or reanalysis temperature")
    print("3. Climate sensitivity: Apply specific warming assumptions")
    print("4. Consistency: Use same temperature across multiple variables/models")
    print("5. Preprocessing: Apply custom detrending or bias correction")
    print("6. Research scenarios: Test sensitivity to temperature trajectory")


if __name__ == "__main__":
    print("METEOR Custom Temperature Training Example")
    print("=" * 50)

    example_with_data_getter()
    example_direct_training()
    example_use_cases()

    print("\nNote: This example creates synthetic data for demonstration.")
    print("In practice, replace with your actual climate data and temperature series.")
