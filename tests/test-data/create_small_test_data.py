#!/usr/bin/env python3
"""
Create small, low-resolution test data files for faster testing.
This script takes the existing test data and creates much smaller versions
by coarsening the spatial resolution while preserving the temporal structure.
"""

import xarray as xr
import numpy as np
import os


def create_small_test_data():
    """Create small test data files by coarsening the spatial resolution of real PDRMIP data."""

    # Load original test data
    print("Loading original test data...")
    base_ds = xr.open_dataset("pdrmip-base_T42_ANN.nc")
    co2x2_ds = xr.open_dataset("pdrmip-co2x2_T42_ANN.nc")

    print(f"Original base file size: {base_ds.nbytes / 1024**2:.1f} MB")
    print(f"Original co2x2 file size: {co2x2_ds.nbytes / 1024**2:.1f} MB")
    print(f"Original dimensions - Base: {base_ds.dims}")
    print(f"Original dimensions - Co2x2: {co2x2_ds.dims}")

    # Define coarsening factors to reduce spatial resolution dramatically
    # Original: 64 lat x 128 lon = 8,192 grid cells
    # Target: ~8 lat x 12 lon = 96 grid cells (85x reduction)
    lat_coarsen = 8  # 64 -> 8 latitude points
    lon_coarsen = 11  # 128 -> ~12 longitude points

    print(f"Coarsening by factors: lat={lat_coarsen}, lon={lon_coarsen}")

    # Coarsen the base dataset
    print("Coarsening base dataset...")
    base_small = base_ds.coarsen(
        lat=lat_coarsen, lon=lon_coarsen, boundary="trim"
    ).mean()

    # Coarsen the co2x2 dataset
    print("Coarsening co2x2 dataset...")
    co2x2_small = co2x2_ds.coarsen(
        lat=lat_coarsen, lon=lon_coarsen, boundary="trim"
    ).mean()

    # Create output directory if it doesn't exist
    os.makedirs("small", exist_ok=True)

    # Save small files
    output_base = "small/pdrmip-base_small.nc"
    output_co2x2 = "small/pdrmip-co2x2_small.nc"

    print(f"Saving coarsened base dataset to {output_base}...")
    base_small.to_netcdf(output_base)

    print(f"Saving coarsened co2x2 dataset to {output_co2x2}...")
    co2x2_small.to_netcdf(output_co2x2)

    # Create a composite file for the sulfate test by combining aspects of both datasets
    print("Creating coarsened composite dataset...")
    # Use base data structure but add some signal from co2x2 to simulate anomaly
    composite_small = base_small.copy()

    # Create anomaly-like signal by adding differences between co2x2 and base
    # This preserves realistic climate relationships
    time_overlap = min(len(base_small.year), len(co2x2_small.year))

    # Add temperature anomaly signal (use only overlapping time period to avoid NaNs)
    temp_diff = co2x2_small.tas[:time_overlap] - base_small.tas[:time_overlap]
    composite_small["tas"][:time_overlap] = (
        base_small.tas[:time_overlap] + 0.5 * temp_diff
    )

    # Add precipitation anomaly signal (use only overlapping time period to avoid NaNs)
    precip_ratio = co2x2_small.pr[:time_overlap] / base_small.pr[:time_overlap]
    # Avoid division by zero and invalid operations
    precip_ratio = xr.where(
        (base_small.pr[:time_overlap] > 0) & np.isfinite(precip_ratio),
        precip_ratio,
        1.0,
    )
    composite_small["pr"][:time_overlap] = base_small.pr[:time_overlap] * (
        1.0 + 0.3 * (precip_ratio - 1.0)
    )

    # Ensure no NaN values in the composite dataset
    composite_small = composite_small.fillna(0.0)

    output_composite = "small/pdrmip-composite_small.nc"
    print(f"Saving coarsened composite dataset to {output_composite}...")
    composite_small.to_netcdf(output_composite)

    # Print size comparison
    print(f"\nSize comparison:")
    print(
        f"Original base: {base_ds.nbytes / 1024**2:.1f} MB -> Small: {base_small.nbytes / 1024:.1f} KB"
    )
    print(
        f"Original co2x2: {co2x2_ds.nbytes / 1024**2:.1f} MB -> Small: {co2x2_small.nbytes / 1024:.1f} KB"
    )
    print(f"Reduction factor: ~{base_ds.nbytes / base_small.nbytes:.0f}x")

    print("\nTest data validation:")
    # Quick validation
    test_base = xr.open_dataset(output_base)
    test_co2x2 = xr.open_dataset(output_co2x2)
    test_composite = xr.open_dataset(output_composite)

    print(f"Base dims: {test_base.dims}")
    print(f"Co2x2 dims: {test_co2x2.dims}")
    print(f"Composite dims: {test_composite.dims}")
    print(f"Variables: {list(test_base.data_vars)}")
    print(f"Coordinates: {list(test_base.coords)}")

    # Check that temporal structure is preserved
    print(f"\nTemporal structure preserved:")
    print(
        f"Base years: {test_base.year.values[0]} to {test_base.year.values[-1]} ({len(test_base.year)} years)"
    )
    print(
        f"Co2x2 years: {test_co2x2.year.values[0]} to {test_co2x2.year.values[-1]} ({len(test_co2x2.year)} years)"
    )
    print(
        f"Composite years: {test_composite.year.values[0]} to {test_composite.year.values[-1]} ({len(test_composite.year)} years)"
    )

    # Check spatial reduction
    print(f"\nSpatial reduction:")
    print(
        f"Original spatial grid: {len(base_ds.lat)} x {len(base_ds.lon)} = {len(base_ds.lat) * len(base_ds.lon)} points"
    )
    print(
        f"Coarsened spatial grid: {len(test_base.lat)} x {len(test_base.lon)} = {len(test_base.lat) * len(test_base.lon)} points"
    )
    print(
        f"Spatial reduction factor: {(len(base_ds.lat) * len(base_ds.lon)) / (len(test_base.lat) * len(test_base.lon)):.0f}x"
    )

    print("\nCoarsened test data files created successfully!")
    print("These files preserve temporal structure and realistic climate relationships")
    print("while dramatically reducing computational requirements.")

    return output_base, output_co2x2, output_composite


if __name__ == "__main__":
    create_small_test_data()
