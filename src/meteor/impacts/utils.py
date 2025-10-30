"""
Utility functions for METEOR-impacts
====================================

This module provides common utility functions used across different
impact calculators and processing workflows.
"""

import warnings
from typing import Union, Optional, Tuple
import numpy as np
import xarray as xr


def validate_temperature_data(
    data: xr.DataArray,
    expected_units: str = "celsius",
    temp_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Validate temperature data for impact calculations.

    Args:
        data: Temperature data to validate
        expected_units: Expected units ("celsius" or "kelvin")
        temp_range: Optional tuple of (min, max) expected temperatures

    Raises:
        ValueError: If data validation fails
        UserWarning: If data seems suspicious but might be valid
    """
    if not isinstance(data, xr.DataArray):
        raise ValueError("Temperature data must be an xarray.DataArray")

    temp_min = float(data.min())
    temp_max = float(data.max())

    if expected_units.lower() == "kelvin":
        if temp_range is None:
            temp_range = (150, 400)  # Reasonable range for Earth temperatures in Kelvin

        if temp_min < temp_range[0] or temp_max > temp_range[1]:
            warnings.warn(
                f"Temperature values ({temp_min:.1f}K to {temp_max:.1f}K) "
                f"outside expected range {temp_range}K for Earth's climate"
            )

    elif expected_units.lower() == "celsius":
        if temp_range is None:
            temp_range = (-80, 60)  # Reasonable range for Earth temperatures in Celsius

        if temp_min < temp_range[0] or temp_max > temp_range[1]:
            warnings.warn(
                f"Temperature values ({temp_min:.1f}°C to {temp_max:.1f}°C) "
                f"outside expected range {temp_range}°C for Earth's climate"
            )

    else:
        raise ValueError(f"Unknown temperature units: {expected_units}")


def convert_temperature_units(
    data: xr.DataArray, from_units: str, to_units: str
) -> xr.DataArray:
    """
    Convert temperature data between different units.

    Args:
        data: Temperature data to convert
        from_units: Source units ("celsius", "kelvin", "fahrenheit")
        to_units: Target units ("celsius", "kelvin", "fahrenheit")

    Returns:
        Converted temperature data

    Raises:
        ValueError: If units are not recognized
    """
    from_units = from_units.lower()
    to_units = to_units.lower()

    if from_units == to_units:
        return data.copy()

    # Convert to Celsius as intermediate step
    if from_units == "kelvin":
        celsius_data = data - 273.15
    elif from_units == "fahrenheit":
        celsius_data = (data - 32) * 5 / 9
    elif from_units == "celsius":
        celsius_data = data.copy()
    else:
        raise ValueError(f"Unknown source units: {from_units}")

    # Convert from Celsius to target units
    if to_units == "celsius":
        result = celsius_data
    elif to_units == "kelvin":
        result = celsius_data + 273.15
    elif to_units == "fahrenheit":
        result = celsius_data * 9 / 5 + 32
    else:
        raise ValueError(f"Unknown target units: {to_units}")

    # Update attributes
    result.attrs.update(data.attrs)
    result.attrs["units"] = to_units

    return result


def check_monthly_dimension(data: xr.DataArray, dim_name: str = "month") -> None:
    """
    Check that data has a properly structured monthly dimension.

    Args:
        data: Data to check
        dim_name: Name of the monthly dimension

    Raises:
        ValueError: If monthly dimension is missing or improperly structured
    """
    if dim_name not in data.dims:
        raise ValueError(f"Data must have a '{dim_name}' dimension")

    n_months = data.sizes[dim_name]

    if n_months < 12:
        warnings.warn(
            f"Monthly data contains only {n_months} months. "
            "Some calculations may be less accurate with incomplete years.",
            UserWarning,
        )


def ensure_spatial_coordinates(
    data: xr.DataArray, lat_name: Optional[str] = "lat", lon_name: Optional[str] = "lon"
) -> Tuple[str, str]:
    """
    Ensure data has spatial coordinates and return their names.

    Args:
        data: Data to check
        lat_name: Preferred name for latitude coordinate
        lon_name: Preferred name for longitude coordinate

    Returns:
        Tuple of (actual_lat_name, actual_lon_name)

    Raises:
        ValueError: If spatial coordinates cannot be found
    """
    # Common latitude coordinate names (filter out None)
    lat_variants = [
        name
        for name in [lat_name, "lat", "latitude", "y", "lat_rho"]
        if name is not None
    ]
    # Common longitude coordinate names (filter out None)
    lon_variants = [
        name
        for name in [lon_name, "lon", "longitude", "x", "lon_rho"]
        if name is not None
    ]

    actual_lat = None
    actual_lon = None

    # Find latitude coordinate
    for lat_var in lat_variants:
        if lat_var in data.coords or lat_var in data.dims:
            actual_lat = lat_var
            break

    # Find longitude coordinate
    for lon_var in lon_variants:
        if lon_var in data.coords or lon_var in data.dims:
            actual_lon = lon_var
            break

    if actual_lat is None:
        raise ValueError(f"Could not find latitude coordinate. Tried: {lat_variants}")

    if actual_lon is None:
        raise ValueError(f"Could not find longitude coordinate. Tried: {lon_variants}")

    return actual_lat, actual_lon


def calculate_global_mean(
    data: xr.DataArray,
    lat_name: Optional[str] = None,
    lon_name: Optional[str] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Calculate area-weighted global mean of spatial data.

    Args:
        data: Spatial data to average
        lat_name: Name of latitude coordinate (auto-detected if None)
        lon_name: Name of longitude coordinate (auto-detected if None)
        weights: Optional custom weights (if None, uses cosine of latitude)

    Returns:
        Global mean with spatial dimensions removed
    """
    # Auto-detect coordinate names if not provided
    if lat_name is None or lon_name is None:
        actual_lat, actual_lon = ensure_spatial_coordinates(data, lat_name, lon_name)
        if lat_name is None:
            lat_name = actual_lat
        if lon_name is None:
            lon_name = actual_lon

    if weights is None:
        # Use cosine of latitude as weights
        lat_rad = np.deg2rad(data[lat_name])
        weights = np.cos(lat_rad)

        # Broadcast weights to match data dimensions
        if lon_name in data.dims:
            weights = weights * xr.ones_like(data[lon_name])

    # Calculate weighted mean
    weighted_data = data * weights
    global_mean = weighted_data.sum(dim=[lat_name, lon_name]) / weights.sum()

    # Update attributes
    global_mean.attrs.update(data.attrs)
    global_mean.attrs["operation"] = "area_weighted_global_mean"

    return global_mean


def create_monthly_time_axis(
    start_year: int, n_months: int, dim_name: str = "month"
) -> xr.DataArray:
    """
    Create a monthly time axis for METEOR data.

    Args:
        start_year: Starting year (e.g., 1850)
        n_months: Number of months
        dim_name: Name of the dimension

    Returns:
        DataArray with monthly time coordinates
    """
    # Create month indices (0, 1, 2, ...)
    month_indices = np.arange(n_months)

    # Convert to decimal years (1850.0, 1850.083, 1850.167, ...)
    decimal_years = start_year + month_indices / 12.0

    return xr.DataArray(
        decimal_years,
        dims=[dim_name],
        coords={dim_name: month_indices},
        attrs={"long_name": "Time", "units": "decimal years", "start_year": start_year},
    )


def group_by_season(
    data: xr.DataArray, month_dim: str = "month", seasons: Optional[dict] = None
) -> xr.Dataset:
    """
    Group monthly data by seasons.

    Args:
        data: Monthly data to group
        month_dim: Name of the monthly dimension
        seasons: Custom season definitions (dict mapping season names to month lists)
                If None, uses standard meteorological seasons

    Returns:
        Dataset with seasonal means
    """
    if seasons is None:
        seasons = {
            "DJF": [11, 0, 1],  # Dec, Jan, Feb
            "MAM": [2, 3, 4],  # Mar, Apr, May
            "JJA": [5, 6, 7],  # Jun, Jul, Aug
            "SON": [8, 9, 10],  # Sep, Oct, Nov
        }

    check_monthly_dimension(data, month_dim)

    # Group by calendar month
    month_grouper = data[month_dim] % 12

    seasonal_data = {}

    for season_name, month_list in seasons.items():
        # Select months for this season
        season_mask = month_grouper.isin(month_list)
        season_data = data.where(season_mask, drop=True)

        # Calculate seasonal mean
        seasonal_data[season_name] = season_data.mean(dim=month_dim)

    return xr.Dataset(seasonal_data)
