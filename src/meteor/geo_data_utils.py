"""Utility functions for handling geographic data."""

import logging

import numpy as np
import regionmask
import xarray as xr

LOGGER = logging.getLogger(__name__)


def get_time_name(ds):
    """
    Get name of temporal dimension

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    str
        The name of the temporal dimension of the dataset,
        provided it's either time or year

    Raises
    ------
    RuntimeError
        If there is no dimension called time or year
        in the dataset
    """
    for time_name in ["time", "year", "month"]:
        if time_name in ds.coords:
            return time_name
    raise RuntimeError("Couldn't find a time coordinate")


def get_lat_name(ds):
    """
    Get name of latitude dimension

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str
        The name of the latitudinal dimension of the dataset,
        provided it's either lat or latitude

    Raises
    ------
    RuntimeError
        If there is no dimension called lat or latitude
        in the dataset
    """
    # Common latitude coordinate names
    lat_variants = ["lat", "latitude", "y", "lat_rho"]

    for lat_name in lat_variants:
        if lat_name in ds.coords or lat_name in ds.dims:
            return lat_name

    raise RuntimeError(f"Couldn't find a latitude coordinate. Tried: {lat_variants}")


def get_lon_name(ds):
    """
    Get name of longitude dimension

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str
        The name of the latitudinal dimension of the dataset,
        provided it's either lat or latitude

    Raises
    ------
    RuntimeError
        If there is no dimension called lat or latitude
        in the dataset
    """
    # Common latitude coordinate names
    lat_variants = ["lon", "longitude", "x", "lon_rho"]

    for lat_name in lat_variants:
        if lat_name in ds.coords or lat_name in ds.dims:
            return lat_name

    raise RuntimeError(f"Couldn't find a longitude coordinate. Tried: {lat_variants}")


def get_weights_for_ds(ds, lat_name=None, lon_name=None, weights=None):
    """
    Generate latitude-based weights for an xarray dataset or dataarray.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Data with lat/lon coordinates
    lat_name : str, optional
        Name of latitude coordinate (if None, auto-detected)
    lon_name : str, optional
        Name of longitude coordinate (if None, auto-detected)
    weights : xarray.DataArray, optional
        Custom weights for averaging (if None, uses cosine of latitude)

    Returns
    -------
    xarray.DataArray
        Weights for latitude dimension (and longitude if applicable)
    """
    if weights is None:
        # Get coordinate names if not provided
        if lat_name is None:
            lat_name = get_lat_name(ds)
        if lon_name is None and len(ds.shape) > 1:
            lon_name = get_lon_name(ds)
        # Create weights if not provided
        lat = ds[lat_name]
        weights = np.cos(np.deg2rad(lat))

        # Broadcast to longitude if needed
        if lon_name in ds.dims:
            weights = weights * xr.ones_like(ds[lon_name])
    return weights


def apply_weights_and_do_spatial_mean(
    dataset, weights, normalize_weights=True, lat_name=None, lon_name=None
):
    """
    Apply weights to dataset and compute spatial mean.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        Data to compute weighted mean over
    weights : xarray.DataArray
        Weights for averaging
    normalize_weights : bool, optional
        Whether to normalize weights by their mean (default True)
    lat_name : str, optional
        Name of latitude coordinate (if None, auto-detected)
    lon_name : str, optional
        Name of longitude coordinate (if None, auto-detected)

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Weighted mean with spatial dimensions removed
    """
    if lat_name is None:
        lat_name = get_lat_name(dataset)
    if lon_name is None:
        lon_name = get_lon_name(dataset)
    # Normalize weights
    if normalize_weights:
        weights = weights / weights.mean()

    # Calculate weighted mean
    spatial_dims = [lat_name, lon_name]
    weighted_data = dataset * weights
    result = weighted_data.mean(spatial_dims, skipna=True)
    return result


def global_mean(
    ds,
    weights=None,
    normalize_weights=True,
):
    """
    Calculate latitude weighted global mean of xarray dataset or dataarray.

    This is the universal global mean function for METEOR. It can handle both
    Datasets and DataArrays with flexible coordinate detection and weighting.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Data to compute global mean over
    weights : xarray.DataArray, optional
        Custom weights for averaging (if None, uses cosine of latitude)
    normalize_weights : bool, optional
        Whether to normalize weights by their mean (default True for backward compatibility)

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Global mean with spatial dimensions removed
    """
    # Create weights if not provided
    weights = get_weights_for_ds(ds, weights=weights)

    # Calculate weighted mean
    result = apply_weights_and_do_spatial_mean(
        ds, weights, normalize_weights=normalize_weights
    )

    # Update attributes if possible
    if hasattr(ds, "attrs"):
        result.attrs.update(ds.attrs)
    result.attrs["operation"] = "area_weighted_global_mean"

    return result


def create_region_mask(ds, bbox=None, mask=None):
    """
    Create a 2D boolean mask for a custom region.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Data with lat/lon coordinates
    bbox : dict, optional
        Bounding box: {'lat': (min, max), 'lon': (min, max)}
    mask : numpy.ndarray or xarray.DataArray, optional
        Pre-computed boolean mask

    Returns
    -------
    xarray.DataArray
        2D boolean mask (True = inside region)
    """
    if bbox is None and mask is None:
        raise ValueError("Must provide either bbox or mask")

    lat_name = get_lat_name(ds)
    lon_name = get_lon_name(ds)

    if bbox is not None:
        # Create mask from bounding box
        lat = ds[lat_name]
        lon = ds[lon_name]

        lat_min, lat_max = bbox["lat"]
        lon_min, lon_max = bbox["lon"]

        # Create boolean mask for latitude
        mask_lat = (lat >= lat_min) & (lat <= lat_max)

        # Handle longitude wrapping (for regions crossing 0° or 180°)
        if lon_min > lon_max:
            # Region crosses the prime meridian (e.g., -10° to 10° stored as 350° to 10°)
            mask_lon = (lon >= lon_min) | (lon <= lon_max)
        else:
            # Normal case: region doesn't wrap
            mask_lon = (lon >= lon_min) & (lon <= lon_max)

        # Combine masks
        region_mask = mask_lat & mask_lon
        return region_mask

    # Use provided mask
    if isinstance(mask, np.ndarray):
        # Convert to xarray with appropriate coordinates
        coords = {lat_name: ds[lat_name], lon_name: ds[lon_name]}
        mask = xr.DataArray(mask, coords=coords, dims=[lat_name, lon_name])
    return mask


def regional_mean(
    ds,
    region_code=None,
    region_mask=None,
    weights=None,
    normalize_weights=True,
):
    """
    Calculate area-weighted regional mean for a specific AR6 region or custom region.

    This function uses the IPCC AR6 reference regions from regionmask to
    calculate spatial averages over specific regions (e.g., 'EAS' for East Asia),
    or accepts a custom boolean mask for arbitrary regions.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Data to compute regional mean over
    region_code : str, optional
        AR6 region abbreviation (e.g., 'WNA', 'NEU', 'EAS', 'ARP', etc.)
        Use list_ar6_regions() to see all available regions
    region_mask : xarray.DataArray, optional
        Custom 2D boolean mask (True = inside region)
    weights : xarray.DataArray, optional
        Custom weights for averaging (if None, uses cosine of latitude)
    normalize_weights : bool, optional
        Whether to normalize weights by their mean (default True)

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Regional mean with spatial dimensions removed
        Includes 'region_code' and 'region_name' in attributes

    Examples
    --------
    >>> # Calculate mean temperature over East Asia
    >>> tas_eas = regional_mean(temperature_data, region_code='EAS')
    >>>
    >>> # Calculate mean precipitation over custom region
    >>> custom_mask = create_region_mask(data, bbox={'lat': (45, 55), 'lon': (5, 20)})
    >>> pr_custom = regional_mean(precipitation_data, region_mask=custom_mask)

    See Also
    --------
    create_region_mask : Create custom region masks
    list_ar6_regions : List all available AR6 regions
    global_mean : Calculate global mean
    extract_point : Extract time series at a specific point
    """
    # Handle AR6 region code
    if region_code is None and region_mask is None:
        raise ValueError("Must provide either region_code or region_mask")

    # Auto-detect coordinates
    lat_name = get_lat_name(ds)
    lon_name = get_lon_name(ds)
    # Handle custom region mask first
    if region_mask is not None:
        # Apply mask
        regional_data = ds.where(region_mask)

        # Create weights
        weights = get_weights_for_ds(regional_data, lat_name, lon_name, weights=weights)

        # Apply mask to weights
        weights = weights.where(region_mask)

        result = apply_weights_and_do_spatial_mean(
            regional_data, weights, normalize_weights, lat_name, lon_name
        )

        # Update attributes
        if hasattr(ds, "attrs"):
            result.attrs.update(ds.attrs)
        result.attrs["operation"] = "area_weighted_regional_mean"
        result.attrs["region_type"] = "custom"

        return result
    # Load AR6 regions
    ar6_regions = regionmask.defined_regions.ar6.all

    # Create mask for the data grid
    mask = ar6_regions.mask(ds)

    # Find the region number for the given code
    region_number = None
    region_name = None
    for region in ar6_regions:
        if region.abbrev == region_code:
            region_number = region.number
            region_name = region.name
            break

    if region_number is None:
        raise ValueError(
            f"Region code '{region_code}' not found in AR6 regions. "
            f"Use list_ar6_regions() to see available regions."
        )

    # Apply mask to data (keep only the specified region)
    regional_data = ds.where(mask == region_number)

    weights = get_weights_for_ds(regional_data, lat_name, lon_name, weights=weights)

    # Apply mask to weights as well
    weights = weights.where(mask == region_number)

    result = apply_weights_and_do_spatial_mean(
        regional_data, weights, normalize_weights, lat_name, lon_name
    )

    # Update attributes
    if hasattr(ds, "attrs"):
        result.attrs.update(ds.attrs)
    result.attrs["operation"] = "area_weighted_regional_mean"
    result.attrs["region_code"] = region_code
    result.attrs["region_name"] = region_name

    return result


def extract_point(
    ds,
    lat_point,
    lon_point,
    method="nearest",
):
    """
    Extract time series at a specific latitude/longitude point.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Data to extract point from
    lat_point : float
        Latitude of the point (degrees North, -90 to 90)
    lon_point : float
        Longitude of the point (degrees East, -180 to 180 or 0 to 360)
    method : str, optional
        Selection method: 'nearest' (default) or 'interp' for interpolation

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Time series at the specified point
        Includes 'lat_point' and 'lon_point' in attributes

    Examples
    --------
    >>> # Extract temperature at Mumbai (19.08°N, 72.88°E)
    >>> tas_mumbai = extract_point(temperature_data, lat_point=19.08, lon_point=72.88)
    >>>
    >>> # Extract with interpolation instead of nearest neighbor
    >>> tas_mumbai_interp = extract_point(temperature_data, 19.08, 72.88, method='interp')

    See Also
    --------
    regional_mean : Calculate regional mean
    global_mean : Calculate global mean
    """
    lat_name = get_lat_name(ds)
    lon_name = get_lon_name(ds)

    # Handle longitude wrapping (convert -180:180 to 0:360 if needed)
    lon_data = ds[lon_name].values
    if lon_data.max() > 180 and lon_point < 0:
        lon_point = lon_point + 360
    elif lon_data.max() <= 180 <= lon_point:
        lon_point = lon_point - 360

    # Extract point based on method
    if method == "nearest":
        result = ds.sel({lat_name: lat_point, lon_name: lon_point}, method="nearest")
    elif method == "interp":
        result = ds.interp({lat_name: lat_point, lon_name: lon_point})
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'nearest' or 'interp'.")

    # Update attributes
    if hasattr(ds, "attrs"):
        result.attrs.update(ds.attrs)
    result.attrs["operation"] = f"point_extraction_{method}"
    result.attrs["lat_point"] = lat_point
    result.attrs["lon_point"] = lon_point

    return result


def list_ar6_regions():  # pragma: no cover
    """
    List all available IPCC AR6 reference regions.

    Prints a formatted table of region codes and names that can be used
    with the regional_mean() function.

    Returns
    -------
    None
        Prints region information to stdout

    Examples
    --------
    >>> list_ar6_regions()
    AR6 IPCC Reference Regions:
    ============================================================
      GIC    : Greenland/Iceland
      NWN    : N.W. North America
      NEN    : N.E. North America
      ...

    See Also
    --------
    regional_mean : Calculate regional mean using AR6 regions
    """
    ar6_regions = regionmask.defined_regions.ar6.all

    print("AR6 IPCC Reference Regions:")
    print("=" * 60)
    for region in ar6_regions:
        print(f"  {region.abbrev:6s} : {region.name}")
    print("=" * 60)
    print(f"Total: {len(ar6_regions)} regions")
