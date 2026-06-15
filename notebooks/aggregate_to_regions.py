import numpy as np
import xarray as xr
import regionmask

def compute_regional_means(da):
    """
    Compute AR6 regional and global means from gridded data.

    Parameters
    ----------
    da : xarray.DataArray
        Input data on a regular latitude–longitude grid.
        Must contain:
        - dimension: "gridcell"
        - coordinates: "lat", "lon" (1D, aligned with gridcell)

        Typical examples include temperature, precipitation, or other
        gridded climate variables.

    Returns
    -------
    da_regions : xarray.DataArray
        Data aggregated to AR6 land regions plus a global mean.
        Contains a new dimension "mask" representing regions, with
        appropriate CF-style metadata:
        - AR6 region indices
        - an additional entry for GLOBAL (-1)

    Notes
    -----
    - Regional means are computed using area-weighted averages
      over grid cells belonging to each AR6 land region.
    - The global mean is computed independently using the same
      area-weighting but without applying a regional mask.
    """

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    lat = da["lat"]
    lon = da["lon"]

    # ------------------------------------------------------------------
    # Define AR6 land regions and compute region mask
    # regionmask assigns a region number to each grid cell
    # ------------------------------------------------------------------
    ar6 = regionmask.defined_regions.ar6.land
    ar6_mask = ar6.mask(da.gridcell)

    # ------------------------------------------------------------------
    # Area weights (proportional to grid-cell area on a sphere)
    # ------------------------------------------------------------------
    weights = np.cos(np.deg2rad(lat))
    weights.name = "area_weight"

    # ------------------------------------------------------------------
    # Regional aggregation
    # Weighted mean over grid cells within each AR6 region
    # ------------------------------------------------------------------
    da_ar6 = (
        (da * weights)
        .groupby(ar6_mask)
        .sum(dim="gridcell")
        / weights.groupby(ar6_mask).sum(dim="gridcell")
    )

    # ------------------------------------------------------------------
    # Global aggregation
    # Computed separately (not via regionmask)
    # ------------------------------------------------------------------
    da_global = (
        da
        .weighted(weights)
        .mean(dim="gridcell")
        .expand_dims(mask=[-1])  # use -1 to label GLOBAL
    )

    # ------------------------------------------------------------------
    # Combine regional and global results
    # ------------------------------------------------------------------
    da_regions = xr.concat(
        [da_ar6, da_global],
        dim="mask"
    )

    # ------------------------------------------------------------------
    # Attach region metadata following CF conventions
    # ------------------------------------------------------------------
    flag_values = np.concatenate([ar6.numbers, [-1]])
    flag_meanings = " ".join(ar6.names + ["GLOBAL"])

    da_regions["mask"].attrs = {
        "standard_name": "region",
        "flag_values": flag_values,
        "flag_meanings": flag_meanings,
    }

    return da_regions
