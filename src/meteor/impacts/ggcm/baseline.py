"""
Load the AgMERRA 1980-2010 climatological baseline bundled with METEOR.

The two netCDF files bundled in meteor/impacts/ggcm/data/ are the
pre-processed AgMERRA averages required by the GGCMI Phase 2 emulator
(Franke et al., 2020) as the historical reference climate baseline.
They define the valid input ranges for temperature and precipitation and
set the reference point for the anomaly inputs to the polynomial.
"""

import importlib.resources

import numpy as np
import xarray as xr


def load_agmerra_baseline():
    """Load the bundled AgMERRA 1980-2010 climatological means.

    Returns
    -------
    temp_baseline : np.ndarray, shape (360, 720)
        Annual mean temperature in degrees Celsius.
        Row 0 = 89.75 N, row 359 = -89.75 S, step 0.5 deg.
        Column 0 = -179.75 W, column 719 = 179.75 E, step 0.5 deg.
    precip_baseline : np.ndarray, shape (360, 720)
        Annual mean precipitation in mm/yr, floor-clipped at 1 mm/yr
        to avoid division-by-zero in the precipitation ratio computation.
    """
    data_pkg = importlib.resources.files("meteor.impacts.ggcm.data")

    with importlib.resources.as_file(
        data_pkg / "agmerra-tavg-avg-1980-2010-05deg-adjlon.nc4"
    ) as path:
        with xr.open_dataset(path) as ds:
            # Ocean cells arrive as NaN once xarray applies the fill mask
            temp_baseline = np.nan_to_num(
                ds["tavg"].isel(time=0).values.astype(np.float64), nan=0.0
            )

    with importlib.resources.as_file(
        data_pkg / "agmerra-prate-avg-1980-2010-05deg-adjlon.nc4"
    ) as path:
        with xr.open_dataset(path) as ds:
            precip_mmday = np.nan_to_num(
                ds["prate"].isel(time=0).values.astype(np.float64), nan=1.0 / 365.25
            )
    precip_baseline = precip_mmday * 365.25

    # Floor precipitation to avoid division by zero in the precip ratio
    precip_baseline[precip_baseline < 1] = 1.0

    return temp_baseline, precip_baseline
