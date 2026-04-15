"""
Load the AgMERRA 1980-2010 climatological baseline bundled with METEOR.

The two netCDF4 files bundled in meteor/impacts/ggcm/data/ are the
pre-processed AgMERRA averages required by the GGCMI Phase 2 emulator
(Franke et al., 2020) as the historical reference climate baseline.
They define the valid input ranges for temperature and precipitation and
set the reference point for the anomaly inputs to the polynomial.
"""

import importlib.resources

import netCDF4 as netcdf
import numpy as np


def load_agmerra_baseline():
    """Load the bundled AgMERRA 1980-2010 climatological means.

    Returns
    -------
    T_agmerra : np.ndarray, shape (360, 720)
        Annual mean temperature in degrees Celsius.
        Row 0 = 89.75 N, row 359 = -89.75 S, step 0.5 deg.
        Column 0 = -179.75 W, column 719 = 179.75 E, step 0.5 deg.
    W_agmerra : np.ndarray, shape (360, 720)
        Annual mean precipitation in mm/yr, floor-clipped at 1 mm/yr
        to avoid division-by-zero in the precipitation ratio computation.
    """
    data_pkg = importlib.resources.files("meteor.impacts.ggcm.data")

    with importlib.resources.as_file(
        data_pkg / "agmerra-tavg-avg-1980-2010-05deg-adjlon.nc4"
    ) as p:
        nc = netcdf.Dataset(str(p), "r")
        raw = nc.variables["tavg"][0, :, :]  # MaskedArray, °C
        # Convert to plain float64; fill masked ocean cells with 0 °C
        T_agmerra = np.ma.filled(raw, 0.0).astype(np.float64)
        nc.close()

    with importlib.resources.as_file(
        data_pkg / "agmerra-prate-avg-1980-2010-05deg-adjlon.nc4"
    ) as p:
        nc = netcdf.Dataset(str(p), "r")
        raw = nc.variables["prate"][0, :, :]  # MaskedArray, mm/day
        # Convert to mm/yr; fill masked ocean cells with a small positive value
        W_agmerra = np.ma.filled(raw, 1.0 / 365.25).astype(np.float64) * 365.25
        nc.close()

    # Floor precipitation to avoid divison by zero in the W ratio
    W_agmerra[W_agmerra < 1] = 1.0

    return T_agmerra, W_agmerra
