"""
Vendored polynomial evaluator from the GGCMI Phase 2 emulator.

Faithfully reproduces the core mathematics of Franke et al. (2020)
(https://doi.org/10.5194/gmd-13-3995-2020) without external dependencies
on the original ggcm_emulator package.
"""

import netCDF4 as netcdf
import numpy as np


def load_coefficients(filepath):
    """Load the K coefficient tensor from a GGCMI Phase 2 nc4 file.

    Parameters
    ----------
    filepath : str
        Path to a file of the form
        ``{CropModel}_{crop}_ggcmi_phase2_emulator_{variant}.nc4``.

    Returns
    -------
    K : np.ndarray, shape (35, 360, 720)
        Polynomial coefficients; spatially varying per 0.5-degree grid cell.
    """
    nc = netcdf.Dataset(filepath, "r")
    K = np.array(nc.variables["K_rf"][:, :, :], dtype=np.float64)
    nc.close()
    return K


def get_yields(K, Ca, Ta, Wa, Na, T_agmerra, W_agmerra):
    """Evaluate the 35-term GGCMI Phase 2 crop yield polynomial.

    Implements Equation (1) from Franke et al. (2020).  Input bounds are
    enforced by clamping; out-of-bounds offsets are returned so callers can
    diagnose extrapolation.

    Parameters
    ----------
    K : np.ndarray, shape (35, 360, 720)
        Coefficient tensor loaded by :func:`load_coefficients`.
    Ca : float
        Global annual mean CO2 concentration (ppm). Valid range: 360-810.
    Ta : np.ndarray, shape (360, 720)
        Annual mean temperature in degrees Celsius on the 0.5-degree grid.
        Must be absolute temperature (not an anomaly).
    Wa : np.ndarray, shape (360, 720)
        Annual mean precipitation in mm/yr on the 0.5-degree grid.
        Must be absolute precipitation.
    Na : float
        Uniform nitrogen application (kg N / ha / yr). Valid range: 10-200.
    T_agmerra : np.ndarray, shape (360, 720)
        AgMERRA 1980-2010 temperature baseline in degrees Celsius.
    W_agmerra : np.ndarray, shape (360, 720)
        AgMERRA 1980-2010 precipitation baseline in mm/yr.

    Returns
    -------
    Yield : np.ndarray, shape (360, 720)
        Estimated crop yield in t dry matter / ha / yr.  Clipped to >= 0.
    W_oob : np.ndarray, shape (360, 720)
        Precipitation out-of-bounds offset (mm/yr).  Negative = below lower
        bound; positive = above upper bound.
    T_oob : np.ndarray, shape (360, 720)
        Temperature out-of-bounds offset (degrees Celsius), same convention.
    """
    # Clamp inputs to valid ranges
    C_san = min(max(360.0, Ca), 810.0)
    T_san = np.minimum(np.maximum(T_agmerra - 1.0, Ta), T_agmerra + 6.0)
    W_san = np.minimum(np.maximum(0.5 * W_agmerra, Wa), 1.3 * W_agmerra)
    N_san = min(max(10.0, Na), 200.0)

    T_oob = Ta - T_san
    W_oob = Wa - W_san

    # Transform inputs for polynomial evaluation
    C = C_san
    T = T_san - T_agmerra  # temperature anomaly from AgMERRA baseline
    W = W_san / W_agmerra  # precipitation ratio relative to AgMERRA baseline
    N = N_san

    # 35-term third-order polynomial (Eq. 1, Franke et al. 2020)
    # Python indices are shifted by -1 from the paper's 1-based notation.
    Yield = (
        K[0]
        + K[1] * C
        + K[2] * T
        + K[3] * W
        + K[4] * N
        + K[5] * C**2
        + K[6] * C * T
        + K[7] * C * W
        + K[8] * C * N
        + K[9] * T**2
        + K[10] * T * W
        + K[11] * T * N
        + K[12] * W**2
        + K[13] * W * N
        + K[14] * N**2
        + K[15] * C**3
        + K[16] * C**2 * T
        + K[17] * C**2 * W
        + K[18] * C**2 * N
        + K[19] * C * T**2
        + K[20] * C * T * W
        + K[21] * C * T * N
        + K[22] * C * W**2
        + K[23] * C * W * N
        + K[24] * C * N**2
        + K[25] * T**3
        + K[26] * T**2 * W
        + K[27] * T**2 * N
        + K[28] * T * W**2
        + K[29] * T * W * N
        + K[30] * T * N**2
        + K[31] * W**3
        + K[32] * W**2 * N
        + K[33] * W * N**2
    )

    # Yield is non-negative by definition
    Yield[Yield < 0] = 0.0

    return Yield, W_oob, T_oob
