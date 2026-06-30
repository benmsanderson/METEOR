"""
Vendored polynomial evaluator from the GGCMI Phase 2 emulator.

Faithfully reproduces the core mathematics of Franke et al. (2020)
(https://doi.org/10.5194/gmd-13-3995-2020) without external dependencies
on the original ggcm_emulator package.
"""

import numpy as np
import xarray as xr


def load_coefficients(filepath):
    """Load the coefficient tensor from a GGCMI Phase 2 nc4 file.

    Parameters
    ----------
    filepath : str
        Path to a file of the form
        ``{CropModel}_{crop}_ggcmi_phase2_emulator_{variant}.nc4``.

    Returns
    -------
    coefficients : np.ndarray, shape (35, 360, 720)
        Polynomial coefficients (``K_rf`` in the source files); spatially
        varying per 0.5-degree grid cell.
    """
    with xr.open_dataset(filepath) as ds:
        return ds["K_rf"].values.astype(np.float64)


def get_yields(
    coefficients,
    co2_ppm,
    temp_c,
    precip_mmyr,
    n_fertilizer,
    temp_baseline,
    precip_baseline,
):
    """Evaluate the 35-term GGCMI Phase 2 crop yield polynomial.

    Implements Equation (1) from Franke et al. (2020).  Input bounds are
    enforced by clamping; out-of-bounds offsets are returned so callers can
    diagnose extrapolation.

    Parameters
    ----------
    coefficients : np.ndarray, shape (35, 360, 720)
        Coefficient tensor loaded by :func:`load_coefficients`.
    co2_ppm : float
        Global annual mean CO2 concentration (ppm). Valid range: 360-810.
    temp_c : np.ndarray, shape (360, 720)
        Annual mean temperature in degrees Celsius on the 0.5-degree grid.
        Must be absolute temperature (not an anomaly).
    precip_mmyr : np.ndarray, shape (360, 720)
        Annual mean precipitation in mm/yr on the 0.5-degree grid.
        Must be absolute precipitation.
    n_fertilizer : float
        Uniform nitrogen application (kg N / ha / yr). Valid range: 10-200.
    temp_baseline : np.ndarray, shape (360, 720)
        AgMERRA 1980-2010 temperature baseline in degrees Celsius.
    precip_baseline : np.ndarray, shape (360, 720)
        AgMERRA 1980-2010 precipitation baseline in mm/yr.

    Returns
    -------
    yields : np.ndarray, shape (360, 720)
        Estimated crop yield in t dry matter / ha / yr.  Clipped to >= 0.
    precip_oob : np.ndarray, shape (360, 720)
        Precipitation out-of-bounds offset (mm/yr).  Negative = below lower
        bound; positive = above upper bound.
    temp_oob : np.ndarray, shape (360, 720)
        Temperature out-of-bounds offset (degrees Celsius), same convention.
    """
    # Clamp inputs to the valid ranges of Franke et al. (2020)
    co2_clamped = min(max(360.0, co2_ppm), 810.0)
    temp_clamped = np.minimum(
        np.maximum(temp_baseline - 1.0, temp_c), temp_baseline + 6.0
    )
    precip_clamped = np.minimum(
        np.maximum(0.5 * precip_baseline, precip_mmyr), 1.3 * precip_baseline
    )
    n_clamped = min(max(10.0, n_fertilizer), 200.0)

    temp_oob = temp_c - temp_clamped
    precip_oob = precip_mmyr - precip_clamped

    # Transform inputs for polynomial evaluation
    # Paper symbols: C = co2, T = temp_anom, W = precip_ratio, N = nitrogen
    co2 = co2_clamped
    temp_anom = temp_clamped - temp_baseline
    precip_ratio = precip_clamped / precip_baseline
    nitrogen = n_clamped

    # 35-term third-order polynomial (Eq. 1, Franke et al. 2020)
    # Python indices are shifted by -1 from the paper's 1-based notation.
    yields = (
        coefficients[0]
        + coefficients[1] * co2
        + coefficients[2] * temp_anom
        + coefficients[3] * precip_ratio
        + coefficients[4] * nitrogen
        + coefficients[5] * co2**2
        + coefficients[6] * co2 * temp_anom
        + coefficients[7] * co2 * precip_ratio
        + coefficients[8] * co2 * nitrogen
        + coefficients[9] * temp_anom**2
        + coefficients[10] * temp_anom * precip_ratio
        + coefficients[11] * temp_anom * nitrogen
        + coefficients[12] * precip_ratio**2
        + coefficients[13] * precip_ratio * nitrogen
        + coefficients[14] * nitrogen**2
        + coefficients[15] * co2**3
        + coefficients[16] * co2**2 * temp_anom
        + coefficients[17] * co2**2 * precip_ratio
        + coefficients[18] * co2**2 * nitrogen
        + coefficients[19] * co2 * temp_anom**2
        + coefficients[20] * co2 * temp_anom * precip_ratio
        + coefficients[21] * co2 * temp_anom * nitrogen
        + coefficients[22] * co2 * precip_ratio**2
        + coefficients[23] * co2 * precip_ratio * nitrogen
        + coefficients[24] * co2 * nitrogen**2
        + coefficients[25] * temp_anom**3
        + coefficients[26] * temp_anom**2 * precip_ratio
        + coefficients[27] * temp_anom**2 * nitrogen
        + coefficients[28] * temp_anom * precip_ratio**2
        + coefficients[29] * temp_anom * precip_ratio * nitrogen
        + coefficients[30] * temp_anom * nitrogen**2
        + coefficients[31] * precip_ratio**3
        + coefficients[32] * precip_ratio**2 * nitrogen
        + coefficients[33] * precip_ratio * nitrogen**2
    )

    # Yield is non-negative by definition
    yields[yields < 0] = 0.0

    return yields, precip_oob, temp_oob
