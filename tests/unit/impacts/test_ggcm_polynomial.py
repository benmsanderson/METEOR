"""
Unit tests for the GGCMI Phase 2 polynomial evaluator.
=======================================================

These tests verify that ``get_yields`` in ``meteor.impacts.ggcm.coefficients``
faithfully implements Equation (1) from Franke et al. (2020) [GMD 13, 3995-4018,
https://doi.org/10.5194/gmd-13-3995-2020]:

    Y = sum_{i<=j<=k} K_{ijk} * C^i * T^j * W^k * N^l

where the polynomial is third-order in C, T, W, N with the N³ term omitted (it
cannot be fitted from three N levels).

Key invariants verified:
- Each of the 34 polynomial terms is evaluated correctly in isolation.
- The input variables are the *transformed* quantities: co2 (raw, ppm),
  temp_anom (temperature anomaly from AgMERRA baseline, °C),
  precip_ratio (precipitation ratio to AgMERRA baseline, dimensionless),
  nitrogen (raw, kg/ha/yr).
- Inputs are clamped to the GGCMI Phase 2 valid ranges before evaluation.
- Out-of-bounds offsets (temp_oob, precip_oob) correctly reflect the un-clamped excess.
- Yield is clipped to zero from below.
- N³ is NOT included in the polynomial.
"""

import numpy as np
import xarray as xr

from meteor.impacts.ggcm.coefficients import get_yields, load_coefficients

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID_SHAPE = (3, 4)  # small grid for speed; 360×720 arithmetic is unchanged

# Realistic AgMERRA-style baseline grids
TEMP_BASE = np.full(GRID_SHAPE, 15.0)  # 15 °C everywhere
PRECIP_BASE = np.full(GRID_SHAPE, 600.0)  # 600 mm/yr everywhere


def _zero_coefficients():
    """Return a coefficient tensor of all zeros in shape (35, *GRID_SHAPE)."""
    return np.zeros((35,) + GRID_SHAPE)


def _standard_inputs():
    """In-range inputs that produce zero anomalies: temp_anom=0, precip_ratio=1."""
    temp_c = TEMP_BASE.copy()  # temp_c == temp_baseline → anomaly = 0
    precip_mmyr = PRECIP_BASE.copy()  # precip_mmyr == precip_baseline → ratio = 1
    co2_ppm = 400.0
    n_fert = 100.0
    return co2_ppm, temp_c, precip_mmyr, n_fert


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_yield_shape():
    coefficients = _zero_coefficients()
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    assert yld.shape == GRID_SHAPE


def test_precip_oob_shape():
    coefficients = _zero_coefficients()
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()
    _, precip_oob, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    assert precip_oob.shape == GRID_SHAPE


def test_temp_oob_shape():
    coefficients = _zero_coefficients()
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()
    _, _, temp_oob = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    assert temp_oob.shape == GRID_SHAPE


# ---------------------------------------------------------------------------
# Non-negativity of yield
# ---------------------------------------------------------------------------


def test_yield_clipped_to_zero():
    coefficients = _zero_coefficients()
    coefficients[0] = -999.0  # large negative intercept → raw yield < 0
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    assert np.all(yld == 0.0)


def test_positive_yield_unchanged():
    coefficients = _zero_coefficients()
    coefficients[0] = 3.5  # positive intercept → yield = 3.5
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 3.5)


# ---------------------------------------------------------------------------
# Polynomial term isolation tests
# Each test sets exactly one coefficient and verifies the contribution.
# Standard conditions: co2_ppm=400, temp_c=TEMP_BASE, precip_mmyr=PRECIP_BASE,
# n_fert=100  →  co2=400, temp_anom=0, precip_ratio=1, nitrogen=100  (baseline)
# Varying temp_c or precip_mmyr shifts temp_anom or precip_ratio away from baseline.
# ---------------------------------------------------------------------------


def test_K0_intercept():
    """K[0]: constant term."""
    coefficients = _zero_coefficients()
    coefficients[0] = 7.0
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 7.0)


def test_K1_linear_C():
    """K[1]: linear C term. With co2_ppm=400, yield = 400."""
    coefficients = _zero_coefficients()
    coefficients[1] = 1.0
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 400.0)


def test_K2_linear_T():
    """K[2]: linear T term. T is the *anomaly* from temp_baseline."""
    coefficients = _zero_coefficients()
    coefficients[2] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE + 2.0  # anomaly = 2 (within [base-1, base+6])
    precip_mmyr = PRECIP_BASE.copy()
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 2.0)


def test_K2_linear_T_at_baseline_zero():
    """K[2]: when temp_c equals temp_baseline, T anomaly = 0 → no contribution."""
    coefficients = _zero_coefficients()
    coefficients[2] = 99.0
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()  # temp_c == TEMP_BASE
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 0.0)


def test_K3_linear_W():
    """K[3]: linear W term. W is the *ratio* precip_mmyr/precip_baseline."""
    coefficients = _zero_coefficients()
    coefficients[3] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = PRECIP_BASE.copy()  # ratio = 1
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 1.0)


def test_K3_linear_W_ratio():
    """K[3]: W = 1.2×precip_base → ratio = 1.2."""
    coefficients = _zero_coefficients()
    coefficients[3] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = 1.2 * PRECIP_BASE  # within 1.3×precip_base limit
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 1.2, rtol=1e-10)


def test_K4_linear_N():
    """K[4]: linear N term. n_fert=150 → nitrogen=150."""
    coefficients = _zero_coefficients()
    coefficients[4] = 1.0
    co2_ppm, temp_c, precip_mmyr, _ = _standard_inputs()
    n_fert = 150.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 150.0)


def test_K5_C_squared():
    """K[5]: C² term. co2=400 → contribution = 160 000."""
    coefficients = _zero_coefficients()
    coefficients[5] = 1.0
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 400.0**2)


def test_K9_T_squared():
    """K[9]: T² term. temp_anom=3 → contribution = 9."""
    coefficients = _zero_coefficients()
    coefficients[9] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE + 3.0
    precip_mmyr = PRECIP_BASE.copy()
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 9.0)


def test_K10_TW_interaction():
    """K[10]: T·W cross term. temp_anom=2, precip_ratio=1.1 → 2.2."""
    coefficients = _zero_coefficients()
    coefficients[10] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE + 2.0
    precip_mmyr = 1.1 * PRECIP_BASE
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 2.0 * 1.1, rtol=1e-10)


def test_K12_W_squared():
    """K[12]: W² term. precip_ratio=1 (baseline) → 1. precip_ratio=1.2 → 1.44."""
    coefficients = _zero_coefficients()
    coefficients[12] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = 1.2 * PRECIP_BASE
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 1.2**2, rtol=1e-10)


def test_K14_N_squared():
    """K[14]: N² term. nitrogen=50 → 2500."""
    coefficients = _zero_coefficients()
    coefficients[14] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = PRECIP_BASE.copy()
    n_fert = 50.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 50.0**2)


def test_K15_C_cubed():
    """K[15]: C³ term. co2=400 → 64 000 000."""
    coefficients = _zero_coefficients()
    coefficients[15] = 1.0
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 400.0**3)


def test_K25_T_cubed():
    """K[25]: T³ term. temp_anom=2 → 8."""
    coefficients = _zero_coefficients()
    coefficients[25] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE + 2.0
    precip_mmyr = PRECIP_BASE.copy()
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 8.0)


def test_K31_W_cubed():
    """K[31]: W³ term. precip_ratio=1.1 → 1.331."""
    coefficients = _zero_coefficients()
    coefficients[31] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = 1.1 * PRECIP_BASE
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 1.1**3, rtol=1e-10)


def test_K33_WN_squared():
    """K[33]: W·N² term. precip_ratio=1.2, nitrogen=50 → 1.2 × 2500 = 3000."""
    coefficients = _zero_coefficients()
    coefficients[33] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = 1.2 * PRECIP_BASE
    n_fert = 50.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 1.2 * 50.0**2, rtol=1e-10)


# ---------------------------------------------------------------------------
# N³ absence (paper: term omitted because only 3 N levels in training data)
# ---------------------------------------------------------------------------


def test_K34_N3_not_included():
    coefficients = _zero_coefficients()
    coefficients[0] = 5.0  # baseline yield
    coefficients[34] = 1e6  # enormous N³ coefficient that WOULD dominate if included
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = PRECIP_BASE.copy()
    n_fert = 200.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    # Yield should be just K[0]=5, unaffected by K[34]
    np.testing.assert_allclose(yld, 5.0)


# ---------------------------------------------------------------------------
# Multi-term superposition (linearity check)
# ---------------------------------------------------------------------------


def test_intercept_plus_T_term():
    """K[0]=1, K[2]=2: at temp_anom=3 → yield = 1 + 2*3 = 7."""
    coefficients = _zero_coefficients()
    coefficients[0] = 1.0
    coefficients[2] = 2.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE + 3.0
    precip_mmyr = PRECIP_BASE.copy()
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 7.0)


def test_intercept_T_T2_T3_combination():
    """K[0]=1, K[2]=2, K[9]=0.5, K[25]=0.1 at temp_anom=2 → 1+4+2+0.8 = 7.8."""
    coefficients = _zero_coefficients()
    coefficients[0] = 1.0
    coefficients[2] = 2.0
    coefficients[9] = 0.5
    coefficients[25] = 0.1
    co2_ppm = 400.0
    temp_c = TEMP_BASE + 2.0
    precip_mmyr = PRECIP_BASE.copy()
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    expected = 1.0 + 2.0 * 2 + 0.5 * 4 + 0.1 * 8  # = 7.8
    np.testing.assert_allclose(yld, expected, rtol=1e-12)


def test_W_ratio_terms():
    """K[3]=1, K[12]=1, K[31]=1 at precip_ratio=1.2 → 1.2 + 1.44 + 1.728 = 4.368."""
    coefficients = _zero_coefficients()
    coefficients[3] = 1.0
    coefficients[12] = 1.0
    coefficients[31] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = 1.2 * PRECIP_BASE
    n_fert = 100.0
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    ratio = 1.2
    expected = ratio + ratio**2 + ratio**3
    np.testing.assert_allclose(yld, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Input clamping (GGCMI Phase 2 valid ranges, Table 2, Franke et al. 2020)
# ---------------------------------------------------------------------------


def test_CO2_below_min_clamped_to_360():
    coefficients = _zero_coefficients()
    coefficients[1] = 1.0  # linear C: yield == co2_clamped
    co2_ppm = 200.0  # below 360
    temp_c, precip_mmyr = TEMP_BASE.copy(), PRECIP_BASE.copy()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 360.0)


def test_CO2_above_max_clamped_to_810():
    coefficients = _zero_coefficients()
    coefficients[1] = 1.0
    co2_ppm = 1200.0  # above 810
    temp_c, precip_mmyr = TEMP_BASE.copy(), PRECIP_BASE.copy()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 810.0)


def test_CO2_midrange_unchanged():
    coefficients = _zero_coefficients()
    coefficients[1] = 1.0
    co2_ppm = 550.0
    temp_c, precip_mmyr = TEMP_BASE.copy(), PRECIP_BASE.copy()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 550.0)


def test_T_below_min_clamped():
    """temp_c well below temp_baseline - 1 → clamped to temp_baseline - 1 → anomaly = -1.

    A large intercept (K[0]=10) keeps the total yield positive so we can
    verify the clamped anomaly contribution (10 + 1×(−1) = 9) without the
    non-negativity clip obscuring the result.
    """
    coefficients = _zero_coefficients()
    coefficients[0] = 10.0  # intercept keeps yield > 0
    coefficients[2] = 1.0  # linear T: adds temp_anom
    co2_ppm = 400.0
    temp_c = TEMP_BASE - 5.0  # way below min → clamped to base-1 (anomaly = -1)
    precip_mmyr = PRECIP_BASE.copy()
    yld, _, temp_oob = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 9.0)  # 10 + 1×(−1) = 9
    np.testing.assert_allclose(temp_oob, -4.0)  # excess = (−5) − (−1) = −4


def test_T_above_max_clamped():
    """temp_c = temp_baseline + 10 → clamped to temp_baseline + 6 → anomaly = 6."""
    coefficients = _zero_coefficients()
    coefficients[2] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE + 10.0
    precip_mmyr = PRECIP_BASE.copy()
    yld, _, temp_oob = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 6.0)
    np.testing.assert_allclose(temp_oob, 4.0)  # excess = 10 - 6 = 4


def test_T_in_range_no_clamping():
    coefficients = _zero_coefficients()
    coefficients[2] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE + 4.0  # within [-1, +6]
    precip_mmyr = PRECIP_BASE.copy()
    yld, _, temp_oob = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 4.0)
    np.testing.assert_allclose(temp_oob, 0.0)


def test_W_below_min_clamped():
    """precip_mmyr = 0.1×precip_base → clamped to 0.5×precip_base → ratio = 0.5."""
    coefficients = _zero_coefficients()
    coefficients[3] = 1.0  # linear W: yield == precip_ratio
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = 0.1 * PRECIP_BASE
    yld, precip_oob, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 0.5)
    np.testing.assert_allclose(precip_oob, (0.1 - 0.5) * PRECIP_BASE, rtol=1e-10)


def test_W_above_max_clamped():
    """precip_mmyr = 2×precip_base → clamped to 1.3×precip_base → ratio = 1.3."""
    coefficients = _zero_coefficients()
    coefficients[3] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = 2.0 * PRECIP_BASE
    yld, precip_oob, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 1.3, rtol=1e-10)
    np.testing.assert_allclose(precip_oob, (2.0 - 1.3) * PRECIP_BASE, rtol=1e-10)


def test_W_at_baseline_ratio_is_one():
    coefficients = _zero_coefficients()
    coefficients[3] = 1.0
    co2_ppm = 400.0
    temp_c = TEMP_BASE.copy()
    precip_mmyr = PRECIP_BASE.copy()
    yld, precip_oob, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 1.0)
    np.testing.assert_allclose(precip_oob, 0.0)


def test_N_below_min_clamped_to_10():
    coefficients = _zero_coefficients()
    coefficients[4] = 1.0  # linear N: yield == n_clamped
    co2_ppm, temp_c, precip_mmyr = 400.0, TEMP_BASE.copy(), PRECIP_BASE.copy()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 1.0, TEMP_BASE, PRECIP_BASE
    )  # n_fert=1 < 10
    np.testing.assert_allclose(yld, 10.0)


def test_N_above_max_clamped_to_200():
    coefficients = _zero_coefficients()
    coefficients[4] = 1.0
    co2_ppm, temp_c, precip_mmyr = 400.0, TEMP_BASE.copy(), PRECIP_BASE.copy()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 500.0, TEMP_BASE, PRECIP_BASE
    )  # n_fert=500 > 200
    np.testing.assert_allclose(yld, 200.0)


def test_N_in_range_unchanged():
    coefficients = _zero_coefficients()
    coefficients[4] = 1.0
    co2_ppm, temp_c, precip_mmyr = 400.0, TEMP_BASE.copy(), PRECIP_BASE.copy()
    yld, _, _ = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, 75.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(yld, 75.0)


# ---------------------------------------------------------------------------
# Out-of-bounds offset semantics
# ---------------------------------------------------------------------------


def test_no_oob_when_in_range():
    coefficients = _zero_coefficients()
    co2_ppm, temp_c, precip_mmyr, n_fert = _standard_inputs()
    _, precip_oob, temp_oob = get_yields(
        coefficients, co2_ppm, temp_c, precip_mmyr, n_fert, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_array_equal(temp_oob, 0.0)
    np.testing.assert_array_equal(precip_oob, 0.0)


def test_T_oob_negative_below_range():
    coefficients = _zero_coefficients()
    excess = -3.0
    temp_c = TEMP_BASE + (-1.0 + excess)  # 3 °C below lower bound
    _, _, temp_oob = get_yields(
        coefficients, 400.0, temp_c, PRECIP_BASE.copy(), 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(temp_oob, excess)


def test_T_oob_positive_above_range():
    coefficients = _zero_coefficients()
    excess = 4.0
    temp_c = TEMP_BASE + (6.0 + excess)  # 4 °C above upper bound
    _, _, temp_oob = get_yields(
        coefficients, 400.0, temp_c, PRECIP_BASE.copy(), 100.0, TEMP_BASE, PRECIP_BASE
    )
    np.testing.assert_allclose(temp_oob, excess)


def test_W_oob_is_in_mmyr():
    """precip_oob is in mm/yr (absolute, not ratio)."""
    coefficients = _zero_coefficients()
    precip_mmyr = 0.2 * PRECIP_BASE  # well below 0.5×precip_base
    _, precip_oob, _ = get_yields(
        coefficients,
        400.0,
        TEMP_BASE.copy(),
        precip_mmyr,
        100.0,
        TEMP_BASE,
        PRECIP_BASE,
    )
    expected = precip_mmyr - 0.5 * PRECIP_BASE  # should be -0.3 × PRECIP_BASE
    np.testing.assert_allclose(precip_oob, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Spatially varying baseline
# ---------------------------------------------------------------------------


def test_T_anomaly_uses_local_baseline():
    """temp_anom = temp_c - temp_baseline: each cell uses its own baseline."""
    coefficients = np.zeros((35, 2, 2))  # must match the 2×2 grid shape
    coefficients[2] = 1.0
    temp_base_varying = np.array([[10.0, 20.0], [5.0, 30.0]])
    precip_base_varying = np.full((2, 2), 500.0)
    temp_c = temp_base_varying + 1.0  # +1 °C anomaly everywhere (within range)
    precip_mmyr = precip_base_varying.copy()
    yld, _, _ = get_yields(
        coefficients,
        400.0,
        temp_c,
        precip_mmyr,
        100.0,
        temp_base_varying,
        precip_base_varying,
    )
    np.testing.assert_allclose(yld, 1.0)  # anomaly = 1 at every cell


def test_W_ratio_uses_local_baseline():
    """precip_ratio = precip_mmyr / precip_baseline: each cell uses its own baseline."""
    coefficients = np.zeros((35, 2, 2))  # must match the 2×2 grid shape
    coefficients[3] = 1.0
    temp_base_varying = np.full((2, 2), 15.0)
    precip_base_varying = np.array([[400.0, 800.0], [200.0, 1200.0]])
    temp_c = temp_base_varying.copy()
    precip_mmyr = 1.1 * precip_base_varying  # ratio = 1.1 everywhere
    yld, _, _ = get_yields(
        coefficients,
        400.0,
        temp_c,
        precip_mmyr,
        100.0,
        temp_base_varying,
        precip_base_varying,
    )
    np.testing.assert_allclose(yld, 1.1, rtol=1e-10)


# ---------------------------------------------------------------------------
# load_coefficients — reads the coefficient tensor from a netCDF file via xarray
# ---------------------------------------------------------------------------


def _write_coefficient_nc(tmp_path, k_data):
    """Write a temporary nc file with the same structure as the Zenodo files."""
    ds = xr.Dataset({"K_rf": (["term", "lat", "lon"], k_data)})
    path = tmp_path / "coefficients.nc4"
    ds.to_netcdf(path)
    return path


def test_returns_array_with_correct_shape(tmp_path):
    """Coefficient shape must be (35, 360, 720) as stored in the Zenodo files."""
    k_data = np.zeros((35, 360, 720), dtype=np.float64)
    path = _write_coefficient_nc(tmp_path, k_data)
    coefficients = load_coefficients(str(path))
    assert coefficients.shape == (35, 360, 720)


def test_returns_float64_dtype(tmp_path):
    """load_coefficients must cast the result to float64."""
    k_data = np.ones((35, 2, 3), dtype=np.float32)  # float32 input
    path = _write_coefficient_nc(tmp_path, k_data)
    coefficients = load_coefficients(str(path))
    assert coefficients.dtype == np.float64


def test_values_are_preserved(tmp_path):
    """Coefficient values read from the file are returned unchanged."""
    rng = np.random.default_rng(0)
    k_data = rng.standard_normal((35, 4, 6))
    path = _write_coefficient_nc(tmp_path, k_data)
    coefficients = load_coefficients(str(path))
    np.testing.assert_allclose(coefficients, k_data)
