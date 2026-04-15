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
- The input variables are the *transformed* quantities: C (raw, ppm),
  T (temperature anomaly from AgMERRA baseline, °C),
  W (precipitation ratio to AgMERRA baseline, dimensionless), N (raw, kg/ha/yr).
- Inputs are clamped to the GGCMI Phase 2 valid ranges before evaluation.
- Out-of-bounds offsets (T_oob, W_oob) correctly reflect the un-clamped excess.
- Yield is clipped to zero from below.
- N³ is NOT included in the polynomial.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from meteor.impacts.ggcm.coefficients import get_yields, load_coefficients

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID_SHAPE = (3, 4)  # small grid for speed; 360×720 arithmetic is unchanged

# Realistic AgMERRA-style baseline grids
T_BASE = np.full(GRID_SHAPE, 15.0)  # 15 °C everywhere
W_BASE = np.full(GRID_SHAPE, 600.0)  # 600 mm/yr everywhere


def _zero_K():
    """Return a K tensor of all zeros in shape (35, *GRID_SHAPE)."""
    return np.zeros((35,) + GRID_SHAPE)


def _standard_inputs():
    """In-range inputs that produce zero anomalies: T=0, W=1."""
    Ta = T_BASE.copy()  # Ta == T_agmerra → anomaly = 0
    Wa = W_BASE.copy()  # Wa == W_agmerra → ratio = 1
    Ca = 400.0
    Na = 100.0
    return Ca, Ta, Wa, Na


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


class TestOutputShape:
    """Verify shape and dtype of all returned arrays."""

    def test_yield_shape(self):
        K = _zero_K()
        Ca, Ta, Wa, Na = _standard_inputs()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        assert yld.shape == GRID_SHAPE

    def test_w_oob_shape(self):
        K = _zero_K()
        Ca, Ta, Wa, Na = _standard_inputs()
        _, w_oob, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        assert w_oob.shape == GRID_SHAPE

    def test_t_oob_shape(self):
        K = _zero_K()
        Ca, Ta, Wa, Na = _standard_inputs()
        _, _, t_oob = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        assert t_oob.shape == GRID_SHAPE


# ---------------------------------------------------------------------------
# Non-negativity of yield
# ---------------------------------------------------------------------------


class TestYieldNonNegativity:
    """Yield is clipped to zero from below (Franke et al. Eq. 1 discussion)."""

    def test_yield_clipped_to_zero(self):
        K = _zero_K()
        K[0] = -999.0  # large negative intercept → raw yield < 0
        Ca, Ta, Wa, Na = _standard_inputs()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        assert np.all(yld == 0.0)

    def test_positive_yield_unchanged(self):
        K = _zero_K()
        K[0] = 3.5  # positive intercept → yield = 3.5
        Ca, Ta, Wa, Na = _standard_inputs()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 3.5)


# ---------------------------------------------------------------------------
# Polynomial term isolation tests
# Each test sets exactly one coefficient and verifies the contribution.
# Standard conditions: Ca=400, Ta=T_BASE, Wa=W_BASE, Na=100
#   → C=400, T=0, W=1, N=100  (at AgMERRA baseline)
# Varying Ta or Wa shifts T or W away from baseline.
# ---------------------------------------------------------------------------


class TestPolynomialTerms:
    """Test that each polynomial term is evaluated with the correct formula."""

    # ---- degree-0 ----

    def test_K0_intercept(self):
        """K[0]: constant term."""
        K = _zero_K()
        K[0] = 7.0
        Ca, Ta, Wa, Na = _standard_inputs()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 7.0)

    # ---- degree-1 ----

    def test_K1_linear_C(self):
        """K[1]: linear C term. With Ca=400, yield = 400."""
        K = _zero_K()
        K[1] = 1.0
        Ca, Ta, Wa, Na = _standard_inputs()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 400.0)

    def test_K2_linear_T(self):
        """K[2]: linear T term.  T is the *anomaly* from T_agmerra."""
        K = _zero_K()
        K[2] = 1.0
        Ca = 400.0
        Ta = T_BASE + 2.0  # anomaly = 2 (within [T-1, T+6])
        Wa = W_BASE.copy()
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 2.0)

    def test_K2_linear_T_at_baseline_zero(self):
        """K[2]: when Ta equals T_agmerra, T anomaly = 0 → no contribution."""
        K = _zero_K()
        K[2] = 99.0
        Ca, Ta, Wa, Na = _standard_inputs()  # Ta == T_BASE
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 0.0)

    def test_K3_linear_W(self):
        """K[3]: linear W term.  W is the *ratio* Wa/W_agmerra."""
        K = _zero_K()
        K[3] = 1.0
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = W_BASE.copy()  # ratio = 1
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 1.0)

    def test_K3_linear_W_ratio(self):
        """K[3]: W = 1.2×W_base → ratio = 1.2."""
        K = _zero_K()
        K[3] = 1.0
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = 1.2 * W_BASE  # within 1.3×W_base limit
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 1.2, rtol=1e-10)

    def test_K4_linear_N(self):
        """K[4]: linear N term. Na=150 → N=150."""
        K = _zero_K()
        K[4] = 1.0
        Ca, Ta, Wa, _ = _standard_inputs()
        Na = 150.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 150.0)

    # ---- degree-2 ----

    def test_K5_C_squared(self):
        """K[5]: C² term. C=400 → contribution = 160 000."""
        K = _zero_K()
        K[5] = 1.0
        Ca, Ta, Wa, Na = _standard_inputs()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 400.0**2)

    def test_K9_T_squared(self):
        """K[9]: T² term. T=3 → contribution = 9."""
        K = _zero_K()
        K[9] = 1.0
        Ca = 400.0
        Ta = T_BASE + 3.0
        Wa = W_BASE.copy()
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 9.0)

    def test_K10_TW_interaction(self):
        """K[10]: T·W cross term. T=2, W=1.1 → 2.2."""
        K = _zero_K()
        K[10] = 1.0
        Ca = 400.0
        Ta = T_BASE + 2.0
        Wa = 1.1 * W_BASE
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 2.0 * 1.1, rtol=1e-10)

    def test_K12_W_squared(self):
        """K[12]: W² term. W=1 (baseline) → 1. W=1.2 → 1.44."""
        K = _zero_K()
        K[12] = 1.0
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = 1.2 * W_BASE
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 1.2**2, rtol=1e-10)

    def test_K14_N_squared(self):
        """K[14]: N² term. N=50 → 2500."""
        K = _zero_K()
        K[14] = 1.0
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = W_BASE.copy()
        Na = 50.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 50.0**2)

    # ---- degree-3 ----

    def test_K15_C_cubed(self):
        """K[15]: C³ term. C=400 → 64 000 000."""
        K = _zero_K()
        K[15] = 1.0
        Ca, Ta, Wa, Na = _standard_inputs()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 400.0**3)

    def test_K25_T_cubed(self):
        """K[25]: T³ term. T=2 → 8."""
        K = _zero_K()
        K[25] = 1.0
        Ca = 400.0
        Ta = T_BASE + 2.0
        Wa = W_BASE.copy()
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 8.0)

    def test_K31_W_cubed(self):
        """K[31]: W³ term. W=1.1 → 1.331."""
        K = _zero_K()
        K[31] = 1.0
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = 1.1 * W_BASE
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 1.1**3, rtol=1e-10)

    def test_K33_WN_squared(self):
        """K[33]: W·N² term. W=1.2, N=50 → 1.2 × 2500 = 3000."""
        K = _zero_K()
        K[33] = 1.0
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = 1.2 * W_BASE
        Na = 50.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 1.2 * 50.0**2, rtol=1e-10)


# ---------------------------------------------------------------------------
# N³ absence (paper: term omitted because only 3 N levels in training data)
# ---------------------------------------------------------------------------


class TestN3TermOmitted:
    """Verify that the N³ term is NOT included in the polynomial.

    Since K has shape (35,...), K[34] corresponds to N³ in a full polynomial.
    Setting it to a large value should have no effect on the yield.
    """

    def test_K34_N3_not_included(self):
        K = _zero_K()
        K[0] = 5.0  # baseline yield
        K[34] = 1e6  # enormous N³ coefficient that WOULD dominate if included
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = W_BASE.copy()
        Na = 200.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        # Yield should be just K[0]=5, unaffected by K[34]
        np.testing.assert_allclose(yld, 5.0)


# ---------------------------------------------------------------------------
# Multi-term superposition (linearity check)
# ---------------------------------------------------------------------------


class TestPolynomialSuperposition:
    """Verify that multiple non-zero K values combine additively."""

    def test_intercept_plus_T_term(self):
        """K[0]=1, K[2]=2: at T=3 → yield = 1 + 2*3 = 7."""
        K = _zero_K()
        K[0] = 1.0
        K[2] = 2.0
        Ca = 400.0
        Ta = T_BASE + 3.0
        Wa = W_BASE.copy()
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 7.0)

    def test_intercept_T_T2_T3_combination(self):
        """K[0]=1, K[2]=2, K[9]=0.5, K[25]=0.1  at T=2 → 1+4+2+0.8 = 7.8."""
        K = _zero_K()
        K[0] = 1.0
        K[2] = 2.0
        K[9] = 0.5
        K[25] = 0.1
        Ca = 400.0
        Ta = T_BASE + 2.0
        Wa = W_BASE.copy()
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        expected = 1.0 + 2.0 * 2 + 0.5 * 4 + 0.1 * 8  # = 7.8
        np.testing.assert_allclose(yld, expected, rtol=1e-12)

    def test_W_ratio_terms(self):
        """K[3]=1, K[12]=1, K[31]=1 at W=1.2 → 1.2 + 1.44 + 1.728 = 4.368."""
        K = _zero_K()
        K[3] = 1.0
        K[12] = 1.0
        K[31] = 1.0
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = 1.2 * W_BASE
        Na = 100.0
        yld, _, _ = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        w = 1.2
        expected = w + w**2 + w**3
        np.testing.assert_allclose(yld, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Input clamping (GGCMI Phase 2 valid ranges, Table 2, Franke et al. 2020)
# ---------------------------------------------------------------------------


class TestInputClamping:
    """
    Input bounds (from Table 2 of Franke et al. 2020):
      CO2: [360, 810] ppm
      T:   [T_agmerra − 1, T_agmerra + 6] °C
      W:   [0.5 × W_agmerra, 1.3 × W_agmerra]
      N:   [10, 200] kg N/ha/yr
    """

    # -- CO2 --

    def test_CO2_below_min_clamped_to_360(self):
        K = _zero_K()
        K[1] = 1.0  # linear C: yield == C_san
        Ca = 200.0  # below 360
        Ta, Wa = T_BASE.copy(), W_BASE.copy()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 360.0)

    def test_CO2_above_max_clamped_to_810(self):
        K = _zero_K()
        K[1] = 1.0
        Ca = 1200.0  # above 810
        Ta, Wa = T_BASE.copy(), W_BASE.copy()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 810.0)

    def test_CO2_midrange_unchanged(self):
        K = _zero_K()
        K[1] = 1.0
        Ca = 550.0
        Ta, Wa = T_BASE.copy(), W_BASE.copy()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 550.0)

    # -- Temperature --

    def test_T_below_min_clamped(self):
        """Ta well below T_agmerra - 1 → T_san = T_agmerra - 1 → T_anomaly = -1.

        A large intercept (K[0]=10) keeps the total yield positive so we can
        verify the clamped anomaly contribution (10 + 1×(−1) = 9) without the
        non-negativity clip obscuring the result.
        """
        K = _zero_K()
        K[0] = 10.0  # intercept keeps yield > 0
        K[2] = 1.0  # linear T: adds T_anomaly
        Ca = 400.0
        Ta = T_BASE - 5.0  # way below min → clamped to T_base - 1 (anomaly = -1)
        Wa = W_BASE.copy()
        yld, _, t_oob = get_yields(K, Ca, Ta, Wa, 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 9.0)  # 10 + 1×(−1) = 9
        np.testing.assert_allclose(t_oob, -4.0)  # excess = (−5) − (−1) = −4

    def test_T_above_max_clamped(self):
        """Ta = T_agmerra + 10 → T_san = T_agmerra + 6 → T_anomaly = 6."""
        K = _zero_K()
        K[2] = 1.0
        Ca = 400.0
        Ta = T_BASE + 10.0
        Wa = W_BASE.copy()
        yld, _, t_oob = get_yields(K, Ca, Ta, Wa, 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 6.0)
        np.testing.assert_allclose(t_oob, 4.0)  # excess = 10 - 6 = 4

    def test_T_in_range_no_clamping(self):
        K = _zero_K()
        K[2] = 1.0
        Ca = 400.0
        Ta = T_BASE + 4.0  # within [-1, +6]
        Wa = W_BASE.copy()
        yld, _, t_oob = get_yields(K, Ca, Ta, Wa, 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 4.0)
        np.testing.assert_allclose(t_oob, 0.0)

    # -- Precipitation --

    def test_W_below_min_clamped(self):
        """Wa = 0.1×W_base → W_san = 0.5×W_base → W_ratio = 0.5."""
        K = _zero_K()
        K[3] = 1.0  # linear W: yield == W_ratio
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = 0.1 * W_BASE
        yld, w_oob, _ = get_yields(K, Ca, Ta, Wa, 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 0.5)
        np.testing.assert_allclose(w_oob, (0.1 - 0.5) * W_BASE, rtol=1e-10)

    def test_W_above_max_clamped(self):
        """Wa = 2×W_base → W_san = 1.3×W_base → W_ratio = 1.3."""
        K = _zero_K()
        K[3] = 1.0
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = 2.0 * W_BASE
        yld, w_oob, _ = get_yields(K, Ca, Ta, Wa, 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 1.3, rtol=1e-10)
        np.testing.assert_allclose(w_oob, (2.0 - 1.3) * W_BASE, rtol=1e-10)

    def test_W_at_baseline_ratio_is_one(self):
        K = _zero_K()
        K[3] = 1.0
        Ca = 400.0
        Ta = T_BASE.copy()
        Wa = W_BASE.copy()
        yld, w_oob, _ = get_yields(K, Ca, Ta, Wa, 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 1.0)
        np.testing.assert_allclose(w_oob, 0.0)

    # -- Nitrogen --

    def test_N_below_min_clamped_to_10(self):
        K = _zero_K()
        K[4] = 1.0  # linear N: yield == N_san
        Ca, Ta, Wa = 400.0, T_BASE.copy(), W_BASE.copy()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, 1.0, T_BASE, W_BASE)  # Na=1 < 10
        np.testing.assert_allclose(yld, 10.0)

    def test_N_above_max_clamped_to_200(self):
        K = _zero_K()
        K[4] = 1.0
        Ca, Ta, Wa = 400.0, T_BASE.copy(), W_BASE.copy()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, 500.0, T_BASE, W_BASE)  # Na=500 > 200
        np.testing.assert_allclose(yld, 200.0)

    def test_N_in_range_unchanged(self):
        K = _zero_K()
        K[4] = 1.0
        Ca, Ta, Wa = 400.0, T_BASE.copy(), W_BASE.copy()
        yld, _, _ = get_yields(K, Ca, Ta, Wa, 75.0, T_BASE, W_BASE)
        np.testing.assert_allclose(yld, 75.0)


# ---------------------------------------------------------------------------
# Out-of-bounds offset semantics
# ---------------------------------------------------------------------------


class TestOutOfBoundsOffsets:
    """
    T_oob = Ta - T_san  (negative when Ta < lower bound, positive when > upper)
    W_oob = Wa - W_san  (same sign convention, in mm/yr)
    In-range inputs produce zero offsets.
    """

    def test_no_oob_when_in_range(self):
        K = _zero_K()
        Ca, Ta, Wa, Na = _standard_inputs()
        _, w_oob, t_oob = get_yields(K, Ca, Ta, Wa, Na, T_BASE, W_BASE)
        np.testing.assert_array_equal(t_oob, 0.0)
        np.testing.assert_array_equal(w_oob, 0.0)

    def test_T_oob_negative_below_range(self):
        K = _zero_K()
        excess = -3.0
        Ta = T_BASE + (-1.0 + excess)  # 3 °C below lower bound
        _, _, t_oob = get_yields(K, 400.0, Ta, W_BASE.copy(), 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(t_oob, excess)

    def test_T_oob_positive_above_range(self):
        K = _zero_K()
        excess = 4.0
        Ta = T_BASE + (6.0 + excess)  # 4 °C above upper bound
        _, _, t_oob = get_yields(K, 400.0, Ta, W_BASE.copy(), 100.0, T_BASE, W_BASE)
        np.testing.assert_allclose(t_oob, excess)

    def test_W_oob_is_in_mmyr(self):
        """W_oob is in mm/yr (absolute, not ratio)."""
        K = _zero_K()
        Wa = 0.2 * W_BASE  # well below 0.5×W_base
        _, w_oob, _ = get_yields(K, 400.0, T_BASE.copy(), Wa, 100.0, T_BASE, W_BASE)
        expected_w_oob = Wa - 0.5 * W_BASE  # should be -0.3 × W_BASE
        np.testing.assert_allclose(w_oob, expected_w_oob, rtol=1e-10)


# ---------------------------------------------------------------------------
# Spatially varying baseline
# ---------------------------------------------------------------------------


class TestSpatiallyVaryingBaseline:
    """The baseline grids can have different values per cell; verify correct use."""

    def test_T_anomaly_uses_local_baseline(self):
        """T = Ta - T_agmerra: each cell uses its own baseline temperature."""
        K = np.zeros((35, 2, 2))  # K must match the 2×2 grid shape
        K[2] = 1.0
        T_base_varying = np.array([[10.0, 20.0], [5.0, 30.0]])
        W_base_varying = np.full((2, 2), 500.0)
        Ta = T_base_varying + 1.0  # +1 °C anomaly everywhere (within valid range)
        Wa = W_base_varying.copy()
        yld, _, _ = get_yields(K, 400.0, Ta, Wa, 100.0, T_base_varying, W_base_varying)
        np.testing.assert_allclose(yld, 1.0)  # anomaly = 1 at every cell

    def test_W_ratio_uses_local_baseline(self):
        """W = Wa / W_agmerra: each cell uses its own baseline precipitation."""
        K = np.zeros((35, 2, 2))  # K must match the 2×2 grid shape
        K[3] = 1.0
        T_base_varying = np.full((2, 2), 15.0)
        W_base_varying = np.array([[400.0, 800.0], [200.0, 1200.0]])
        Ta = T_base_varying.copy()
        Wa = 1.1 * W_base_varying  # ratio = 1.1 everywhere (within [0.5, 1.3])
        yld, _, _ = get_yields(K, 400.0, Ta, Wa, 100.0, T_base_varying, W_base_varying)
        np.testing.assert_allclose(yld, 1.1, rtol=1e-10)


# ---------------------------------------------------------------------------
# load_coefficients — reads K tensor from a netCDF4 file
# ---------------------------------------------------------------------------


class TestLoadCoefficients:
    """Test load_coefficients reads the K_rf variable from a netCDF4 file."""

    def _make_mock_nc(self, k_data):
        """Return a mock netCDF4.Dataset whose K_rf variable returns k_data."""
        mock_var = MagicMock()
        mock_var.__getitem__.return_value = k_data
        mock_nc = MagicMock()
        mock_nc.variables = {"K_rf": mock_var}
        return mock_nc

    def test_returns_array_with_correct_shape(self):
        """K shape must be (35, 360, 720) as stored in the Zenodo files."""
        k_data = np.zeros((35, 360, 720), dtype=np.float64)
        mock_nc = self._make_mock_nc(k_data)
        with patch(
            "meteor.impacts.ggcm.coefficients.netcdf.Dataset", return_value=mock_nc
        ):
            K = load_coefficients("fake.nc4")
        assert K.shape == (35, 360, 720)

    def test_returns_float64_dtype(self):
        """load_coefficients must cast the result to float64."""
        k_data = np.ones((35, 2, 3), dtype=np.float32)  # float32 input
        mock_nc = self._make_mock_nc(k_data)
        with patch(
            "meteor.impacts.ggcm.coefficients.netcdf.Dataset", return_value=mock_nc
        ):
            K = load_coefficients("fake.nc4")
        assert K.dtype == np.float64

    def test_values_are_preserved(self):
        """Coefficient values read from the file are returned unchanged."""
        rng = np.random.default_rng(0)
        k_data = rng.standard_normal((35, 4, 6))
        mock_nc = self._make_mock_nc(k_data)
        with patch(
            "meteor.impacts.ggcm.coefficients.netcdf.Dataset", return_value=mock_nc
        ):
            K = load_coefficients("fake.nc4")
        np.testing.assert_allclose(K, k_data)

    def test_dataset_opened_in_read_mode(self):
        """The file must be opened with mode='r'."""
        k_data = np.zeros((35, 2, 2))
        mock_nc = self._make_mock_nc(k_data)
        with patch(
            "meteor.impacts.ggcm.coefficients.netcdf.Dataset", return_value=mock_nc
        ) as mock_ds:
            load_coefficients("path/to/coeff.nc4")
        mock_ds.assert_called_once_with("path/to/coeff.nc4", "r")

    def test_dataset_is_closed_after_read(self):
        """The netCDF4 dataset must be closed even on success."""
        k_data = np.zeros((35, 2, 2))
        mock_nc = self._make_mock_nc(k_data)
        with patch(
            "meteor.impacts.ggcm.coefficients.netcdf.Dataset", return_value=mock_nc
        ):
            load_coefficients("fake.nc4")
        mock_nc.close.assert_called_once()
