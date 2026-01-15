import random

import numpy as np
import xarray as xr

from meteor import pattern_logic_lib


def test_make_anom():
    cnt_data = np.zeros(shape=(3, 3, 3))
    exp_data = np.random.rand(3, 3, 3)
    ds_exp = xr.DataArray(
        exp_data,
        coords=(
            np.arange(
                3,
            ),
            np.array([-90, 0, 90]),
            np.array([0, 120, 240]),
        ),
        dims=("year", "lat", "lon"),
    )
    ds_cnt = xr.DataArray(
        cnt_data,
        coords=(
            np.arange(
                3,
            ),
            np.array([-90, 0, 90]),
            np.array([0, 120, 240]),
        ),
        dims=("year", "lat", "lon"),
    )
    test = pattern_logic_lib.make_anom(ds_exp, ds_cnt)
    assert np.array_equal(test.values, ds_exp.values)
    # Test with Dataset inputs
    ds_exp = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.random.randn(10, 3, 3),
                dims=["year", "lat", "lon"],
                coords={
                    "year": range(2000, 2010),
                    "lat": [0, 1, 2],
                    "lon": [0, 1, 2],
                },
            )
        }
    )

    ds_cnt = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.random.randn(10, 3, 3),
                dims=["year", "lat", "lon"],
                coords={
                    "year": range(2000, 2010),
                    "lat": [0, 1, 2],
                    "lon": [0, 1, 2],
                },
            )
        }
    )

    result = pattern_logic_lib.make_anom(ds_exp, ds_cnt)

    # Should return a dataset
    assert isinstance(result, xr.Dataset)

    # Should have same variables as input
    assert "tas" in result

    # Should have same dimensions
    assert set(result["tas"].dims) == set(["time"]).union(
        set(ds_exp["tas"].dims) - set(["year"])
    )


def test_pmodel_function():
    """Test the pmodel function."""
    # Test with simple parameters
    ain = (1.0, 2.0, 0.5, 1.0, 0.0, 1.0)
    pars = pattern_logic_lib.make_params(ain)

    n_times = 10

    result = pattern_logic_lib.pmodel(pars, n_times)

    # Should return an array
    assert isinstance(result, np.ndarray)

    # Should have correct length
    assert len(result) == n_times

    # Should be real values
    assert np.all(np.isreal(result))


def test_residual_function():
    """Test the residual function to hit lines 406-408."""
    # Create simple test data to test the residual function
    # Mock parameters object
    ain = (1.0, 2.0, 0.5, 1.0)
    pars = pattern_logic_lib.make_params(ain)
    mode_weights = np.array([0.5, 0.8])
    test_data = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 time steps, 2 modes

    # Test the residual calculation directly
    # This should hit the return statement at lines 406-408
    result = pattern_logic_lib.residual(pars, mode_weights, test_data)

    # Should return an array with the same shape as test_data
    assert isinstance(result, np.ndarray)
    assert result.shape == test_data.shape


def test_expfun_function():
    """Test the expfun function."""
    # Test with simple parameters
    t = np.array([0, 1, 2, 3])
    ain = (1.0, 2.0, 0.5, 1.0, 0.0, 1.0)
    pars = pattern_logic_lib.make_params(ain)
    result = pattern_logic_lib.expfun(t, pars)

    # Should return array same shape as t
    assert result.shape == t.shape

    # Should be real values
    assert np.all(np.isreal(result))

    # At t=0, should be 0
    assert np.isclose(result[0], 0)


def test_expotas():
    s1 = 5
    t1 = 25
    assert pattern_logic_lib.expotas(0, s1, t1) == 0
    assert pattern_logic_lib.expotas(random.randint(0, 200), s1, t1) < s1
    # Test with zero coefficient
    result = pattern_logic_lib.expotas(1.0, 0.0, 1.0)
    assert result == 0.0

    # Test with scalar inputs
    time = 1.0
    coeff = 2.0
    decay_time = 3.0

    result = pattern_logic_lib.expotas(time, coeff, decay_time)

    # Should return expected exponential value
    expected = coeff * (1 - np.exp(-time / decay_time))
    assert np.isclose(result, expected)

    # Test with array inputs
    times = np.array([0, 1, 2, 5, 10])
    results = pattern_logic_lib.expotas(times, coeff, decay_time)

    # Should be same shape as input
    assert results.shape == times.shape

    # Should be monotonically increasing (for positive coeff)
    assert np.all(np.diff(results) >= 0)

    # At t=0, should be 0
    result_t0 = pattern_logic_lib.expotas(0, coeff, decay_time)
    assert np.isclose(result_t0, 0)

    # At t=infinity, should approach coeff
    result_large_t = pattern_logic_lib.expotas(1000, coeff, decay_time)
    assert np.isclose(result_large_t, coeff, rtol=1e-3)

    # Test with very small decay time
    result = pattern_logic_lib.expotas(1.0, 1.0, 0.001)
    assert np.isclose(result, 1.0, rtol=1e-3)

    # Test with very large decay time
    result = pattern_logic_lib.expotas(1.0, 1.0, 1000.0)
    assert result < 0.01  # Should be very small


def test_make_amat_function():
    """Test the make_amat function."""
    # Test with simple parameters
    ain = (1.0, 2.0, 0.5, 1.0, 0.0, 1.0)
    pars = pattern_logic_lib.make_params(ain)

    nt = 10  # Number of time steps

    result = pattern_logic_lib.make_amat(pars, nt)

    # Should return a matrix
    assert isinstance(result, np.ndarray)

    # Should have expected shape (nt x nt)
    assert result.shape == (nt, len(ain) // 2)

    # Should be real values
    assert np.all(np.isreal(result))


def test_make_pmat_function():
    """Test the make_pmat function."""
    # Test with simple parameters
    tauvec = np.array([1.0, 2.0, 3.0])
    nt = 5

    result = pattern_logic_lib.make_pmat(tauvec, nt)

    # Should return a matrix
    assert isinstance(result, np.ndarray)

    # Should have expected shape
    assert result.shape[0] == len(tauvec)  # Parameter dimension
    assert result.shape[1] == nt  # Time dimension

    # Should be real values
    assert np.all(np.isreal(result))


def test_weights():
    empty_data = np.zeros(shape=(3, 3, 3))
    empty_array = xr.DataArray(
        empty_data,
        coords=(
            np.arange(
                3,
            ),
            np.array([-90, 0, 90]),
            np.array([0, 120, 240]),
        ),
        dims=("year", "lat", "lon"),
    )
    assert np.allclose(pattern_logic_lib.wgt(empty_array), np.array([0, 1, 0]))


def test_pattern_logic_lib_functions_numerical_stability():
    """Test numerical stability of pattern_logic_lib functions."""
    # Test make_anom with extreme values
    extreme_exp = xr.DataArray(
        np.array([[[1e10]], [[-1e10]], [[1e-10]]]),
        coords=([0, 1, 2], [0], [0]),
        dims=("year", "lat", "lon"),
    )

    extreme_cnt = xr.DataArray(
        np.array([[[0]], [[0]], [[0]]]),
        coords=([0, 1, 2], [0], [0]),
        dims=("year", "lat", "lon"),
    )

    extreme_anom = pattern_logic_lib.make_anom(extreme_exp, extreme_cnt)
    assert isinstance(extreme_anom, xr.DataArray)
    assert np.isfinite(extreme_anom.values).all()
