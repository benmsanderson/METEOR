import random

import numpy as np
import pytest
import xarray as xr

from meteor import prpatt


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
    test = prpatt.make_anom(ds_exp, ds_cnt)
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

    result = prpatt.make_anom(ds_exp, ds_cnt)

    # Should return a dataset
    assert isinstance(result, xr.Dataset)

    # Should have same variables as input
    assert "tas" in result

    # Should have same dimensions
    assert set(result["tas"].dims) == set(["time"]).union(
        set(ds_exp["tas"].dims) - set(["year"])
    )


# def test_pmodel_function():
#     """Test the pmodel function."""
#     # Test with simple parameters
#     pars = Mock()
#     pars.s1 = 1.0
#     pars.t1 = 2.0
#     pars.s2 = 0.5
#     pars.t2 = 1.0
#     pars.s3 = 0.0
#     pars.t3 = 1.0

#     n_times = 10

#     result = prpatt.pmodel(pars, n_times)

#     # Should return an array
#     assert isinstance(result, np.ndarray)

#     # Should have correct length
#     assert len(result) == n_times

#     # Should be real values
#     assert np.all(np.isreal(result))

# def test_residual_function():
#     """Test the residual function to hit lines 406-408."""
#     # Create simple test data to test the residual function
#     # Mock parameters object
#     class MockParams:
#         def __getitem__(self, key):
#             return 1.0

#     mock_pars = MockParams()
#     mode_weights = np.array([0.5, 0.8])
#     test_data = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 time steps, 2 modes

#     # Test the residual calculation directly
#     # This should hit the return statement at lines 406-408
#     result = prpatt.residual(mock_pars, mode_weights, test_data)

#     # Should return an array with the same shape as test_data
#     assert isinstance(result, np.ndarray)
#     assert result.shape == test_data.shape

# def test_expfun_function():
#     """Test the expfun function."""
#     # Test with simple parameters
#     t = np.array([0, 1, 2, 3])
#     pars = Mock()
#     pars.s1 = 1.0
#     pars.t1 = 2.0
#     pars.s2 = 0.5
#     pars.t2 = 1.0
#     pars.s3 = 0.0  # Zero third component
#     pars.t3 = 1.0
#     result = prpatt.expfun(t, pars)

#     # Should return array same shape as t
#     assert result.shape == t.shape

#     # Should be real values
#     assert np.all(np.isreal(result))

#     # At t=0, should be 0
#     assert np.isclose(result[0], 0)


def test_expotas():
    s1 = 5
    t1 = 25
    assert prpatt.expotas(0, s1, t1) == 0
    assert prpatt.expotas(random.randint(0, 200), s1, t1) < s1
    # Test with zero coefficient
    result = prpatt.expotas(1.0, 0.0, 1.0)
    assert result == 0.0

    # Test with scalar inputs
    time = 1.0
    coeff = 2.0
    decay_time = 3.0

    result = prpatt.expotas(time, coeff, decay_time)

    # Should return expected exponential value
    expected = coeff * (1 - np.exp(-time / decay_time))
    assert np.isclose(result, expected)

    # Test with array inputs
    times = np.array([0, 1, 2, 5, 10])
    results = prpatt.expotas(times, coeff, decay_time)

    # Should be same shape as input
    assert results.shape == times.shape

    # Should be monotonically increasing (for positive coeff)
    assert np.all(np.diff(results) >= 0)

    # At t=0, should be 0
    result_t0 = prpatt.expotas(0, coeff, decay_time)
    assert np.isclose(result_t0, 0)

    # At t=infinity, should approach coeff
    result_large_t = prpatt.expotas(1000, coeff, decay_time)
    assert np.isclose(result_large_t, coeff, rtol=1e-3)

    # Test with very small decay time
    result = prpatt.expotas(1.0, 1.0, 0.001)
    assert np.isclose(result, 1.0, rtol=1e-3)

    # Test with very large decay time
    result = prpatt.expotas(1.0, 1.0, 1000.0)
    assert result < 0.01  # Should be very small


# def test_make_amat_function():
#     """Test the make_amat function."""
#     # Test with simple parameters
#     pars = Mock()
#     pars.s1 = 1.0
#     pars.t1 = 2.0
#     pars.s2 = 0.5
#     pars.t2 = 1.0
#     pars.s3 = 0.0
#     pars.t3 = 1.0

#     nt = 10  # Number of time steps

#     result = prpatt.make_amat(pars, nt)

#     # Should return a matrix
#     assert isinstance(result, np.ndarray)

#     # Should have expected shape (nt x nt)
#     assert result.shape == (nt, nt)

#     # Should be real values
#     assert np.all(np.isreal(result))


def test_make_pmat_function():
    """Test the make_pmat function."""
    # Test with simple parameters
    tauvec = np.array([1.0, 2.0, 3.0])
    nt = 5

    result = prpatt.make_pmat(tauvec, nt)

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
    assert np.allclose(prpatt.wgt(empty_array), np.array([0, 1, 0]))


def test_get_dim_names():
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
    assert prpatt.get_time_name(empty_array) == "year"
    assert prpatt.get_lat_name(empty_array) == "lat"
    assert prpatt.get_lon_name(empty_array) == "lon"
    empty_array = empty_array.rename(
        {"year": "time", "lat": "latitude", "lon": "longitude"}
    )
    assert prpatt.get_time_name(empty_array) == "time"
    assert prpatt.get_lat_name(empty_array) == "latitude"
    assert prpatt.get_lon_name(empty_array) == "longitude"
    empty_array = empty_array.rename(
        {"time": "seconds", "latitude": "deg_NS", "longitude": "deg_EW"}
    )
    with pytest.raises(RuntimeError, match="Couldn't find a latitude coordinate"):
        prpatt.get_lat_name(empty_array)
    with pytest.raises(RuntimeError, match="Couldn't find a longitude coordinate"):
        prpatt.get_lon_name(empty_array)
    with pytest.raises(RuntimeError, match="Couldn't find a time coordinate"):
        prpatt.get_time_name(empty_array)


def test_global_mean():
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
    result = prpatt.global_mean(empty_array)
    assert np.allclose(result, empty_data.mean(0))
    # Should have time dimension only
    assert result.dims == ("year",)

    # Should have correct length
    assert result.sizes["year"] == empty_array.sizes["year"]


def test_prpatt_functions_numerical_stability():
    """Test numerical stability of prpatt functions."""
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

    extreme_anom = prpatt.make_anom(extreme_exp, extreme_cnt)
    assert isinstance(extreme_anom, xr.DataArray)
    assert np.isfinite(extreme_anom.values).all()


def test_regional_functions():
    """Test regional functions in prpatt module."""
    data = np.random.rand(4, 5, 6)
    da = xr.DataArray(
        data,
        coords=(
            np.arange(4),
            np.linspace(-90, 90, 5),
            np.linspace(0, 360, 6, endpoint=False),
        ),
        dims=("time", "lat", "lon"),
        attrs={"units": "K"},
    )

    # Test get_weights_for_ds
    weights = prpatt.get_weights_for_ds(da)
    assert isinstance(weights, xr.DataArray)
    assert weights.shape == (5, 6)

    # Test apply_weights_and_do_spatial_mean
    weighted_mean = prpatt.apply_weights_and_do_spatial_mean(da, weights)
    assert isinstance(weighted_mean, xr.DataArray)
    assert weighted_mean.shape == (4,)

    # TODO add more asserts here
    region_mask = prpatt.create_region_mask(
        da, bbox={"lat": (-30, 30), "lon": (0, 180)}
    )
    assert isinstance(region_mask, xr.DataArray)

    region_mask_2 = prpatt.create_region_mask(da, mask=region_mask)
    assert np.allclose(region_mask.values, region_mask_2.values)

    with pytest.raises(ValueError, match="Must provide either bbox or mask"):
        prpatt.create_region_mask(da)

    regional_mean1 = prpatt.regional_mean(da, region_mask=region_mask)
    assert regional_mean1.shape == (4,)
    regional_mean2 = prpatt.regional_mean(da, region_code="NEU")
    assert regional_mean2.shape == (4,)
    assert not np.allclose(regional_mean1.values, regional_mean2.values)
    point_data1 = prpatt.extract_point(da, lat_point=60, lon_point=120)
    assert point_data1.shape == (4,)
    assert not np.allclose(point_data1.values, regional_mean1.values)
    point_data2 = prpatt.extract_point(da, lat_point=60, lon_point=120, method="interp")
    assert point_data2.shape == (4,)
    assert not np.allclose(point_data2.values, point_data1.values)
    with pytest.raises(
        ValueError, match="Unknown method 'invalid_method'. Use 'nearest' or 'interp'"
    ):
        prpatt.extract_point(da, lat_point=60, lon_point=120, method="invalid_method")


def test_numerical_edge_cases():
    """Test numerical edge cases to improve coverage."""

    # Test global mean with edge case data
    edge_case_data = xr.DataArray(
        np.array([[[0.0, 1e-15], [1e15, -1e15]]]),  # Very small and very large numbers
        dims=["time", "lat", "lon"],
        coords={"time": [2000], "lat": [0, 1], "lon": [0, 1]},
    )

    # Test that global_mean handles extreme values
    result = prpatt.global_mean(edge_case_data)
    assert isinstance(result, xr.DataArray)
    assert np.isfinite(result.values).all()  # Should not produce inf or nan

    # Test with all-zero data
    zero_data = xr.DataArray(
        np.zeros((1, 2, 2)),
        dims=["time", "lat", "lon"],
        coords={"time": [2000], "lat": [0, 1], "lon": [0, 1]},
    )

    zero_result = prpatt.global_mean(zero_data)
    assert isinstance(zero_result, xr.DataArray)
    assert zero_result.values[0] == 0.0
