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


def test_expotas():
    s1 = 5
    t1 = 25
    assert prpatt.expotas(0, s1, t1) == 0
    assert prpatt.expotas(random.randint(0, 200), s1, t1) < s1


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
    assert np.allclose(
        prpatt.wgt2(empty_array),
        np.array([[0.01, 0.01, 0.01], [1, 1, 1], [0.01, 0.01, 0.01]]),
    )
    assert np.allclose(
        prpatt.wgt3(empty_array),
        np.array(
            [
                [[0.01, 0.01, 0.01], [1, 1, 1], [0.01, 0.01, 0.01]],
                [[0.01, 0.01, 0.01], [1, 1, 1], [0.01, 0.01, 0.01]],
                [[0.01, 0.01, 0.01], [1, 1, 1], [0.01, 0.01, 0.01]],
            ]
        ),
    )


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
    empty_array = empty_array.rename({"year": "time", "lat": "latitude"})
    assert prpatt.get_time_name(empty_array) == "time"
    assert prpatt.get_lat_name(empty_array) == "latitude"
    empty_array = empty_array.rename({"time": "seconds", "latitude": "deg_NS"})
    with pytest.raises(RuntimeError, match="Couldn't find a latitude coordinate"):
        prpatt.get_lat_name(empty_array)
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
    assert np.allclose(prpatt.global_mean(empty_array), empty_data.mean(0))


def test_prpatt_functions_edge_cases():
    """Test edge cases in prpatt functions to improve coverage."""
    # Test expotas with edge values
    assert prpatt.expotas(100, 1, 25) < 1  # Very high delta_tas
    assert prpatt.expotas(0.001, 10, 25) < 10  # Very small delta_tas

    # Test with negative delta_tas (cooling) - can produce negative results
    cooling_result = prpatt.expotas(-5, 5, 25)
    assert cooling_result < 5
    assert isinstance(cooling_result, (float, np.floating))  # Just check it's a number

    # Test with extreme temperature values
    extreme_result = prpatt.expotas(10, 5, 50)  # High base temperature
    assert extreme_result < 5


def test_prpatt_functions_weight_edge_cases():
    """Test weight functions with edge cases."""
    # Test with single latitude point (pole)
    pole_data = xr.DataArray(
        np.ones((1, 3)),
        coords=(np.array([90]), np.array([0, 120, 240])),
        dims=("lat", "lon"),
    )

    pole_weights = prpatt.wgt(pole_data)
    assert isinstance(pole_weights, (np.ndarray, xr.DataArray))
    assert len(pole_weights) == 1

    # Test with uniform latitude grid
    uniform_data = xr.DataArray(
        np.ones((5, 5)),
        coords=(np.linspace(-90, 90, 5), np.linspace(-180, 180, 5)),
        dims=("lat", "lon"),
    )

    uniform_weights = prpatt.wgt(uniform_data)
    assert isinstance(uniform_weights, (np.ndarray, xr.DataArray))
    if hasattr(uniform_weights, "__len__"):
        assert len(uniform_weights) == 5


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


def test_prpatt_coordinate_name_variations():
    """Test coordinate name detection with variations."""
    # Test alternative longitude names - use functions that actually exist
    alt_data = xr.DataArray(
        np.ones((3, 3)),
        coords=(np.array([-90, 0, 90]), np.array([0, 120, 240])),
        dims=("latitude", "longitude"),
    )

    lat_name = prpatt.get_lat_name(alt_data)
    assert lat_name == "latitude"

    # Test with 'lat' name
    std_data = xr.DataArray(
        np.ones((3, 3)),
        coords=(np.array([-90, 0, 90]), np.array([0, 120, 240])),
        dims=("lat", "lon"),
    )

    std_lat_name = prpatt.get_lat_name(std_data)
    assert std_lat_name == "lat"

    # Test that invalid coordinate names raise errors
    invalid_data = xr.DataArray(
        np.ones((3, 3)),
        coords=(np.array([-90, 0, 90]), np.array([0, 120, 240])),
        dims=("y_coord", "x_coord"),
    )

    try:
        prpatt.get_lat_name(invalid_data)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass  # Expected behavior
