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
