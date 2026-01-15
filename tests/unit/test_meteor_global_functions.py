import numpy as np
import pytest
import xarray as xr

from meteor import meteor


def test_calculate_residual_and_do_crude_nan_cut():
    field = xr.DataArray(
        data=np.ones((10, 12, 13)),
        dims=["time", "lat", "lon"],
        coords=dict(
            lon=np.arange(13),
            lat=np.arange(12),
            time=np.arange(10),
        ),
    )

    to_subtract = xr.DataArray(
        data=np.zeros((10, 12, 13)),
        dims=["time", "lat", "lon"],
        coords=dict(
            lon=np.arange(13),
            lat=np.arange(12),
            time=np.arange(10),
        ),
    )
    assert np.allclose(
        meteor.calculate_residual_and_do_crude_nan_cut(field, to_subtract), field
    )

    field.loc[8:, :, :] = np.nan
    assert meteor.calculate_residual_and_do_crude_nan_cut(field, to_subtract).shape == (
        8,
        12,
        13,
    )

    field.loc[2, :, :] = np.nan
    with pytest.raises(
        ValueError,
        match="The dataset you are trying to emulate includes NaN values scattered throughout. METEOR does not currently support emulation such datasets",
    ):
        meteor.calculate_residual_and_do_crude_nan_cut(field, to_subtract)
