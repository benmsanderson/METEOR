"""
Precipitation reference: aggregation commutes with windowing.

``_load_transform_reference`` fetches gridded CMIP6 data at *generation* time
purely to fit the precipitation gamma transform, and the timeseries path then
aggregates it per location. Because that aggregation is per-timestep and
linear, it commutes with both the output-window slice and the twelve-month
baseline mean -- so the aggregated 1D series can be baked into a bundle at
training time and still reproduce current behaviour for any requested window.

These tests pin that equivalence. They use synthetic fields: the claim is about
the algebra of the reduction, not about any particular CMIP6 model.
"""

import numpy as np
import pytest
import xarray as xr

from meteor.geo_data_utils import global_mean, regional_mean
from meteor.precipitation_transform import (
    fit_distribution_parameters_1d,
    fit_distribution_parameters_1d_seasonal,
)

N_LAT, N_LON = 24, 48
LATS = np.linspace(-86.25, 86.25, N_LAT)
LONS = np.linspace(3.75, 356.25, N_LON)
COMPOSITE_START_YEAR = 1850
# Span the real historical+ssp composite range so the test windows below are
# genuinely inside the data; an out-of-range window would slice to nothing and
# make the comparisons vacuously true.
N_YEARS = 200


@pytest.fixture(name="gridded_reference")
def fixture_gridded_reference():
    """
    Synthetic gridded monthly precipitation composite.

    Returns
    -------
    xr.DataArray
        Strictly positive field with dims (month, lat, lon).
    """
    rng = np.random.default_rng(31)
    n_month = N_YEARS * 12
    month = np.arange(n_month)
    seasonal = 1.0 + 0.4 * np.sin(2 * np.pi * month / 12)
    field = (
        seasonal[:, None, None]
        * (1.0 + 0.3 * np.cos(np.deg2rad(LATS))[None, :, None])
        * rng.gamma(shape=3.0, scale=0.5, size=(n_month, N_LAT, N_LON))
    )
    return xr.DataArray(
        field,
        coords={"month": month, "lat": LATS, "lon": LONS},
        dims=("month", "lat", "lon"),
    )


def _aggregate(field, location):
    """Reduce a field to a location the way the timeseries path does."""
    if location == "global":
        return global_mean(field)
    return regional_mean(field, region_code=location)


@pytest.mark.parametrize("location", ["global", "NEU", "SEA"])
@pytest.mark.parametrize("window", [(1900, 1949), (1990, 2039)])
def test_aggregation_commutes_with_window_slice(gridded_reference, location, window):
    """Slicing then aggregating equals aggregating then slicing."""
    start_year, end_year = window
    start = (start_year - COMPOSITE_START_YEAR) * 12
    stop = (end_year - COMPOSITE_START_YEAR + 1) * 12

    meteor_path = _aggregate(gridded_reference.isel(month=slice(start, stop)), location)
    bundle_path = _aggregate(gridded_reference, location).isel(month=slice(start, stop))
    np.testing.assert_allclose(
        np.asarray(bundle_path.values), np.asarray(meteor_path.values), rtol=1e-12
    )


@pytest.mark.parametrize("location", ["global", "NEU"])
def test_aggregation_commutes_with_first_year_baseline(gridded_reference, location):
    """The pr first-year baseline is the same computed either way."""
    start = (1990 - COMPOSITE_START_YEAR) * 12
    assert start + 12 <= gridded_reference.sizes["month"]

    # METEOR: mean the gridded first year, then aggregate.
    gridded_baseline = gridded_reference.isel(month=slice(start, start + 12)).mean(
        dim="month"
    )
    meteor_value = float(np.asarray(_aggregate(gridded_baseline, location).values))

    # Bundle: aggregate once, then mean the first twelve months.
    bundle_value = float(
        _aggregate(gridded_reference, location)
        .isel(month=slice(start, start + 12))
        .mean(dim="month")
        .values
    )
    assert bundle_value == pytest.approx(meteor_value, rel=1e-12)


@pytest.mark.parametrize("location", ["global", "NEU"])
def test_gamma_fit_from_baked_series_matches_gridded_path(gridded_reference, location):
    """Fitted transform parameters agree, seasonal and non-seasonal."""
    start = (1990 - COMPOSITE_START_YEAR) * 12
    stop = (2039 - COMPOSITE_START_YEAR + 1) * 12
    assert stop <= gridded_reference.sizes["month"]

    meteor_series = _aggregate(
        gridded_reference.isel(month=slice(start, stop)), location
    )
    baked_series = _aggregate(gridded_reference, location).isel(
        month=slice(start, stop)
    )

    for fit in (
        fit_distribution_parameters_1d,
        fit_distribution_parameters_1d_seasonal,
    ):
        from_gridded = fit(meteor_series, "gamma")
        from_baked = fit(baked_series, "gamma")
        assert set(from_gridded) == set(from_baked)
        for key, expected in from_gridded.items():
            np.testing.assert_allclose(
                np.asarray(from_baked[key], dtype=float),
                np.asarray(expected, dtype=float),
                rtol=1e-10,
                err_msg=f"{fit.__name__}[{key}] differs at {location}",
            )
