import numpy as np
import pytest
import xarray as xr

from meteor import geo_data_utils


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
    assert geo_data_utils.get_time_name(empty_array) == "year"
    assert geo_data_utils.get_lat_name(empty_array) == "lat"
    assert geo_data_utils.get_lon_name(empty_array) == "lon"
    empty_array = empty_array.rename(
        {"year": "time", "lat": "latitude", "lon": "longitude"}
    )
    assert geo_data_utils.get_time_name(empty_array) == "time"
    assert geo_data_utils.get_lat_name(empty_array) == "latitude"
    assert geo_data_utils.get_lon_name(empty_array) == "longitude"
    empty_array = empty_array.rename(
        {"time": "seconds", "latitude": "deg_NS", "longitude": "deg_EW"}
    )
    with pytest.raises(RuntimeError, match="Couldn't find a latitude coordinate"):
        geo_data_utils.get_lat_name(empty_array)
    with pytest.raises(RuntimeError, match="Couldn't find a longitude coordinate"):
        geo_data_utils.get_lon_name(empty_array)
    with pytest.raises(RuntimeError, match="Couldn't find a time coordinate"):
        geo_data_utils.get_time_name(empty_array)


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
    result = geo_data_utils.global_mean(empty_array)
    assert np.allclose(result, empty_data.mean(0))
    # Should have time dimension only
    assert result.dims == ("year",)

    # Should have correct length
    assert result.sizes["year"] == empty_array.sizes["year"]


def test_regional_functions():
    """Test regional functions in geo_data_utils module."""
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
    da_alt_lon = xr.DataArray(
        data,
        coords=(
            np.arange(4),
            np.linspace(-90, 90, 5),
            np.linspace(-180, 180, 6, endpoint=False),
        ),
        dims=("time", "lat", "lon"),
    )

    # Test get_weights_for_ds
    weights = geo_data_utils.get_weights_for_ds(da)
    assert isinstance(weights, xr.DataArray)
    assert weights.shape == (5, 6)

    ds_flat = da.mean(dim=["time", "lon"]).squeeze()
    weights_one_d = geo_data_utils.get_weights_for_ds(ds_flat)
    assert weights_one_d.shape == (5,)
    assert np.allclose(weights_one_d.values, weights.mean(dim="lon").values)

    # Test apply_weights_and_do_spatial_mean
    weighted_mean = geo_data_utils.apply_weights_and_do_spatial_mean(da, weights)
    assert isinstance(weighted_mean, xr.DataArray)
    assert weighted_mean.shape == (4,)

    # TODO add more asserts here
    region_mask = geo_data_utils.create_region_mask(
        da, bbox={"lat": (-30, 30), "lon": (0, 180)}
    )
    assert isinstance(region_mask, xr.DataArray)

    region_mask_2 = geo_data_utils.create_region_mask(da, mask=region_mask)
    assert np.allclose(region_mask.values, region_mask_2.values)
    # Test crossing prime meridian
    region_mask_3 = geo_data_utils.create_region_mask(
        da, bbox={"lat": (-30, 30), "lon": (-30, 30)}
    )
    assert not np.allclose(region_mask.values, region_mask_3.values)

    with pytest.raises(ValueError, match="Must provide either bbox or mask"):
        geo_data_utils.create_region_mask(da)

    regional_mean1 = geo_data_utils.regional_mean(da, region_mask=region_mask)
    assert regional_mean1.shape == (4,)
    assert regional_mean1.attrs["units"] == "K"
    assert regional_mean1.attrs["operation"] == "area_weighted_regional_mean"
    regional_mean2 = geo_data_utils.regional_mean(da, region_code="NEU")
    assert regional_mean2.shape == (4,)
    assert not np.allclose(regional_mean1.values, regional_mean2.values)
    with pytest.raises(
        ValueError,
        match="Region code 'XYZ' not found in AR6 regions. "
        r"Use list_ar6_regions\(\) to see available regions.",
    ):
        geo_data_utils.regional_mean(da, region_code="XYZ")
    point_data1 = geo_data_utils.extract_point(da, lat_point=60, lon_point=120)
    assert point_data1.shape == (4,)
    assert not np.allclose(point_data1.values, regional_mean1.values)
    point_data2 = geo_data_utils.extract_point(
        da, lat_point=60, lon_point=120, method="interp"
    )
    assert point_data2.shape == (4,)
    assert not np.allclose(point_data2.values, point_data1.values)
    point_data3 = geo_data_utils.extract_point(da, lat_point=60, lon_point=-240)
    point_data4 = geo_data_utils.extract_point(da_alt_lon, lat_point=60, lon_point=240)
    assert np.allclose(point_data1.values, point_data3.values)
    assert not np.allclose(point_data4.values, point_data3.values)
    with pytest.raises(
        ValueError, match="Unknown method 'invalid_method'. Use 'nearest' or 'interp'"
    ):
        geo_data_utils.extract_point(
            da, lat_point=60, lon_point=120, method="invalid_method"
        )
    with pytest.raises(
        ValueError, match="Must provide either region_code or region_mask"
    ):
        geo_data_utils.regional_mean(da)


def test_numerical_edge_cases():
    """Test numerical edge cases to improve coverage."""

    # Test global mean with edge case data
    edge_case_data = xr.DataArray(
        np.array([[[0.0, 1e-15], [1e15, -1e15]]]),  # Very small and very large numbers
        dims=["time", "lat", "lon"],
        coords={"time": [2000], "lat": [0, 1], "lon": [0, 1]},
        attrs={"units": "K"},
    )

    # Test that global_mean handles extreme values
    result = geo_data_utils.global_mean(edge_case_data)
    assert isinstance(result, xr.DataArray)
    assert np.isfinite(result.values).all()  # Should not produce inf or nan
    assert result.attrs["units"] == "K"
    assert result.attrs["operation"] == "area_weighted_global_mean"

    # Test with all-zero data
    zero_data = xr.DataArray(
        np.zeros((1, 2, 2)),
        dims=["time", "lat", "lon"],
        coords={"time": [2000], "lat": [0, 1], "lon": [0, 1]},
    )

    zero_result = geo_data_utils.global_mean(zero_data)
    assert isinstance(zero_result, xr.DataArray)
    assert zero_result.values[0] == 0.0


def test_global_mean_weighting_effect():
    """Test that latitude weighting affects results."""
    # Create simple test data
    lats = np.array([0, 30, 60])  # Different latitudes for weighting test
    lons = np.array([0, 180])

    # Data that's latitude-dependent (higher values at equator)
    data_values = np.array([[3, 3], [2, 2], [1, 1]])  # shape: (lat, lon)

    test_data = xr.DataArray(
        data_values, dims=["lat", "lon"], coords={"lat": lats, "lon": lons}
    )

    # With latitude weighting, equatorial values should have more influence
    weighted_mean = geo_data_utils.global_mean(test_data)

    # Simple unweighted mean
    unweighted_mean = test_data.mean()

    # Weighted mean should be higher due to higher equatorial values
    assert float(weighted_mean) > float(unweighted_mean)


def test_global_mean_custom_weights():
    """Test with custom weights."""
    # Create simple test data
    lats = np.array([0, 30, 60])
    lons = np.array([0, 180])
    data_values = np.array([[3, 3], [2, 2], [1, 1]])

    test_data = xr.DataArray(
        data_values, dims=["lat", "lon"], coords={"lat": lats, "lon": lons}
    )

    # Equal weights (should give simple average)
    equal_weights = xr.ones_like(test_data)
    result = geo_data_utils.global_mean(
        test_data, weights=equal_weights, normalize_weights=False
    )

    expected = test_data.mean()
    np.testing.assert_almost_equal(float(result), float(expected))


def test_global_mean_auto_coordinate_detection():
    """Test automatic coordinate detection."""
    # Create simple test data
    lats = np.array([0, 30, 60])
    lons = np.array([0, 180])
    data_values = np.array([[3, 3], [2, 2], [1, 1]])

    test_data = xr.DataArray(
        data_values, dims=["lat", "lon"], coords={"lat": lats, "lon": lons}
    )

    # Rename coordinates
    renamed_data = test_data.rename({"lat": "latitude", "lon": "longitude"})

    result = geo_data_utils.global_mean(renamed_data)
    assert result.ndim == 0  # Spatial dims should be removed


def test_extend_temperature_anomaly_timeseries_for_scaling():
    # Create a simple temperature anomaly timeseries with a gap
    temp_anomaly = xr.DataArray(
        [0.1, 0.2, 0.3], dims=["year"], coords={"year": [2000, 2001, 2002]}
    )
    target_years = np.array([1999, 2000, 2001, 2002, 2003])
    annual_temp_prediction_gm_anomaly = xr.DataArray(
        [0.1, 0.2, 0.3, 0.4, 0.5], dims=["year"], coords={"year": target_years}
    )

    extended = geo_data_utils.extend_temeperature_anomaly_timeseries_for_scaling(
        annual_temp_prediction_gm_anomaly, temp_anomaly
    )
    assert extended.shape == (5,)
    assert extended[1] == 0.1  # Base year value should be unchanged
    assert extended[0] == 0.1  # Year before base should be same as base
    assert extended[2] == 0.2
    assert extended[3] == 0.3
    assert (
        extended[4] == 0.5
    )  # Year after last should be same as annual_temp_prediction_gm_anomaly
    assert extended.coords["year"].values.tolist() == target_years.tolist()


def test_find_time_dim_and_cut():
    # Create a DataArray with time dimension and extra dimensions
    data = np.random.rand(10, 5, 5)
    n_time = 8
    n_lat = 5
    n_lon = 5

    cut_data = geo_data_utils.find_time_dim_and_cut(data, n_time, n_lat, n_lon)
    assert cut_data.shape == (n_time, n_lat, n_lon)
    # Test with time as second dimension
    data2 = np.random.rand(5, 10, 5)
    cut_data2 = geo_data_utils.find_time_dim_and_cut(data2, n_time, n_lat, n_lon)
    assert cut_data2.shape == (n_time, n_lat, n_lon)

    # Test with time as third dimension
    data3 = np.random.rand(5, 5, 10)
    cut_data3 = geo_data_utils.find_time_dim_and_cut(data3, n_time, n_lat, n_lon)
    assert cut_data3.shape == (n_time, n_lat, n_lon)
