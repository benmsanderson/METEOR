"""
Unit tests for precipitation transform functions.
"""

import numpy as np
import pytest
import xarray as xr

from meteor.precipitation_transform import (
    apply_distribution_transform,
    apply_empirical_quantile_mapping,
    fit_distribution_parameters_1d,
    fit_distribution_parameters_3d,
)


def test_fit_distribution_parameters_1d():
    """Test fitting distribution parameters for 1D data."""

    data = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    params = fit_distribution_parameters_1d(data, distribution="gamma")
    assert "shape" in params and "scale" in params
    assert params["shape"] > 0
    assert params["scale"] > 0
    data_neg = np.array([-1.0, 0.5, 1.0])
    with pytest.raises(
        ValueError,
        match="Gamma distribution requires positive data. "
        "Ensure you're using absolute precipitation, not anomalies.",
    ):
        params = fit_distribution_parameters_1d(data_neg, distribution="gamma")
    with pytest.raises(
        ValueError, match="Weibull distribution requires positive data."
    ):
        params = fit_distribution_parameters_1d(data_neg, distribution="weibull")
    with pytest.raises(
        ValueError, match="Generalized Gamma distribution requires positive data."
    ):
        params = fit_distribution_parameters_1d(data_neg, distribution="gengamma")

    params = fit_distribution_parameters_1d(data, distribution="weibull")
    assert np.allclose(params["shape"], 1.1054055920260153)
    assert np.allclose(params["scale"], 1.3668311960654322)

    params = fit_distribution_parameters_1d(data, distribution="lognorm")
    assert np.allclose(params["shape"], 1.1983190724023456)
    assert np.allclose(params["scale"], 0.7860030855966228)

    params = fit_distribution_parameters_1d(data, distribution="gengamma")
    assert np.allclose(params["a"], 0.0002357964916252947)
    assert np.allclose(params["c"], 3154.006251745962)
    assert np.allclose(params["scale"], 3.0065872015276325)

    with pytest.raises(
        ValueError,
        match="Unknown distribution: unknown_dist. "
        "Supported: 'gaussian', 'gamma', 'weibull', 'lognorm', 'gengamma'",
    ):
        params = fit_distribution_parameters_1d(data, distribution="unknown_dist")


def test_fit_distribution_parameters_3d():
    """Test fitting distribution parameters for 3D data."""

    data = np.ones(1)
    with pytest.raises(
        ValueError,
        match=r"Data must be 3D \(n_time, n_lat, n_lon\) or "
        r"4D \(n_ensemble, n_time, n_lat, n_lon\), got shape \(1,\)",
    ):
        params = fit_distribution_parameters_3d(data, distribution="gamma")

    data = np.ones((1, 1, 3, 2)) * 4  # 4D data: (n_ensemble, n_time, n_lat, n_lon)
    with pytest.raises(ValueError, match="Unknown distribution: unknown_dist"):
        params = fit_distribution_parameters_3d(data, distribution="unknown_dist")
    params = fit_distribution_parameters_3d(data, distribution="gamma")
    assert params["shape"].shape == (3, 2)
    assert params["scale"].shape == (3, 2)
    assert np.all(params["shape"] > 0)
    assert np.all(params["scale"] > 0)
    data_xarray = xr.DataArray(
        np.ones((1, 2, 2)) * 5,
        dims=("ntime", "nlat", "nlon"),
        coords={
            "ntime": np.arange(1),
            "nlat": np.arange(2),
            "nlon": np.arange(2),
        },
    )
    params = fit_distribution_parameters_3d(data_xarray, distribution="gaussian")
    print(params)
    assert params["mean"].shape == (2, 2)
    assert np.allclose(params["mean"], 5.0)
    assert params["std"].shape == (2, 2)
    assert np.allclose(params["std"].all(), 0.0)


def test_apply_distribution_transform():
    """Test applying distribution transform to data."""

    data = np.array([[0.1, 0.5, 1.0, 2.0, 3.0]])
    params_gauss = fit_distribution_parameters_1d(data, distribution="gaussian")
    params_gamma = fit_distribution_parameters_1d(data, distribution="gamma")
    transformed = apply_distribution_transform(
        data, params_gauss, params_gamma, target_dist="gamma"
    )
    assert transformed.shape == data.shape
    assert np.all(transformed >= 0)

    data_3d = np.ones((2, 2, 2)) * 4  # 3D data: (n_time, n_lat, n_lon)
    params_gauss_3d = fit_distribution_parameters_3d(data_3d, distribution="gaussian")
    params_3d = fit_distribution_parameters_3d(data_3d, distribution="gamma")
    transformed_3d = apply_distribution_transform(
        data_3d, params_gauss_3d, params_3d, target_dist="gamma"
    )
    assert transformed_3d.shape == data_3d.shape

    with pytest.raises(
        ValueError, match="Only 'gamma' distribution supported for 3D data, got weibull"
    ):
        apply_distribution_transform(
            data_3d, params_gauss_3d, params_3d, target_dist="weibull"
        )

    with pytest.raises(
        ValueError,
        match=r"For spatial parameters, expected 3D or 4D data, got shape \(2,\)",
    ):
        apply_distribution_transform(
            np.array([1.0, 2.0]), params_gauss_3d, params_3d, target_dist="gamma"
        )


def test_apply_empirical_quantile_mapping():
    """Test applying empirical quantile mapping."""

    data = np.array([[0.1, 0.5, 1.0, 2.0, 3.0]])
    reference = np.array([[0.2, 0.6, 1.5, 2.5, 4.0]])
    transformed = apply_empirical_quantile_mapping(data, reference)
    assert transformed.shape == data.shape
    assert np.all(transformed >= 0)

    data = np.ones((2, 2)) * 4  # 3D data: (n_time, n_lat, n_lon)
    reference = np.ones((2, 2)) * 5
    transformed = apply_empirical_quantile_mapping(data, reference)
    assert transformed.shape == data.shape

    data_xarray = xr.DataArray(
        np.ones((2, 2)) * 5,
        dims=("n_realisations", "n_time"),
        coords={
            "n_realisations": np.arange(2),
            "n_time": np.arange(2),
        },
    )
    transformed = apply_empirical_quantile_mapping(data_xarray, reference)


def test_fit_distribution_parameters_1d_lognorm_with_zeros():
    """lognorm fit adds 1e-6 offset when data contains zeros."""
    data = np.array([0.0, 0.5, 1.0, 2.0, 3.0])  # zero triggers offset path
    params = fit_distribution_parameters_1d(data, distribution="lognorm")
    assert "shape" in params and "scale" in params
    assert params["shape"] > 0
    assert params["scale"] > 0


def test_apply_distribution_transform_xarray_input_1d():
    """apply_distribution_transform accepts xarray input and returns xarray."""
    data = xr.DataArray(
        np.array([[0.1, 0.5, 1.0, 2.0, 3.0]]),
        dims=["n_realisations", "n_time"],
    )
    params_gauss = {"mean": 1.2, "std": 0.8}
    params_gamma = {"shape": 1.5, "scale": 0.8}
    transformed = apply_distribution_transform(
        data, params_gauss, params_gamma, target_dist="gamma"
    )
    assert isinstance(transformed, xr.DataArray)
    assert transformed.shape == data.shape
    assert np.all(transformed.values >= 0)


def test_apply_distribution_transform_1d_weibull():
    """apply_distribution_transform maps Gaussian → Weibull for 1D scalar params."""
    data = np.array([[0.1, 0.5, 1.0, 2.0, 3.0]])
    params_gauss = {"mean": 1.2, "std": 0.8}
    params_weibull = fit_distribution_parameters_1d(
        np.array([0.1, 0.5, 1.0, 2.0, 3.0]), distribution="weibull"
    )
    transformed = apply_distribution_transform(
        data, params_gauss, params_weibull, target_dist="weibull"
    )
    assert transformed.shape == data.shape
    assert np.all(np.isfinite(transformed))


def test_apply_distribution_transform_1d_lognorm():
    """apply_distribution_transform maps Gaussian → LogNorm for 1D scalar params."""
    data = np.array([[0.1, 0.5, 1.0, 2.0, 3.0]])
    params_gauss = {"mean": 1.2, "std": 0.8}
    params_lognorm = fit_distribution_parameters_1d(
        np.array([0.1, 0.5, 1.0, 2.0, 3.0]), distribution="lognorm"
    )
    transformed = apply_distribution_transform(
        data, params_gauss, params_lognorm, target_dist="lognorm"
    )
    assert transformed.shape == data.shape
    assert np.all(np.isfinite(transformed))


def test_apply_distribution_transform_1d_gengamma():
    """apply_distribution_transform maps Gaussian → GenGamma for 1D scalar params."""
    data = np.array([[0.1, 0.5, 1.0, 2.0, 3.0]])
    params_gauss = {"mean": 1.2, "std": 0.8}
    params_gengamma = fit_distribution_parameters_1d(
        np.array([0.1, 0.5, 1.0, 2.0, 3.0]), distribution="gengamma"
    )
    transformed = apply_distribution_transform(
        data, params_gauss, params_gengamma, target_dist="gengamma"
    )
    assert transformed.shape == data.shape
    assert np.all(np.isfinite(transformed))


def test_apply_distribution_transform_1d_unknown_raises():
    """apply_distribution_transform raises ValueError for unknown 1D target_dist."""
    data = np.array([[0.1, 0.5, 1.0, 2.0, 3.0]])
    params_gauss = {"mean": 1.2, "std": 0.8}
    params_gamma = {"shape": 1.5, "scale": 0.8}
    with pytest.raises(ValueError, match="Unknown target distribution: bad_dist"):
        apply_distribution_transform(
            data, params_gauss, params_gamma, target_dist="bad_dist"
        )


def test_apply_distribution_transform_1d_wrong_ndim_raises():
    """apply_distribution_transform raises ValueError when 1D data is not 2D array."""
    data = np.array([0.1, 0.5, 1.0, 2.0, 3.0])  # 1D, not (n_real, n_time)
    params_gauss = {"mean": 1.2, "std": 0.8}
    params_gamma = {"shape": 1.5, "scale": 0.8}
    with pytest.raises(ValueError, match="For scalar parameters"):
        apply_distribution_transform(
            data, params_gauss, params_gamma, target_dist="gamma"
        )


def test_apply_empirical_quantile_mapping_xarray_target():
    """apply_empirical_quantile_mapping accepts xarray target_data."""
    data = np.array([[0.1, 0.5, 1.0, 2.0, 3.0]])
    reference = xr.DataArray(np.array([0.2, 0.6, 1.5, 2.5, 4.0]), dims=["time"])
    transformed = apply_empirical_quantile_mapping(data, reference)
    assert transformed.shape == data.shape
    assert np.all(transformed >= 0)
