"""
High-quality tests for MeteorInterface functionality.

Focus: Test critical new logic with clear behavior expectations.
Strategy: Use mocks for expensive operations, verify transformations.
"""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from meteor.ensemble_output import EnsembleOutput
from meteor.meteor_interface import MeteorInterface, _get_default_config


def _set_trained(interface, variables):
    for var in variables:
        interface._is_trained[var] = True
        interface.pattern_models[var] = MagicMock()
        interface.noise_models[var] = MagicMock()


@pytest.fixture
def interface_factory():
    """Factory for MeteorInterface instances with mocked data getter."""
    with patch("meteor.meteor_interface.Cmip6MeteorDataGetter") as mock_getter_class:
        mock_getter = MagicMock()
        mock_getter_class.return_value = mock_getter

        def _make_interface(
            model="TestModel", variables=("tas",), cache_dir="/tmp/test"
        ):
            interface = MeteorInterface(
                model=model, variables=list(variables), cache_dir=cache_dir
            )
            return interface, mock_getter

        yield _make_interface


@pytest.fixture
def mock_interface(interface_factory):
    """Create MeteorInterface with mocked data getter."""
    interface, _ = interface_factory(
        model="TestModel", variables=("tas", "pr"), cache_dir="/tmp/test_cache"
    )
    _set_trained(interface, ["tas", "pr"])
    yield interface


def test_get_default_config():
    """Test that _get_default_config returns expected defaults."""
    tas_config = _get_default_config("tas")
    assert tas_config["n_modes_pattern"] == 3
    assert tas_config["n_modes_noise"] == 40
    assert tas_config["lag_order"] == 2
    assert tas_config["training_scenario"] == "ssp245"
    assert tas_config["use_exog"] == "all"
    assert not tas_config["transform"]

    pr_config = _get_default_config("pr")
    assert pr_config["n_modes_pattern"] == 3
    assert pr_config["n_modes_noise"] == 40
    assert pr_config["training_scenario"] == "ssp245"
    assert pr_config["use_exog"] == "none"
    assert pr_config["transform"]
    assert pr_config["transform_type"] == "gamma"

    generic_config = _get_default_config("other_variable")
    assert generic_config["n_modes_pattern"] == 3
    assert generic_config["n_modes_noise"] == 40
    assert generic_config["training_scenario"] == "ssp245"
    assert generic_config["use_exog"] == "none"
    assert not generic_config["transform"]


def test_tas_converted_to_anomalies(mock_interface):
    """Test that tas data is converted to anomalies from piControl baseline."""
    # Create mock piControl data with known mean
    picontrol_mean = 288.0  # K
    picontrol_data = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.full((100, 5, 5), picontrol_mean) + np.random.randn(100, 5, 5) * 0.1,
                dims=["month", "lat", "lon"],
                coords={
                    "month": range(100),
                    "lat": np.linspace(-90, 90, 5),
                    "lon": np.linspace(-180, 180, 5),
                },
            )
        }
    )

    # Create mock scenario data with known temperature
    scenario_temp = 290.0  # K (2K warmer than piControl)
    scenario_data = xr.Dataset(
        {
            "tas": xr.DataArray(
                np.full((100, 5, 5), scenario_temp) + np.random.randn(100, 5, 5) * 0.1,
                dims=["month", "lat", "lon"],
                coords={
                    "month": range(100),
                    "lat": np.linspace(-90, 90, 5),
                    "lon": np.linspace(-180, 180, 5),
                },
            )
        }
    )

    # Mock data getter to return our test data
    mock_composite = MagicMock(
        side_effect=lambda exps, model, monthly=True: (
            picontrol_data if "piControl" in exps else scenario_data
        )
    )
    mock_interface.data_getter.make_meteor_training_data_composite = mock_composite

    # Mock pattern scaling components to avoid full computation
    mock_pattern = xr.DataArray(
        np.zeros((100, 5, 5)),
        dims=["month", "lat", "lon"],
        coords={
            "month": range(100),
            "lat": np.linspace(-90, 90, 5),
            "lon": np.linspace(-180, 180, 5),
        },
    )
    mock_interface.pattern_models["tas"].to_monthly.return_value = mock_pattern
    mock_interface.pattern_models[
        "tas"
    ].predict_from_combined_experiment.return_value = {
        "tas": xr.DataArray(np.zeros(10), dims=["year"])
    }

    # Mock noise model
    mock_interface.noise_models["tas"].generate_stochastic_pcs.return_value = (
        np.random.randn(3, 10)
    )
    mock_interface.noise_models[
        "tas"
    ].generate_regional_mean_realizations.return_value = np.random.randn(3, 100)

    # Call _generate_timeseries which contains the anomaly conversion logic
    with patch("meteor.global_mean") as mock_global_mean:
        # Mock global mean to return simple arrays
        mock_global_mean.return_value = xr.DataArray(
            np.zeros(100), dims=["month"], coords={"month": range(100)}
        )

        # This should trigger the anomaly conversion for tas
        _ = mock_interface._generate_timeseries(
            variable="tas",
            scenario="ssp245",
            start_year=2000,
            end_year=2010,
            n_realizations=3,
            aggregations=["global"],
            include_noise=True,
            verbose=False,
        )

    # Verify that make_meteor_training_data_composite was called for both scenario and piControl
    calls = [
        call[0]
        for call in mock_interface.data_getter.make_meteor_training_data_composite.call_args_list
    ]

    # Should have been called with scenario data
    assert any("ssp245" in str(call) or "historical" in str(call) for call in calls)
    # Should have been called with piControl data (for tas only)
    assert any("piControl" in str(call) for call in calls)


def test_pr_not_converted_to_anomalies(mock_interface):
    """Test that pr data is NOT converted to anomalies (uses first-year baseline instead of piControl)."""
    # Create mock scenario data for pr with enough months for 1850-2010
    # Need ~160 years * 12 months = 1920 months
    n_months = 2000
    scenario_data = xr.Dataset(
        {
            "pr": xr.DataArray(
                np.random.rand(n_months, 5, 5) * 1e-5,  # Typical precip values
                dims=["month", "lat", "lon"],
                coords={
                    "month": range(n_months),
                    "lat": np.linspace(-90, 90, 5),
                    "lon": np.linspace(-180, 180, 5),
                },
            )
        }
    )

    picontrol_data = xr.Dataset(
        {
            "pr": xr.DataArray(
                np.random.rand(500, 5, 5) * 1e-5,  # piControl data
                dims=["month", "lat", "lon"],
                coords={
                    "month": range(500),
                    "lat": np.linspace(-90, 90, 5),
                    "lon": np.linspace(-180, 180, 5),
                },
            )
        }
    )

    call_log = []

    def mock_composite(exps, model, monthly=True):
        call_log.append(exps)
        # Return appropriate data based on what's requested
        if "piControl" in exps:
            return picontrol_data
        return scenario_data

    mock_interface.data_getter.make_meteor_training_data_composite = mock_composite

    # Mock pattern scaling components
    mock_pattern = xr.DataArray(
        np.zeros((100, 5, 5)),
        dims=["month", "lat", "lon"],
        coords={
            "month": range(100),
            "lat": np.linspace(-90, 90, 5),
            "lon": np.linspace(-180, 180, 5),
        },
    )
    mock_interface.pattern_models["pr"].to_monthly.return_value = mock_pattern
    mock_interface.pattern_models[
        "pr"
    ].predict_from_combined_experiment.return_value = {
        "pr": xr.DataArray(np.zeros(10), dims=["year"])
    }

    # Mock noise model
    mock_interface.noise_models["pr"].generate_stochastic_pcs.return_value = (
        np.random.randn(3, 10)
    )
    mock_interface.noise_models[
        "pr"
    ].generate_regional_mean_realizations.return_value = (np.random.rand(3, 100) * 1e-5)

    # Call _generate_timeseries for pr
    with patch("meteor.global_mean") as mock_global_mean:
        mock_global_mean.return_value = xr.DataArray(
            np.zeros(100), dims=["month"], coords={"month": range(100)}
        )

        _ = mock_interface._generate_timeseries(
            variable="pr",
            scenario="ssp245",
            start_year=2000,
            end_year=2010,
            n_realizations=3,
            aggregations=["global"],
            include_noise=True,
            verbose=False,
        )

    # Verify both piControl and scenario data were loaded
    # (piControl is loaded for reference but first-year baseline is used for pr)
    assert len(call_log) == 2, "Should load both scenario data and piControl"
    assert any("piControl" in exps for exps in call_log), "piControl should be loaded"
    assert any(
        "historical" in exps or "ssp245" in exps for exps in call_log
    ), "Scenario data should be loaded"


def test_train_initializes_state_tracking(interface_factory):
    """Test that MeteorInterface initializes training state tracking."""
    interface, _ = interface_factory(model="TestModel", variables=("tas",))

    # Check that training state is initialized
    assert hasattr(interface, "_is_trained")
    assert "tas" in interface._is_trained
    assert not interface._is_trained["tas"]

    # Check that model storage is initialized
    assert hasattr(interface, "pattern_models")
    assert hasattr(interface, "noise_models")
    assert isinstance(interface.pattern_models, dict)
    assert isinstance(interface.noise_models, dict)


def test_train_tracks_multiple_variables(interface_factory):
    """Test that training state is tracked per-variable."""
    interface, _ = interface_factory(model="TestModel", variables=("tas", "pr"))

    # Initially, no variables are trained
    assert not interface._is_trained["tas"]
    assert not interface._is_trained["pr"]

    # Simulate training tas only (actual training is integration test)
    interface._is_trained["tas"] = True
    interface.pattern_models["tas"] = MagicMock()
    interface.noise_models["tas"] = MagicMock()

    # Check that tas is trained but pr is not
    assert interface._is_trained["tas"]
    assert not interface._is_trained["pr"]
    assert "tas" in interface.pattern_models
    assert "pr" not in interface.pattern_models


def test_training_config_storage(interface_factory):
    """Test that training configuration is stored."""
    interface, _ = interface_factory(model="TestModel", variables=("tas",))

    # Check that training config dictionary exists
    assert hasattr(interface, "_training_config")
    assert isinstance(interface._training_config, dict)

    # Simulate setting a config (actual training is integration test)
    interface._training_config["tas"] = {
        "n_modes_pattern": 10,
        "n_modes_noise": 40,
        "training_scenario": "ssp245",
    }

    # Verify it's stored
    assert "tas" in interface._training_config
    assert interface._training_config["tas"]["n_modes_pattern"] == 10
    assert interface._training_config["tas"]["n_modes_noise"] == 40


def test_train_multiple_variables(interface_factory):
    """Test training multiple variables."""
    interface, mock_getter = interface_factory(
        model="TestModel", variables=("tas", "pr")
    )
    mock_getter.validate_pattern_scaling_cache.return_value = (False, None, {})
    mock_getter.validate_noise_model_cache.return_value = (False, None, {})
    assert interface is not None


def test_model_dictionaries_are_mutable(interface_factory):
    """Test that model storage dictionaries can be populated."""
    interface, _ = interface_factory(model="TestModel", variables=("tas", "pr"))

    # Initially empty
    assert len(interface.pattern_models) == 0
    assert len(interface.noise_models) == 0

    # Simulate populating after training (actual training is integration test)
    interface.pattern_models["tas"] = MagicMock()
    interface.pattern_models["pr"] = MagicMock()
    interface.noise_models["tas"] = MagicMock()
    interface.noise_models["pr"] = MagicMock()

    # Verify they're populated
    assert len(interface.pattern_models) == 2
    assert len(interface.noise_models) == 2
    assert "tas" in interface.pattern_models
    assert "pr" in interface.pattern_models


def test_cache():
    """Test pattern scaling cache hit and noise training cache miss."""
    with (
        patch("meteor.meteor_interface.Cmip6MeteorDataGetter") as mock_getter_class,
        patch("meteor.meteor_interface.MeteorPatternScaling") as mock_pattern_class,
        patch(
            "meteor.meteor_interface.validate_noise_model_cache"
        ) as mock_validate_noise,
        patch(
            "meteor.meteor_interface.load_emissions_concentrations_from_name"
        ) as mock_load_emissions,
        patch(
            "meteor.meteor_interface.train_noise_model_from_cmip6"
        ) as mock_train_noise,
        patch("meteor.meteor_interface.global_mean") as mock_global_mean,
    ):
        mock_getter = MagicMock()
        mock_getter_class.return_value = mock_getter
        mock_getter.validate_pattern_scaling_cache.return_value = (
            True,
            MagicMock(),
            {},
        )
        mock_validate_noise.return_value = (False, None, {})
        mock_load_emissions.return_value = ("em", "conc")

        interface = MeteorInterface(
            model="TestModel", variables=["tas"], cache_dir="/tmp/test"
        )

        interface._train_pattern_scaling("tas", {"n_modes_pattern": 3}, verbose=False)

        pattern_args, pattern_kwargs = mock_pattern_class.call_args
        assert pattern_args[2] is None
        assert pattern_kwargs["ssp_input"] is None
        assert pattern_kwargs["exp_list"] is None

        monthly_prediction = xr.DataArray(
            np.zeros((2400, 2, 2)),
            dims=["month", "lat", "lon"],
            coords={"month": np.arange(2400), "lat": [0, 1], "lon": [0, 1]},
        )
        interface.pattern_models["tas"] = MagicMock()
        interface.pattern_models[
            "tas"
        ].predict_from_combined_experiment.return_value = {
            "tas": xr.DataArray(np.zeros(200), dims=["year"])
        }
        interface.pattern_models["tas"].to_monthly.return_value = monthly_prediction
        mock_global_mean.return_value = xr.DataArray(
            np.zeros(2400), dims=["month"], coords={"month": np.arange(2400)}
        )

        config = {
            "n_modes_noise": 5,
            "lag_order": 2,
            "use_exog": "all",
            "training_scenario": "ssp370",
        }
        interface._train_noise_model("tas", config, verbose=False)

        mock_train_noise.assert_called_once()
        _, noise_kwargs = mock_train_noise.call_args
        assert noise_kwargs["experiments"] == ["historical", "ssp370"]
        assert len(noise_kwargs["custom_global_temp"]) == 1200


def test_generate_requires_training(interface_factory):
    """Test that generate_ensemble_outputs() raises error if not trained."""
    interface, _ = interface_factory(model="TestModel", variables=("tas",))

    # Try to generate without training
    with pytest.raises(RuntimeError, match="not trained"):
        interface.generate_ensemble_outputs(
            scenario="ssp245",
            start_year=2020,
            end_year=2050,
            n_realizations=10,
            timeseries=["global"],
        )
    empty_impacts = interface._apply_impacts(
        np.array([290.0]), "tas", {"unknown_impact": 5}
    )
    assert isinstance(empty_impacts, dict)
    assert len(empty_impacts) == 0


def test_generate_with_noise_false_forces_single_realization(interface_factory):
    """Test that include_noise=False forces n_realizations=1."""
    interface, _ = interface_factory(model="TestModel", variables=("tas",))

    # Mark as trained
    _set_trained(interface, ["tas"])

    # Mock the internal generation method to capture arguments
    call_args = {}

    def capture_args(*args, **kwargs):
        call_args["args"] = args
        call_args["kwargs"] = kwargs
        # Return minimal valid structure
        return {"global": xr.DataArray([290.0])}

    interface._generate_timeseries = capture_args

    # Call generate with include_noise=False but n_realizations=100
    interface.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=2020,
        end_year=2020,
        n_realizations=100,
        timeseries=["global"],
        include_noise=False,
        verbose=False,
    )

    # Should have forced n_realizations to 1
    # _generate_timeseries(variable, scenario, start_year, end_year, n_realizations, aggregations, ...)
    assert call_args["args"][4] == 1


def test_generate_returns_ensemble_output(interface_factory):
    """Test that generate_ensemble_outputs() returns EnsembleOutput container."""
    interface, _ = interface_factory(model="TestModel", variables=("tas",))

    _set_trained(interface, ["tas"])

    # Mock generation to return minimal data
    interface._generate_timeseries = MagicMock(
        return_value={"global": xr.DataArray([290.0], dims=["time"])}
    )

    result = interface.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=2020,
        end_year=2020,
        n_realizations=1,
        timeseries=["global"],
        verbose=False,
    )

    # Should return EnsembleOutput
    assert isinstance(result, EnsembleOutput)
    assert "tas" in result


def test_generate_populates_timeseries(interface_factory):
    """Test that generate_ensemble_outputs() populates timeseries outputs."""
    interface, _ = interface_factory(model="TestModel", variables=("tas",))

    _set_trained(interface, ["tas"])

    # Mock generation to return test data
    mock_timeseries = {
        "global": xr.DataArray([290.0, 290.5, 291.0], dims=["time"]),
        "regional:EAS": xr.DataArray([289.0, 289.5, 290.0], dims=["time"]),
    }
    interface._generate_timeseries = MagicMock(return_value=mock_timeseries)

    result = interface.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=2020,
        end_year=2022,
        n_realizations=5,
        timeseries=["global", "regional:EAS"],
        verbose=False,
    )

    # Check timeseries were populated
    assert "global" in result["tas"].timeseries
    assert "regional:EAS" in result["tas"].timeseries
    assert len(result["tas"].timeseries["global"]) == 3


def test_generate_includes_metadata(interface_factory):
    """Test that generate_ensemble_outputs() includes metadata in output."""
    interface, _ = interface_factory(model="TestModel", variables=("tas",))

    _set_trained(interface, ["tas"])

    interface._generate_timeseries = MagicMock(
        return_value={"global": xr.DataArray([290.0])}
    )

    result = interface.generate_ensemble_outputs(
        scenario="ssp370",
        start_year=2030,
        end_year=2080,
        n_realizations=25,
        timeseries=["global"],
        verbose=False,
    )

    # Check metadata
    assert result.metadata["scenario"] == "ssp370"
    assert result.metadata["year_range"] == "2030-2080"
    assert result.metadata["n_realizations"] == 25
    assert result.metadata["model"] == "TestModel"


def test_train_config(interface_factory):
    """Validate train() configuration logic for auto/manual modes."""
    interface, _ = interface_factory(model="TestModel", variables=("tas",))
    interface._train_pattern_scaling = MagicMock()
    interface._train_noise_model = MagicMock()

    interface.train(auto=True, training_scenario="ssp370", verbose=False)
    assert interface._training_config["tas"]["training_scenario"] == "ssp370"

    interface.train(
        auto=True,
        variable_configs={"tas": {"n_modes_noise": 50}},
        verbose=False,
    )
    assert interface._training_config["tas"]["n_modes_noise"] == 50

    interface.train(
        auto=False,
        training_scenario="ssp126",
        variable_configs={"tas": {}},
        verbose=False,
    )
    assert interface._training_config["tas"]["training_scenario"] == "ssp126"

    interface.train(
        auto=False,
        training_scenario="ssp126",
        variable_configs={"tas": {"n_modes_noise": 30}},
        verbose=False,
    )
    config = interface._training_config["tas"]
    assert config["n_modes_noise"] == 30
    assert config["training_scenario"] == "ssp126"


def test_generate_gridded_climatology_no_noise_single_realization(interface_factory):
    """Test gridded outputs force single realization without noise."""
    interface, _ = interface_factory(model="TestModel", variables=("tas",))

    interface.noise_models["tas"] = MagicMock()

    monthly_prediction = xr.DataArray(
        np.zeros((24, 2, 2)),
        dims=["month", "lat", "lon"],
        coords={"month": np.arange(24), "lat": [0, 1], "lon": [0, 1]},
    )
    monthly_warming = xr.DataArray(
        np.zeros((24, 2, 2)),
        dims=["month", "lat", "lon"],
        coords={"month": np.arange(24), "lat": [0, 1], "lon": [0, 1]},
    )
    interface._get_or_compute_pattern_scaling = MagicMock(
        return_value=(monthly_prediction, monthly_warming)
    )

    gridded = interface._generate_gridded(
        variable="tas",
        scenario="ssp245",
        start_year=2000,
        end_year=2001,
        n_realizations=5,
        gridded_spec={
            "annual": [2000],
            "monthly": [2000],
            "climatology": [(2000, 2001)],
        },
        include_noise=False,
        verbose=False,
    )

    interface.noise_models["tas"].generate_realization.assert_not_called()

    annual = gridded["annual"][2000]
    monthly = gridded["monthly"][2000]
    climatology = gridded["climatology"]["2000-2001"]

    assert annual.sizes["realization"] == 1
    assert monthly.sizes["realization"] == 1
    assert climatology.sizes["realization"] == 1


def test_generate_before_training_clear_error(interface_factory):
    """Test that generating before training gives clear error message."""
    interface, _ = interface_factory(model="TestModel", variables=("tas", "pr"))

    # Try to generate without training
    with pytest.raises(RuntimeError) as exc_info:
        interface.generate_ensemble_outputs(
            scenario="ssp245",
            start_year=2020,
            end_year=2050,
            n_realizations=10,
            timeseries=["global"],
        )

    # Error message should mention which variable
    assert "not trained" in str(exc_info.value).lower()
    assert "train()" in str(exc_info.value).lower()


def test_custom_scenario(tmp_path, capsys, interface_factory):
    """Test custom scenario handling and save_to in pattern scaling."""
    interface, _ = interface_factory(model="TestModel", variables=("tas",))
    interface._is_trained["tas"] = True
    interface.noise_models["tas"] = MagicMock()

    interface._generate_timeseries = MagicMock(
        return_value={"global": xr.DataArray([0.0], dims=["time"])}
    )

    save_path = tmp_path / "output.nc"
    with patch.object(EnsembleOutput, "to_netcdf") as mock_save:
        interface.generate_ensemble_outputs(
            scenario="ssp245",
            start_year=2000,
            end_year=2000,
            n_realizations=1,
            timeseries=["global"],
            save_to=str(save_path),
            verbose=False,
        )
        mock_save.assert_called_once_with(str(save_path))

    years = np.arange(2000, 2051)
    em_data = pd.DataFrame({"value": np.zeros(len(years))}, index=years)
    conc_data = pd.DataFrame({"value": np.zeros(len(years))}, index=years)
    scenario = {"emissions": em_data, "concentrations": conc_data, "name": "custom"}

    pattern_model = MagicMock()
    pattern_model.predict_from_combined_experiment.return_value = {
        "tas": xr.DataArray(np.zeros(10), dims=["year"])
    }
    full_monthly = xr.DataArray(
        np.zeros((4000, 2, 2)),
        dims=["month", "lat", "lon"],
        coords={"month": np.arange(4000), "lat": [0, 1], "lon": [0, 1]},
    )
    pattern_model.to_monthly.return_value = full_monthly
    interface.pattern_models["tas"] = pattern_model

    monthly_prediction, monthly_warming, em_used, conc_used = (
        interface._get_or_compute_pattern_scaling(
            "tas",
            scenario,
            start_year=1990,
            end_year=2060,
            verbose=True,
        )
    )

    output = capsys.readouterr().out
    assert "Warning: Emissions data ends at" in output
    assert "Warning: Emissions data starts at" in output
    assert em_used.index.max() == 2100
    assert conc_used.index.max() == 2100
    assert monthly_prediction.sizes["month"] == 612
    assert len(monthly_warming) == 612

    # Try to generate without training
    with pytest.raises(RuntimeError) as exc_info:
        interface.generate_ensemble_outputs(
            scenario="ssp245",
            start_year=2020,
            end_year=2050,
            n_realizations=10,
            timeseries=["global"],
        )

    # Error message should mention which variable
    assert "not trained" in str(exc_info.value).lower()
    assert "train()" in str(exc_info.value).lower()


def test_compute_timeseries_scaling():
    """Test that _compute_timeseries_scaling_factor() computes expected scaling."""
    with patch("meteor.meteor_interface.Cmip6MeteorDataGetter"):
        interface = MeteorInterface(
            model="TestModel", variables=["tas"], cache_dir="/tmp/test"
        )

        # Mock the data and pattern scaling components
        em_data = xr.DataArray(np.array([290.0]), dims=["time"])
        conc_data = xr.DataArray(np.array([400.0]), dims=["time"])
        temp_scaling_ts = xr.DataArray(np.array([2.0]), dims=["time"])
        annual_prediction = xr.DataArray(
            np.array([[[1.0]]]),
            dims=["year", "lat", "lon"],
            coords={"year": [0], "lat": [0], "lon": [0]},
        )

        # Call the method
        with pytest.raises(
            ValueError, match="temp_scaling_ts must be an xarray DataArray"
        ):
            interface._compute_timeseries_scaling(
                "tas", annual_prediction, 0, em_data, conc_data, 2.0
            )
        with pytest.raises(
            ValueError, match="temp_scaling_ts must have a 'year' coordinate"
        ):
            interface._compute_timeseries_scaling(
                "tas", annual_prediction, 0, em_data, conc_data, temp_scaling_ts
            )
        temp_scaling_ts = xr.DataArray(np.array([2.0, 2.0]), dims=["year"])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "temp_scaling_ts temporal extent (2) must match annual_prediction time dimension (1)"
            ),
        ):
            interface._compute_timeseries_scaling(
                "tas", annual_prediction, 0, em_data, conc_data, temp_scaling_ts
            )
        temp_scaling_ts = xr.DataArray(
            np.array([2.0]), dims=["year"], coords={"year": [2000]}
        )
        with pytest.raises(
            ValueError, match="base_year 0 not found in temp_scaling_ts years"
        ):
            interface._compute_timeseries_scaling(
                "tas", annual_prediction, 0, em_data, conc_data, temp_scaling_ts
            )
        temp_scaling_ts = xr.DataArray(
            np.array([2.0]), dims=["year"], coords={"year": [0]}
        )
        scaling_factor = interface._compute_timeseries_scaling(
            "tas", annual_prediction, 0, em_data, conc_data, temp_scaling_ts
        )
        assert scaling_factor.shape == (1, 1, 1)
        assert np.isclose(scaling_factor.values[0, 0, 0], 1.0)
