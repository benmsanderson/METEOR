import os

import numpy as np
import xarray as xr

from meteor.ensemble_output import EnsembleOutput, VariableOutput
from meteor.meteor_interface import MeteorInterface


def test_meteor_interface_integration_pr(test_data_dir):
    """Integration test for MeteorInterface with a simple variable and transform."""

    cache_path = os.path.join(test_data_dir, "light_mock_cache")

    # Initialize MeteorInterface
    meteor = MeteorInterface("CanESM5", variables=["pr"], cache_dir=cache_path)
    assert meteor._is_trained["pr"] is False
    # Train the model using the simple dataset
    meteor.train(
        variable_configs={
            "pr": {
                "n_modes_noise": 4,
                "lag_order": 1,
                "n_modes_pattern": 3,
                "use_exog": "none",
                "use_picontrol_baseline": True,
                "transform": True,
                "transform_type": "gamma",
            }
        },
        auto=False,
        verbose=False,
    )
    assert meteor._is_trained["pr"] is True

    # Generate ensemble output
    ensemble = meteor.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=2015,
        end_year=2035,
        n_realizations=10,
        timeseries=["global", "regional:EAS"],
        gridded={
            "annual": list(range(2020, 2026)),
            "monthly": [2025],
            "climatology": [(2020, 2025)],
        },
        impacts=["hdd:point:59.9,10.8"],
    )

    # Check that the ensemble output has the expected shape
    assert isinstance(ensemble, EnsembleOutput)
    assert ensemble.variables is not None
    assert ensemble.metadata["scenario"] == "ssp245"
    assert ensemble.metadata["n_realizations"] == 10
    assert ensemble.metadata["model"] == "CanESM5"
    assert ensemble.metadata["year_range"] == "2015-2035"
    assert ensemble.variables["pr"].gridded["annual"][2021].shape == (
        10,
        3,
        3,
    )  # ensemble, year, lat, lon
    assert ensemble.variables["pr"].gridded["monthly"][2025].shape == (
        10,
        12,
        3,
        3,
    )  # ensemble, month, lat, lon
    assert ensemble.variables["pr"].gridded["climatology"]["2020-2025"].shape == (
        10,
        3,
        3,
    )  # ensemble, lat, lon
    # assert ensemble.pr.sizes["lat"] == 5
    # assert ensemble.pr.sizes["ensemble"] == 10
    # assert ensemble.pr.sizes["time"] == 102  # 2000 to 2100 inclusive
    temp_ts_mock = xr.DataArray(
        data=np.linspace(0, 4, 351),  # Mock temperature time series
        dims=["year"],
        coords={"year": np.arange(1750, 2101)},  # Years from 1750 to 2100
    )

    ensemble_scaled = meteor.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=2015,
        end_year=2035,
        n_realizations=2,
        timeseries=["global", "regional:EAS"],
        gridded={"annual": list(range(2020, 2026))},
        temp_scaling_ts=temp_ts_mock,
    )
    assert isinstance(ensemble_scaled, EnsembleOutput)
    assert ensemble_scaled.variables is not None
    assert ensemble.metadata["scenario"] == "ssp245"
    assert ensemble.metadata["n_realizations"] == 10
    assert ensemble.metadata["model"] == "CanESM5"
    assert ensemble.metadata["year_range"] == "2015-2035"
    assert isinstance(ensemble.variables["pr"], VariableOutput)


def test_meteor_interface_integration_tas(test_data_dir):
    """Integration test for MeteorInterface with a simple variable and transform."""

    cache_path = os.path.join(test_data_dir, "light_mock_cache")

    # Initialize MeteorInterface
    meteor = MeteorInterface("CanESM5", variables="tas", cache_dir=cache_path)
    assert meteor._is_trained["tas"] is False
    repr = meteor.__repr__()
    print(repr)
    assert (
        repr
        == "MeteorInterface(model='CanESM5', variables=['tas'], status='not trained')"
    )
    # Train the model using the simple dataset
    meteor.train(
        variable_configs={"tas": {"n_modes_noise": 4, "lag_order": 1}}, verbose=False
    )
    repr = meteor.__repr__()
    print(repr)
    assert (
        repr == "MeteorInterface(model='CanESM5', variables=['tas'], status='trained')"
    )
    assert meteor._is_trained["tas"] is True

    # Generate ensemble output
    ensemble = meteor.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=2015,
        end_year=2035,
        n_realizations=10,
        timeseries=["global", "regional:EAS"],
        gridded={"annual": list(range(2020, 2026))},
        impacts={"tas": {"degree_days": {"hdd_base": 18.0}}},
        verbose=False,
    )

    # Check that the ensemble output has the expected shape
    assert isinstance(ensemble, EnsembleOutput)
    assert ensemble.variables is not None
    print(ensemble.variables)
    print(ensemble.metadata)
    assert ensemble.metadata["scenario"] == "ssp245"
    assert ensemble.metadata["n_realizations"] == 10
    assert ensemble.metadata["model"] == "CanESM5"
    assert ensemble.metadata["year_range"] == "2015-2035"
    assert isinstance(ensemble.variables["tas"], VariableOutput)
    print(ensemble.variables["tas"].impacts)
    assert ensemble.variables["tas"].impacts["hdd"]["global"].shape == (
        10,
        21,
    )  # ensemble, time
    # assert ensemble.tas.sizes["lat"] == 5
    # assert ensemble.tas.sizes["ensemble"] == 10
    # assert ensemble.tas.sizes["time"] == 102  # 2000 to 2100 inclusive
    temp_ts_mock = xr.DataArray(
        data=np.linspace(0, 4, 351),  # Mock temperature time series
        dims=["year"],
        coords={"year": np.arange(1750, 2101)},  # Years from 1750 to 2100
    )

    ensemble_scaled = meteor.generate_ensemble_outputs(
        scenario="ssp245",
        start_year=2015,
        end_year=2035,
        n_realizations=2,
        timeseries=["global", "regional:EAS"],
        gridded={"annual": list(range(2020, 2026))},
        temp_scaling_ts=temp_ts_mock,
    )
    assert isinstance(ensemble_scaled, EnsembleOutput)
    assert ensemble_scaled.variables is not None
    assert ensemble.metadata["scenario"] == "ssp245"
    assert ensemble.metadata["n_realizations"] == 10
    assert ensemble.metadata["model"] == "CanESM5"
    assert ensemble.metadata["year_range"] == "2015-2035"
    assert isinstance(ensemble.variables["tas"], VariableOutput)
