import os

from meteor.ensemble_output import EnsembleOutput, VariableOutput
from meteor.meteor_interface import MeteorInterface


def test_meteor_interface_integration_pr(test_data_dir):
    """Integration test for MeteorInterface with a simple variable and transform."""

    cache_path = os.path.join(test_data_dir, "light_mock_cache")

    # Initialize MeteorInterface
    meteor = MeteorInterface("CanESM5", variables=["pr"], cache_dir=cache_path)
    assert meteor._is_trained["pr"] is False
    # Train the model using the simple dataset
    meteor.train(variable_configs={"pr": {"n_modes_noise": 4, "lag_order": 1}})
    assert meteor._is_trained["pr"] is True

    # Generate ensemble output
    ensemble = meteor.generate(
        scenario="ssp245",
        start_year=2015,
        end_year=2035,
        n_realizations=10,
        timeseries=["global", "regional:EAS"],
        gridded={"annual": list(range(2020, 2026))},
        impacts=["hdd:point:59.9,10.8"],
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
    assert isinstance(ensemble.variables["pr"], VariableOutput)
    print(ensemble.variables["pr"].timeseries)
    # assert False
    # assert ensemble.pr.sizes["lat"] == 5
    # assert ensemble.pr.sizes["ensemble"] == 10
    # assert ensemble.pr.sizes["time"] == 102  # 2000 to 2100 inclusive


def test_meteor_interface_integration_tas(test_data_dir):
    """Integration test for MeteorInterface with a simple variable and transform."""

    cache_path = os.path.join(test_data_dir, "light_mock_cache")

    # Initialize MeteorInterface
    meteor = MeteorInterface("CanESM5", variables="tas", cache_dir=cache_path)
    assert meteor._is_trained["tas"] is False
    # Train the model using the simple dataset
    meteor.train(
        variable_configs={"tas": {"n_modes_noise": 4, "lag_order": 1}}, verbose=False
    )
    assert meteor._is_trained["tas"] is True

    # Generate ensemble output
    ensemble = meteor.generate(
        scenario="ssp245",
        start_year=2015,
        end_year=2035,
        n_realizations=10,
        timeseries=["global", "regional:EAS"],
        gridded={"annual": list(range(2020, 2026))},
        impacts=["hdd:point:59.9,10.8"],
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
    print(ensemble.variables["tas"].timeseries)
    # assert False
    # assert ensemble.tas.sizes["lat"] == 5
    # assert ensemble.tas.sizes["ensemble"] == 10
    # assert ensemble.tas.sizes["time"] == 102  # 2000 to 2100 inclusive
