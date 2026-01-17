"""
SCM input handling utilities for METEOR.
"""

import os

import pandas as pd
from ciceroscm import input_handler


def parse_scenario_input(scenario):
    """
    Parse scenario input and return emissions/concentrations info.

    Parameters
    ----------
    scenario : str or dict
        Scenario specification

    Returns
    -------
    dict
        Dictionary with keys:
        - 'type': 'ssp' or 'custom'
        - 'name': scenario name
        - 'emissions': emissions file path or DataFrame (if custom)
        - 'concentrations': concentrations file path or DataFrame (if custom)
    """
    if isinstance(scenario, str):
        # Standard SSP scenario
        return {
            "type": "ssp",
            "name": scenario,
            "emissions": None,
            "concentrations": None,
        }
    elif isinstance(scenario, dict):
        # Custom scenario
        if "emissions" not in scenario:
            raise ValueError("Custom scenario dict must include 'emissions' key")

        # Get emissions (path or DataFrame)
        emissions = scenario["emissions"]

        # Get concentrations (path, DataFrame, or use base_scenario)
        if "concentrations" in scenario:
            concentrations = scenario["concentrations"]
        else:
            # Use base_scenario concentrations (default: ssp245)
            base_scenario = scenario.get("base_scenario", "ssp245")
            cscm_data_dir = os.path.join(os.path.dirname(__file__), "default_scm_data")
            concentrations = os.path.join(
                cscm_data_dir, f"{base_scenario}_conc_RCMIP.txt"
            )

        # Get scenario name for labeling
        scenario_name = scenario.get("name", "custom")

        return {
            "type": "custom",
            "name": scenario_name,
            "emissions": emissions,
            "concentrations": concentrations,
        }
    else:
        raise TypeError(f"scenario must be str or dict, got {type(scenario)}")


def load_emissions_concentrations_from_name(scenario_name):
    """
    Load emissions and concentrations for a standard SSP scenario.

    Parameters
    ----------
    scenario_name : str
        Name of the SSP scenario (e.g., 'ssp245')

    Returns
    -------
    tuple
        (emissions_data, concentrations_data) as DataFrames
    """
    cscm_data_dir = os.path.join(os.path.dirname(__file__), "default_scm_data")
    conc_file = os.path.join(cscm_data_dir, f"{scenario_name}_conc_RCMIP.txt")
    if not os.path.exists(conc_file):
        raise FileNotFoundError(
            f"Concentration file not found: {conc_file}\n"
            f"Available scenarios should be in: {cscm_data_dir}"
        )
    em_file = os.path.join(cscm_data_dir, f"{scenario_name}_em_RCMIP.txt")
    # TODO: Consider just dropping this as it should never happen.
    if not os.path.exists(em_file):
        raise FileNotFoundError(  # pragma: no cover
            f"Emission file not found: {em_file}\n"
            f"Available scenarios should be in: {cscm_data_dir}"
        )
    ih = input_handler.InputHandler({})
    conc_data = input_handler.read_inputfile(conc_file)
    em_data = ih.read_emissions(em_file)
    return em_data, conc_data


def load_emissions_concentrations(emissions_spec, concentrations_spec, verbose=False):
    """
    Load emissions and concentrations from files or DataFrames.

    Parameters
    ----------
    emissions_spec : str or pd.DataFrame
        Path to emissions file or DataFrame
    concentrations_spec : str or pd.DataFrame
        Path to concentrations file or DataFrame
    verbose : bool
        Print loading messages

    Returns
    -------
    tuple
        (emissions_data, concentrations_data) as DataFrames
    """
    # Load emissions
    if isinstance(emissions_spec, pd.DataFrame):
        em_data = emissions_spec
        if verbose:
            print("      → Using provided emissions DataFrame")
    elif isinstance(emissions_spec, str):
        ih = input_handler.InputHandler({})
        em_data = ih.read_emissions(emissions_spec)
        if verbose:
            print(f"      → Loaded emissions from {os.path.basename(emissions_spec)}")
    else:
        raise TypeError(
            f"emissions must be str path or DataFrame, got {type(emissions_spec)}"
        )

    # Load concentrations
    if isinstance(concentrations_spec, pd.DataFrame):
        conc_data = concentrations_spec
        if verbose:
            print("      → Using provided concentrations DataFrame")
    elif isinstance(concentrations_spec, str):
        conc_data = input_handler.read_inputfile(concentrations_spec)
        if verbose:
            print(
                f"      → Loaded concentrations from {os.path.basename(concentrations_spec)}"
            )
    else:
        raise TypeError(
            f"concentrations must be str path or DataFrame, got {type(concentrations_spec)}"
        )

    return em_data, conc_data


def load_ssp_config(scenario="ssp245", nystart=1750, nyend=2100):
    """
    Load CICERO-SCM forcing data and create configuration for pattern scaling.

    Loads concentration and emission data for a specified SSP scenario
    from the default METEOR data directory and creates a configuration
    dictionary for use with METEOR pattern scaling models.

    Parameters
    ----------
    scenario : str, optional
        SSP scenario name. Default is "ssp245".
        Common options: "ssp126", "ssp245", "ssp370", "ssp585"
    nystart : int, optional
        Start year for the simulation. Default is 1750.
    nyend : int, optional
        End year for the simulation. Default is 2100.

    Returns
    -------
    dict
        Configuration dictionary with keys:
        - emstart: Emission start year (1850)
        - nystart: Simulation start year
        - nyend: Simulation end year
        - conc_run: Whether to run with concentrations (False)
        - concentrations_data: Loaded concentration data
        - emissions_data: Loaded emission data

    Examples
    --------
    >>> data_getter = Cmip6MeteorDataGetter(exps=["piControl"], flds=["tas"])
    >>> config = data_getter.load_ssp_config("ssp245")
    >>> print(config.keys())
    dict_keys(['emstart', 'nystart', 'nyend', 'conc_run', 'concentrations_data', 'emissions_data'])
    """
    print(f"📥 Loading CICERO-SCM forcing data for {scenario}...")

    # Load concentration data
    scen_em, scen_conc = load_emissions_concentrations_from_name(scenario)

    ssp_config = {
        "emstart": 1850,
        "nystart": nystart,
        "nyend": nyend,
        "conc_run": False,
        "concentrations_data": scen_conc,
        "emissions_data": scen_em,
    }

    print(f"   ✅ Loaded {len(scen_conc)} concentration records")
    print(f"   ✅ Loaded {len(scen_em)} emission records")
    print(f"   ✅ Config: {nystart}-{nyend}, emissions start: 1850")

    return ssp_config
