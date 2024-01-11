"""
SCM_FORCER_ENGINE

Module to facility ciceroscm to create forcing time series
for pattern scaling
"""

import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from ciceroscm import concentrations_emissions_handler, input_handler


def aerosol_priority_mapping(comps):
    """
    Produce aerosol mapping

    Parameters
    ----------
    comps : list
            List of the components that have forcing experiments

    Returns
    -------
    dict
        Where aerosol components that don't have direct mappings
        from the forcing experiments are mapped to what component
        there forcings should be taken from
    """
    aerosols = {"SO4_IND": "SO2", "BMB_AEROS_BC": "BC", "BMB_AEROS_OC": "OC"}
    missing_keys = list(set(aerosols.values()) - set(comps))
    if len(missing_keys) == 3:
        return {
            "SO4_IND": "CO2",
            "BMB_AEROS_BC": "CO2",
            "BMB_AEROS_OC": "CO2",
            "SO2": "CO2",
            "BC": "CO2",
            "OC": "CO2",
        }
    for miss in missing_keys:
        if "BC" not in missing_keys:
            aerosols[miss] = "BC"
            aerosols[
                [aermiss for aermiss, value in aerosols.items() if value == miss][0]
            ] = "BC"
        elif "OC" not in missing_keys:
            aerosols[miss] = "OC"
            aerosols[
                [aermiss for aermiss, value in aerosols.items() if value == miss][0]
            ] = "OC"
        else:
            aerosols[miss] = "SO2"
            aerosols[
                [aermiss for aermiss, value in aerosols.items() if value == miss][0]
            ] = "SO2"
    return aerosols


@dataclass
class ScmEngineConfigurations:
    """
    Dataclass for scm engine configurations
    """

    # pylint: disable=too-many-instance-attributes
    gaspam_data: pd.DataFrame
    concentrations_data: pd.DataFrame
    emissions_data: pd.DataFrame
    emstart: int = 2000
    nystart: int = 1950
    nyend: int = 2100
    idtm: int = 24
    nat_ch4_data: pd.DataFrame = None
    nat_n2o_data: pd.DataFrame = None
    conc_run: bool = True

    def __post_init__(self):
        """
        Read in and set sensible defaults for natural emissions,
        if they are missing, setting concentrations and emissions
        to flat values around the emission base year
        """
        if self.nat_ch4_data is None:
            self.nat_ch4_data = pd.DataFrame(
                data={"CH4": np.ones(self.nyend - self.nystart + 1) * 242.09},
                index=np.arange(self.nystart, self.nyend + 1),
            )
        if self.nat_n2o_data is None:
            self.nat_n2o_data = pd.DataFrame(
                data={"N2O": np.ones(self.nyend - self.nystart + 1) * 242.09},
                index=np.arange(self.nystart, self.nyend + 1),
            )

        # TODO : Possibly take this out or make more flexible
        if self.conc_run:
            self.concentrations_data.loc[self.emstart : self.emstart + 6].iloc[
                :
            ] = self.concentrations_data.loc[self.emstart, :]
            self.emissions_data.loc[self.emstart : self.emstart + 6].iloc[
                :
            ] = self.emissions_data.loc[self.emstart, :]


def run_single_experiment(pamset, input_h):
    """
    Run the concentrations_emissions part of the SCM

    Parameters
    ----------
    pamset : dict
             Pamset to the concentrations handler, can be empty
    input_h : input_handler.InputHandler
         Defining the Concetrations Emissions handler that this object will
         run

    Returns
    -------
    dict
        Forcing dictionary from concentrations_emissions_handler calculations
    """
    ce_handler = concentrations_emissions_handler.ConcentrationsEmissionsHandler(
        input_h, pamset
    )
    ce_handler.reset_with_new_pams(pamset)
    # print(pamset)
    for year in range(pamset["nystart"], pamset["nyend"] + 1):
        ce_handler.emi2conc(year)
        ce_handler.conc2forc(year, 0, 0)
    return ce_handler.forc


class ScmEngineForPatternScaling:
    """
    Class to support handling scm-scaling and forcing
    timeseries creation for a pattern scaling object

    Attributes
    ----------
    cfg : ScmEngineConfigurations
          Dataclass to hold the configurations that define the input handler
          and options in a convenient way.
    input_h : input_handler.InputHandler
         Defining the Concetrations Emissions handler that this object can
         run to either do scaling experiments, or to run and
         split into forcerrs
    """

    def __init__(self, cfg=None):
        """
        Parameters
        ----------
        cfg : dict
              Optional dictionary with values for the ScmEngineConfiguration
              contents
        """
        if cfg is None:
            ih_temp = input_handler.InputHandler({})
            em_set = ih_temp.read_emissions(
                os.path.join(
                    os.path.dirname(__file__), "default_scm_data", "ssp245_em_RCMIP.txt"
                )
            )
            conc_set = input_handler.read_inputfile(
                os.path.join(
                    os.path.dirname(__file__),
                    "default_scm_data",
                    "ssp245_conc_RCMIP.txt",
                )
            )
            self.cfg = ScmEngineConfigurations(
                gaspam_data=input_handler.read_components(
                    os.path.join(
                        os.path.dirname(__file__),
                        "default_scm_data",
                        "gases_vupdate_2022_AR6.txt",
                    )
                ),
                concentrations_data=conc_set,
                emissions_data=em_set,
            )
        else:
            if "concentrations_data" not in cfg:
                cfg["concentrations_data"] = input_handler.read_inputfile(
                    os.path.join(
                        os.path.dirname(__file__),
                        "default_scm_data",
                        "ssp245_conc_RCMIP.txt",
                    )
                )
            if "emissions_data" not in cfg:
                ih_temp = input_handler.InputHandler({})
                cfg["emissions_data"] = ih_temp.read_emissions(
                    os.path.join(
                        os.path.dirname(__file__),
                        "default_scm_data",
                        "ssp245_em_RCMIP.txt",
                    )
                )
            self.cfg = ScmEngineConfigurations(
                gaspam_data=input_handler.read_components(
                    os.path.join(
                        os.path.dirname(__file__),
                        "default_scm_data",
                        "gases_vupdate_2022_AR6.txt",
                    )
                ),
                **cfg,
            )

        self.input_h = input_handler.InputHandler(asdict(self.cfg))

    def run_to_get_scaling(self, exps):
        """
        Get per experiment scaling for a list of experiments

        Parameters
        ----------
        exps : list
               list of experiment names to be included
               the experiment names should be either base
               or on the format forcerxperturbationscaling
               so2 can be called sul

        Returns
        -------
        np.ndarray
               Mapped along the experiments with the scaling per
               each experiment at the same index
        """
        scaling = np.zeros(len(exps))
        run_dict = {k: v for k, v in asdict(self.cfg).items() if not k.endswith("data")}
        run_dict["emstart"] = 2101
        run_dict["nyend"] = self.cfg.emstart + 5
        base_forcing = run_single_experiment(run_dict, self.input_h)
        for i, exp in enumerate(exps):
            run_dict_exp = run_dict.copy()
            if exp == "base":
                scaling[i] = 0
                continue
            species = exp.split("x")[0].upper()
            if species == "SUL":
                species = "SO2"
            multiplicator = int(exp.split("x")[1])
            em_here = self.cfg.emissions_data.copy()
            conc_here = self.cfg.concentrations_data.copy()

            if species in conc_here.columns:
                conc_here.loc[self.cfg.emstart : self.cfg.emstart + 6, species] = (
                    conc_here[species][self.cfg.emstart] * multiplicator * np.ones(7)
                )
            if species in em_here.columns:
                em_here.loc[self.cfg.emstart : self.cfg.emstart + 6, species] = (
                    em_here[species][self.cfg.emstart] * multiplicator * np.ones(7)
                )
            run_dict_exp["gaspam_data"] = self.cfg.gaspam_data
            run_dict_exp["nat_ch4_data"] = self.cfg.nat_ch4_data
            run_dict_exp["nat_n2o_data"] = self.cfg.nat_n2o_data
            run_dict_exp["emissions_data"] = em_here
            run_dict_exp["concentrations_data"] = conc_here
            ih_here = input_handler.InputHandler(run_dict_exp)
            exp_forcing = run_single_experiment(run_dict, ih_here)
            scaling[i] = (
                exp_forcing["Total_forcing"][-1] - base_forcing["Total_forcing"][-1]
            )
        return scaling

    def run_and_return_per_forcer_results(self, exps):
        """
        Run scm and separate forcing timeseries per
        forcer in the experiment list

        Parameters
        ----------
        exps : list
               list of experiment names to be included
               the experiment names should be either base
               or on the format forcerxperturbationscaling
               so2 can be called sul

        Returns
        -------
        dict
             Dictionary including the forcing time series
             per forcer experiment. Forcers are matched to the
             compound that has been perturbed by name. Aerosols
             are mapped to other aerosols if they are included
             otherwise all forcers for which no particular
             experiment is included is mapped to CO2
        """
        # TODO : Deal with if co2 experiment is not included
        # TODO : Deal with several experiments for the same component
        forcing_total = run_single_experiment(asdict(self.cfg), self.input_h)
        forcing = {}
        comps = [exp.split("x")[0].upper() for exp in exps]
        aerosol_mapping = aerosol_priority_mapping(comps)
        for i, exp in enumerate(exps):
            forcing[exp] = np.zeros(len(forcing_total["Total_forcing"]))
            if exp.split("x")[0].upper() == "CO2":
                co2_name = exp
                forcing[exp] = forcing_total["Total_forcing"]
            if exp.split("c")[0].upper() == "SUL":
                comps[i] = "SO2"

        for comp, forc_series in forcing_total.items():
            print(comp)
            if comp in comps:
                print("incomp")
                forcing[exps[comps.index(comp)]] = (
                    forcing[exps[comps.index(comp)]] + forc_series
                )
                forcing[co2_name] = forcing[co2_name] - forc_series
            elif comp in aerosol_mapping:
                print("inaer")
                forcing[exps[comps.index(aerosol_mapping[comp])]] = (
                    forcing[exps[comps.index(aerosol_mapping[comp])]] + forc_series
                )
                forcing[co2_name] = forcing[co2_name] - forc_series

        return forcing
