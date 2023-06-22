# Calibrate per experiment
import os
import sys

import numpy as np
import pandas as pd

from ciceroscm import concentrations_emissions_handler, input_handler
from dataclasses import dataclass, asdict
# Run with some dataset get forcing per combined tracer and combine patterns


@dataclass
class ScmEngineConfigurations:
    """
    Dataclass for scm engine configurations
    """
    gaspam_data : pd.DataFrame
    concentrations_data : pd.DataFrame
    emissions_data : pd.DataFrame
    emstart: int = 2000
    nystart: int = 1950
    nyend : int = 2100
    idtm : int = 24
    nat_ch4_data : pd.DataFrame = None
    nat_n2o_data : pd.DataFrame = None
    conc_run : bool = True

    def __post_init__(self):
        if self.nat_ch4_data is None:
            self.nat_ch4_data = pd.DataFrame(data={"CH4": np.ones(self.nyend-self.nystart+1)*242.09}, index = np.arange(self.nystart, self.nyend + 1))
        if self.nat_n2o_data is None:
            self.nat_n2o_data = pd.DataFrame(data={"N2O": np.ones(self.nyend-self.nystart+1)*242.09}, index = np.arange(self.nystart, self.nyend + 1))
        
        self.concentrations_data.loc[self.emstart:self.emstart+6].iloc[:] = self.concentrations_data.loc[self.emstart,:]
        self.emissions_data.loc[self.emstart:self.emstart+6].iloc[:] = self.emissions_data.loc[self.emstart,:]
    

def run_single_experiment(pamset, ih):
    """
    Method to run the concentrations_emissions part of the SCM

    
    """
    ce_handler = concentrations_emissions_handler.ConcentrationsEmissionsHandler(ih, pamset)
    ce_handler.reset_with_new_pams(pamset)
    for yr in range(pamset["nystart"], pamset["nyend"]+1):
        ce_handler.emi2conc(yr)
        forc, fn, fs = ce_handler.conc2forc(yr, 0, 0)
    return ce_handler.forc

class ScmEngineForPatternScaling:
    """
    Class to support handling scm-scaling and forcing
    timeseries creation for a pattern scaling object
    """
    def __init__(self, cfg = None):
        ih_temp = input_handler.InputHandler({})
        em_set = ih_temp.read_emissions(os.path.join(os.path.dirname(__file__), "default_scm_data", "ssp245_em_RCMIP.txt"))
        conc_set = input_handler.read_inputfile(os.path.join(os.path.dirname(__file__), "default_scm_data", "ssp245_conc_RCMIP.txt"))
        
        if cfg is None:
            self.cfg = ScmEngineConfigurations(gaspam_data=input_handler.read_components(os.path.join(os.path.dirname(__file__), "default_scm_data", "gases_vupdate_2022_AR6.txt")), concentrations_data = conc_set, emissions_data = em_set)
        else:
            self.cfg = ScmEngineConfigurations(gaspam_data=input_handler.read_components(os.path.join(os.path.dirname(__file__), "default_scm_data", "gases_vupdate_2022_AR6.txt")), concentrations_data = conc_set, emissions_data = em_set, **cfg)
        
        self.ih = input_handler.InputHandler(asdict(self.cfg))

    def run_to_get_scaling(self, exps):
        scaling = np.zeros(len(exps))
        run_dict = {k : v for k, v in asdict(self.cfg).items() if not k.endswith("data")}
        run_dict["conc_run"] = True
        run_dict["emstart"] = 2101
        run_dict["nyend"] = self.cfg.emstart + 5
        base_forcing = run_single_experiment(run_dict, self.ih)
        subst_len = len(self.cfg.concentrations_data.loc[self.cfg.emstart:, "CO2"])
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
                conc_here.loc[self.cfg.emstart:self.cfg.emstart + 6, species] = conc_here[species][self.cfg.emstart]*multiplicator*np.ones(7)
            if species in em_here.columns:
                em_here.loc[self.cfg.emstart:self.cfg.emstart + 6, species] = em_here[species][self.cfg.emstart]*multiplicator*np.ones(7)
            run_dict_exp["gaspam_data"] = self.cfg.gaspam_data
            run_dict_exp["nat_ch4_data"] = self.cfg.nat_ch4_data
            run_dict_exp["nat_n2o_data"] = self.cfg.nat_n2o_data
            run_dict_exp["emissions_data"] = em_here
            run_dict_exp["concentrations_data"] = conc_here
            ih_here = input_handler.InputHandler(run_dict_exp)
            exp_forcing = run_single_experiment(run_dict, ih_here)
            scaling[i] = exp_forcing['Total_forcing'][-1] - base_forcing['Total_forcing'][-1]
               
        return scaling


        
    def pick_out_forcing_results(self, forcer, results):
        pass
