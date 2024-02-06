"""
METEOR
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from . import prpatt, scm_forcer_engine

LOGGER = logging.getLogger(__name__)


def read_training_data(get_training_file_from_exp, exp_list):
    """
    Read training data into xarray

    Parameters
    ----------
    get_training_file_from_exp: function
              Funtion that returns the path to the training
              file given the experiment name
    exp_list: list
              List of experiments to include

    Returns
    -------
    xarray dataset
         A dataset containing the experiments as an extra dimension
    """
    # Might need to be rewritten to account for several models in files...
    for i, exp in enumerate(exp_list):
        tmp = xr.open_dataset(get_training_file_from_exp(exp))
        if not i:
            dac = tmp
        else:
            dac = xr.concat([dac, tmp], "expt")
    dac = dac.assign_coords({"expt": exp_list})
    ctrl = exp_list.index("base")
    varis = dac.data_vars
    dacanom = dac
    for var in varis:
        dacanom[var] = dac[var].isel(ens=0).drop_vars("ens") - dac[var][
            ctrl, :, :, :, :
        ].mean(dim="year", skipna=True).isel(ens=0).drop_vars("ens")
    dacanom = dacanom.rename({"year": "time"})
    return dacanom


class MeteorPatternScaling:
    """
    Pattern scaling descriptor class

    Handles pattern scaling, defining
    and calculating it from a dataset
    Then provides routines to apply it
    to new data

    Attributes
    ----------
    exp_forc_dict: dict
             Dict of experiments included in the pattern scaling
             with the forcing scaling size of the experiments as
             values
    exp_list : list
             List of the experiments used for ordering
    daconom : xarray dataset
             Input data from the experiments belonging to the pattern
    patternflds: dict
             Dictionary of with variables for pattern scaling as
             keys, and their truncation length for PCAs as values
    pattern_dict: dict
             Dictionary with the PCAs and patterns for the various
             experiments and variables. A nested dictionary
             with patterns for the experiment of the object.
             First keyset: The experiments that the pattern is defined by,
             Second keyset: The variables for which patterns are produced.
             Third keyset: neweof - a synthetic PCA for the data from
             the calculated response timescales,
             orgeof -  orginal PCA object from the data, and if data allows,
             outp - the lmfit parameter fit using the original
             PCA object and timescales
    name: str
          Name of the pattern, to be printed on plots etc.
    """

    def __init__(
        self, name, patternflds, get_training_file_from_exp, exp_list, tmscl=None
    ):  # pylint: disable=too-many-arguments
        """
        Initialise Pattern Scaling object

        Defining the patternscaling object from lists of experiments

        Parameters
        ----------
        name : str
               name of the model/dataset for that this patter belongs to
        patternflds : dict
                    keys are names of the varibles to be considered
                    Values are truncation lengths for each field
        get_training_file_from_exp : function
                    Function that defines how to get find the location
                    of the training data input file for a given experiment
        exp_list : dict
                   List with experiment names
        tmscl : list
                Intial guess for timescales to fit the pattern to if
                None is sent, [2,50] will be used
        """
        sefps = scm_forcer_engine.ScmEngineForPatternScaling(None)
        scaling = sefps.run_to_get_scaling(exp_list)
        self.exp_forc_dict = {exp: scaling[i] for i, exp in enumerate(exp_list)}
        self.dacanom = read_training_data(get_training_file_from_exp, exp_list)
        self.exp_list = exp_list
        self.patternflds = patternflds
        if tmscl is None:
            tmscl = [2, 50]
        self.pattern_dict = self._make_pattern_dict()
        self.name = name

    def _make_pattern_dict(self):
        """
        Make a pattern scale dictionary

        Making a pattern scaling dictionary to define the object at initialisation

        Returns
        -------
        dict
            A nested dictionary that with the experiments of the objects.
            First keyset: The experiments that the pattern is defined by.
            Second keyset: The variables for which patterns are produced.
            Third keyset: neweof, a synthetic PCA for the data from the
            calculated response timescales,
            orgeof, orginal PCA object from the data, and if data allows,
            outp, the lmfit parameter fit using the original PCA object
            and timescales
        """
        pattern_dict = {}
        for j, exp in enumerate(self.exp_forc_dict.keys()):
            pattern_dict[exp] = {}
            for fld, trnc in self.patternflds.items():
                pattern_dict[exp][fld] = {}
                # The :100? Flexible?
                anomaly_data = self.dacanom[fld][j, :100, :, :]
                if not np.isnan(np.mean(anomaly_data)):
                    (out, orgeof, neweof) = prpatt.get_timescales(anomaly_data, trnc)

                    pattern_dict[exp][fld]["neweof"] = neweof
                    pattern_dict[exp][fld]["orgeof"] = orgeof
                    pattern_dict[exp][fld]["outp"] = out
                else:  # pragma: no cover
                    pattern_dict[exp][fld]["neweof"] = np.nan
                    pattern_dict[exp][fld]["orgeof"] = np.nan
        return pattern_dict

    def predict_from_forcing_profile(self, forc_timeseries, fld, exp="co2x2"):
        """
        Make prediction from experiment and a forcing profile

        Parameters
        ----------
        forc_timeseries : np.array
            Array of with forcing timeseries for which to create predictions from the pattern
        fld : str
            Variable to make prediction for
        exp : str
            Experiment that defines the stepfunction response for the forcer in question

        Returns
        -------
        xarray dataarray
             Prediction object for the variable given the forcing time series

        !Todo: Add tests to check that variable and experiment are in the patterns
        patternfld and exp_lists
        """
        # Add something to account for the forcing strength of the experiment
        convolved_pca = prpatt.imodel_filter(
            self.pattern_dict[exp][fld]["outp"].params,
            forc_timeseries,
            forc_step=self.exp_forc_dict[exp],
        )
        predicted = prpatt.rmodel(self.pattern_dict[exp][fld]["orgeof"], convolved_pca)
        return predicted

    def predict_from_combined_experiment(
        self, emissions_data, concentrations_data, flds, conc_run=False
    ):
        """
        Predict the combined patterns for given flds for the given emissions and concentrations

        Parameters
        ----------
        emissions_data : pd.DataFrame
                         Emissions data on the format used by the ciceroscm input_handler
        concentrations_data : pd.DataFrame
                         Concentrations data on the format used by the ciceroscm input_handler
        flds : list
               Fields for which to calculate patterns
        conc_run : Bool
                   Whether experiment should be a concentrations run

        Returns
        -------
        dict
            keys are flds, values are predicted per fld combined patterns
        """
        # Setup and run scm-run to get forcing time series per forcing experiment
        # Run and make predictions per experiment
        # Combine predictions to full pattern
        cfg = {
            "conc_run": conc_run,
            "nystart": emissions_data.index[0],
            "emstart": emissions_data.index[0] + 100,
            "nyend": 2100,
            "concentrations_data": concentrations_data,
            "emissions_data": emissions_data,
        }
        sefps = scm_forcer_engine.ScmEngineForPatternScaling(cfg)
        forcing_series = sefps.run_and_return_per_forcer_results(self.exp_list)
        predicted = {}
        for exp in self.exp_list:
            if exp == "base":
                continue
            for fld in flds:
                if fld not in predicted:
                    predicted[fld] = self.predict_from_forcing_profile(
                        forcing_series[exp], fld, exp
                    )
                    predicted[fld]["time"] = pd.to_datetime(
                        predicted[fld]["time"], format="%Y"
                    )

                else:
                    tmp = self.predict_from_forcing_profile(
                        forcing_series[exp], fld, exp
                    )
                    tmp["time"] = pd.to_datetime(tmp["time"], format="%Y")
                    predicted[fld] = predicted[fld] + tmp

        return predicted