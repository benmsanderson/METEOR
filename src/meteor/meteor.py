"""
METEOR
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from . import prpatt, scm_forcer_engine

LOGGER = logging.getLogger(__name__)


def read_training_data(get_training_data, exp_list, from_file=True):
    """
    Read training data into xarray

    Parameters
    ----------
    get_training_data: function
              Funtion that returns the path to the training
              file given the experiment name
    exp_list: list
              List of experiments to include
    from_file: bool
               Whehther to get data from file or not
    Returns
    -------
    xarray dataset
         A dataset containing the experiments as an extra dimension
    """
    # Might need to be rewritten to account for several models in files...
    for i, exp in enumerate(exp_list):
        if from_file:
            tmp = xr.open_dataset(get_training_data(exp))
        else:
            tmp = get_training_data(exp)
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
             Third keyset:
             pattern_full - pattern of impulse response timeseries and spatial
             patterns per mode
             outp - the lmfit parameter fit using the original
             PCA object and timescales
    name: str
          Name of the pattern, to be printed on plots etc.
    """

    def __init__(
        self,
        name,
        patternflds,
        get_training_file_from_exp,
        exp_list,
        from_file=True,
        ssp_input=None,
        anom_timescales=None,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Initialise Pattern Scaling object

        Defining the patternscaling object from lists of experiments

        Parameters
        ----------
        name : str
               name of the model/dataset for that this patter belongs to
        patternflds : dict
                    keys are names of the varibles to be considered
                    Values are number of timescales to fit
        get_training_file_from_exp : function
                    Function that defines how to get find the location
                    of the training data input file for a given experiment
        exp_list : dict
                   List with experiment names
        """
        sefps = scm_forcer_engine.ScmEngineForPatternScaling(None)
        scaling = sefps.run_to_get_scaling(exp_list)
        self.exp_forc_dict = {exp: scaling[i] for i, exp in enumerate(exp_list)}
        self.dacanom = read_training_data(
            get_training_file_from_exp, exp_list, from_file=from_file
        )
        self.exp_list = exp_list
        self.patternflds = patternflds
        self.pattern_dict = self._make_pattern_dict()
        if anom_timescales is None or not isinstance(anom_timescales, dict):
            anom_timescales = {}
            for fld in patternflds:
                anom_timescales[fld] = 1
        else:
            for fld in patternflds:
                if fld not in anom_timescales:
                    anom_timescales[fld] = 1
        self.anom_timescales = anom_timescales
        if "xanom" in "-".join(exp_list):
            self._add_patterns_for_residual_exp(ssp_input)
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
            Third keyset: pattern split in temporal and spatial part per mode,
            and if data allows, outp, the lmfit parameter fit of timescales
        """
        pattern_dict = {}
        for j, exp in enumerate(self.exp_forc_dict.keys()):
            pattern_dict[exp] = {}
            for fld, trnc in self.patternflds.items():
                pattern_dict[exp][fld] = {}
                if exp.split("x") == "anom":
                    continue
                # The :100? Flexible?
                # anomaly_data is the time x lat x lon data for variable fld and expt j

                anomaly_data = self.dacanom[fld][j, :100, :, :]
                if not np.isnan(np.mean(anomaly_data)):
                    # now call get timescales to the fitted timescales and compute the patterns
                    # out is the lmfit object
                    # pattern_full is the pattern of impulse response timeseries and spatial patterns per mode
                    (out, pattern_full) = prpatt.get_timescales(anomaly_data, trnc)
                    pattern_dict[exp][fld]["pattern_full"] = pattern_full
                    pattern_dict[exp][fld]["outp"] = out.params
                else:  # pragma: no cover
                    pattern_dict[exp][fld]["pattern_full"] = np.nan
        return pattern_dict

    def _add_patterns_for_residual_exp(
        self, ssp_input
    ):  # pylint: disable=too-many-locals
        """
        Add patterns for an experiment predicted from residual
        """
        anom_exps = [exp for exp in self.exp_list if "xanom" in exp]
        if len(anom_exps) != 1:
            LOGGER.error(
                "Adding a residual pattern can only be done for a MeteorPatternScaling instance with exactly one xanom experiment"
            )
            raise RuntimeError(
                "MeteorPatternScaling objects can only handle a single xanom experiment"
            )
        exp = anom_exps[0]
        exp_index = self.exp_list.index(exp)
        if ssp_input is None:
            ssp_input = {
                "emstart": 1850,
                "nystart": 1750,
                "nyend": 2100,
                "conc_run": False,
            }
        elif "nystart" not in ssp_input:
            ssp_input["nystart"] = 1750
        sefps = scm_forcer_engine.ScmEngineForPatternScaling(ssp_input)
        start_index = ssp_input["emstart"] - ssp_input["nystart"]
        em_len = ssp_input["nyend"] - ssp_input["emstart"] + 1
        forcing_series = sefps.run_and_return_per_forcer_results(self.exp_list)
        forcing_of_residual = xr.DataArray(
            data=forcing_series[exp][start_index:].copy(),
            coords={"time": np.arange(len(forcing_series[exp][start_index:]))},
        )
        forcing_series[exp] = None
        predicted_without = self._predict_combined_experiment_from_forcer_series(
            forcing_series, self.patternflds.keys(), ssp_input["nystart"]
        )  # [100:, :, :]
        for fld in self.patternflds:
            predicted_without_fld = predicted_without[fld].isel(
                time=slice(start_index, start_index + em_len)
            )
            predicted_without_fld = predicted_without_fld.assign_coords(
                time=np.arange(em_len)
            )
            residual = (
                self.dacanom[fld][exp_index, :em_len, :, :] - predicted_without_fld
            )
            (out, pattern_full) = prpatt.get_timescales_from_anomaly(
                residual, forcing_of_residual, n_modes=self.anom_timescales[fld]
            )
            self.pattern_dict[exp][fld]["pattern_full"] = pattern_full
            self.pattern_dict[exp][fld]["outp"] = out

    def predict_from_forcing_profile(
        self, forc_timeseries, fld, exp="co2x2", year_0=1850
    ):
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
        year_0 : int
            Start year of forcing timeseries

        Returns
        -------
        xarray dataarray
             Prediction object for the variable given the forcing time series

        !Todo: Add tests to check that variable and experiment are in the patterns
        patternfld and exp_lists
        """
        # Add something to account for the forcing strength of the experiment
        convolved_pca = prpatt.imodel_filter(
            self.pattern_dict[exp][fld]["outp"],
            forc_timeseries,
            forc_step=self.exp_forc_dict[exp],
            year_0=year_0,
        )
        predicted = prpatt.rmodel(
            self.pattern_dict[exp][fld]["pattern_full"], convolved_pca
        )
        return predicted

    def predict_from_combined_experiment(
        self,
        emissions_data,
        concentrations_data,
        flds,
        conc_run=False,
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
        predicted = self._predict_combined_experiment_from_forcer_series(
            forcing_series, flds, cfg["nystart"]
        )
        return predicted

    def _predict_combined_experiment_from_forcer_series(
        self, forcing_series, flds, nystart
    ):
        """
        Predict the combined patterns for given flds for the given experiment split forcing series

        Parameters
        ----------
        forcing series : dict
                        Dictionary including the forcing time series
                        per forcer experiment.
        flds : list
               Fields for which to calculate patterns
        nystart : int
                   Whether ex

        Returns
        -------
        dict
            keys are flds, values are predicted per fld combined patterns
        """
        predicted = {}

        for exp in self.exp_list:
            if exp == "base":
                continue
            if forcing_series[exp] is None:
                continue
            for fld in flds:
                if fld not in predicted:
                    predicted[fld] = self.predict_from_forcing_profile(
                        forcing_series[exp], fld, exp, year_0=nystart
                    )
                    predicted[fld]["time"] = pd.to_datetime(
                        predicted[fld]["time"], format="%Y"
                    )

                else:
                    tmp = self.predict_from_forcing_profile(
                        forcing_series[exp], fld, exp, year_0=nystart
                    )
                    tmp["time"] = pd.to_datetime(tmp["time"], format="%Y")
                    predicted[fld] = predicted[fld] + tmp
        return predicted
