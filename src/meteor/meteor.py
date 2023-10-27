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
        InitialisePatternScaling

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
        self.pattern_dict = self._make_pattern_dict(tmscl)
        self.name = name

    def _make_pattern_dict(self, tmscl):
        """
        Make a pattern scale dictionary

        Making a pattern scaling dictionary to define the object at initialisation

        Parameters
        ----------
        tmscl : list
                Array of initial guesses for timescales for the pattern scaling
                pattern, all should be ints or floats.

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
                    (out, orgeof, neweof) = prpatt.get_timescales(
                        anomaly_data, tmscl, trnc
                    )

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
        forcing_series,forcing_total = sefps.run_and_return_per_forcer_results(self.exp_list)
        predicted = {}
        for exp in self.exp_list:
            if exp == "base":
                continue
            for fld in flds:
                if fld not in predicted:
                    predicted[fld] = self.predict_from_forcing_profile(
                        forcing_series[exp], fld, exp
                    )
                    predicted[fld]['time']=pd.to_datetime(predicted[fld]['time'],format='%Y')
                    

                else:
                    tmp=self.predict_from_forcing_profile(
                        forcing_series[exp], fld, exp
                    )
                    tmp['time']=pd.to_datetime(tmp['time'],format='%Y')
                    predicted[fld] = predicted[fld] + tmp

        return predicted,forcing_series,forcing_total

    def make_prediction_plot(
        self, ax, forc_timeseries, fld, exp="co2x2"
    ):  # pragma: no cover
        """
        Make plot with prediction for forcing time series

        Parameters
        ----------
        ax : matplotlib.axes
             Axes on which to do plotting
        forc_timeseries : np.array
            Array of with forcing timeseries for which to create predictions from the pattern
        fld : str
            Variable to make prediction for
        exp : str
            Experiment that defines the stepfunction response for the forcer in question
        """
        sim_data = self.predict_from_forcing_profile(forc_timeseries, fld, exp)
        mean_f_var = sim_data.weighted(
            prpatt.wgt(self.dacanom[fld][self.exp_list.index(exp), :100, :, :])
        ).mean(("lat", "lon"))
        plt.plot(mean_f_var, ax=ax)

    # Method to combine forcing timeseries for various components?

    def plot_global_mean_values(self, ax, fld, exp):  # pragma: no cover
        """
        Make plot of global mean values of pattern

        Parameters
        ----------
        ax : matplotlib.axes
             Axes on which to do plotting
        fld : str
            Name of field variable to plot
        exp : str
            Experiment for which to plot flobal mean pattern
        """
        data = self.dacanom[fld][self.exp_list.index(exp), :100, :, :]
        trun = self.patternflds[fld]
        ax.plot(prpatt.global_mean(data), label="Original Data")
        ax.plot(
            prpatt.global_mean(prpatt.recon(self.pattern_dict[exp][fld]["orgeof"])),
            label="EOF reconstruction (t=" + str(trun) + ")",
        )
        ax.plot(
            prpatt.global_mean(prpatt.recon(self.pattern_dict[exp][fld]["neweof"])),
            label="P-R fit to PCs (t=" + str(trun) + ")",
        )
        ax.set_xlabel("time (years)")
        ax.legend()
        ax.set_title(self.name)

    def plot_pca_map(self, fld, exp, comps_to_show=20):  # pragma: no cover
        """
        Make maps of principal components

        Parameters
        ----------
        fld : str
            Name of field variable to plot
        exp : str
            Experiment for which to plot principal components
        comps_to_show : int
            Maximal number of principal components to show
        """
        comps_to_show = min(comps_to_show, self.patternflds[fld])
        plothandle, ax = plt.subplots(comps_to_show, 1)
        plt.set_cmap("bwr")

        for i in range(comps_to_show):
            self.pattern_dict[exp][fld]["orgeof"]["v"][i, :, :].plot(
                ax=ax[i], cmap="bwr"
            )
        return plothandle

    def plot_reconstructed_globmean(self, ax, fld, exp):  # pragma: no cover
        """
        Make plot of global mean values of pattern

        Parameters
        ----------
        ax : matplotlib.axes
             Axes on which to do plotting
        fld : str
            Name of field variable to plot
        exp : str
            Experiment for which to plot global mean of original data, and pattern fields
        """
        raw_data = self.dacanom[fld][self.exp_list.index(exp), :100, :, :]
        pca_pattern = prpatt.recon(self.pattern_dict[exp][fld]["orgeof"])
        synthetic_pattern = prpatt.recon(self.pattern_dict[exp][fld]["neweof"])

        raw_data.weighted(prpatt.wgt(raw_data)).mean(("lat", "lon")).plot(
            color="cyan", ax=ax
        )
        pca_pattern.weighted(prpatt.wgt(raw_data)).mean(("lat", "lon")).plot(
            color="k", ax=ax
        )
        synthetic_pattern.weighted(prpatt.wgt(raw_data)).mean(("lat", "lon")).plot(
            color="red", ax=ax
        )
