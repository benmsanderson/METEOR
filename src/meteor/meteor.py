"""
METEOR
"""

import logging
import os
import pickle  # nosec - Used for trusted model serialization only

import numpy as np
import pandas as pd
import xarray as xr

from . import prpatt, scm_forcer_engine

LOGGER = logging.getLogger(__name__)


class Meteor:
    """
    Main METEOR emulator class.

    This class orchestrates pattern scaling and noise generation
    to create an ensemble of climate realisations based on a training set.
    """

    def __init__(self, pattern_scaling_model, noise_generator):
        """
        Initialise the METEOR emulator.

        Parameters
        ----------
        pattern_scaling_model : object
            A fitted pattern scaling model. It is assumed to have a `predict`
            method that takes a global mean trajectory (xr.DataArray) and
            returns a spatio-temporal field (xr.DataArray).
        noise_generator : object
            A fitted noise generator with a `generate_noise` method.
        """
        self.pattern_scaling_model_ = pattern_scaling_model
        self.noise_generator_ = noise_generator
        self.X_train_ = None  # pylint: disable=invalid-name
        self.ensemble_ = None

    # pylint: disable=invalid-name
    def fit(self, X_train):
        """
        "Fits" the model by providing the training data.

        In a complete workflow, this method might also train the pattern
        scaling and noise models. Here, we assume they are pre-trained and
        we just store the training data required for generating new
        realisations.

        Parameters
        ----------
        X_train : xr.DataArray
            Training data with dimensions ('member', 'time', 'lat', 'lon').
        """
        self.X_train_ = X_train
        # The initial ensemble is the training data itself.
        self.ensemble_ = X_train.copy(deep=True)
        return self

    def add_realisation(self):
        """
        Generate and add a new realisation to the ensemble.

        This implements the corrected logic where a single mean warming
        pattern is calculated from the ensemble-mean trajectory of the
        training data. A new, unique noise realisation is then added to this
        mean pattern to create the new ensemble member.
        """
        if self.X_train_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X_train) first.")

        # generate a new noise realisation
        new_noise = self.noise_generator_.generate_noise()

        # The mean signal is the forced response component, which is assumed to be
        # the same for all realisations. We estimate this from the mean
        # global-mean temperature change across all training members.
        mean_global_mean_trajectory = self.X_train_.mean(dim=["lat", "lon", "member"])
        mean_warming_pattern = self.pattern_scaling_model_.predict(
            mean_global_mean_trajectory
        )

        # A new realisation is the sum of the mean warming pattern and a new
        # noise realisation.
        new_realisation = mean_warming_pattern + new_noise

        # add the new realisation to the ensemble
        self.ensemble_ = xr.concat([self.ensemble_, new_realisation], dim="member")


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


def calculate_residual_and_do_crude_nan_cut(daconom_field, predicted_without_fld):
    """
    Calculate a residual between input data field, and predicted field
    Includes light handling of the data field possibly missing data in the end years

    Parameters
    ----------
    daconom_field : xr.DataArray
        Data field with time, lat and lon dimensions
    predicted_without_fld : xr.DataArray
        Data field of predicted values for data without residual forcing
        should have same dimensions of same size as daconom_field

    Returns
    -------
    xr.DataArray
        The result of subtracting predicted_without_fld from the daconom_field
        If the daconom_fld has nan values for the last timesteps (i.e. missing
        data for the latest years), both that and the predicted_without_fld
        will be cut to omit those years before the subtractions is conducted
        If the daconom_field has nans scattered throughout the data, an error
        will be raised.

    Raises
    ------
    ValueError
        If the daconom_field has nans scattered throughout the data,
        (i.e. data missing for some years, present for later years) a
        ValueError will be raised.
    """
    if np.isnan(daconom_field.values).sum() > 0:
        gm = prpatt.global_mean(daconom_field)
        tot_len = len(gm.values)
        tot_nan_yrs = np.isnan(gm.values).sum()
        nan_in_last = np.isnan(gm.values[-tot_nan_yrs:]).sum()
        if nan_in_last < tot_nan_yrs:
            raise ValueError(
                "The dataset you are trying to emulate includes NaN values scattered throughout. METEOR does not currently support emulation such datasets"
            )
        LOGGER.warning(
            "Cutting residual dataset to avoid nans at the end of the dataset"
        )
        daconom_field = daconom_field[: (tot_len - tot_nan_yrs), :, :]
        predicted_without_fld = predicted_without_fld[: (tot_len - tot_nan_yrs), :, :]
    return daconom_field - predicted_without_fld


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
        patternflds=None,
        get_training_file_from_exp=None,
        exp_list=None,
        from_file=True,
        ssp_input=None,
        anom_timescales=None,
        cache_dir=None,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Initialise Pattern Scaling object

        Defining the patternscaling object from lists of experiments

        Parameters
        ----------
        name : str
               name of the model/dataset for that this patter belongs to
        patternflds : dict, optional
                    keys are names of the varibles to be considered
                    Values are number of timescales to fit.
                    Required when training from scratch (cache miss or no cache_dir).
                    Not needed when loading from cache.
        get_training_file_from_exp : function, optional
                    Function that defines how to get find the location
                    of the training data input file for a given experiment.
                    Required when training from scratch (cache miss or no cache_dir).
                    Not needed when loading from cache.
        exp_list : dict, optional
                   List with experiment names.
                   Required when training from scratch (cache miss or no cache_dir).
                   Not needed when loading from cache.
        anom_timescales : dict
            Optional parameter
            Like patternfields should have fields as values, and number
            of timescales to fit from the anomaly experiments for that field
            as values. If this parameter is not sent, the 1 timescale per field
            will be assumed if anomaly experiments are included.
        cache_dir : str, optional
            Directory to cache the trained pattern scaling model. If provided,
            the model will be saved after training and loaded from cache if
            it already exists. When loading from cache, training parameters
            (patternflds, get_training_file_from_exp, exp_list) are not required.

        Examples
        --------
        >>> # First time: train and cache
        >>> model = MeteorPatternScaling(
        ...     "cesm2-model",
        ...     patternflds={"tas": 3},
        ...     get_training_file_from_exp=lambda key: training_data[key],
        ...     exp_list=["base", "co2x4"],
        ...     cache_dir="./cache"
        ... )
        >>>
        >>> # Subsequent times: load from cache (no training data needed!)
        >>> model = MeteorPatternScaling(
        ...     "cesm2-model",
        ...     cache_dir="./cache"
        ... )
        """
        self.name = name

        # Try to load from cache if cache_dir is provided
        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, f"{name}_pattern_scaling.pkl")
            if os.path.exists(cache_path):
                print(f"📦 Loading cached pattern scaling model from {cache_path}")
                self.load_model(cache_path)
                return

        # If not loaded from cache, validate required parameters and train the model
        if patternflds is None:
            raise ValueError(
                "patternflds is required when training a new model "
                "(not loading from cache)"
            )
        if get_training_file_from_exp is None:
            raise ValueError(
                "get_training_file_from_exp is required when training a new model "
                "(not loading from cache)"
            )
        if exp_list is None:
            raise ValueError(
                "exp_list is required when training a new model "
                "(not loading from cache)"
            )

        sefps = scm_forcer_engine.ScmEngineForPatternScaling(None)
        scaling = sefps.run_to_get_scaling(exp_list)
        self.exp_forc_dict = {exp: scaling[i] for i, exp in enumerate(exp_list)}
        self.dacanom = read_training_data(
            get_training_file_from_exp, exp_list, from_file=from_file
        )
        self.exp_list = exp_list
        self.patternflds = patternflds
        self.pattern_dict = self._make_pattern_dict()
        if "xanom" in "-".join(exp_list):
            if anom_timescales is None or not isinstance(anom_timescales, dict):
                anom_timescales = {}
                for fld in patternflds:
                    anom_timescales[fld] = 1
            else:
                for fld in patternflds:
                    if fld not in anom_timescales:
                        anom_timescales[fld] = 1

            if all(value == 0 for value in anom_timescales.values()):
                anom_exps = [anomexp for anomexp in exp_list if "xanom" in anomexp]
                for anomexp in anom_exps:
                    self.exp_list.remove(anomexp)
                    self.dacanom = self.dacanom.where(
                        self.dacanom.expt != anomexp, drop=True
                    )
                    del self.exp_forc_dict[anomexp]
                    del self.pattern_dict[anomexp]
            else:
                self.anom_timescales = anom_timescales
                self._add_patterns_for_residual_exp(ssp_input)

        # Save to cache if cache_dir is provided
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{name}_pattern_scaling.pkl")
            self.save_model(cache_path)
            print(f"💾 Cached pattern scaling model to {cache_path}")

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
            if self.anom_timescales[fld] == 0:
                # TODO: Delete also the input data from daconom for this?
                del self.pattern_dict[exp][fld]
                continue
            predicted_without_fld = predicted_without[fld].isel(
                time=slice(start_index, start_index + em_len)
            )
            predicted_without_fld = predicted_without_fld.assign_coords(
                time=np.arange(em_len)
            )
            residual = calculate_residual_and_do_crude_nan_cut(
                self.dacanom[fld][exp_index, :em_len, :, :], predicted_without_fld
            )
            (out, pattern_full) = prpatt.get_timescales_from_anomaly(
                residual, forcing_of_residual, n_modes=self.anom_timescales[fld]
            )
            self.pattern_dict[exp][fld]["pattern_full"] = pattern_full
            self.pattern_dict[exp][fld]["outp"] = out

    def predict_from_forcing_profile(
        self,
        forc_timeseries,
        fld,
        exp="co2x2",
        year_0=1850,
        return_patterns_per_mode=False,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
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
        return_patterns_per_mode : bool
            Option to have predictions returned separately per mode. The predicted patterns
            will then have a separate dimension for the modes.

        Returns
        -------
        xarray dataarray
             Prediction object for the variable given the forcing time series

        !Todo: Add tests to check that variable and experiment are in the patterns
        patternfld and exp_lists
        """
        convolved_pca = prpatt.imodel_filter(
            self.pattern_dict[exp][fld]["outp"],
            forc_timeseries,
            forc_step=self.exp_forc_dict[exp],
            year_0=year_0,
        )
        if not return_patterns_per_mode:
            predicted = prpatt.rmodel(
                self.pattern_dict[exp][fld]["pattern_full"], convolved_pca
            )
        else:
            predicted = prpatt.recon_separately(
                self.pattern_dict[exp][fld]["pattern_full"], convolved_pca
            )
        return predicted

    def predict_from_combined_experiment(
        self,
        emissions_data,
        concentrations_data,
        flds,
        conc_run=False,
        return_patterns_per_mode=False,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
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
        return_patterns_per_mode : bool
            Option to have predictions returned separately per mode. The predicted patterns
            will then have a separate dimension for the modes.

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
            forcing_series,
            flds,
            cfg["nystart"],
            return_patterns_per_mode=return_patterns_per_mode,
        )
        return predicted

    def _predict_combined_experiment_from_forcer_series(
        self, forcing_series, flds, nystart, return_patterns_per_mode=False
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
                if fld not in self.pattern_dict[exp].keys():
                    continue
                if fld not in predicted:
                    predicted[fld] = self.predict_from_forcing_profile(
                        forcing_series[exp],
                        fld,
                        exp,
                        year_0=nystart,
                        return_patterns_per_mode=return_patterns_per_mode,
                    )
                    predicted[fld]["time"] = pd.to_datetime(
                        predicted[fld]["time"], format="%Y"
                    )

                else:
                    tmp = self.predict_from_forcing_profile(
                        forcing_series[exp],
                        fld,
                        exp,
                        year_0=nystart,
                        return_patterns_per_mode=return_patterns_per_mode,
                    )
                    tmp["time"] = pd.to_datetime(tmp["time"], format="%Y")
                    if return_patterns_per_mode:
                        predicted[fld] = xr.concat((predicted[fld], tmp), dim="mode")
                    else:
                        predicted[fld] = predicted[fld] + tmp
        return predicted

    # pylint: disable=too-many-locals
    def to_monthly(self, annual_prediction, start_year=None):
        """
        Convert annual prediction output to monthly intervals.

        This method expands annual climate predictions to monthly resolution
        by repeating each annual value 12 times. This allows easy combination
        with monthly noise generator output.

        Parameters
        ----------
        annual_prediction : xr.DataArray
            Annual climate prediction with dimensions (time, lat, lon) where
            time represents years
        start_year : int, optional
            Starting year for the monthly time coordinate. If None, uses
            integer indices starting from 0.

        Returns
        -------
        xr.DataArray
            Monthly climate prediction with dimensions (month, lat, lon)
            where each annual value is repeated for 12 consecutive months

        Examples
        --------
        >>> # Get annual prediction from METEOR
        >>> annual_pred = pattern_model.predict_from_combined_experiment(...)
        >>> # Convert to monthly for combining with noise
        >>> monthly_pred = pattern_model.to_monthly(annual_pred['tas'])
        >>> # Now can add monthly noise
        >>> full_monthly = monthly_pred + noise_realization
        """
        # Validate input
        if not isinstance(annual_prediction, xr.DataArray):
            raise ValueError("annual_prediction must be an xarray DataArray")

        if "time" not in annual_prediction.dims:
            raise ValueError("annual_prediction must have a 'time' dimension")

        # Get dimensions
        time_dim = annual_prediction.get_axis_num("time")
        n_years = annual_prediction.shape[time_dim]
        n_months = n_years * 12

        # Create expanded array by repeating each year 12 times
        # Use numpy repeat along the time axis
        expanded_values = np.repeat(annual_prediction.values, 12, axis=time_dim)

        # Create monthly time coordinate
        if start_year is not None:
            # Create proper monthly time coordinate based on years
            months = []
            for year_idx in range(n_years):
                year = start_year + year_idx
                for month in range(12):
                    months.append(year * 12 + month)  # Year-month index
            monthly_coord = np.array(months)
        else:
            # Use simple integer indexing
            monthly_coord = np.arange(n_months)

        # Create new DataArray with monthly dimensions
        # Replace time dimension with month dimension
        new_dims = list(annual_prediction.dims)
        new_dims[time_dim] = "month"

        # Create new coordinates
        new_coords = {}
        for coord_name, coord_values in annual_prediction.coords.items():
            if coord_name == "time":
                new_coords["month"] = monthly_coord
            else:
                new_coords[coord_name] = coord_values

        # Create the monthly DataArray
        monthly_prediction = xr.DataArray(
            expanded_values,
            dims=new_dims,
            coords=new_coords,
            attrs=annual_prediction.attrs.copy(),
        )

        # Update attributes to indicate monthly conversion
        monthly_prediction.attrs["converted_to_monthly"] = True
        if "description" in monthly_prediction.attrs:
            monthly_prediction.attrs["description"] = (
                monthly_prediction.attrs["description"]
                + " (converted from annual to monthly by repeating values)"
            )

        return monthly_prediction

    def save_model(self, filepath):
        """
        Save the fitted pattern scaling model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        model_data = {
            "name": self.name,
            "exp_forc_dict": self.exp_forc_dict,
            "exp_list": self.exp_list,
            "patternflds": self.patternflds,
            "pattern_dict": self.pattern_dict,
            "dacanom": self.dacanom,
        }

        # Include anom_timescales if it exists
        if hasattr(self, "anom_timescales"):
            model_data["anom_timescales"] = self.anom_timescales

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"✅ Pattern scaling model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a fitted pattern scaling model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)  # nosec - Loading trusted model files only

        self.name = model_data["name"]
        self.exp_forc_dict = model_data["exp_forc_dict"]
        self.exp_list = model_data["exp_list"]
        self.patternflds = model_data["patternflds"]
        self.pattern_dict = model_data["pattern_dict"]
        self.dacanom = model_data["dacanom"]

        # Load anom_timescales if it exists
        if "anom_timescales" in model_data:
            self.anom_timescales = model_data["anom_timescales"]

        print(f"✅ Pattern scaling model loaded from {filepath}")
