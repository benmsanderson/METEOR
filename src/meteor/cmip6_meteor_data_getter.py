"""
Module to get CMIP6 data and convert to format that can be used for METEOR
"""

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

cmip6_to_meteor_exp_remapper = {
    "base": "piControl",
    "co2x4": "abrupt-4xCO2",
    "co2x8": "abrupt-4xCO2",
    "co2x16": "abrupt-4xCO2",
    "1pc": "1pctCO2",
}


def multiply_along_axis(array_a, array_b, axis):
    """
    Multiply to arrays along a given axis

    Pure infrastructure function to make multiplying along an axis that is not the last
    axis along two np.ndarray objects without encountering broadcasting issues

    Parameters
    ----------
    array_a : np.ndarray
        First array to multiply
    array_b : np.ndarray
        Second array to multiply
    axis : int
        Axis to multiply along

    Returns
    -------
    np.ndarray
        Array/matrix with result of multiplication, and the multiplication axis
        back where it was
    """
    return np.swapaxes(np.swapaxes(array_a, axis, -1) * array_b, -1, axis)


def year_mean_monthly(monthly_data):
    """
    Calculate yearmean from monthly data

    Weighting by days in month and calculating the yearly mean of an array of
    monthly data. The data are assumed to be January to December per year, and
    will assume for simplicity that the data follows a no-leap calendar

    Parameters
    ----------
    monthly_data : np.ndarray
        1 or multiple dimensional np.ndarray with the first dimension being time
        and on monthly resolutions running from January to December for each year

    Returns
    -------
    np.ndarray
        Weighted year averages for the monthly_data, the output-dimension will be
        the same as for the monthly_data, except that the first time dimension
        will be 1/12th as long as before including only yearly mean values
    """
    month_weights = np.tile(
        np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) / 365.0 * 12.0,
        monthly_data.shape[0] // 12,
    )
    mul_weigths = multiply_along_axis(monthly_data, month_weights, 0)
    return np.mean(
        mul_weigths.reshape(
            -1,
            12,
            monthly_data.shape[1],
            monthly_data.shape[2],
        ),
        axis=1,
    )


def year_mean_monthly_xarray(monthly_xarray):
    """
    Calculate yearmean from monthly xarray dataarray

    Essentially just a wrapper for xarray around the numpy based
    year_mean_monthly function above

    Parameters
    ----------
    monthly_xarray : xr.dataArray
        with the first dimension being time followed by lat and lon
        and on monthly resolutions running from January to December for each year

    Returns
    -------
    xr.dataArray
        with same dimensions as before but with the time dimension 12 times
        fewer entries, corresponding to yearly values, this coordinate now
        also no longer has an associated coordinate as it has changed
    """
    return xr.apply_ufunc(
        year_mean_monthly,
        monthly_xarray,
        input_core_dims=[
            ["time", "lat", "lon"],
        ],
        output_core_dims=[
            ["time", "lat", "lon"],
        ],
        exclude_dims=set(("time",)),
        dask="allowed",
    )


def make_xarray_with_correct_dims(fld_names, fld_values):
    """
    Make a dataset for a list of dataArrays over the same dimensions

    Parameters
    ----------
    fld_names : list
        containing names of the fields, the ordering should be
        the same as for the fld_values
    fld_values : list
        containing the xr.DataArrays for each of the fields in the
        fld_names list

    Returns
    -------
    xr.Dataset
    """
    ds = xr.Dataset(
        data_vars={fld_names[i]: fld_values[i] for i in range(len(fld_names))},
    )
    return ds


def initialise_dataframe_and_models(
    df_all1, flds, exps, mdl_skipmbrs=None
):  # pylint: disable=too-many-locals, too-many-branches
    """
    Intialise a dataframe with complete data for the first full data ensemble
    member from a cmip6 data list

    Parameters
    ----------
    df_all1: list
        two-dimensional list of pd.DataFrame where the first dimension runs over the
        experiments and the second over the fields in fld
    flds: list
        of names of fields
    exps: list
        of names of experiments
    mdl_skipmbrs : dict
        of models and a list of ensemble members to skip for that model. If none
        is sent, it defaults to skipping NorESM2-Lm which has insufficient data
        for pr for the historical experiment even though there is a file
    Returns
    -------
    list
        Consisting of first a list with dataframes with models and first ensemble members
        with full data per experiment and field, and second a list of the models
        that have the full data
    """
    mdls1 = df_all1[0][0].source_id.unique()
    mdls1.sort()
    df_all = []
    cnames = df_all1[0][0].columns
    if mdl_skipmbrs is None:
        mdl_skipmbrs = {"NorESM2-LM": ["r1i1p1f1"]}
    for i in range(len(exps)):
        # tmp = []
        # for fld in flds:
        #    tmp.append(pd.DataFrame(columns=cnames))
        tmp = [pd.DataFrame(columns=cnames) for j in range(len(flds))]
        df_all.append(tmp)

    mdls = []

    n = 0
    for mdl in mdls1:

        # Test that one ensemble member has all data:
        sufficient_data = True
        for i in range(len(exps)):
            # find first variable for expt/model
            for j in range(len(flds)):
                if "historical" in exps:
                    ii = exps.index("historical")
                    hist_tmp = df_all1[ii][j].query(
                        "source_id=='" + mdl + "' & experiment_id == 'historical'"
                    )
                    hmb = hist_tmp.member_id.unique()
                else:
                    hmb = ["zzzzzzzz"]  # dummy value to avoid another if statement
                tmp = df_all1[i][j].query("source_id=='" + mdl + "'")
                mmbs = tmp.member_id.unique()
                if len(mmbs) > 0:
                    mmb = mmbs[0]
                    if hmb[0] in mmbs:
                        mmb = hmb[0]

                    tt = df_all1[i][j].query(f"source_id=='{mdl}' & member_id=='{mmb}'")
                    df_all[i][j].loc[n] = tt.values[0]
                else:
                    mmb = -1
                    df_all[i][j].loc[n] = None
                    sufficient_data = False
            # add model to final list

        if sufficient_data:
            mdls.append(mdl)
            n = n + 1
            # print(f"Model {mdl} has full data")

    return df_all, mdls


class Cmip6MeteorDataGetter:
    """
    Cmip6MeteorDataGetter class

    Objects of this class holds a shortlist of CMIP6 data links to
    data for one ensemble member for all models that have complete data for
    the full combination of the classes fields and experiments all monthly
    data.

    In various methods it can return yearly mean data for a single
    dataset, or a full METEOR training dataset

    Attributes
    ----------
    flds: list
        List of variable fields for the instance to consider
    exps: list
        List of experiments for the instance to consider


    """

    def __init__(
        self,
        flds=None,
        exps=None,
        dbe=None,
    ):
        """
        Initialise data getter object

        Parameters
        ----------
        flds : list
            Of variable fields to include (monthly data)
            If nothing is sent for this it will be set to
            ['tas', 'pr']
        exps : list
            Of experiments to include. If nothing is sent for
            this it will be set to ['piControl', 'abrupt4xCO2']
        dbe : list
            Of CMIP6 project names for each of the experiments. This should be the same length
            as exps, and have values corresponding to ecah experiment in the same order.
            If nothing is set, it will be set to a list of all entries equal to
            'CMIP' with the same lenght as exps
        df_all : list
            Two dimensional list along experiments and fields, every entry
            is a pandas.DataSet with lines with data placement for one
            ensemble memeber for each model that has data for the full
            combination of fields and experiments
        models : list
            List of models with full data available
        gcs : gcfs.GCSFileSystem
              A GCSFileSystem to load data
        """
        dbe = self._set_fld_exps_dbe(flds, exps, dbe)

        df = pd.read_csv(
            "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv",
            low_memory=False,
        )
        df_all1 = []
        for i, exp in enumerate(self.exps):
            df_ta1 = []
            for fld in self.flds:
                tmp = df.query(
                    f"activity_id=='{dbe[i]}' & table_id == 'Amon' & variable_id == '{fld}' & experiment_id == '{exp}'"
                )
                df_ta1.append(tmp)
            df_all1.append(df_ta1)
        self.df_all, self.models = initialise_dataframe_and_models(
            df_all1, flds=self.flds, exps=self.exps
        )
        self.gcs = gcsfs.GCSFileSystem(token="anon")  # nosec

    def _set_fld_exps_dbe(self, flds, exps, dbe):
        """
        Private method to set flds and exps and dbe and
        take care of if they are not set then have defaults

        Parameters
        ----------
        flds : list
            Of variable fields to include (monthly data)
            If None is sent for this it will be set to
            ['tas', 'pr']
        exps : list
            Of experiments to include. If None is sent for
            this it will be set to ['piControl', 'abrupt4xCO2']
        dbe : list
            Of CMIP6 project names for each of the experiments. This should be the same length
            as exps, and have values corresponding to ecah experiment in the same order.
            If None is sent, it will be set to a list of all entries equal to
            'CMIP' with the same lenght as exps

        Returns
        -------
        list
            dbe, If None was sent, a list of all entries equal to
            'CMIP' with the same lenght as exps will be returned
        """
        if not flds:
            flds = ["tas", "pr"]
        if not exps:
            exps = ["piControl", "abrupt-4xCO2"]
        if not dbe:
            dbe = ["CMIP" for i in range(len(exps))]
        self.flds = flds
        self.exps = exps
        return dbe

    def get_models_avail(self):
        """
        Get list of available models

        Returns
        -------
        list
            List of available models. They have full data for the dataGetters flds and
            experiment combinations for at least one ensemble member.
        """
        return self.models.copy()

    def check_if_model_has_data(self, model):
        """
        Check if a model has full data for this dataGetter object

        Parameters
        ----------
        model: str
            Name of model

        Returns
        -------
        bool
            True if the model has full data for at least on ensemble member
            False otherwise
        """
        if model in self.models:
            return True
        return False

    def get_single_var_mod_data(self, exp, fld, model):
        """
        Get mean data for a single variable, model and experiment combination

        Parameters
        ----------
        exp: str
            Name of experiment for which you want data
        fld: str
            Name of field for which you want data
        model: str
            Name of model for which you want data

        Returns
        -------
        xr.DataArrray
            Data downloaded from zstore

        Raises
        ------
        KeyError
            If either exp, or fld is not in the instances self.exps and self.flds or if the model
            is not among the models that has complete data for all the combinations of experiments
            and fields that the instance holds.
        """
        if exp not in self.exps:
            raise KeyError(
                f"This datagetter does not handle data from the {exp} experiment, available options are {self.exps} "
            )
        if fld not in self.flds:
            raise KeyError(
                f"This datagetter does not handle {fld} data, available options are {self.flds}"
            )
        if not self.check_if_model_has_data(model):
            raise KeyError(f"No or incomplete data for {model}")
        zstore_ref = (
            self.df_all[self.exps.index(exp)][self.flds.index(fld)]
            .loc[self.models.index(model)]
            .zstore
        )

        if zstore_ref is np.nan:
            raise KeyError(f"No zstore ref for {model}")

        mapper = self.gcs.get_mapper(zstore_ref)
        fld = xr.open_zarr(mapper, decode_times=False).sortby("time")
        return fld

    def get_single_var_mod_data_yearmean(self, exp, fld, model):
        """
        Get yearly mean data for a single variable, model and experiment combination

        Parameters
        ----------
        exp: str
            Name of experiment for which you want data
        fld: str
            Name of field for which you want data
        model: str
            Name of model for which you want data

        Returns
        -------
        xr.DataArrray
            Data converted from monthly to yearly mean data and including and extra flat ens dimension
        """
        ds = self.get_single_var_mod_data(exp, fld, model)
        if ds is None:
            return None

        var_yearly = year_mean_monthly_xarray(ds[fld])
        var_yearly = var_yearly.assign_coords(
            {"time": np.arange(len(ds.time.values) // 12)}
        ).rename({"time": "year"})
        var_yearly = var_yearly.expand_dims(
            dim={"ens": np.array([1])}
        )  # .assign_coords({'ens':1})
        return var_yearly

    def make_meteor_training_data(self, exp, model, exp_mapper=None):
        """
        Make xr.dataset with data and format used for meteor

        Parameters
        ----------
        exp : str
            Experiment name used by METEOR, this will be remapped to an experiment
            in the cmip6 set by the exp_mapper dictionary
        model : str
            Name of model for which to find and format training data
        exp_mapper : dict
            Dictionary that maps experiments of the METEOR type (keys) to experiments
            getable by the CMIP6DataGetter (values)

        Returns
        -------
        xr.Dataset
            Dataset with yearly data on the format usable for METEOR
        """
        if not exp_mapper:
            exp_mapper = cmip6_to_meteor_exp_remapper
        fld_values = []
        for fld in self.flds:
            if exp in exp_mapper:
                fld_values.append(
                    self.get_single_var_mod_data_yearmean(exp_mapper[exp], fld, model)
                )
            else:
                fld_values.append(
                    self.get_single_var_mod_data_yearmean(exp, fld, model)
                )
        training_data = make_xarray_with_correct_dims(self.flds, fld_values)
        return training_data

    def make_meteor_training_data_composite(self, exps, model):
        """
        Make xr.dataset with data and format used for meteor

        Parameters
        ----------
        exps : list
            Lists with experiments to be glued together, should be ordered
            the same order as the experiments are meant to be concatenated
        model : str
            Name of model for which to find and format training data
        Returns
        -------
        xr.Dataset
            Dataset with yearly data on the format usable for METEOR
        """
        fld_values = []
        for fld in self.flds:
            value = None
            for exp in exps:
                if value is None:
                    value = self.get_single_var_mod_data_yearmean(exp, fld, model)
                else:
                    next_dataset = self.get_single_var_mod_data_yearmean(
                        exp, fld, model
                    )
                    start_year = value["year"].values[-1] + 1
                    end_year_plus = start_year + next_dataset.sizes["year"]
                    # print(next_dataset.sizes["year"])
                    # print(len(range(start_year, end_year_plus)))
                    # print(range(start_year, end_year_plus))
                    next_dataset = next_dataset.assign_coords(
                        {"year": np.arange(start_year, end_year_plus)}
                    )
                    value = xr.concat(
                        [value, next_dataset],
                        dim="year",
                    )
            fld_values.append(value)
        return make_xarray_with_correct_dims(self.flds, fld_values)
