"""
METEOR
"""
import logging

import xarray as xr
from . import prpatt
import numpy as np
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

def read_training_data(get_training_file_from_exp, exp_list):
    """
    Reading training data into xarray
    
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
    for i,exp in enumerate(exp_list):
        print(get_training_file_from_exp(exp))
        tmp=xr.open_dataset(get_training_file_from_exp(exp))
        if i==0:
            dac=tmp
        else:
            dac=xr.concat([dac,tmp],'expt')
    dac=dac.assign_coords({"expt": self.exp_list})
    ctrl=expts.index("base")
    varis=dac.data_vars
    dacanom=dac
    for var in varis:
        dacanom[var]=dac[var]-dac[var][ctrl,:,:,:,:].mean(dim='year',skipna=True)
    dacanom=dacanom.rename({'year': 'time'})


class MeteorPatternScaling:
    """
    Pattern scaling descriptor class

    Handles pattern scaling, defining
    and calculating it from a dataset
    Then provides routines to apply it
    to new data

    Attributes
    ----------
    exp_list: list
             List of experiments included in the pattern scaling
    daconom : xarray dataset
             Input data from the experiments belonging to the pattern
    patternflds: dict
             Dictionary of with variables for pattern scaling as
             keys, and their truncation length for PCAs as values
    od: dict
        Dictionary with the PCAs and patterns for the various experiments
        and variables. A nested dictionary that with the experiments of the 
        objects. First keyset: The experiments that the pattern is defined by, 
        second keyset: The variables for which patterns are produced. 
        Third keyset: neweof, a synthetic PCA for the data from the calculated 
        response timescales,
        orgeof, orginal PCA object from the data, and if data allows,
        outp, the lmfit parameter fit using the original PCA object and timescales
    name: str
        Name of the pattern, to be printed on plots etc.
    """

    def __init__(self, name, patternflds, get_training_file_from_exp, exp_list, tmscl=[2,50]):
        """
        Initialise MeteorPatternScaling

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
        exp_list : list
                   List of experiments to add to the pattern scaling
        """
        self.exp_list = exp_list
        self.daconom = read_training_data(get_training_file_from_exp, self.exp_list)
        self.patterflds = patternflds
        self.od = self._make_pattern_dict(tmscl)
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
            A nested dictionary that with the experiments of the objects. First
            keyset: The experiments that the pattern is defined by, second
            keyset: The variables for which patterns are produced. Third keyset:
            neweof, a synthetic PCA for the data from the calculated response timescales,
            orgeof, orginal PCA object from the data, and if data allows,
            outp, the lmfit parameter fit using the original PCA object and timescales
        """
        od = dict()
        for j,exp in enumerate(self.exp_list):
            od[exp] = dict()
            for fld,trnc in self.patternflds.items():
                od[exp][fld] = dict()
                #The :100? Flexible?
                X = self.daconom[fld][j,:100,:,:]
                if not np.isnan(np.mean(X)):
                    (ts,out,us,orgeof,neweof)=prpatt.get_timescales(X,tmscl,trnc)

                    od[exp][fld]["neweof"] = neweof
                    od[exp][fld]["orgeof"] = orgeof
                    od[exp][fld]["outp"] = out
                else:
                    od[exp][fld]["neweof"] = np.nan
                    od[exp][fld]["orgeof"] = np.nan
        return od


    def predict_from_forcing_profile(self, forc_timeseries, var, exp= "co2x2"):
        """
        Make prediction from experiment and a forcing profile

        Parameters
        ----------
        forc_timeseries : np.array
            Array of with forcing timeseries for which to create predictions from the pattern
        var : str
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
        #Add something to account for the forcing strength of the experiment
        Xf = prpatt.imodel_fileter(self.od[exp][var]['outp'].params,forc)
        Xsim = prpatt.rmodel(self.od[exp][var]['orgeof'], Xf)
        return Xsim

    def make_prediction_plot(self, ax, forc_timeseries, var, exp="co2x2"):
        """
        Make plot with prediction for forcing time series
        """
        Xsim = self.predict_from_forcing_profile(forc_timeseries,var,exp)
        ts_f = Xsim.weighted(prpatt.wgt(self.dacanom[var][exp_list.index[exp],:100,:,:])).mean(('lat','lon'))
        plt.plot(mean_f_var, ax = ax)
        
    #Method to combine forcing timeseries for various components?    
    

    def plot_global_mean_values(self, ax, fld, exp):
    
      X = self.daconom[fld][self.exps_list.index(exp), :100,:,:]
      ax.plot(prpatt.global_mean(X), label='Original Data')
      try:
          ax.plot(prpatt.global_mean(prpatt.recon(self.od[exp][fld]['orgeof'])),label='EOF reconstruction (t=2)')
          ax.plot(prpatt.global_mean(prpatt.recon(self.od[exp][fld]['neweof'])),label='P-R fit to PCs (t=2)')
          ax.set_xlabel('time (years)')
          ax.legend()
          ax.set_title(mdl)
      except:
          LOGGER.warning("No eof data to plot")
              

    def plot_pca_map(self, fld, exp, comps_to_show = 20):
        comps_to_show = min(comps_to_show, self.patternflds[fld])
        p,ax= plt.subplots(comps_to_show, 1)
        plt.set_cmap("bwr")

        for i in range(comps_to_show):
            self.od[exp][fld]['orgeof']['v'][i,:,:].plot(ax=ax[i], cmap='bwr', vmin=-1e-4, vmax=1e-4)


    def plot_reconstructed_globmean(self, ax, fld,exp):
        Xp = self.daconom[fld][self.exp_list.index(exp),:100,:,:]
        Xrp = prpatt.recon(self.od[exp][fld]['orgeof'])
        Xrp = prpatt.recon(self.od[exp][fld]['neweof'])
        
        Xp.weighted(prpatt.wgt(Xp)).mean(('lat','lon')).plot(color='cyan', ax =ax)
        Xrp.weighted(prpatt.wgt(Xp)).mean(('lat','lon')).plot(color='k', ax=ax)
        Xrs.weighted(prpatt.wgt(Xp)).mean(('lat','lon')).plot(color='red', ax=ax)


    
        #more methods for how to apply the patternscaling to new data
