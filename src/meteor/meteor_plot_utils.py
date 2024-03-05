"""
Plotting library for METEOR
"""

import logging

import matplotlib.pyplot as plt

from . import prpatt

LOGGER = logging.getLogger(__name__)


def make_prediction_plot(pattern, ax, forc_timeseries, fld, exp="co2x2"):
    """
    Make plot with prediction for forcing time series

    Parameters
    ----------
    pattern: meteorpatternscaling
        Pattern to be plotted
    ax : matplotlib.axes
         Axes on which to do plotting
    forc_timeseries : np.array
        Array of with forcing timeseries for which to create predictions from the pattern
    fld : str
        Variable to make prediction for
    exp : str
        Experiment that defines the stepfunction response for the forcer in question
    """
    sim_data = pattern.predict_from_forcing_profile(forc_timeseries, fld, exp)
    mean_f_var = sim_data.weighted(
        prpatt.wgt(pattern.dacanom[fld][pattern.exp_list.index(exp), :100, :, :])
    ).mean(("lat", "lon"))
    ax.plot(mean_f_var)


# Method to combine forcing timeseries for various components?


def plot_global_mean_values(pattern, ax, fld, exp):
    """
    Make plot of global mean values of pattern

    Parameters
    ----------
    pattern: meteorpatternscaling
        Pattern to be plotted
    ax : matplotlib.axes
        Axes on which to do plotting
    fld : str
        Name of field variable to plot
    exp : str
        Experiment for which to plot flobal mean pattern
    """
    data = pattern.dacanom[fld][pattern.exp_list.index(exp), :100, :, :]
    trun = pattern.patternflds[fld]
    ax.plot(prpatt.global_mean(data), label="Original Data")
    ax.plot(
        prpatt.global_mean(prpatt.recon(pattern.pattern_dict[exp][fld]["orgeof"])),
        label="EOF reconstruction (t=" + str(trun) + ")",
    )
    ax.plot(
        prpatt.global_mean(prpatt.recon(pattern.pattern_dict[exp][fld]["neweof"])),
        label="P-R fit to PCs (t=" + str(trun) + ")",
    )
    ax.set_xlabel("time (years)")
    ax.legend()
    ax.set_title(pattern.name)


def plot_pca_map(pattern, fld, exp, comps_to_show=20):
    """
    Make maps of principal components

    Parameters
    ----------
    pattern: meteorpatternscaling
        Pattern to be plotted
    fld : str
        Name of field variable to plot
    exp : str
        Experiment for which to plot principal components
    comps_to_show : int
        Maximal number of principal components to show
    """
    comps_to_show = min(comps_to_show, pattern.patternflds[fld])
    plothandle, ax = plt.subplots(comps_to_show, 1)
    plt.set_cmap("bwr")

    for i in range(comps_to_show):
        pattern.pattern_dict[exp][fld]["orgeof"]["v"][i, :, :].plot(
            ax=ax[i], cmap="bwr"
        )
    return plothandle


def plot_reconstructed_globmean(pattern, ax, fld, exp):
    """
    Make plot of global mean values of pattern

    Parameters
    ----------
    pattern: meteorpatternscaling
        Pattern to be plotted
    ax : matplotlib.axes
        Axes on which to do plotting
    fld : str
        Name of field variable to plot
    exp : str
        Experiment for which to plot global mean of original data, and pattern fields
    """
    raw_data = pattern.dacanom[fld][pattern.exp_list.index(exp), :100, :, :]
    pca_pattern = prpatt.recon(pattern.pattern_dict[exp][fld]["orgeof"])
    synthetic_pattern = prpatt.recon(pattern.pattern_dict[exp][fld]["neweof"])

    raw_data.weighted(prpatt.wgt(raw_data)).mean(("lat", "lon")).plot(
        color="cyan", ax=ax
    )
    pca_pattern.weighted(prpatt.wgt(raw_data)).mean(("lat", "lon")).plot(
        color="k", ax=ax
    )
    synthetic_pattern.weighted(prpatt.wgt(raw_data)).mean(("lat", "lon")).plot(
        color="red", ax=ax
    )
