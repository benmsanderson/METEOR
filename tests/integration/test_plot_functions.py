# TODO : Write these, also figure out why new function is not included in test

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from meteor import MeteorPatternScaling, meteor_plot_utils


def test_various_plot_modules(test_data_dir):
    canesm_basic_pattern = MeteorPatternScaling(
        "pdrmip-CanESM2-basic",
        {"tas": 2, "pr": 10},
        lambda exp: os.path.join(test_data_dir, f"pdrmip-{exp}_T42_ANN.nc"),
        exp_list=["base", "co2x2"],
    )

    plothandle = meteor_plot_utils.plot_pca_map(
        canesm_basic_pattern, "pr", "co2x2", comps_to_show=4
    )
    assert isinstance(plothandle, matplotlib.figure.Figure)
    assert plothandle

    plt.clf()
    fig, axs = plt.subplots(3, 1)
    meteor_plot_utils.make_prediction_plot(
        canesm_basic_pattern, axs[0], np.linspace(0, 2, 100), "tas"
    )
    meteor_plot_utils.plot_global_mean_values(
        canesm_basic_pattern, axs[1], "tas", "co2x2"
    )
    meteor_plot_utils.plot_reconstructed_globmean(
        canesm_basic_pattern, axs[2], "tas", "co2x2"
    )
    assert fig
    for ax in axs:
        assert ax
