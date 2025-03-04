### METEOR
Multivariate Emulation of Time-Evolving and Overlapping Responses

# Installation
```
git clone https://github.com/benmsanderson/METEOR.git
```
# First time setup:
```
cd METEOR
make first-venv
make clean
make virtual-environment
```

# Update to newer version from github
```
git pull
make clean
make virtual-environment

If you are still having trouble with missing packages try this:
rm -r venv
make first-venv
make clean
make virtual-environment
```
## Jupyter Notebooks
The notebooks folder provides simple working examples to run the model within a Jupyter environment, and plot example output.  Installation instructions for installing Jupyterlab can be found at https://jupyter.org/install

Beware that notebooks are tested as part of the test-suite on a one by one basis, so if you add a new notebook and you wnat it to be tested routinely, you need to add it. For certain notebooks the amount of downloading and running needed might not be advisable to have in the overall test suite.

<code>METEOR_single_model_pattern_example.ipynb</code> produces a single pattern using CanESM2 data for the base and co2x2 experiments in PDRMIP, using input data text files stored in the <code>tests/test-data</code> folder

<code>CMIP6_demo_with_residual.ipynb</code> demonstrates how to levarage the inbuilt aerosol forcing from residuals using the cmip6_meteor_data_getter functionality to get data in a minimal way.

<code>CMIP6_mmdemo_with_archiving.ipynb</code> provides demonstration of the cmip6_meteor_data_getter functionality to get data as xarrays directly from the CMIP6 zarrstore to use in pattern making and predictions, archiving inputs to save runtime on multiple runs. Finally these datasets are used to provide multimodel demonstration of the METEOR emulation with both CO2-based green house gas responses, and aerosol response from residuals.

<code>METEOR_paper_figures.ipynb</code> documents the code for Figures 1 and 3--7 of the main manuscript for deriving GHG and aerosol residula patterm and globally aggregated evaluation of METEOR, including supplementary figures showing individual behaviour of the models used as trainign data. It also illustrates reading and downloading (local save) training data, as well as running METEOR with this data and how to vary its inputs.

<code>METEOR_paper_figures_spatial.ipynb</code> documents the code for Figures 8 and 9 of the main manuscript for the spatial evaluation of METEOR. 

<code>METEOR_paper_figures_timescale_eval.ipynb</code> documents the code for the supplementary figures 1--3, illustrating an evaluation of the performance of METEOR under a varying selection of timescales. 

Nota bene: The <code>METEOR_paper_figures_*.ipynb</code> python notebooks are excluded from the test suite and are not tested by the automated tests. Modifications or relying on them for production use is at your own risk, so proceed with caution.

## scripts
This folder contains example scripts and an example notebook, which might not work out of the box, but require data. 

<code>test_install.py</code> is a test installation script that should work

<code>test_cmip6_read_indata.py</code> demonstrates the cmip6_meteor_data_getter functionality only to make plots of cmip6 input data for a comprehensive list of models. Depending on the specs on your laptop you may
run into memory issues for some of the higher resolution models, but otherwise it should work.

<code>make_pattern_4xco2_plots.py</code> is a script demonstrating pattern generation from piControl and abrupt-4xco2 experiments using the cmip6datagetter for inputs, should work anywhere.

<code>make_cmip6_test_plots.py</code> is a similar script, but used for testing, so might not be a good one to look at for a random user.

<code>make_noresm_test.py</code> is a more comprehensive test script which relays on locally available data, which you would need to download if not run on cicero's internal servers.

<code>METEOR_multi_model_pattern_example.ipynb</code> is a notebook which similarly relys on locally available data and which produces patterns and plots for four models (CanESM2, GISS-E2-R, NorESM1 and MIROC5) and three different PDRMIP experiments (base, co2x2 and sulx5) again relying on data locally available where you run. Demonstrating aerosol forcing not from a residual, but from a direct aerosol forcing experiment.

## Development
* To start developing make sure you have a github account and that you are part of the ciceroOslo team.
* If you haven't already, [setup your github account with an ssh key](https://docs.github.com/en/enterprise-server@3.0/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
* Find a suitable place where you want to start your developement either on windows or under /div/nobakcup/users/yourusername/ and change to that directory
* Once in the preferred directory with a terminal do:
```
git clone git@github.com:benmsanderson/meteor.git
```
* To make your own branch (which you really should)
```
git checkout -b your-cool-branch-name
```
* Whenever you log in or want to check stuff
```
git status
```

It will tell you the branch you are on, changes since last etc
* To commit your code changes
```
git add path-of-file-that-changed
```

Repeat this for all the files that you would want to commit the changes for
```
git commit -m "A small message to describe the changes"
```

```
git push
```

(The last one is to push the changes to the github version. The first time you do this on a new branch you will need to set where to push to, but how to do that will be suggested when you just do git push)
* To get new changes that have happened on the main branch is always good before you commit. To do so do:
```
git checkout main
git pull
git checkout your-cool-branch-name
git merge main
```

If all goes well this will fill your terminal with a merge message in your default editor, which is likely vim. The message there is likely ok as it is, so to just use that as a commit message for the merge type: <code>:wq</code> which will just save and quit vim and complete the merge with the original commit message.

Then finally just push your code to the web.
```
git push
```

The last part is just to pushed this new version of your branch again

### Test suite and environment
The code comes with a suite of tests and tools. To use this you must do:
```
make first-venv
make virtual-environment
```

This should only be necessary the first time you setup the code
You can load this environment with
```
source venv/bin/activate
```

Later to update you should do:
`make virtual-environment`

Or if you know you need updates, but aren't getting them:
```
make clean
make virtual-environment
```

After this you should be able to run the automatic tests
`make test` will only run the tests
`make checks` will run the tests and formatting checks
`make format-hecks` will run formatting checks only

Before your code branch can be merged into the main code, it has to pass all the tests
(The makefile also has an option to run only the formatting checks)
Tests are located in tests in tests/test-data/ data for testing against fortran runs and test input data are stored. In tests/unit there are unit tests for certain methods. In test/integration there are integration tests of the code.
When you develop new code, try to think about what can be done to test and validate that your code does what you expect it to do, and try to integrate such tests into the automatic testing scheme.

## General code flow
`meteor.py` contains the main control module, `prpatt.py` is a library with functions to fit parameters to define synthetic PC timeseries to processes PDRMIP output, as well as functions to convolve synthetic PCAs with user-defined forcing timeseries. `scm_forcer_engine.py` provides a wrapper to enable using the ciceroscm simple climate to estimate forcing strength from emissions. `meteor_plot_utils.py` provides some simple plotting functionality. `cmip6_meteor_data_getter.py` has functions to define and access CMIP6 data as xarrays from cloud storage, meaning you can use cmip6 data with meteor with out going through a huge download and data wrangling process before going ahead. This can also be leveraged on it's own to get cmip6 data to plot and explore.
