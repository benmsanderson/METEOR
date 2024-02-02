### METEOR
Multivariate Emulation of Time-Emergent fOrced Response


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
```
## Jupyter Notebooks
The notebooks folder provides simple working examples to run the model within a Jupyter environment, and plot example output.  Installation instructions for installing Jupyterlab can be found at https://jupyter.org/install
<code>METEOR_single_model_pattern_example.ipynb</code> produces a single pattern using CanESM2 data for the base and co2x2 experiments in PDRMIP, using input data text files stored in the <code>tests/test-data</code> folder

## scripts
This folder contains example scripts and an example notebook, which might not work out of the box, but require data. 
test_install.py is a test installation script that should work
make_noresm_test.py is a more comprehensive test script which relays on locally available data, which you would need to download if not run on cicero's internal servers.
`METEOR_multi_model_pattern_example.ipynb`is a notebook which similarly relays on locally available data and which produces patterns and plots for four models (CanESM2, GISS-E2-R, NorESM1 and MIROC5) and three different PDRMIP experiments (base, co2x2 and sulx5).

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

(The last one is to push the changes to the github version. The first time youi do this on a new branch you will need to set where to push to, but how to do that will be suggested when you just do git push)
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
`make checks` will run the tests and formatting tests

Before your code branch can be merged into the main code, it has to pass all the tests
(The makefile also has an option to run only the formatting checks)
Tests are located in tests in tests/test-data/ data for testing against fortran runs and test input data are stored. In tests/unit there are unit tests for certain methods. In test/integration there are integration tests of the code.
When you develop new code, try to think about what can be done to test and validate that your code does what you expect it to do, and try to integrate such tests into the automatic testing scheme.

## General code flow
`meteor.py` contains the main control module, prpatt is a library with functions to fit parameters to define synthetic PC timeseries to processes PDRMIP output, as well as functions to run convolve synthetic PCA with a user-defined forcing timeseries. `scm_forcer_engine` provides a wrapper to enable using the ciceroscm simple climate to estimate forcing strength from emissions.
