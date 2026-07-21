
Steps to produce the FastMIP Phase 2 Tier 1 METEOR contributions.
April 2026 by Maura Dewey (maura.dewey@cicero.oslo.no)

	1. Train METEOR for all the models and scenarios. This is done with the METEOR venv (for example, within a tmux session on nac)
		a. cd to METEOR folder and activate venv with: "source venv/bin/activate"
		b. cd to scripts folder and run: "python FastMIP_phase2_trainMETEOR.py > output.txt 2>&1" - The list of ESMs that are emulated, the scenarios that are run, and the choices for subsetting the FAIR ensemble are made in the top of this python script. (NOTE: If you need to change the training settings, ie. which scenario or ESM, it needs to be done in this script. Possible update/to-do would be to put that in a yml file.)
		c. Disconnect from the tmux session and let it run. This will train METEOR and create all the raw METEOR output for the given list of ESMs and scenarios (takes ~10 hrs for Tier 1, if also training new METEOR instances)

	2. Create FastMIP specific output files. This is done with a conda env fastmip_env
		a. setup modules, deactivate METEOR venv, and activate conda environment with: "FastMIP_env_setup.sh"
		b. cd to scripts folder and create FastMIP specific output files with: "python FastMIP_phase2_makeoutput.py"
		c. disconnect from the tmux session and let it run (takes a couple hours). 
		d. Once the output is done, update netCDF attributes with: "python FastMIP_phase2_attrs.py"

	3. Create plots 
		The notebook FastMIP_phase2_plots.ipynb will create all the plots from the Fastmip phase2 example notebook (https://github.com/sarasita/fastMIP/blob/main/requested_output/tour_through_requested_output.ipynb) with METEOR results. 


Files on fastmip branch:

./FastMIP_README.md 
./conda_env_fastmip.yml (conda environment file for processing output, needed to use for python-cdo for regridding)
./FastMIP_env_setup.sh (to clear modules and activate conda environment on nac server)

./scripts/FastMIP_phase2_trainMETEOR.py (trains METEOR for all Tier 1 models and creates 200 member ensembles - 20 FAIR ensemble members x 10 METEOR noise model realizations. The subselection of FAIR members used is saved in a pickle in ./data/FASTMIP_phase2/FAIR_data)
./scripts/FastMIP_phase2_makeoutput.py (makes requested FastMIP output files - subset 10 member ensemble and bulk statistics across full 200.)
./scripts/FastMIP_phase2_attrs.py (updates netCDF attributes to FastMIP protocol)

./notebooks/FastMIP_phase2_plots.py (makes all plots as in https://github.com/sarasita/fastMIP/blob/main/requested_output/tour_through_requested_output.ipynb)


Data files are not tracked with git, but the following structure is required and/or created with above scripts:

./data/FASTMIP_phase2/FAIR_data (contains full FAIR ensemble files (climate_assessment_forced.csv), grid file for regridding (g025.txt), and the pickles of the FAIR subset.)
./data/FASTMIP_phase2/scenario_data (contains the emission and concentration data files needed for the SCM)
./data/FASTMIP_phase2/METEOR_emulations/raw (where METEOR output is saved initially)
./data/FastMIP_phase2/METEOR_emulations/aggregated (where regridded and combined tmp files are saved)
./data/FASTMIP_phase2/METEOR_emulations/processed (where FastMIP output is saved)

Initial emulations where done for 4 scenarios (L, M, H, VL)
The full CMIP7 scenario list and short_name markers are:
scenarios_list = ['SSP1 - Very Low Emissions', 'SSP2 - Low Emissions', 'SSP2 - Medium-Low Emissions', 'SSP2 - Medium Emissions', 'SSP3 - High Emissions', 'SSP5 - Medium-Low Emissions_a', 'SSP2 - Low Overshoot_a']
scenarios_short = ['VL','L','ML','M','H','HL','LN']