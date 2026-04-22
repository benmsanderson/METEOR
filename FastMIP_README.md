
Steps to produce the FastMIP Phase 2 METEOR contributions.
April 2026 by Maura Dewey (maura.dewey@cicero.oslo.no)

	1. Run METEOR for all the scenarios. This is done with the METEOR venv:
		(for example, within a tmux session on nac)
		a. cd to METEOR folder and activate venv with: "source venv/bin/activate"
		b. cd to scripts folder and run: "python FastMIP_phase2_trainMETEOR.py > output.txt 2>&1"
		c. then you can disconnect from the tmux session and let it run. This will create all the raw METEOR output for FastMIP (takes ~10 hrs if also training new METEOR instances)

	2. Create FastMIP specific output files. This is done with a conda env fastmip_env (can be in the same tmux session)
		a. deactivate venv
		b. might need to make sure conda/bash also works in the tmux session with: "source ~/.bashrc"
		c. setup modules and conda environment with: ". FastMIP_env_setup.sh"
		d. cd to scripts folder and create FastMIP specific output files with: "python FastMIP_phase2_makeoutput.py"
		e. you can disconnect from the tmux session and let it run (takes a couple hours). 

	3. Create plots 
		The notebook FastMIP_phase2_plots.ipynb will create all the plots from the Fastmip phase2 example notebook (https://github.com/sarasita/fastMIP/blob/main/requested_output/tour_through_requested_output.ipynb) with METEOR results. 

Files on fastmip branch:

./FastMIP_README.md 
./conda_env_fastmip.yml (conda environment file for processing output, needed to use for python-cdo for regridding)
./FastMIP_env_setup.sh (to clear modules and activate conda environment on nac server)

./scripts/FastMIP_phase2_trainMETEOR.py (trains METEOR for all Tier 1 models and creates 200 member ensembles - 20 FAIR ensemble members x 10 METEOR noise model realizations. The subselection of Fair members used is saved in a pickle in ./data/FASTMIP_phase2/FAIR_data)
./scripts/FastMIP_phase2_makeoutput.py (makes requested FastMIP output files - subset 10 member ensemble and bulk statistics across full 200.)

./notebooks/FastMIP_phase2_plots.py (makes all plots as in https://github.com/sarasita/fastMIP/blob/main/requested_output/tour_through_requested_output.ipynb)


Data files are not tracked with git, but the following structure is required/created with above scripts:

./data/FASTMIP_phase2/FAIR_data (contains full FAIR ensemble files (climate_assessment_forced.csv), grid file for regridding (g025.txt), and the pickles of the FAIR subset.)
./data/FASTMIP_phase2/scenario_data (contains the emission and concentration data files needed for the SCM)
./data/FASTMIP_phase2/METEOR_emulations/raw (where METEOR output is saved initially)
./data/FastMIP_phase2/METEOR_emulations/aggregated (where regridded and combined tmp files are saved)
./data/FASTMIP_phase2/METEOR_emulations/processed (where FastMIP output is saved)

Initial emulations where done for 4 scenarios (L, M, H, VL)
The full scenario list and short_name markers are:
scenarios_list = ['SSP1 - Very Low Emissions', 'SSP2 - Low Emissions', 'SSP2 - Medium-Low Emissions', 'SSP2 - Medium Emissions', 'SSP3 - High Emissions', 'SSP5 - Medium-Low Emissions_a', 'SSP2 - Low Overshoot_a']
scenarios_short = ['VL','L','ML','M','H','HL','LN']