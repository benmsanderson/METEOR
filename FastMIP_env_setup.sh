#!/bin/bash

# load modules and conda environment to run FastMIP_phase2_makeoutput.py, making the FastMIP phase 2 output files for METEOR. 
# The same conda environment can be used as a kernel for FastMIP_phase2_plots.ipynb.
# Note: this will deactivate the METEOR venv!
# module versions as of 2026-04
# the environment can be created from conda_env_fastmip.yml 

module purge
module load Anaconda3/2023.09-0
source deactivate
conda deactivate
conda activate /div/no-backup-nac/users/maurad/METEOR/fastmip_env/