#!/bin/bash

# load modules and conda environment to run FastMIP_phase2_makeoutput.py, the FastMIP phase 2 output files for METEOR. The same conda environment can be used as a kernel for FastMIP_phase2_plots.ipynb.
# Note: this will deactivate the METEOR venv!
# module versions as of 2026-04

module purge
module load Anaconda3/2023.09-0
source deactivate
conda deactivate
conda activate /div/no-backup-nac/users/maurad/METEOR/fastmip_env/