#!/bin/bash

# Path to the setup script (if needed)
SETUP="/home/ge84yes/master_thesis/HSBI/simulation_env/setup_conda_env.sh"

# File containing the list of servers
SERVERS="servers.txt"


# Setting up the conda environment on each listed server
parallel -u --sshloginfile ${SERVERS} --nonall ' \
ENV_NAME="master_thesis"; \
echo "- Initializing conda environment $ENV_NAME on $(hostname)"; \
source /home/ge84yes/miniconda3/etc/profile.d/conda.sh; \
if conda info --envs | grep -q $ENV_NAME; then \
    echo "-- Conda environment $ENV_NAME already exists on $(hostname). Skipping ..."; \
else \
    conda create --name $ENV_NAME python=3.8 -y && echo "-- Successfully created $ENV_NAME environment on $(hostname)."; \
fi; \
conda install -c conda-forge brian2 && echo "-- Successfully set up $ENV_NAME environment on $(hostname).";
 \
'

