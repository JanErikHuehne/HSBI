#!/bin/bash
# run_simulations.sh
param1="$1"
param2="$2"
param3="$3"

echo "Running simulation with parameters: $param1 $param2 $param3"

parallel --verbose -u --sshloginfile ${SERVERS} --nonall ' \
GIT_MAIN_PATH="/home/ge84yes/HSBI"; \
ENV_NAME="master_thesis"; \
echo "- Initializing conda environment $ENV_NAME on $(hostname)"; \
source /home/ge84yes/miniconda3/etc/profile.d/conda.sh; \
if conda info --envs | grep -q $ENV_NAME; then \
    echo "-- Conda environment $ENV_NAME already exists on $(hostname). Skipping ..."; \
else \
    conda create --name $ENV_NAME python=3.8 -y && echo "-- Successfully created $ENV_NAME environment on $(hostname)."; \
fi; \
conda activate master_thesis;
# conda install -c -y conda-forge brian2 && echo "-- Successfully set up $ENV_NAME environment on $(hostname).";
cd $GIT_MAIN_PATH; \
if [ "$(pwd)" == "$GIT_MAIN_PATH" ]; then \
    git reset --hard origin/main &&  chmod +x run_simulations.sh && echo "-- Successfully reset git repository in $GIT_MAIN_PATH to origin/main on $(hostname)"; \
else \
    echo "Failed to change directory to $GIT_MAIN_PATH"; \
fi;'