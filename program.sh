#!/bin/bash
# path to the github repository main folder
export GIT_MAIN_PATH="/home/ge84yes/HSBI"
# name of the conda environment 
export ENV_NAME="master_thesis"
export WOR_DIR="test"
# name of server list file
SERVERS="servers.txt"

SIMULATION_RELATIVE_PATH="run_simulations.sh"


init_server="echo \"\$(hostname)(\$(date +%T)): initializing conda environment $ENV_NAME\"; \
source /home/ge84yes/miniconda3/etc/profile.d/conda.sh; \
if conda info --envs | grep -q $ENV_NAME; then \
echo \"\$(hostname)(\$(date +%T)): conda environment $ENV_NAME already exists on. Skipping ...\"; \
else conda create --name $ENV_NAME python=3.8 -y > /dev/null && echo \"\$(hostname)(\$(date +%T)): successfully created $ENV_NAME environment\"; fi; \
conda activate master_thesis; \
if [[ \"\$CONDA_DEFAULT_ENV\" == \"$ENV_NAME\" ]]; then echo \"\$(hostname)(\$(date +%T)): successfully activated $ENV_NAME environment\"; \
else echo \"\$(hostname)(\$(date +%T)): Failed to activate $ENV_NAME environment \"; fi; \
if conda list | grep -q 'brian2'; then \
echo \"\$(hostname)(\$(date +%T)): brian2 package already installed in $ENV_NAME. Skipping installation ...\"; \
else conda install -c conda-forge brian2 -y > /dev/null && echo \"\$(hostname)(\$(date +%T)): successfully installed brian2 in $ENV_NAME environment\"; fi; \
cd $GIT_MAIN_PATH; \
if [ \$(pwd) == \"$GIT_MAIN_PATH\" ]; then \
git fetch --all > /dev/null && git reset --hard origin/main > /dev/null && chmod +x run_simulations.sh && echo \"\$(hostname)(\$(date +%T)): successfully reset git repository in $GIT_MAIN_PATH to origin/main\"; \
else echo 'Failed to change directory to $GIT_MAIN_PATH \"$(pwd)\"' ; fi;"

# Setting up the conda environment on each listed server
parallel --verbose -u --sshloginfile ${SERVERS} --nonall "$init_server"


# first we need to sample parmeters that are stored in PARAMETERS_FILE
# File containing the simulation parameters
eval "$(conda shell.bash hook)"
conda activate thesis
python parallel_python/hsbi.py
echo "MAIN $(hostname)($(date +%T)): generating simulation parameters..."
PARAMETERS_FILE="parameters.txt"
# Use GNU Parallel to execute the simulations
parallel -u --sshloginfile ${SERVERS} -j 16 --nonall "cd $GIT_MAIN_PATH;source /home/ge84yes/miniconda3/etc/profile.d/conda.sh; conda activate master_thesis; python3 bash/run_simulation.py $WOR_DIR {}" :::: $PARAMETERS_FILE
# We collect the simulation results
# after running the simulation we want to collect the results from the result dir 
 

