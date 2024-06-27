#!/bin/bash
# path to the github repository main folder
export GIT_MAIN_PATH="/home/ge84yes/HSBI"
export SIMULATIONS="/home/ge84yes/HSBI/analysis/run_analysis.sh"
# name of the conda environment 
export ENV_NAME="master_thesis"
# the working directory for data storage 
export WOR_DIR="/home/ge84yes/data/analysis"
export PARAMETERS_FILE="/home/ge84yes/master_thesis/HSBI/analysis/analysis_set.txt"
# name of server list file
SERVERS="servers.txt"
#export SBI_METRICS=("rate_e" "rate_i" "cv_isi" "f_w-blow" "w_creep" "wmean_ee" "wmean_ie" "mean_fano_t" "mean_fano_s" "auto_cov" "std_fr" "std_rate_spatial")

SIMULATION_RELATIVE_PATH="/home/ge84yes/master_thesis/HSBI/analysis/analysis_parallel.py"
logfile="/home/ge84yes/data/analysis.log"
rm "$logfile"
touch "$logfile"

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
if conda list | grep -q 'scipy'; then \
echo \"\$(hostname)(\$(date +%T)): scipy package already installed in $ENV_NAME. Skipping installation ...\"; \
else conda install scipy -y > /dev/null && echo \"\$(hostname)(\$(date +%T)): successfully installed scipy in $ENV_NAME environment\"; fi; \
if conda list | grep -q 'h5py'; then \
echo \"\$(hostname)(\$(date +%T)): h5py package already installed in $ENV_NAME. Skipping installation ...\"; \
else conda install anaconda::h5py -y > /dev/null && echo \"\$(hostname)(\$(date +%T)): successfully installed h5py in $ENV_NAME environment\"; fi; \
if conda list | grep -q 'tqdm'; then \
echo \"\$(hostname)(\$(date +%T)): tqdm package already installed in $ENV_NAME. Skipping installation ...\"; \
else conda install tqdm -y > /dev/null && echo \"\$(hostname)(\$(date +%T)): successfully installed tqdm in $ENV_NAME environment\"; fi; \
cd $GIT_MAIN_PATH; \
if [ \$(pwd) == \"$GIT_MAIN_PATH\" ]; then \
git fetch --all > /dev/null && git reset --hard origin/main > /dev/null && chmod +x run_simulations.sh && echo \"\$(hostname)(\$(date +%T)): successfully reset git repository in $GIT_MAIN_PATH to origin/main\"; \
else echo 'Failed to change directory to $GIT_MAIN_PATH \"$(pwd)\"' ; fi;
chmod +x analysis/run_analysis.sh"

# Setting up the conda environment on each listed server
parallel --verbose -u --sshloginfile ${SERVERS} --nonall "$init_server"  >> "$logfile" 2>&1
echo "MAIN $(hostname)($(date +%T)) - INFO - SETUP SUCCESSFUL" >> "$logfile" 2>&1

eval "$(conda shell.bash hook)"
conda activate thesis
python create_parameters.py  >> "$logfile" 2>&1
parallel -u  --sshloginfile ${SERVERS} "cd $GIT_MAIN_PATH;source /home/ge84yes/miniconda3/etc/profile.d/conda.sh; conda activate master_thesis;$SIMULATIONS $WOR_DIR {}" :::: $PARAMETERS_FILE   >> "$logfile" 2>&1