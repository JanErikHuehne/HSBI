#!/bin/bash
# path to the github repository main folder
export GIT_MAIN_PATH="/home/ge84yes/HSBI"
export SIMULATIONS="/home/ge84yes/HSBI/bash/run_simulation.sh"
# name of the conda environment 
export ENV_NAME="master_thesis"
# the working directory for data storage 
export WOR_DIR="/home/ge84yes/data"
export PARAMETERS_FILE="/home/ge84yes/data/parameters.txt"
# name of server list file
SERVERS="servers.txt"
#export SBI_METRICS=("rate_e" "rate_i" "cv_isi" "f_w-blow" "w_creep" "wmean_ee" "wmean_ie" "mean_fano_t" "mean_fano_s" "auto_cov" "std_fr" "std_rate_spatial")
export SBI_METRICS=("f_w-blow" "end")
SBI_AMETRICS=("rate_e" "rate_i")
SIMULATION_RELATIVE_PATH="run_simulations.sh"
logfile="/home/ge84yes/data/run.log"
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
cd $GIT_MAIN_PATH; \
if [ \$(pwd) == \"$GIT_MAIN_PATH\" ]; then \
git fetch --all > /dev/null && git reset --hard origin/main > /dev/null && chmod +x run_simulations.sh && echo \"\$(hostname)(\$(date +%T)): successfully reset git repository in $GIT_MAIN_PATH to origin/main\"; \
else echo 'Failed to change directory to $GIT_MAIN_PATH \"$(pwd)\"' ; fi;
chmod +x bash/run_simulation.sh;
chmod +x bash/run_analysis.sh"

# Setting up the conda environment on each listed server
parallel --verbose -u --sshloginfile ${SERVERS} --nonall "$init_server"  >> "$logfile" 2>&1
echo "MAIN $(hostname)($(date +%T)) - INFO - SETUP SUCCESSFUL" >> "$logfile" 2>&1

# first we need to sample parmeters that are stored in PARAMETERS_FILE
# File containing the simulation parameters
eval "$(conda shell.bash hook)"
conda activate thesis
 python parallel_python/collect_simulations.py --working_dir "$WOR_DIR" --metrics "$applied_metrics_str"  >> "$logfile" 2>&1
for metric in "${SBI_METRICS[@]}"; do
    echo "MAIN $(hostname)($(date +%T)) - INFO - NEW ROUND ${SBI_AMETRICS[*]}"  >> "$logfile" 2>&1
    echo "MAIN $(hostname)($(date +%T)) - INFO - generating simulation parameters..."  >> "$logfile" 2>&1
    applied_metrics_str=$(IFS=' '; echo "${SBI_AMETRICS[*]}"); 
    exit_code=1
    while [ "$exit_code" -ne 0 ]; do
        python parallel_python/hsbi.py --train --working_dir "$WOR_DIR" "$applied_metrics_str"  >> "$logfile" 2>&1
        python parallel_python/hsbi.py --sample --working_dir "$WOR_DIR" "$applied_metrics_str"  >> "$logfile" 2>&1
        echo "MAIN $(hostname)($(date +%T)) - INFO - Triggering simulations ..."  >> "$logfile" 2>&1
        parallel -u  --sshloginfile ${SERVERS} "cd $GIT_MAIN_PATH;source /home/ge84yes/miniconda3/etc/profile.d/conda.sh; conda activate master_thesis;nice -n 19 $SIMULATIONS $WOR_DIR {}" :::: $PARAMETERS_FILE   >> "$logfile" 2>&1
        echo "MAIN $(hostname)($(date +%T)) - INFO - Collecting simulations ..."  >> "$logfile" 2>&1
        python parallel_python/collect_simulations.py --working_dir "$WOR_DIR" --metrics "$applied_metrics_str"  >> "$logfile" 2>&1
        exit_code=$?
        if [ "$exit_code" -ne 0 ]; then
            echo "MAIN $(hostname)($(date +%T)) - INFO - Next Round with Same Metrics ..."  >> "$logfile" 2>&1
        else 
            echo "MAIN $(hostname)($(date +%T)) - INFO - Next Round with Additional Metrics ..."  >> "$logfile" 2>&1
        fi
    done
    SBI_AMETRICS+=("$metric")
done
echo "MAIN $(hostname)($(date +%T)) - INFO - All metric performed "  >> "$logfile" 2>&1


 

