export PARAMETERS_FILE="/home/ge84yes/data/run_1/eval_parameters.txt"
export SERVERS="servers.txt"
export logfile="/home/ge84yes/data/run_1/eval.log"
export SIMULATIONS="/home/ge84yes/HSBI/bash/eval.sh"
export GIT_MAIN_PATH="/home/ge84yes/HSBI"
export WOR_DIR="/home/ge84yes/data/run_1"
export ENV_NAME="master_thesis"

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
if conda list | grep -q 'brian2tools'; then \
echo \"\$(hostname)(\$(date +%T)): brian2tools package already installed in $ENV_NAME. Skipping installation ...\"; \
else conda install brian2tools -y > /dev/null && echo \"\$(hostname)(\$(date +%T)): successfully installed scipy in $ENV_NAME environment\"; fi; \
if conda list | grep -q 'h5py'; then \
echo \"\$(hostname)(\$(date +%T)): h5py package already installed in $ENV_NAME. Skipping installation ...\"; \
else conda install anaconda::h5py -y > /dev/null && echo \"\$(hostname)(\$(date +%T)): successfully installed h5py in $ENV_NAME environment\"; fi; \
cd $GIT_MAIN_PATH; \
if [ \$(pwd) == \"$GIT_MAIN_PATH\" ]; then \
git fetch --all > /dev/null && git reset --hard origin/main > /dev/null && chmod +x run_simulations.sh && echo \"\$(hostname)(\$(date +%T)): successfully reset git repository in $GIT_MAIN_PATH to origin/main\"; \
else echo 'Failed to change directory to $GIT_MAIN_PATH \"$(pwd)\"' ; fi;
chmod +x bash/run_simulation.sh"

# Setting up the conda environment on each listed server
parallel --verbose -u --sshloginfile ${SERVERS} --nonall "$init_server"  >> "$logfile" 2>&1
echo "MAIN $(hostname)($(date +%T)) - INFO - SETUP SUCCESSFUL" >> "$logfile" 2>&1
parallel -u  --sshloginfile ${SERVERS} "cd $GIT_MAIN_PATH;source /home/ge84yes/miniconda3/etc/profile.d/conda.sh; conda activate master_thesis;$SIMULATIONS $WOR_DIR {}" :::: $PARAMETERS_FILE   >> "$logfile" 2>&1