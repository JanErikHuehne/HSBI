SECONDS=0
# Check if at least one parameter is provided (the working directory)
if [ $# -lt 1 ]; then
    echo "Usage: $0 working_directory [plasticity parameters ...]"
    exit 1
fi

working_directory="$1"
shift 

# Check if the working directory is a valid directory
if [ ! -d "$working_directory" ]; then
    echo "Error: Provided working directory '$working_directory' does not exist or is not a directory."
    exit 1
fi
python_script="analysis/network_analysis.py"
python "$python_script" --working_dir "$working_directory" "$@"
