SECONDS=0
MEMORY_LIMIT=2000
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
python_script="parallel_python/run_simulation.py" 
python "$python_script" --working_dir "$working_directory" "$@" &
PID=$!

while true; do
    # Check if the process is still running
    if ! kill -0 $PID 2>/dev/null; then
        echo "Process $PID has terminated"
        break
    fi

    # Get the RSS (Resident Set Size) memory usage of the process in KB
    MEM_USAGE=$(ps -o rss= -p $PID)

    if [ -z "$MEM_USAGE" ]; then
        echo "Could not get memory usage for PID $PID"
        sleep 1
        continue
    fi

    # Convert memory usage to MB
    MEM_USAGE_MB=$((MEM_USAGE / 1024))

    echo "Memory usage of process (PID $PID): $MEM_USAGE_MB MB"

    # Check if memory usage exceeds the limit
    if [ $MEM_USAGE_MB -gt $MEMORY_LIMIT ]; then
        echo "Memory usage of process (PID $PID) exceeds $MEMORY_LIMIT MB, terminating process"
        kill -9 $PID
        break
    fi

    # Sleep for a while before checking again
    sleep 1
done
