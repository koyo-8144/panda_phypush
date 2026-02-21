#!/bin/bash

# Define how many times you want to run the script
NUM_RUNS=10
SCRIPT_NAME="push_cube_inference.py"

echo "Starting automated data collection: $NUM_RUNS runs of $SCRIPT_NAME"

for i in $(seq 1 $NUM_RUNS)
do
    echo "=========================================================="
    echo "  STARTING RUN $i OF $NUM_RUNS"
    echo "=========================================================="
    
    # Run the Python script
    python $SCRIPT_NAME
    
    # Capture the exit code of the python script
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ Error detected in run $i (Exit code: $EXIT_CODE)."
        # Optional: Uncomment the next line if you want the loop to stop completely on an error
        # exit $EXIT_CODE
    else
        echo "✅ Run $i completed successfully."
    fi

    if [ $i -lt $NUM_RUNS ]; then
        echo "⏳ Waiting 3 seconds for robot controller to fully reset..."
        sleep 3
    fi
done

echo "=========================================================="
echo " ALL $NUM_RUNS RUNS COMPLETED!"
echo "=========================================================="