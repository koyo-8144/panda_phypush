#!/bin/bash

# --- 0. EXPERIMENT SETTINGS ---
# OBJECT="nolid_cube_1.856" # "wooden_cube_0.348" # "blue_cylinder_0.175" #"colored_cubes_0.989"
# SURFACE="wood_smooth" # "wood_rough"  # "green_rub"
OBJECT="colored_cubes_video" # "wooden_cube_0.348" # "blue_cylinder_0.175" #"colored_cubes_0.989"
SURFACE="wood_smooth" # "wood_rough"  # "green_rub"
NUM_RUNS=1
SCRIPT_NAME="collect_model_input.py"

# --- 1. PROMPT FOR GROUND TRUTH ---
echo "=========================================================="
echo " DATA COLLECTION SETUP"
echo "=========================================================="
echo "Object:  $OBJECT"
echo "Surface: $SURFACE"
echo "----------------------------------------------------------"
read -p "Enter Ground Truth Mass (M_GT) in kg (or type 'None'): " M_GT
read -p "Enter Ground Truth Friction (MU_GT) (or type 'None'): " MU_GT
echo "=========================================================="

echo "Starting automated data collection: $NUM_RUNS runs..."

for i in $(seq 1 $NUM_RUNS)
do
    echo "----------------------------------------------------------"
    echo "  STARTING RUN $i OF $NUM_RUNS"
    echo "----------------------------------------------------------"
    
    # Pass all variables to Python
    python $SCRIPT_NAME \
        --run_id $i \
        --m_gt "$M_GT" \
        --mu_gt "$MU_GT" \
        --object "$OBJECT" \
        --surface "$SURFACE"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ Error detected in run $i (Exit code: $EXIT_CODE)."
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