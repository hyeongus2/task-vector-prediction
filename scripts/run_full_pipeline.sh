#!/bin/bash
# scripts/run_full_pipeline.sh
# This script runs the full end-to-end experiment pipeline:
# 1. Train the model.
# 2. Run prediction on the saved task vectors.
# 3. Run analysis to compare results.

# Exit immediately if any command fails, making the script safer.
set -e

# --- Configuration ---
# Define the config file to use for this pipeline run.
CONFIG_FILE="configs/vitb32_cifar10_sgd_lora.yaml"


echo "--- 1. Starting Training ---"
# Run the training script. The '| tee' command shows the output on the console
# AND saves it to a temporary log file for parsing.
python train.py --config ${CONFIG_FILE} | tee .train_output.log

# Extract the unique output directory path from the log file.
# It finds the line with 'Output directory:', then takes the last word on that line.
EXP_DIR=$(grep 'Output directory:' .train_output.log | awk '{print $NF}')

# Safety check: Ensure the directory path was actually found.
if [ -z "$EXP_DIR" ]; then
    echo "Error: Could not find 'Output directory:' in the training log. Aborting."
    exit 1
fi

echo "Training finished. Results saved to: ${EXP_DIR}"
echo "----------------------------------------------------"


echo "--- 2. Starting Prediction ---"
# Run the prediction script, passing the experiment directory we just found.
python predict.py --exp_dir ${EXP_DIR}
echo "Prediction finished."
echo "----------------------------------------------------"


echo "--- 3. Starting Analysis ---"
# Run the analysis script, also passing the same experiment directory.
python analyze.py --exp_dir ${EXP_DIR}
echo "Analysis finished."
echo "----------------------------------------------------"


# Clean up the temporary log file.
rm .train_output.log

echo "Full pipeline completed successfully!"