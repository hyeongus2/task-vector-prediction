#!/bin/bash
# scripts/train_vitb32_eurosat_adamw_lora.sh
# This is a template script for running a training experiment.
# For a new experiment, COPY this file, rename it, and edit the variables below.

# --- EDIT THESE VARIABLES FOR EACH EXPERIMENT ---

# 1. Set the path to the configuration file you want to use.
CONFIG="configs/vitb32_eurosat_adamw_lora.yaml"

# 2. (Optional) Set any command-line overrides.
#    Leave empty '()' to run with the config file's values as is.
OVERRIDES=(
    # --- Example Overrides (uncomment and edit as needed) ---
    # "finetuning.lr=0.01"
    # "finetuning.epochs=5"
    # "data.batch_size=128"
    # "seed=123"
)

# 3. (Optional) Set the wandb run ID to resume a stopped experiment.
#    Leave empty "" to start a new run.
RESUME_ID="" # Example: "3a1b2c3d"

# ------------------------------------------------


# --- Do not edit below this line ---
set -e
echo "--- Starting Experiment ---"
echo "Config: ${CONFIG}"

# Build the base command
CMD="python train.py --config ${CONFIG}"

# Add overrides if they exist
if [ ${#OVERRIDES[@]} -gt 0 ]; then
    echo "Overrides: ${OVERRIDES[*]}"
    CMD+=" --set ${OVERRIDES[@]}"
fi

# Add resume ID if it exists
if [ -n "$RESUME_ID" ]; then
    echo "Attempting to resume from wandb run ID: ${RESUME_ID}"
    CMD+=" --resume_id ${RESUME_ID}"
fi

# Execute the command
echo "Running command: ${CMD}"
eval ${CMD}