#!/bin/bash
# scripts/analyze_vitb32_eurosat_sgd_lora.sh
# This is a template script for running an analysis job.
# For a new analysis, COPY this file, rename it, and edit the variables below.

# --- EDIT THESE VARIABLES FOR EACH ANALYSIS ---

# 1. Set the path to the experiment output directory you want to analyze.
#    This is the folder with the unique wandb ID.
#    Example: "outputs/vitb32_eurosat_sgd_lora/kth4zbgz"
EXP_DIR="outputs/vitb32_eurosat_sgd_lora/pifedpnb"

# 2. Set the prediction space. This is a REQUIRED choice.
#    - 'adapter': Predict the trajectory of LoRA A & B matrices directly. (Recommended)
#    - 'operational': Predict the trajectory of the effective B@A product.
PREDICTION_SPACE="adapter"

# 3. Set the parameters for the trajectory prediction model.
K=3 # The number of exponential terms to use.
N=6 # The number of early tau vectors to use for fitting.

# ------------------------------------------------


# --- Do not edit below this line ---
set -e # Exit immediately if a command exits with a non-zero status.

# Safety checks
if [[ -z "$EXP_DIR" || "$EXP_DIR" == *"YOUR_RUN_ID_HERE"* ]]; then
    echo "Error: Please set the EXP_DIR variable in this script."
    exit 1
fi
if [[ -z "$PREDICTION_SPACE" ]]; then
    echo "Error: Please set the PREDICTION_SPACE variable ('adapter' or 'operational')."
    exit 1
fi

echo "--- Starting Analysis ---"
echo "Experiment Directory: ${EXP_DIR}"
echo "Prediction Space:     ${PREDICTION_SPACE}"
echo "Fitting with k=${K} and N=${N}"

# Build and execute the command
python analyze.py \
    --exp_dir "${EXP_DIR}" \
    --prediction_space "${PREDICTION_SPACE}" \
    --k ${K} \
    --N ${N}

echo "Analysis script finished."