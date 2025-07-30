#!/bin/bash

# [1] Activate virtual environment (if .venv exists)
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment: .venv"
else
    echo "No .venv found. Make sure your environment is ready."
fi

# [2] Experiment name and config path
CONFIG_PATH="configs/vit_tiny_cifar10_full_sgd.yaml"
TASK="train.py"
# TASK="analyze.py"
OVERRIDES=""

# [3] Running
echo "Running experiment..."
python $TASK --config $CONFIG_PATH --overrides $OVERRIDES

# [4] Finished
echo "Experiment finished."
