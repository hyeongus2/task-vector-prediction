#!/bin/bash

# [1] Activate virtual environment (if .venv exists)
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment: .venv"
else
    echo "No .venv found. Make sure your environment is ready."
fi

# [2] Experiment name and config path
EXP_NAME="vit_cifar10_sgd"
CONFIG_PATH="configs/vit_tiny_cifar10_sgd.yaml"

# [3] Running
echo "Running experiment: $EXP_NAME"
python train.py --config $CONFIG_PATH

# [4] Finished
echo "Experiment $EXP_NAME finished."
