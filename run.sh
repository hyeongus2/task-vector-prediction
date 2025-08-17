#!/usr/bin/env bash

# Usage:
#   bash run.sh --mode foreground
#   bash run.sh --mode background
MODE="background"  # default
if [[ "$1" == "--mode" && -n "$2" ]]; then
    MODE="$2"
fi

# [1] Activate virtual environment (if .venv exists)
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment: .venv"
else
    echo "No .venv found. Make sure your environment is ready."
    exit 1
fi

# [2] Create logs directory if not exists
mkdir -p logs

# [3] Experiment configs
CONFIGS=(
    "configs/vit_tiny_cifar10_full_sgd.yaml"
    "configs/vit_tiny_cifar10_full_adam.yaml"
    "configs/vit_tiny_mnist_full_sgd.yaml"
    "configs/vit_tiny_mnist_full_adam.yaml"
    "configs/resnet18_cifar10_full_sgd.yaml"
    "configs/resnet18_cifar10_full_adam.yaml"
    "configs/resnet18_mnist_full_sgd.yaml"
    "configs/resnet18_mnist_full_adam.yaml"
)
TRAIN="train.py"
ANALYZE="analyze.py"
OVERRIDES=""  # e.g., "train.lr=0.01 train.batch_size=64"

MAX_PARALLEL=2
CURRENT_PARALLEL=0

# [4] Launch training
for CONFIG_PATH in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
    LOG_FILE="logs/${CONFIG_NAME}.log"

    echo ""
    echo "==============================="
    echo "Running: $CONFIG_NAME"
    echo "Log file: $LOG_FILE"
    echo "==============================="

    if [ -z "$OVERRIDES" ]; then
        CMD="python $TRAIN --config \"$CONFIG_PATH\""
    else
        CMD="python $TRAIN --config \"$CONFIG_PATH\" --overrides \"$OVERRIDES\""
    fi

    if [[ "$MODE" == "background" ]]; then
        nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    else
        nohup bash -c "$CMD" > "$LOG_FILE" 2>&1
    fi
    echo "Started in ${MODE^^}"

    ((CURRENT_PARALLEL++))
    if (( CURRENT_PARALLEL >= MAX_PARALLEL )); then
        wait
        CURRENT_PARALLEL=0
    fi
done

# [5] Wait for remaining jobs
wait
echo ""
echo "All training experiments completed."

# [6] Run analyze.py for each config
for CONFIG_PATH in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
    ANALYZE_LOG="logs/analyze_${CONFIG_NAME}.log"

    echo ""
    echo "-------------------------------"
    echo "Analyzing: $CONFIG_NAME"
    echo "-------------------------------"

    CMD="python $ANALYZE --config \"$CONFIG_PATH\""

    if [[ "$MODE" == "background" ]]; then
        nohup bash -c "$CMD" > "$ANALYZE_LOG" 2>&1 &
    else
        nohup bash -c "$CMD" > "$ANALYZE_LOG" 2>&1
    fi

    ((CURRENT_PARALLEL++))
    if (( CURRENT_PARALLEL >= MAX_PARALLEL )); then
        wait
        CURRENT_PARALLEL=0
    fi
done

# [7] Wait for remaining analyze jobs
wait
echo ""
echo "All analyses completed."
