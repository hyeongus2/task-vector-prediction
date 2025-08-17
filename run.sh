#!/usr/bin/env bash
# run.sh
# - Launch train and/or analyze across multiple configs with bounded parallelism
# - Uses a small PID pool and wait -n (with fallback) to keep up to MAX_PARALLEL jobs alive
# - Comments intentionally in English

set -u -o pipefail  # avoid 'set -e' so a single job failure won't kill the whole orchestrator

# -------------------------------
# CLI parsing
# -------------------------------
MODE="background"            # background | foreground (affects nohup usage only)
PHASE="both"                 # both | train | analyze
MAX_PARALLEL=2
OVERRIDES=""

usage() {
  cat <<EOF
Usage: bash run.sh [--phase both|train|analyze] [--mode background|foreground] [--max-parallel N] [--overrides "k=v ..."]

Examples:
  bash run.sh --phase analyze --mode background
  bash run.sh --phase train --mode foreground --max-parallel 4
  bash run.sh --overrides "train.lr=0.01 train.batch_size=64"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-background}"; shift 2;;
    --phase) PHASE="${2:-both}"; shift 2;;
    --max-parallel) MAX_PARALLEL="${2:-2}"; shift 2;;
    --overrides) OVERRIDES="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done

# -------------------------------
# Env & paths
# -------------------------------
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "Activated virtual environment: .venv"
else
  echo "No .venv found. Make sure your environment is ready."
  exit 1
fi

mkdir -p logs

CONFIGS=(
  "configs/vit_tiny_cifar10_full_sgd.yaml"
  "configs/vit_tiny_cifar10_full_adam.yaml"
  # "configs/vit_tiny_mnist_full_sgd.yaml"
  # "configs/vit_tiny_mnist_full_adam.yaml"
  # "configs/resnet18_cifar10_full_sgd.yaml"
  # "configs/resnet18_cifar10_full_adam.yaml"
  # "configs/resnet18_mnist_full_sgd.yaml"
  # "configs/resnet18_mnist_full_adam.yaml"
)

TRAIN="train.py"
ANALYZE="analyze.py"

# -------------------------------
# Minimal bounded-parallel launcher
# -------------------------------
PIDS=()        # parallel job pids
DESCS=()       # labels for logging (optional)

launch_job () {
  # $1 = command; $2 = logfile; $3 = desc
  local CMD="$1"
  local LOG="$2"
  local DESC="$3"

  if [[ "$MODE" == "background" ]]; then
    nohup bash -lc "$CMD" > "$LOG" 2>&1 &
  else
    # In foreground mode we still launch jobs in background to keep concurrency,
    # but we do not use nohup; we will 'wait' them explicitly.
    bash -lc "$CMD" > "$LOG" 2>&1 &
  fi

  local PID=$!
  PIDS+=("$PID")
  DESCS+=("$DESC")
  echo "Launched [$DESC] (pid=$PID) -> $CMD"
}

wait_for_slot () {
  # Keep at most MAX_PARALLEL concurrent jobs
  while (( ${#PIDS[@]} >= MAX_PARALLEL )); do
    if wait -n 2>/dev/null; then
      # A job finished; prune it from our arrays
      prune_finished
    else
      # 'wait -n' may not be available (bash < 5): fallback to waiting first pid
      local PID0="${PIDS[0]}"
      wait "$PID0" || true
      prune_finished
    fi
  done
}

prune_finished () {
  # Remove finished PIDs from arrays
  local NEW_PIDS=()
  local NEW_DESCS=()
  for i in "${!PIDS[@]}"; do
    local PID="${PIDS[$i]}"
    if kill -0 "$PID" 2>/dev/null; then
      NEW_PIDS+=("$PID")
      NEW_DESCS+=("${DESCS[$i]}")
    else
      echo "Finished [${DESCS[$i]}] (pid=$PID)"
    fi
  done
  PIDS=("${NEW_PIDS[@]}")
  DESCS=("${NEW_DESCS[@]}")
}

wait_all () {
  # Wait for all remaining jobs to complete
  if (( ${#PIDS[@]} == 0 )); then
    return
  fi
  echo "Waiting for ${#PIDS[@]} job(s) to finish..."
  for PID in "${PIDS[@]}"; do
    wait "$PID" || true
  done
  PIDS=()
  DESCS=()
}

# -------------------------------
# Phase: TRAIN
# -------------------------------
if [[ "$PHASE" == "both" || "$PHASE" == "train" ]]; then
  for CONFIG_PATH in "${CONFIGS[@]}"; do
    NAME=$(basename "$CONFIG_PATH" .yaml)
    LOG="logs/${NAME}.log"

    echo ""
    echo "==============================="
    echo "Training: $NAME"
    echo "Log:      $LOG"
    echo "==============================="

    if [[ -z "$OVERRIDES" ]]; then
      CMD="python $TRAIN --config \"$CONFIG_PATH\""
    else
      CMD="python $TRAIN --config \"$CONFIG_PATH\" --overrides \"$OVERRIDES\""
    fi

    wait_for_slot
    launch_job "$CMD" "$LOG" "train:$NAME"
  done

  wait_all
  echo ""
  echo "All training experiments completed."
else
  echo "[Skip] Training phase skipped (--phase=$PHASE)."
fi

# -------------------------------
# Phase: ANALYZE
# -------------------------------
if [[ "$PHASE" == "both" || "$PHASE" == "analyze" ]]; then
  for CONFIG_PATH in "${CONFIGS[@]}"; do
    NAME=$(basename "$CONFIG_PATH" .yaml)
    LOG="logs/analyze_${NAME}.log"

    echo ""
    echo "-------------------------------"
    echo "Analyzing: $NAME"
    echo "Log:       $LOG"
    echo "-------------------------------"

    if [[ -z "$OVERRIDES" ]]; then
      CMD="python $ANALYZE --config \"$CONFIG_PATH\""
    else
      CMD="python $ANALYZE --config \"$CONFIG_PATH\" --overrides \"$OVERRIDES\""
    fi

    wait_for_slot
    launch_job "$CMD" "$LOG" "analyze:$NAME"
  done

  wait_all
  echo ""
  echo "All analyses completed."
else
  echo "[Skip] Analyze phase skipped (--phase=$PHASE)."
fi
