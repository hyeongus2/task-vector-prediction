#!/usr/bin/env bash
set -euo pipefail

# Output directory for generated YAMLs
OUTDIR="configs"
mkdir -p "$OUTDIR"

# ===============================
# Dataset presets (HF datasets)
# ===============================
declare -A DSET_NAME  DSET_IMG  DSET_LBL  DSET_SPLIT
DSET_NAME[eurosat]="tanganke/eurosat"; DSET_IMG[eurosat]="image"; DSET_LBL[eurosat]="label";     DSET_SPLIT[eurosat]="auto_split"
DSET_NAME[cifar10]="cifar10";          DSET_IMG[cifar10]="img";   DSET_LBL[cifar10]="label";     DSET_SPLIT[cifar10]="pre_split"   # train & test exist
DSET_NAME[food101]="ethz/food101";     DSET_IMG[food101]="image"; DSET_LBL[food101]="label";     DSET_SPLIT[food101]="auto_split"

# ===============================
# Model ids (HF model hub)
# ===============================
MODEL_ID_BASE="openai/clip-vit-base-patch32"
MODEL_ID_LARGE="openai/clip-vit-large-patch14"

# ===============================
# Heuristics (epochs, lr, lora params)
# ===============================
epochs_for() {
  case "$1" in
    adamw) echo 20 ;;
    mmt)   echo 40 ;;
    sgd)   echo 60 ;;
    *)     echo 40 ;;
  esac
}

lr_for() {
  case "$1" in
    adamw) echo "1.e-4" ;;
    mmt)   echo "1.e-2" ;;
    sgd)   echo "1.e-1" ;;
    *)     echo "1.e-3" ;;
  esac
}

# batch size = 128
batch_size() { echo 128; }

# LoRA params: vitb32 → r=16, alpha=32 / vitl14 → r=32, alpha=64
lora_params_for() {
  local model_tag="$1"   # vitb32|vitl14
  if [[ "$model_tag" == "vitl14" ]]; then
    echo "32 64"   # r alpha
  else
    echo "16 32"
  fi
}

# ===============================
# Emit one YAML with EXACT schema
# ===============================
emit_yaml() {
  local model_tag="$1"   # vitb32|vitl14
  local dkey="$2"        # eurosat|cifar10|food101|caltech101
  local opt="$3"         # adamw|sgd|mmt
  local fmethod="$4"     # lora|full

  # model
  local model_id="$MODEL_ID_BASE"
  if [[ "$model_tag" == "vitl14" ]]; then model_id="$MODEL_ID_LARGE"; fi

  # dataset
  local ds_name="${DSET_NAME[$dkey]}"
  local img_col="${DSET_IMG[$dkey]}"
  local lbl_col="${DSET_LBL[$dkey]}"
  local split_strategy="${DSET_SPLIT[$dkey]}"

  # train settings
  local epochs="$(epochs_for "$opt")"
  local lr="$(lr_for "$opt")"
  local bs="$(batch_size)"

  # optimizer mapping: mmt => sgd + momentum 0.9
  local optimizer_yaml="$opt"
  local momentum="0.0"
  if [[ "$opt" == "mmt" ]]; then optimizer_yaml="sgd"; momentum="0.9"; fi

  # LoRA params by model size
  read -r LORA_R LORA_ALPHA < <(lora_params_for "$model_tag")

  # filename
  local fname="${model_tag}_${dkey}_${opt}_${fmethod}.yaml"
  local fout="$OUTDIR/$fname"

  cat > "$fout" <<EOF
# Basic settings
seed: 42

# Model settings
model_id: ${model_id}

# Data settings
data:
    name: ${ds_name}
    image_column_name: ${img_col}
    label_column_name: ${lbl_col}

    split_strategy: ${split_strategy}
    validation_split_size: 0.2

    batch_size: ${bs}
    num_workers: 4

# Finetuning settings (kept the same for a direct comparison)
finetuning:
    method: ${fmethod}
    optimizer: ${optimizer_yaml}
    epochs: ${epochs}
    lr: ${lr}
    momentum: ${momentum}

    # LoRA settings (kept even if method=full; simply unused)
    lora:
        r: ${LORA_R}
        lora_alpha: ${LORA_ALPHA}
        target_modules: [q_proj, v_proj]
        lora_dropout: 0.0
        bias: none

# Analysis & Save settings
analysis:
    save_tau_every_n_steps: 100
    num_monitoring_elements: 20
    num_analysis_elements: 50

# Logging settings
logging:
    enabled: true
    wandb: true
EOF

  echo "[make_config] wrote $fout"
}

# ===============================
# Grid
# ===============================
DATASETS=(eurosat cifar10 food101)
MODELS=(vitb32 vitl14)
OPTS=(adamw sgd mmt)
METHODS=(lora full)

for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    for o in "${OPTS[@]}"; do
      for f in "${METHODS[@]}"; do
        emit_yaml "$m" "$d" "$o" "$f"
      done
    done
  done
done

echo "[make_config] done. files at $OUTDIR/"
