#!/usr/bin/env bash
# scripts/train_autogpu.sh
#
# ----------------------------------------------------------------------
# train_autogpu.sh
# - Launch selected training jobs, each pinned to exactly one GPU.
# - Queueing guarantees "one job per GPU" using atomic filesystem locks
#   (no artificial delays; does not rely on nvidia-smi process timing).
# - Each job runs in its own tmux session (safe after SSH disconnect).
# - The scheduler self-wraps into a tmux session so queueing survives drops.
#
# Usage examples:
#   # Run everything that matches the user-editable settings below:
#   bash scripts/train_autogpu.sh
#
#   # Only vitb32 on CIFAR-10 with adamw or mmt, LoRA only:
#   MODEL_TAGS="vitb32" DATASETS="cifar10" OPTS="adamw mmt" METHODS="lora" \
#   bash scripts/train_autogpu.sh
#
#   # Preview what would launch (no sessions created):
#   DRY_RUN=1 bash scripts/train_autogpu.sh
#
# Environment filters (space-separated lists; empty means "all"):
#   MODEL_TAGS="vitb32 vitl14"
#   DATASETS="eurosat cifar10 food101"
#   OPTS="adamw sgd mmt"
#   METHODS="lora full"
#
# Other env vars:
#   CONFIG_DIR="configs"   # where YAML files live
#   LOG_DIR="logs"         # where logs go
#   TMUX_PREFIX="run"      # tmux session name prefix
#   HF_HOME=/data/.cache/huggingface  # shared cache (optional, recommended)
#
# Resume options (edit in-file; can be overridden by env):
#   - Set RESUME_SPEC to resume only specific runs.
#     * Name mapping:  "<config_basename>:<run_id> ..."  (e.g., vitb32_eurosat_adamw_full:2ppku504)
#     * Run-id only:   "<run_id> <run_id> ..."  (the script scans outputs/*/<run_id>/checkpoint.pt
#                        to detect which config it belongs to)
#   - RESUME_MODE:
#       STRICT (default): only launch matched checkpoints; skip others.
#       MIXED:            resume matched ones; launch the rest as new runs.
# ----------------------------------------------------------------------

set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ======================= USER-EDITABLE SETTINGS =======================
# Put your fixed values here (no CLI needed). Leave empty to use defaults below.

# Filters (space-separated; examples shown, comment out if not needed)
MODEL_TAGS="vitb32"                    # e.g., "vitb32 vitl14"
DATASETS="eurosat"                     # e.g., "eurosat cifar10 food101"
OPTS="adamw sgd mmt"                   # e.g., "sgd" or "adamw mmt"
METHODS="full"                         # e.g., "lora" or "full"

# Dry run (1 = preview only; 0 = actually launch)
DRY_RUN=0

# Hugging Face caches (recommended; leave empty to use defaults below)
# HF_HOME="/data/.cache/huggingface"

# Resume options (edit in-file; can be overridden by env if desired):
#   - Set RESUME_SPEC to resume only specific runs.
#     * Name mapping:  "<config_basename>:<run_id> ..."  (e.g., vitb32_eurosat_adamw_full:2ppku504)
#     * Run-id only:   "<run_id> <run_id> ..."  (the script scans outputs/*/<run_id>/checkpoint.pt
#                        to detect which config it belongs to)
#   - RESUME_MODE:
#       STRICT (default): only launch matched checkpoints; skip others.
#       MIXED:            resume matched ones; launch the rest as new runs.
RESUME_SPEC=""                         # e.g., 'vitb32_eurosat_adamw_full:2ppku504 3a1b2c3d'
RESUME_MODE="STRICT"                   # STRICT or MIXED
# =====================================================================

# ---------------------- BASIC SANITY CHECKS ---------------------------
command -v tmux >/dev/null 2>&1 || { echo "[ERR] tmux not found."; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "[ERR] nvidia-smi not found."; exit 1; }

# ---------------- CORE DIRECTORIES AND TMUX PREFIX --------------------
# Have sane defaults; can still be overridden via env if desired.
CONFIG_DIR="${CONFIG_DIR:-configs}"
LOG_DIR="${LOG_DIR:-logs}"
TMUX_PREFIX="${TMUX_PREFIX:-run}"
mkdir -p "${LOG_DIR}"

# -------------------- DEFAULTS (FALLBACKS) ----------------------------
# These only apply if the user-editable variables above are empty/unset.
MODEL_TAGS="${MODEL_TAGS:-vitb32}"
DATASETS="${DATASETS:-eurosat}"
OPTS="${OPTS:-adamw sgd mmt}"
METHODS="${METHODS:-lora full}"
DRY_RUN="${DRY_RUN:-0}"

# Hugging Face caches (optional but recommended to avoid duplicate downloads)
export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# In-file resume configuration fallbacks (env can override)
RESUME_SPEC="${RESUME_SPEC:-}"
RESUME_MODE="${RESUME_MODE:-STRICT}"

# ----------------------- SELF-WRAP INTO TMUX --------------------------
# Ensure the entire scheduler (including waiting/queueing) survives SSH disconnect.
: "${SELF_WRAP_TMUX:=1}"   # set to 0 to disable self-wrap
if [[ "${SELF_WRAP_TMUX}" == "1" && -z "${TMUX:-}" ]]; then
  # Resolve absolute script path and current working directory
  SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")"
  WORKDIR="$(pwd)"
  SCHED_PREFIX="${TMUX_PREFIX}"
  RANDHEX="$(openssl rand -hex 3 2>/dev/null || echo $$)"
  SESS_NAME="${SCHED_PREFIX}_sched_$(date +%Y%m%d_%H%M%S)_${RANDHEX}"

  # Safely forward all CLI args into the inner shell command
  TMUXFWD=""
  for a in "$@"; do TMUXFWD+=" $(printf '%q' "$a")"; done

  # Launch scheduler inside tmux; pipe pane output to a scheduler log.
  tmux new -d -s "$SESS_NAME" "bash -lc 'cd \"$WORKDIR\" && \"$SCRIPT_PATH\" $TMUXFWD'"
  tmux pipe-pane -o -t "$SESS_NAME" "tee -a \"${WORKDIR}/${LOG_DIR}/_scheduler_${SESS_NAME}.log\""

  echo "[train_autogpu] Scheduler launched inside tmux: $SESS_NAME"
  echo "Attach with: tmux attach -t $SESS_NAME"
  echo "Scheduler log: ${LOG_DIR}/_scheduler_${SESS_NAME}.log"
  exit 0
fi

# In addition to the tmux-piped log, keep a master log for this shell as well
MASTER_LOG="${LOG_DIR}/_scheduler_master_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${MASTER_LOG}") 2>&1
echo "[sched] Master log: ${MASTER_LOG}"

# ---------------- GPU RESERVATION VIA ATOMIC DIR LOCKS ----------------
# We avoid relying on nvidia-smi process timing by using a simple
# filesystem-based reservation. Exactly one job can hold a GPU index
# because `mkdir` is atomic:
#   - Reserve:  mkdir /tmp/autogpu.gpulocks/gpu<idx>.lock  (success => reserved)
#   - Release:  rmdir /tmp/autogpu.gpulocks/gpu<idx>.lock  (on job exit)
#
# Extra robustness:
#   - Record owner PID + tmux session into files for stale-lock GC.
#   - On each reservation attempt, GC stale locks:
#       * dead PID
#       * tmux session missing
#       * over-age (12h)
LOCK_ROOT="/tmp/autogpu.gpulocks"
mkdir -p "${LOCK_ROOT}"

# Return number of GPUs by counting `nvidia-smi -L` lines.
gpu_count() { nvidia-smi -L | wc -l; }

# Try to reserve a GPU by creating a lock dir. If none available, block until one frees.
# Also: write an owner file (PID, tmux pane id if any, timestamp); GC stale locks safely.
reserve_gpu_blocking() {
  local n sleep_s=3 now ts pid sid ssn ldir warned=0
  n="$(gpu_count)"
  if [[ "$n" -le 0 ]]; then
    echo "[ERR] No GPUs detected." >&2
    return 1
  fi

  while true; do
    now=$(date +%s)
    for ((i=0; i<n; i++)); do
      ldir="${LOCK_ROOT}/gpu${i}.lock"
      if mkdir "$ldir" 2>/dev/null; then
        # Record scheduler PID for fallback GC; tmux session will be recorded after launch.
        echo "$$ ${TMUX_PANE:-no_tmux} $now" > "${ldir}/owner" || true
        echo "$i"   # stdout must be index only
        return 0
      else
        # ---------------------- SAFE STALE GC RULES ----------------------
        if [[ -f "${ldir}/session" ]]; then
          read -r ssn < "${ldir}/session" || true
          if [[ -n "${ssn:-}" ]] && ! tmux has-session -t "$ssn" 2>/dev/null; then
            rm -rf "$ldir"
            continue
          fi
          # Session exists -> do NOT touch (even if PID died)
          continue
        fi
        if [[ -f "${ldir}/owner" ]]; then
          read -r pid sid ts < "${ldir}/owner" || true
          if [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null; then
            rm -rf "$ldir"
            continue
          fi
          if [[ -n "${ts:-}" ]] && [[ $(( now - ts )) -gt 43200 ]]; then
            rm -rf "$ldir"
            continue
          fi
        fi
        # ----------------------------------------------------------------
      fi
    done

    # --- only print queue notice once ---
    if [[ $warned -eq 0 ]]; then
      echo "[queue] All GPU locks busy. Waiting for free slot..." >&2
      warned=1
    fi
    sleep "$sleep_s"
  done
}
# ----------------------------------------------------------------------

# ------------------------- BUILD CANDIDATES ----------------------------
declare -a CANDIDATES=()
for mt in ${MODEL_TAGS}; do
  for ds in ${DATASETS}; do
    for opt in ${OPTS}; do
      for mtd in ${METHODS}; do
        cfg="${CONFIG_DIR}/${mt}_${ds}_${opt}_${mtd}.yaml"
        [[ -f "$cfg" ]] && CANDIDATES+=("$cfg")
      done
    done
  done
done

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "[train_autogpu] No matching configs with current filters."
  exit 0
fi

echo "[train_autogpu] ${#CANDIDATES[@]} configs matched:"
printf ' - %s\n' "${CANDIDATES[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[train_autogpu] DRY_RUN=1 -> not launching."
  exit 0
fi

# --------------------------- RESUME HELPERS ----------------------------
cfg_base() { basename "$1" .yaml; }

find_ckpt() {
  local base="$1"; local rid="$2"
  local p="outputs/${base}/${rid}/checkpoint.pt"
  [[ -f "$p" ]] && { echo "$p"; return 0; }
  return 1
}

detect_base_by_runid() {
  local rid="$1"
  local match
  match="$(find outputs -maxdepth 2 -type d -name "$rid" 2>/dev/null | head -n1 || true)"
  if [[ -n "$match" ]]; then
    local parent; parent="$(dirname "$match")"
    echo "$(basename "$parent")"
    return 0
  fi
  return 1
}

# Parse RESUME_SPEC → RESUME_BY_NAME + RESUME_RUNIDS
declare -A RESUME_BY_NAME=()
declare -a RESUME_RUNIDS=()
if [[ -n "${RESUME_SPEC}" ]]; then
  for tok in ${RESUME_SPEC}; do
    if [[ "$tok" == *:* ]]; then
      name="${tok%%:*}"
      rid="${tok#*:}"
      [[ -n "$name" && -n "$rid" ]] && RESUME_BY_NAME["$name"]="$rid"
    else
      RESUME_RUNIDS+=("$tok")
    fi
  done
  echo "[train_autogpu] Resume requested. MODE=${RESUME_MODE}, name-mapped=${#RESUME_BY_NAME[@]}, auto-runids=${#RESUME_RUNIDS[@]}"
fi

# -------------------------- LAUNCH QUEUE -------------------------------
declare -a LAUNCH_CFGS=()
declare -a LAUNCH_RIDS=()

if [[ -n "${RESUME_SPEC}" && "${RESUME_MODE^^}" == "STRICT" ]]; then
  # STRICT: only resume the provided run ids; do not start new runs for others.
  for base in "${!RESUME_BY_NAME[@]}"; do
    rid="${RESUME_BY_NAME[$base]}"
    cfg_path="${CONFIG_DIR}/${base}.yaml"
    if [[ -f "$cfg_path" ]]; then
      if ckpt="$(find_ckpt "$base" "$rid")"; then
        LAUNCH_CFGS+=("$cfg_path"); LAUNCH_RIDS+=("$rid")
      else
        echo "[WARN] checkpoint not found: outputs/${base}/${rid}/checkpoint.pt — skipping"
      fi
    else
      echo "[WARN] config not found for basename='${base}' — skipping"
    fi
  done
  for rid in "${RESUME_RUNIDS[@]}"; do
    if base="$(detect_base_by_runid "$rid")"; then
      cfg_path="${CONFIG_DIR}/${base}.yaml}"
      # fix: closing brace typo removed
      cfg_path="${CONFIG_DIR}/${base}.yaml"
      if [[ -f "$cfg_path" ]]; then
        if ckpt="$(find_ckpt "$base" "$rid")"; then
          LAUNCH_CFGS+=("$cfg_path"); LAUNCH_RIDS+=("$rid")
        else
          echo "[WARN] checkpoint not found: outputs/${base}/${rid}/checkpoint.pt — skipping"
        fi
      else
        echo "[WARN] config not found for detected base='${base}' (rid=${rid}) — skipping"
      fi
    else
      echo "[WARN] outputs scan could not find a match for run_id='${rid}' — skipping"
    fi
  done
  if [[ ${#LAUNCH_CFGS[@]} -eq 0 ]]; then
    echo "[train_autogpu] Nothing to resume in STRICT mode. Exiting."
    exit 0
  fi
else
  # MIXED (or no RESUME_SPEC): launch all candidates; attach resume ids where possible.
  declare -A CAND_SET=()
  # build candidate set safely (no fragile one-liners)
  for c in "${CANDIDATES[@]}"; do
    base_c="$(cfg_base "$c")"
    CAND_SET["$base_c"]=1    # <<< fixed: no spaces around '='
  done

  declare -A RID_FOR_BASE=()
  for base in "${!RESUME_BY_NAME[@]}"; do
    rid="${RESUME_BY_NAME[$base]}"
    if [[ -n "${CAND_SET[$base]:-}" ]]; then
      if ckpt="$(find_ckpt "$base" "$rid")"; then
        RID_FOR_BASE["$base"]="$rid"
      else
        echo "[WARN] checkpoint not found: outputs/${base}/${rid}/checkpoint.pt — will run fresh"
      fi
    fi
  done
  for rid in "${RESUME_RUNIDS[@]}"; do
    if base="$(detect_base_by_runid "$rid")"; then
      if [[ -n "${CAND_SET[$base]:-}" ]] && ckpt="$(find_ckpt "$base" "$rid")"; then
        RID_FOR_BASE["$base"]="$rid"
      else
        echo "[WARN] matched base='${base}' for rid='${rid}' but no config/ckpt — will run fresh"
      fi
    else
      echo "[WARN] outputs scan could not find base for rid='${rid}' — will run fresh"
    fi
  done
  for cfg in "${CANDIDATES[@]}"; do
    base="$(cfg_base "$cfg")"
    LAUNCH_CFGS+=("$cfg")
    LAUNCH_RIDS+=("${RID_FOR_BASE[$base]:-}")  # empty => start fresh
  done
fi

echo "[train_autogpu] ${#LAUNCH_CFGS[@]} jobs queued:"
for i in "${!LAUNCH_CFGS[@]}"; do
  b="$(cfg_base "${LAUNCH_CFGS[$i]}")"
  r="${LAUNCH_RIDS[$i]:-NEW}"
  echo " - ${b}  (RESUME=${r})"
done

# --------------------------- LAUNCH FUNCTION ---------------------------
launch_tmux_job() {
  local cfg="$1"
  local rid="${2:-}"
  local base
  base="$(basename "$cfg" .yaml)"

  # Reserve a free GPU index via atomic dir lock (blocks until available)
  local gpu_idx
  gpu_idx="$(reserve_gpu_blocking)" || { echo "[ERR] Failed to reserve a GPU."; return 1; }

  local RAND
  RAND="$(openssl rand -hex 3 2>/dev/null || echo $RANDOM)"
  local sess="${TMUX_PREFIX}_${base}_${rid:-new}_${RAND}"
  local logf="${LOG_DIR}/${base}${rid:+_${rid}}.log"

  echo "[train_autogpu] GPU ${gpu_idx} (locked) -> ${cfg} ${rid:+(resume $rid)}"

  local resume_opt=""
  if [[ -n "$rid" ]]; then resume_opt="--resume_id ${rid}"; fi

  # Launch job in tmux. Ensure the GPU lock is always released on any exit.
  if tmux new -d -s "$sess" "bash -lc 'set -euo pipefail; \
    trap \"rmdir \\\"${LOCK_ROOT}/gpu${gpu_idx}.lock\\\" 2>/dev/null || true\" EXIT INT TERM; \
    if [[ ! -f .venv/bin/activate ]]; then echo \"[ERR] .venv not found\"; exit 1; fi; \
    export CUDA_VISIBLE_DEVICES=${gpu_idx}; \
    source .venv/bin/activate; \
    python -u train.py --config \"$cfg\" ${resume_opt} 2>&1 | tee -a \"$logf\"'"; then
    # Record tmux session name into the lock for extra stale detection.
    echo "$sess" > "${LOCK_ROOT}/gpu${gpu_idx}.lock/session" || true
    echo "[tmux] session=${sess}  log=${logf}"
  else
    echo "[ERR] tmux failed to create session: ${sess} — releasing GPU lock"
    rm -rf "${LOCK_ROOT}/gpu${gpu_idx}.lock"
    return 1
  fi
}
# ----------------------------------------------------------------------

# ------------------------------ LAUNCH --------------------------------
for i in "${!LAUNCH_CFGS[@]}"; do
  launch_tmux_job "${LAUNCH_CFGS[$i]}" "${LAUNCH_RIDS[$i]:-}"
done

echo "[train_autogpu] All selected jobs launched (each GPU reserved by lock)."
echo "Tips:"
echo "  - GPU monitor:  watch -n 1 nvidia-smi"
echo "  - Logs:         tail -f ${LOG_DIR}/*.log"
echo "  - tmux list:    tmux ls"
echo "  - attach:       tmux attach -t <session>"
echo "  - stop a job:   tmux kill-session -t <session>"
