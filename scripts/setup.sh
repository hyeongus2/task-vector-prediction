#!/usr/bin/env bash
# scripts/setup.sh
set -euo pipefail

# [0] pick python executable
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python not found. Please install Python 3.9+ and re-run."
  exit 1
fi

# [1] create venv if missing
if [ ! -d ".venv" ]; then
  echo "[1/4] Creating .venv virtual environment..."
  "$PY" -m venv .venv
else
  echo "[1/4] .venv already exists. Skipping creation."
fi

# [2] activate
echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

# [3] upgrade pip and install deps
echo "[3/4] Installing required packages..."
python -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found; installing a minimal set of packages."
  pip install torch torchvision open_clip_torch PyYAML numpy tqdm
fi

# [4] done
echo "[4/4] Setup complete."
echo "To activate later: source .venv/bin/activate"
