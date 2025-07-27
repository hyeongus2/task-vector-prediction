#!/usr/bin/env bash

# setup.sh

# Step 1. Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating .venv virtual environment..."
    python3 -m venv .venv
else
    echo "[1/4] .venv already exists. Skipping creation."
fi

# Step 2. Activate the virtual environment
echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

# Step 3. Upgrade pip and install packages
echo "[3/4] Installing required packages..."
pip install --upgrade pip

# if requirements.txt exists, install from it; otherwise, install manually
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    pip install torch torchvision timm numpy scikit-learn matplotlib pyyaml
fi

# Step 4. Complete setup
echo "[4/4] Setup complete!"
echo "To activate later: source .venv/bin/activate"
