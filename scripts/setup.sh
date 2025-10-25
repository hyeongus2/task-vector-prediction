#!/bin/bash
# scripts/setup.sh
# This script intelligently sets up the Python virtual environment.
#
# Usage:
#   bash scripts/setup.sh        (Auto-detects GPU, defaults to CPU)
#   bash scripts/setup.sh xpu    (Manually installs for Intel GPU)
#   bash scripts/setup.sh cuda   (Manually installs for NVIDIA GPU)
#   bash scripts/setup.sh cpu    (Manually installs for CPU-only)

# Exit immediately if a command fails.
set -e

# --- 1. Smart Hardware Detection ---
HARDWARE_TARGET=""
# Check if a specific target is provided as the first argument
if [ -n "${1:-}" ]; then
    HARDWARE_TARGET=$1
    echo "User specified hardware target: '$HARDWARE_TARGET'"
else
    # Auto-detect hardware in order of preference if no target is specified
    echo "No target specified, attempting auto-detection..."
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected (via nvidia-smi). Setting target to 'cuda'."
        HARDWARE_TARGET="cuda"
    # Check for Intel Arc GPU on Windows using PowerShell (works in Git Bash)
    elif command -v powershell.exe &> /dev/null && (powershell.exe -Command "(Get-CimInstance -ClassName Win32_VideoController).Caption" | grep -q "Arc"); then
        echo "Intel Arc GPU detected (via PowerShell). Setting target to 'xpu'."
        HARDWARE_TARGET="xpu"
    else
        echo "No high-performance GPU detected. Defaulting to 'cpu'."
        HARDWARE_TARGET="cpu"
    fi
fi

# --- 2. Create and Activate Virtual Environment ---
echo "--- [1/4] Setting up virtual environment ---"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in './.venv'..."
    python3 -m venv .venv
else
    echo "Virtual environment './.venv' already exists."
fi

echo "--- [2/4] Activating virtual environment ---"
# Check OS type for correct activation script path and save the command
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source .venv/Scripts/activate
    ACTIVATE_CMD="source .venv/Scripts/activate"
else
    source .venv/bin/activate
    ACTIVATE_CMD="source .venv/bin/activate"
fi

# --- 3. Install Packages ---
echo "--- [3/4] Installing packages ---"
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing PyTorch for '$HARDWARE_TARGET'..."
case "$HARDWARE_TARGET" in
    cuda)
        echo "Installing PyTorch (CUDA 12.1 build)..."
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    xpu)
        echo "Installing PyTorch (Intel XPU build)..."
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
        python -m pip install intel-extension-for-pytorch
        ;;
    cpu|*)
        echo "Installing PyTorch (CPU build)..."
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

echo "Installing remaining packages from requirements.txt..."
if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Installing core ML packages..."
    python -m pip install transformers datasets peft accelerate safetensors numpy pandas matplotlib pillow tqdm pyyaml wandb
fi

# --- 4. Done ---
echo "--- [4/4] Setup complete for '$HARDWARE_TARGET' ---"
# Use the ACTIVATE_CMD variable to show the correct message for the OS.
echo "Environment is ready. To activate it in a new terminal, run: '$ACTIVATE_CMD'"