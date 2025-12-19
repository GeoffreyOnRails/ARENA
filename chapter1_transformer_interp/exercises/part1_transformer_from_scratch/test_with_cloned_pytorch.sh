#!/bin/bash
# Script to test linear_debug.py with the cloned PyTorch version

set -e

# Hardcoded paths
ARENA_DIR="/Users/geoffreyvoyer/dev/ARENA_3.0"
SCRIPT_DIR="/Users/geoffreyvoyer/dev/ARENA_3.0/chapter1_transformer_interp/exercises/part1_transformer_from_scratch"
PYTORCH_CLONE_DIR="/Users/geoffreyvoyer/dev/pytorch"
VENV_ACTIVATE="/Users/geoffreyvoyer/dev/ARENA_3.0/.venv/bin/activate"

echo "=================================================================================="
echo "Testing nn.Linear issue with cloned PyTorch version"
echo "=================================================================================="
echo ""

# Check if venv exists
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "ERROR: Virtual environment not found at $VENV_ACTIVATE"
    exit 1
fi

# Activate virtual environment
source "$VENV_ACTIVATE"

CMAKE_VERSION=$(cmake --version | head -1)
echo "CMake version: $CMAKE_VERSION"
echo ""

# Check current PyTorch version
echo "Current PyTorch version:"
python -c "import torch; print(f'  {torch.__version__}'); print(f'  Path: {torch.__file__}')" 2>/dev/null || echo "  PyTorch not installed"
echo ""

# Check if PyTorch clone exists
if [ ! -d "$PYTORCH_CLONE_DIR" ]; then
    echo "ERROR: PyTorch clone not found at $PYTORCH_CLONE_DIR"
    exit 1
fi

# Get the commit hash of the cloned PyTorch
cd "$PYTORCH_CLONE_DIR"
PYTORCH_COMMIT=$(git rev-parse --short HEAD)
PYTORCH_BRANCH=$(git branch --show-current)
PYTORCH_DATE=$(git log -1 --format=%ci)
echo "Cloned PyTorch info:"
echo "  Branch: $PYTORCH_BRANCH"
echo "  Commit: $PYTORCH_COMMIT"
echo "  Date: $PYTORCH_DATE"
echo ""

# Check if already installed in development mode
CURRENT_TORCH_PATH=$(python -c "import torch; print(torch.__file__)" 2>/dev/null || echo "")
if [[ "$CURRENT_TORCH_PATH" == *"$PYTORCH_CLONE_DIR"* ]]; then
    echo "✓ Cloned PyTorch is already installed in development mode!"
    echo "  Path: $CURRENT_TORCH_PATH"
    echo ""
    echo "Running test..."
    cd "$SCRIPT_DIR"
    python linear_debug.py
    exit 0
fi

# Ask user if they want to proceed with installation
echo "WARNING: Installing PyTorch from source in development mode can take 30-60 minutes"
echo "This will uninstall the current PyTorch and install the cloned version."
echo ""
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. You can run this script again when ready."
    exit 0
fi

# Uninstall current PyTorch
echo ""
echo "Uninstalling current PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Initialize Git submodules (required for PyTorch build)
echo ""
echo "Initializing Git submodules..."
cd "$PYTORCH_CLONE_DIR"
git submodule sync
git submodule update --init --recursive
echo "Git submodules initialized."
echo ""

# Install cloned PyTorch in development mode
echo ""
echo "Installing cloned PyTorch in development mode..."
echo "This will take a while (30-60 minutes)..."
echo ""

# For MPS support on macOS
export USE_MPS=1
export USE_METAL=1

# Install in development mode (this will build from source)
echo "Starting installation (this may take a while)..."
pip install -e . --no-build-isolation 2>&1 | tee /tmp/pytorch_install.log

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'New PyTorch version: {torch.__version__}')
print(f'PyTorch path: {torch.__file__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"

# Run the test
echo ""
echo "=================================================================================="
echo "Running linear_debug.py test..."
echo "=================================================================================="
cd "$SCRIPT_DIR"
python linear_debug.py

echo ""
echo "=================================================================================="
echo "Test completed!"
echo "=================================================================================="
echo ""
echo "To revert to the original PyTorch, run:"
echo "  pip uninstall -y torch torchvision torchaudio"
echo "  pip install torch torchvision torchaudio"
