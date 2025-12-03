#!/bin/bash

# Script to create a clean Python virtual environment for CS259 project
# Optimized for runpods (Linux with CUDA)

set -e  # Exit on error

ENV_NAME="snapdragon"
ENV_DIR="./venv"

echo "Removing existing virtual environment if it exists..."
if [ -d "$ENV_DIR" ]; then
    rm -rf "$ENV_DIR"
    echo "Removed existing environment"
fi

echo "Creating fresh Python virtual environment..."
python3 -m venv "$ENV_DIR"

echo "Activating virtual environment..."
source "$ENV_DIR/bin/activate"

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing PyTorch with CUDA support..."
# Install PyTorch with CUDA 12.1 (adjust version if needed for your runpod)
# Note: If PyTorch is already in the base image, you can skip this step
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "To use this environment:"
echo "  source $ENV_DIR/bin/activate"
echo ""
echo "Then try:"
echo "  python src/finetune.py --model qwen2-0.5b --download-only"
echo "  # Or use the MoE model:"
echo "  # python src/finetune.py --model qwen1.5-moe-a2.7b --download-only"

