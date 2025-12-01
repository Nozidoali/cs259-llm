#!/bin/bash

# Script to create a clean conda environment for CS259 project

echo "Removing existing 'snapdragon' environment if it exists..."
conda env remove -n snapdragon -y 2>/dev/null || echo "No existing environment to remove"

echo "Creating fresh conda environment 'snapdragon'..."
conda create -n snapdragon python=3.9 -y

echo "Activating environment..."
conda activate snapdragon

echo "Installing PyTorch via conda (recommended for macOS)..."
conda install pytorch torchvision -c pytorch -y

echo "Installing other dependencies via pip..."
pip install transformers>=4.30.0
pip install datasets>=2.12.0
pip install accelerate>=0.20.0
pip install "numpy<2.0.0"
pip install huggingface_hub>=0.16.0
pip install evaluate>=0.4.0
pip install python-dotenv>=1.0.0
pip install sentencepiece
pip install protobuf

echo "Skipping bitsandbytes and BLEURT for now (optional)..."
# pip install bitsandbytes>=0.41.0
# pip install git+https://github.com/google-research/bleurt.git

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "To use this environment:"
echo "  conda activate snapdragon"
echo ""
echo "Then try:"
echo "  python src/finetune.py --model qwen2-0.5b --download-only"
echo "  # Or use the MoE model:"
echo "  # python src/finetune.py --model qwen1.5-moe-a2.7b --download-only"

