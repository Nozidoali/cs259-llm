#!/bin/bash

# Script to create a clean conda environment for CS259 project

set -e  # Exit on error

echo "Removing existing 'snapdragon' environment if it exists..."
conda env remove -n snapdragon -y 2>/dev/null || echo "No existing environment to remove"

echo "Creating fresh conda environment 'snapdragon'..."
conda create -n snapdragon python=3.9 -y

echo "Installing PyTorch via conda (recommended for macOS)..."
conda install -n snapdragon pytorch torchvision -c pytorch -y

echo "Installing other dependencies via pip..."
conda run -n snapdragon pip install "transformers>=4.30.0"
conda run -n snapdragon pip install "datasets<2.17.0"
conda run -n snapdragon pip install "accelerate>=0.20.0"
conda run -n snapdragon pip install "numpy<2.0.0"
conda run -n snapdragon pip install "huggingface_hub>=0.16.0"
conda run -n snapdragon pip install "evaluate>=0.4.0"
conda run -n snapdragon pip install "python-dotenv>=1.0.0"
conda run -n snapdragon pip install sentencepiece
conda run -n snapdragon pip install protobuf
conda run -n snapdragon pip install rouge_score nltk    
conda run -n snapdragon pip install tf-keras==2.16.0
conda run -n snapdragon pip install scikit-learn

echo "Installing TensorFlow Metal for Apple Silicon GPU acceleration (for BLEURT)..."
conda run -n snapdragon pip install tensorflow-macos tensorflow-metal

echo "Installing BLEURT..."
conda run -n snapdragon pip install "git+https://github.com/google-research/bleurt.git"

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

