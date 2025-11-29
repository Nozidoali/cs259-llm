#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ENV_NAME="snapdragon"

module unload python

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"
pip install -r requirements.txt
srun --partition=batch \
     --gres=gpu:a100:1 \
     --cpus-per-task=8 \
     --mem=64G \
     --time=04:00:00 \
     --job-name=cs259 \
     --unbuffered \
     python main.py

