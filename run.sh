#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
srun --partition=batch \
     --gres=gpu:a100:1 \
     --cpus-per-task=8 \
     --mem=64G \
     --time=04:00:00 \
     --job-name=cs259 \
     --unbuffered \
     python3 main.py

