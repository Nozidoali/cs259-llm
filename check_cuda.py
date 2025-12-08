#!/usr/bin/env python3
"""
CUDA diagnostic script to verify GPU setup before training.
Run this before starting training to ensure CUDA is properly configured.
"""

import os
import sys

print("=" * 80)
print("CUDA DIAGNOSTIC TOOL")
print("=" * 80)
print()

# Check PyTorch
print("[1/4] Checking PyTorch installation...")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"      Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"      Compute capability: {props.major}.{props.minor}")
    else:
        print("  ✗ CUDA not available for PyTorch")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print()

# Check transformers
print("[2/4] Checking transformers library...")
try:
    import transformers
    print(f"  ✓ Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print()

# Check if TensorFlow is installed (used by BLEURT)
print("[3/4] Checking TensorFlow (used by BLEURT)...")
try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow version: {tf.__version__}")
    
    # Configure TensorFlow to use CPU only (same as in training pipeline)
    tf.config.set_visible_devices([], 'GPU')
    
    # Verify that TensorFlow sees no GPUs after configuration
    gpus = tf.config.get_visible_devices('GPU')
    if len(gpus) == 0:
        print(f"  ✓ TensorFlow correctly configured to use CPU (sees 0 GPUs)")
    else:
        print(f"  ⚠ Warning: TensorFlow sees {len(gpus)} GPU(s) after configuration")
except Exception as e:
    print(f"  ⚠ TensorFlow check failed (this is OK if BLEURT is not used): {e}")

print()

# Test CUDA with PyTorch
print("[4/4] Testing CUDA with PyTorch...")
if torch.cuda.is_available():
    try:
        # Create a small tensor on GPU
        device = torch.device("cuda:0")
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.matmul(x, y)
        print(f"  ✓ Successfully performed matrix multiplication on GPU")
        print(f"  ✓ GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        torch.cuda.empty_cache()
        print(f"  ✓ GPU memory cleared")
    except Exception as e:
        print(f"  ✗ CUDA test failed: {e}")
        sys.exit(1)
else:
    print("  ⚠ Skipping CUDA test (CUDA not available)")

print()
print("=" * 80)
if torch.cuda.is_available():
    print("✓ CUDA is properly configured! You can start training.")
else:
    print("⚠ CUDA is not available. Training will use CPU (very slow).")
print("=" * 80)
print()
print("To start training with CUDA:")
print("  python train.py data/train_qwen2_large.json --conversion-mode preserve_moe")
print()
