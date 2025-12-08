# H100 GPU Training Configuration

## Problem
When running training on GPU servers (H100, A100, etc.), TensorFlow (used by BLEURT for evaluation) was initializing CUDA before PyTorch, causing conflicts and the error:
```
CUDA error: Failed call to cuInit: CUDA_ERROR_UNKNOWN: unknown error
```

## Solution
The fix ensures that **PyTorch initializes CUDA FIRST** and claims the H100 GPU for training, while TensorFlow uses CPU only for BLEURT evaluation.

**Key Strategy:**
1. **PyTorch initializes CUDA immediately** when train.py starts - claims H100 GPU
2. **TensorFlow is configured via Python API** (`tf.config.set_visible_devices([], 'GPU')`) to use CPU only
3. **Warnings are NOT suppressed** - you can see the full status of GPU initialization
4. **No environment variable modification after import** - prevents PyTorch confusion

## Changes Made

### 1. `train.py` - PyTorch CUDA Initialization First
Added immediate PyTorch CUDA initialization at the top of the file:
```python
import torch

# Initialize PyTorch CUDA immediately to claim GPU resources
if torch.cuda.is_available():
    torch.cuda.init()
    print(f"✓ PyTorch CUDA initialized: {torch.cuda.get_device_name(0)}")
```
This ensures PyTorch claims the H100 GPU BEFORE TensorFlow is imported.

### 2. `src/rmoe/trainer.py` - BLEURT with TensorFlow CPU
Updated BLEURT loading to use TensorFlow on CPU via Python API:
```python
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hide GPUs from TensorFlow
```
Includes logging to show when BLEURT is loading and which device it uses.

### 3. `src/rmoe/evaluate.py` - Evaluation with TensorFlow CPU
Applied the same TensorFlow CPU configuration with informative logging.

### 4. `src/evaluation.py` - Main Evaluation with TensorFlow CPU
Applied the same TensorFlow CPU configuration with informative logging.

### 5. **Warnings NOT suppressed** - Full visibility of GPU initialization status

### 5. `check_cuda.py` - Diagnostic Tool (NEW)
Created a diagnostic script to verify CUDA setup before training.

## Usage

### Before Training - Check CUDA Setup
```bash
python check_cuda.py
```

This will:
- ✓ Verify PyTorch can see CUDA
- ✓ List available GPUs (e.g., H100)
- ✓ Check TensorFlow is configured correctly
- ✓ Test GPU computation

### Start Training
```bash
# With the config file
python train.py data/train_qwen2_large.json --conversion-mode preserve_moe
```

### View Logs
All logs are saved to: `workspace/<timestamp>/pipeline.log`
```bash
# View logs in real-time
tail -f workspace/*/pipeline.log

# View tensorboard
tensorboard --logdir workspace/<timestamp>
```

## Why This Works

1. **TensorFlow uses CPU** - BLEURT evaluation runs on CPU (fast enough for evaluation)
2. **PyTorch uses GPU** - All model training happens on GPU (H100/A100)
3. **No conflicts** - They never compete for CUDA resources

## Current Configuration
Your `train_qwen2_large.json` is set to:
- Device: `cuda` ✓
- 32 experts (16 truthfulqa + 16 qmsum)
- 6 epochs per expert
- Full fine-tuning: 100 epochs
- Batch size: 2

## Troubleshooting

If you still see CUDA errors:

1. **Run diagnostic first:**
   ```bash
   python check_cuda.py
   ```

2. **Check CUDA driver:**
   ```bash
   nvidia-smi
   ```

3. **Verify PyTorch CUDA:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Check environment variables:**
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   ```

## Expected Behavior

✓ **PyTorch CUDA initialization at start (FIRST thing you see):**
```
✓ PyTorch CUDA initialized: NVIDIA H100 PCIe
✓ GPU count: 1
✓ CUDA version: 12.1
```

✓ **Normal pipeline logs:**
```
2025-12-08 11:53:29,383 - INFO - Using timestamp from WORKSPACE_TIMESTAMP environment variable: 20251208_115231
2025-12-08 11:53:29,385 - INFO - Work directory: /workspace/cs259-llm/workspace/20251208_115231
2025-12-08 11:53:29,386 - INFO - Training expert for dataset: truthfulqa
```

✓ **PyTorch training on GPU:**
```
Using device: cuda
GPU memory: 40.00 GB / 80.00 GB
```

✓ **BLEURT loading (when evaluation happens):**
```
2025-12-08 12:00:00,000 - INFO - Loading BLEURT model (TensorFlow will use CPU for evaluation)...
2025-12-08 12:00:05,000 - INFO - BLEURT model loaded successfully on CPU
```

✗ **No more CUDA initialization errors**

## Performance Notes

- **Training (PyTorch)**: Uses H100 GPU - Very fast ✓
- **BLEURT Evaluation (TensorFlow)**: Uses CPU - Still fast enough for evaluation ✓
- **Overall**: No performance loss, training is GPU-accelerated

## Questions?

If you encounter any issues, check:
1. Run `python check_cuda.py` for diagnostics
2. Check logs at `workspace/<timestamp>/pipeline.log`
3. Verify GPU with `nvidia-smi`
