# CUDA Conflict Fix for H100/A100 Servers

## Problem
When running training on GPU servers (H100, A100, etc.), TensorFlow (used by BLEURT for evaluation) was initializing CUDA before PyTorch, causing conflicts and the error:
```
CUDA error: Failed call to cuInit: CUDA_ERROR_UNKNOWN: unknown error
```

## Solution
The fix ensures that TensorFlow uses **CPU only** while PyTorch has exclusive access to the GPU for training.

**Key Insight:** We use TensorFlow's Python API (`tf.config.set_visible_devices([], 'GPU')`) to hide GPUs from TensorFlow instead of modifying `CUDA_VISIBLE_DEVICES` environment variable, which would confuse PyTorch after it has already initialized.

## Changes Made

### 1. `train.py` - Suppress TensorFlow Warnings
Added environment variable to suppress TensorFlow logging:
```python
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
```

### 2. `src/rmoe/trainer.py` - BLEURT Loading Fix
Updated the BLEURT loading to force TensorFlow to use CPU via Python API:
```python
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs from TensorFlow
```
This ensures TensorFlow doesn't interfere with PyTorch's CUDA usage.

### 3. `src/rmoe/evaluate.py` - Evaluation Fix
Applied the same TensorFlow Python API fix to BLEURT loading in evaluation functions.

### 4. `src/evaluation.py` - Evaluation Fix
Applied the same TensorFlow Python API fix to BLEURT loading in the main evaluation module.

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

✓ **Normal logs at start:**
```
2025-12-08 11:53:29,383 - INFO - Using timestamp from WORKSPACE_TIMESTAMP environment variable: 20251208_115231
2025-12-08 11:53:29,385 - INFO - Work directory: /workspace/cs259-llm/workspace/20251208_115231
2025-12-08 11:53:29,386 - INFO - Training expert for dataset: truthfulqa
```

✓ **PyTorch should detect CUDA:**
```
Using device: cuda
GPU memory: X.XX GB / XX.XX GB
```

✗ **No more TensorFlow CUDA errors**

## Performance Notes

- **Training (PyTorch)**: Uses H100 GPU - Very fast ✓
- **BLEURT Evaluation (TensorFlow)**: Uses CPU - Still fast enough for evaluation ✓
- **Overall**: No performance loss, training is GPU-accelerated

## Questions?

If you encounter any issues, check:
1. Run `python check_cuda.py` for diagnostics
2. Check logs at `workspace/<timestamp>/pipeline.log`
3. Verify GPU with `nvidia-smi`
