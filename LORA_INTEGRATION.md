# LoRA Integration Summary

## Overview

This document summarizes the integration of LoRA (Low-Rank Adaptation) into the RMoE expert training pipeline. LoRA enables parameter-efficient fine-tuning by training low-rank decomposition matrices instead of full model parameters.

## Changes Made

### 1. Dependencies (`requirements.txt`)

Added the `peft` (Parameter-Efficient Fine-Tuning) library:
```
peft>=0.7.0  # For LoRA training
```

### 2. Expert Training (`src/rmoe/finetune.py`)

**New imports:**
- Added `LoraConfig`, `get_peft_model`, `TaskType` from `peft` library
- Added environment variables to disable TensorFlow and use PyTorch only

**New function parameters:**
- `use_lora` (bool, default=True): Enable/disable LoRA training
- `lora_r` (int, default=8): LoRA rank (number of low-rank dimensions)
- `lora_alpha` (int, default=16): LoRA scaling factor
- `lora_dropout` (float, default=0.05): Dropout probability for LoRA layers
- `lora_target_modules` (list, default=None): Modules to apply LoRA to

**Key modifications:**

1. **Auto-detection of model architecture:**
   ```python
   if model_type in ["qwen2", "qwen"]:
       lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                              "gate_proj", "up_proj", "down_proj"]
   ```

2. **LoRA configuration and application:**
   ```python
   lora_config = LoraConfig(
       r=lora_r,
       lora_alpha=lora_alpha,
       target_modules=lora_target_modules,
       lora_dropout=lora_dropout,
       bias="none",
       task_type=TaskType.CAUSAL_LM,
   )
   model = get_peft_model(model, lora_config)
   ```

3. **Enhanced model saving:**
   - LoRA adapters are saved to the expert directory
   - The LoRA weights are merged with the base model
   - Merged model is saved to `{expert_dir}/merged/` for easy loading
   - This ensures compatibility with the downstream MoE merging process

### 3. Main Training Script (`train.py`)

**Environment variables:**
Added at the top of the script before importing transformers:
```python
os.environ["USE_TF"] = "0"  # Disable TensorFlow
os.environ["USE_TORCH"] = "1"  # Use PyTorch only
```

**LoRA configuration handling:**
```python
lora_config = expert_config.get("lora", {})
use_lora = lora_config.get("enabled", True)

expert_output_dir = train_expert(
    # ... other parameters ...
    use_lora=use_lora,
    lora_r=lora_config.get("r", 8),
    lora_alpha=lora_config.get("alpha", 16),
    lora_dropout=lora_config.get("dropout", 0.05),
    lora_target_modules=lora_config.get("target_modules", None),
)
```

**Path adjustment for LoRA-trained experts:**
When LoRA is used, the script now checks for merged models in the `merged/` subdirectory:
```python
if use_lora:
    adjusted_expert_paths = []
    for expert_path in expert_paths:
        merged_path = expert_path / "merged"
        if merged_path.exists():
            adjusted_expert_paths.append(merged_path)
        else:
            adjusted_expert_paths.append(expert_path)
    expert_paths = adjusted_expert_paths
```

### 4. Example Configuration (`data/train_rmoe_lora.json`)

Created a new example configuration file demonstrating LoRA usage:

```json
{
  "expert_training": {
    "learning_rate": 1e-4,
    "lora": {
      "enabled": true,
      "r": 8,
      "alpha": 16,
      "dropout": 0.05,
      "target_modules": null
    }
  }
}
```

### 5. Documentation (`README.md`)

Added comprehensive documentation about:
- Benefits of LoRA training
- Configuration parameters and their meanings
- How to enable/disable LoRA
- How the LoRA pipeline works

## Benefits

### Memory Efficiency
- **Before (MLP-only fine-tuning)**: Trains all MLP parameters (~30-40% of model)
- **After (LoRA)**: Trains <1% of model parameters
- Allows for larger batch sizes and faster training

### Performance
- Often matches or exceeds full fine-tuning performance
- Better generalization and less overfitting
- More stable training dynamics

### Flexibility
- Easy to switch between LoRA and traditional fine-tuning via config
- LoRA adapters can be saved separately and swapped
- Merged models are fully compatible with existing pipeline

## Usage Examples

### Train with LoRA (default)
```bash
python train.py data/train_rmoe_lora.json
```

### Train without LoRA (traditional MLP-only)
Modify config:
```json
{
  "expert_training": {
    "lora": {
      "enabled": false
    }
  }
}
```

### Adjust LoRA rank for different memory/performance tradeoffs
```json
{
  "expert_training": {
    "lora": {
      "enabled": true,
      "r": 16,        // Higher rank = more parameters, better fit
      "alpha": 32     // Typically 2x the rank
    }
  }
}
```

## Architecture Support

The implementation automatically detects model architecture and applies LoRA to appropriate modules:

- **Qwen2/Qwen**: Attention (q, k, v, o projections) + MLP (gate, up, down projections)
- **Llama**: Attention + MLP projections
- **Generic fallback**: Common projection layers

## File Structure After Training

```
workspace/{timestamp}/
├── experts/
│   ├── truthfulqa/
│   │   ├── adapter_config.json      # LoRA adapter configuration
│   │   ├── adapter_model.safetensors # LoRA adapter weights
│   │   └── merged/                   # Merged model used for MoE
│   │       ├── config.json
│   │       └── model.safetensors
│   └── longbench/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── merged/
│           ├── config.json
│           └── model.safetensors
├── gating_network/
└── rmoe_model/
```

## Troubleshooting

### TensorFlow Import Errors
Fixed by setting environment variables before importing transformers:
```python
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
```

### Memory Issues
If you encounter OOM errors, try:
- Reduce LoRA rank: `"r": 4`
- Reduce batch size
- Enable gradient checkpointing (if not already enabled)

### Model Loading Issues
Ensure the merged model exists:
- Check `{expert_dir}/merged/config.json` exists
- If missing, the LoRA merge may have failed during training

## Future Enhancements

Potential improvements:
1. Support for QLoRA (quantized LoRA) for even more memory efficiency
2. Multiple LoRA adapters per expert
3. Dynamic LoRA rank selection based on available memory
4. LoRA for gating network training
5. LoRA for full fine-tuning stage
