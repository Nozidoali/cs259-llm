# CS259 - Qwen2 Fine-tuning and LongBench Evaluation

## Setup

1. **Clone the repository with submodules:**
```bash
git clone --recurse-submodules https://github.com/Nozidoali/cs259-llm.git
cd cs259-llm
```

If you already cloned without submodules, initialize them:
```bash
git submodule update --init --recursive
```

2. **Configure environment variables:**
```bash
cp .env.example .env
```

Edit `.env` and set:
- `WORK_DIR`: Working directory for models, data, and logs (defaults to current directory)
- `LLAMA_CPP_DIR`: Path to llama.cpp (defaults to `external/llama.cpp`)

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Run Tests

Run benchmarks and evaluations using a JSON config file:

```bash
python test.py data/test.json
```

The config file specifies model parameters, device settings, and evaluation options. Results are saved to the `results/` directory.

### Train Models (Optional)

Fine-tune, convert to GGUF, and push to device in one pipeline:

```bash
python train.py data/train.json
```

You can skip steps using flags:
- `--skip-experts`: Skip expert training
- `--skip-gating`: Skip gating network training
- `--skip-merge`: Skip model merging
- `--skip-finetune`: Skip full fine-tuning
- `--skip-convert`: Skip GGUF conversion

Edit `data/train.json` to configure training parameters, model selection, and device push settings.

#### LoRA Training

The pipeline now supports **LoRA (Low-Rank Adaptation)** for efficient expert training. LoRA dramatically reduces the number of trainable parameters while maintaining or improving model performance.

**Benefits of LoRA:**
- **Memory Efficient**: Only trains a small number of additional parameters (typically <1% of the model)
- **Faster Training**: Reduced memory footprint allows for larger batch sizes
- **Better Generalization**: Often prevents overfitting compared to full fine-tuning
- **Modular**: LoRA adapters can be easily swapped or merged

**Configuration Example** (see `data/train_rmoe_lora.json`):

```json
{
  "expert_training": {
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

**LoRA Parameters:**
- `enabled`: Set to `true` to use LoRA (default: `true`)
- `r`: LoRA rank - lower values = fewer parameters (default: `8`)
- `alpha`: LoRA scaling factor (default: `16`)
- `dropout`: Dropout probability for LoRA layers (default: `0.05`)
- `target_modules`: List of module names to apply LoRA to, or `null` for auto-detection based on model architecture

**How it works:**
1. During expert training, LoRA adapters are added to attention and MLP layers
2. Only the LoRA adapter weights are trained, keeping the base model frozen
3. After training, the LoRA weights are automatically merged back into the model
4. The merged model is saved and used for the MoE merge step

**To disable LoRA and use traditional MLP-only fine-tuning:**
```json
{
  "expert_training": {
    "lora": {
      "enabled": false
    }
  }
}
```







