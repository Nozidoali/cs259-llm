# CS259 - Qwen2 Fine-tuning and LongBench Evaluation

## Setup

1. **Run the setup script to create a clean conda environment:**
```bash
bash setup.sh
```

This will:
- Remove any existing `snapdragon` environment
- Create a fresh `snapdragon` environment with Python 3.9
- Install PyTorch via conda (critical for macOS to avoid threading issues)
- Install TensorFlow Metal for Apple Silicon GPU acceleration (speeds up BLEURT significantly)
- Install BLEURT and all other dependencies

1. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and set:
- `WORK_DIR`: Working directory for models, data, and logs (default: current directory)
- `LLAMA_CPP_DIR`: Path to llama.cpp directory containing `convert_hf_to_gguf.py` (required for GGUF conversion)

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
- `--skip-finetune`: Skip fine-tuning
- `--skip-convert`: Skip GGUF conversion
- `--skip-push`: Skip device push

Edit `data/train.json` to configure training parameters, model selection, and device push settings.





