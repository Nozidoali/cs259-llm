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
- `--skip-finetune`: Skip fine-tuning
- `--skip-convert`: Skip GGUF conversion
- `--skip-push`: Skip device push

Edit `data/train.json` to configure training parameters, model selection, and device push settings.







