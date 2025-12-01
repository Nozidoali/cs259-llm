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
- Install all required dependencies

1. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and set:
- `WORK_DIR`: Working directory for models, data, and logs (default: current directory)
- `LLAMA_CPP_DIR`: Path to llama.cpp directory containing `convert_hf_to_gguf.py` (required for GGUF conversion)

### Fine-tune Models

```bash
# Custom training parameters
python src/finetune.py --model qwen2-0.5b \
    --num-epochs 5 \
    --batch-size 2 \
    --learning-rate 2e-5 \
    --max-length 2048 \
    --use-bleurt

# Or use the MoE model
python src/finetune.py --model qwen1.5-moe-a2.7b \
    --num-epochs 3 \
    --batch-size 1
```

### Convert to GGUF

```bash
python src/convert.py \
    --model models/qwen2-0.5b-instruct-finetuned \
    --output models/gguf/qwen2-0.5b-Q8_0.gguf \
    --quantize tq1_0
```

### Push to Device

```bash
adb push models/gguf/qwen2-0.5b-instruct-finetuned.gguf /data/local/tmp/gguf/
```

Update `scripts/run-cli.sh` line 13 with your model filename.

### Run Benchmarks

**LongBench:**
```bash
python src/longbench_test.py
python src/longbench_eval.py
python src/parse_log.py debug.log
```

**TruthfulQA:**
```bash
python src/truthful_qa_eval.py
```

