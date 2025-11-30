# CS259 - Qwen2 Fine-tuning and LongBench Evaluation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and set:
- `WORK_DIR`: Working directory for models, data, and logs (default: current directory)
- `LLAMA_CPP_DIR`: Path to llama.cpp directory containing `convert_hf_to_gguf.py` (required for GGUF conversion)

## Usage

All Python scripts are in the `src/` directory. **Always run commands from the project root directory.**

Run scripts using:
```bash
python src/script_name.py
```

**Note:** Scripts reference paths relative to the project root (e.g., `./models/`, `./scripts/`, `./prompt_files/`), so running from the project root is required.

### Fine-tune Models

```bash
# Custom training parameters
python src/finetune.py --model qwen2-0.5b \
    --num-epochs 5 \
    --batch-size 2 \
    --learning-rate 2e-5 \
    --max-length 2048 \
    --use-bleurt
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
