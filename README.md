# CS259 - Qwen2 Fine-tuning and LongBench Evaluation

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Fine-tune Qwen2 Models

```bash
# Custom training parameters
python finetune.py --model qwen2-0.5b \
    --num-epochs 5 \
    --batch-size 2 \
    --learning-rate 2e-5 \
    --max-length 2048 \
    --use-bleurt
```

### Convert to GGUF

```bash
# Custom output path
python convert.py \
    --model models/qwen2-0.5b-instruct-finetuned \
    --output models/gguf/qwen2-0.5b-Q8_0.gguf \
    --quantize tq1_0
```

### Push to Device

```bash
adb push models/gguf/qwen2-0.5b-instruct-finetuned.gguf /data/local/tmp/gguf/
```

Update `run-cli.sh` line 13 with your model filename.

### Run Benchmarks

**LongBench:**
```bash
python longbench_test.py
python longbench_eval.py
python parse_log.py debug.log
```

**TruthfulQA:**
```bash
python truthful_qa_eval.py
```
