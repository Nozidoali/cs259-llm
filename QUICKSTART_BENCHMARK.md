# Quick Start Guide - Model Benchmarking

## What Was Created

This benchmarking suite includes:

1. **`benchmark_models.py`** - Main benchmarking script
2. **`list_available_models.py`** - Helper to check HuggingFace repos
3. **`check_benchmark_setup.sh`** - Verify your setup
4. **`BENCHMARK_README.md`** - Full documentation

## Models Included (30+ variants)

### Small Models (100M-3B)
- **GPT-2**: 124M, 355M, 774M
- **Qwen 2.5**: 0.5B, 1.5B, 3B
- **Llama 3.2**: 1B, 3B
- **Phi**: 2.7B, 3.8B

### Large Models (7B-8B) ✨ NEW
- **Qwen 2.5**: 7B
- **Mistral**: 7B v0.3
- **Llama 3.1**: 8B

### Quantization Levels
- `f16` - Full precision (selected models)
- `Q8_0` - 8-bit quantization
- `Q4_K_M` - 4-bit quantization

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Verify installation
huggingface-cli --version
```

### 2. Connect Device

```bash
# Check device connection
adb devices

# Should show:
# List of devices attached
# ABC123      device
```

### 3. Run Setup Check

```bash
./check_benchmark_setup.sh
```

### 4. Test with Debug Mode

```bash
# Test with smallest model (GPT-2, ~2 minutes)
python3 benchmark_models.py --debug
```

### 5. Run Full Benchmark

```bash
# Run all models (several hours)
python3 benchmark_models.py
```

## Key Features

✅ **Automatic Download** - Downloads models from HuggingFace  
✅ **Device Push** - Pushes models to Android via ADB  
✅ **Benchmark** - Runs llama-bench to measure speed  
✅ **CSV Output** - Saves results with metadata  
✅ **Auto Cleanup** - Removes models from device after testing  
✅ **Incremental Save** - Results saved after each model  
✅ **Debug Mode** - Test one model before running all

## Expected Output

### CSV File: `benchmark_results_YYYYMMDD_HHMMSS.csv`

| model_name   | param_size | quantization | layers | model_size_mb | prefill_tokens_per_sec | decode_tokens_per_sec |
| ------------ | ---------- | ------------ | ------ | ------------- | ---------------------- | --------------------- |
| gpt2         | 124M       | Q4_K_M       | 12     | 89.4          | 45.23                  | 12.34                 |
| llama-3.2-1b | 1B         | Q4_K_M       | 16     | 638.4         | 38.67                  | 10.92                 |
| qwen2.5-7b   | 7B         | q4_k_m       | 28     | 4200.1        | 15.34                  | 5.67                  |
| ...          | ...        | ...          | ...    | ...           | ...                    | ...                   |

### Log File: `logs/benchmark_YYYYMMDD_HHMMSS.log`

Detailed logs of every operation including:
- Download progress
- Push status
- Benchmark stdout/stderr
- Parsing results
- Errors and warnings

## Command Options

### Basic

```bash
# Run all models
python3 benchmark_models.py

# Debug mode (single model)
python3 benchmark_models.py --debug
```

### Advanced

```bash
# Custom model directory
python3 benchmark_models.py --models-dir ./my_models

# Specific ADB device
python3 benchmark_models.py --adb-serial 1A2B3C4D

# Custom device path
python3 benchmark_models.py --device-path /sdcard/models/

# Adjust batch size
python3 benchmark_models.py --batch-size 256

# Longer timeout for slow devices
python3 benchmark_models.py --bench-timeout 600
```

## Time Estimates

| Model Size   | Quant  | Download | Push | Benchmark | Total  |
| ------------ | ------ | -------- | ---- | --------- | ------ |
| 124M (GPT-2) | Q4_K_M | 30s      | 10s  | 2-3min    | ~4min  |
| 1B (Llama)   | Q4_K_M | 1min     | 30s  | 3-5min    | ~7min  |
| 3B (Qwen)    | Q4_K_M | 2min     | 1min | 5-8min    | ~12min |
| 7B (Mistral) | Q4_K_M | 3-5min   | 2min | 8-12min   | ~20min |
| 7B (Qwen)    | Q8_0   | 5-8min   | 3min | 10-15min  | ~28min |

**Full suite**: ~4-8 hours depending on network and device speed

## Troubleshooting

### Download Fails

```bash
# Manually download
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --local-dir ./models/benchmark
```

### ADB Issues

```bash
# Check device space
adb shell df -h /data/local/tmp

# Manual push test
bash scripts/push-model.sh models/benchmark/test.gguf /data/local/tmp/gguf/

# Reset ADB
adb kill-server && adb start-server
```

### Benchmark Timeout

```bash
# Increase timeout for 7B models
python3 benchmark_models.py --bench-timeout 900
```

### Out of Space on Device

The script automatically removes models after benchmarking, but if interrupted:

```bash
# Clear device storage
adb shell rm -rf /data/local/tmp/gguf/*.gguf
```

## Storage Requirements

### Local Machine
- **~10-15GB** for all model downloads
- Models persist for future runs

### Android Device
- **~100MB-5GB** temporarily during benchmark
- Automatically cleaned up after each model

## Tips

1. **Start with debug mode** - Verify everything works
2. **Run overnight** - Full suite takes hours
3. **Monitor device temperature** - Pause if overheating
4. **Check logs** - `logs/benchmark_*.log` for issues
5. **CSV updates incrementally** - Safe to interrupt

## What's Next?

After running benchmarks:

1. **Analyze CSV** - Compare speeds across models
2. **Plot results** - Visualize performance vs size
3. **Pick optimal model** - Balance speed and quality
4. **Production deployment** - Use best model for your app

## Support

- **Full docs**: `BENCHMARK_README.md`
- **Check setup**: `./check_benchmark_setup.sh`
- **List models**: `python3 list_available_models.py`
- **View logs**: `tail -f logs/benchmark_*.log`

## Summary

```bash
# 1. Install
pip install huggingface_hub[cli]

# 2. Check setup
./check_benchmark_setup.sh

# 3. Debug test
python3 benchmark_models.py --debug

# 4. Full run
python3 benchmark_models.py

# 5. View results
open benchmark_results_*.csv
```

Happy benchmarking! 🚀
