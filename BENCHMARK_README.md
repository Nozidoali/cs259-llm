# Multi-Model Benchmarking Guide

This guide explains how to use the `benchmark_models.py` script to benchmark multiple GGUF models on Android devices using llama-bench.

## Overview

The benchmarking script automates the following pipeline:
1. **Download** GGUF models from HuggingFace
2. **Push** models to Android device via ADB
3. **Run** llama-bench to measure prefill and decode speeds
4. **Save** results to CSV with metadata
5. **Cleanup** by removing models from device

## Prerequisites

### 1. Install Dependencies

```bash
# Install Python requirements
pip install huggingface_hub[cli]

# Verify huggingface-cli is available
huggingface-cli --version

# Verify adb is available
adb devices
```

### 2. Connect Android Device

Make sure your Android device is connected via ADB:

```bash
# Check device connection
adb devices

# If multiple devices, note the serial number
adb devices -l
```

### 3. Verify Scripts

Ensure these scripts exist:
- `scripts/push-model.sh` - Pushes models to device
- `scripts/run-bench.sh` - Runs llama-bench on device

## Usage

### Basic Usage

Run all models with default settings:

```bash
python benchmark_models.py
```

### Debug Mode (Recommended First Run)

Test with the smallest model (GPT-2) first:

```bash
python benchmark_models.py --debug
```

This will only benchmark GPT-2 with Q4_K_M quantization to verify everything works.

### Common Options

```bash
# Specify custom model download directory
python benchmark_models.py --models-dir ./my_models

# Use specific ADB device (if multiple devices connected)
python benchmark_models.py --adb-serial 1A2B3C4D

# Change device path on Android
python benchmark_models.py --device-path /sdcard/models/

# Adjust batch size for benchmarks
python benchmark_models.py --batch-size 256

# Increase timeout for slower devices
python benchmark_models.py --bench-timeout 600
```

### Full Example

```bash
python benchmark_models.py \
  --models-dir ./downloaded_models \
  --device-path /data/local/tmp/gguf/ \
  --adb-serial 1A2B3C4D \
  --device none \
  --batch-size 128 \
  --bench-timeout 300
```

## Models Included

The script benchmarks the following models:

### Llama 3.2
- 1B Instruct (Q4_K_M, Q8_0, f16)
- 3B Instruct (Q4_K_M, Q8_0)

### Llama 3.1
- 8B Instruct (Q4_K_M, Q8_0)

### Qwen 2.5
- 0.5B Instruct (q4_k_m, q8_0, f16)
- 1.5B Instruct (q4_k_m, q8_0)
- 3B Instruct (q4_k_m, q8_0)
- **7B Instruct (q4_k_m, q8_0)**

### Mistral
- **7B Instruct v0.3 (Q4_K_M, Q8_0)**

### Phi
- Phi-3.5-mini (Q4_K_M, Q8_0)
- Phi-2 (Q4_K_M, Q8_0)

### GPT-2
- GPT-2 (124M) (Q4_K_M, Q8_0)
- GPT-2 Medium (355M) (Q4_K_M, Q8_0)
- GPT-2 Large (774M) (Q4_K_M, Q8_0)

**Total: ~30+ model variants** (including 7B models)

## Output Files

### CSV Results

Results are saved to `benchmark_results_YYYYMMDD_HHMMSS.csv` with the following columns:

- `model_name` - Model identifier
- `param_size` - Number of parameters (e.g., "1B", "124M")
- `quantization` - Quantization level (Q4_K_M, Q8_0, f16)
- `layers` - Number of layers in the model
- `model_size_mb` - File size in megabytes
- `prefill_tokens_per_sec` - Prefill throughput
- `decode_tokens_per_sec` - Decode throughput
- `hf_repo` - HuggingFace repository
- `filename` - GGUF filename
- `timestamp` - ISO timestamp of benchmark

### Log Files

Detailed logs are saved to `logs/benchmark_YYYYMMDD_HHMMSS.log`

## Troubleshooting

### Model Download Issues

If a model fails to download:

```bash
# Manually download using huggingface-cli
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir ./models/benchmark

# List available files in a repo
python list_available_models.py --model llama-3.2-1b
```

### ADB Push Failures

If pushing to device fails:

```bash
# Check available space on device
adb shell df -h /data/local/tmp

# Manually test push
bash scripts/push-model.sh models/benchmark/model.gguf /data/local/tmp/gguf/
```

### Benchmark Timeout

If benchmarks timeout on slower devices:

```bash
# Increase timeout (default 300s)
python benchmark_models.py --bench-timeout 600
```

### Permission Errors on Device

```bash
# Ensure device directory is writable
adb shell mkdir -p /data/local/tmp/gguf
adb shell chmod 777 /data/local/tmp/gguf
```

## Customization

### Adding New Models

Edit `benchmark_models.py` and add to `MODEL_CONFIGS`:

```python
{
    "name": "your-model-name",
    "hf_repo": "username/repo-name-GGUF",
    "quantizations": ["Q4_K_M", "Q8_0"],
    "param_size": "1.5B",
    "layers": 28,
}
```

### Changing Quantization Levels

Modify the `quantizations` list for each model in `MODEL_CONFIGS`.

Common quantization formats:
- `f16` - Full 16-bit precision
- `Q8_0` - 8-bit quantization
- `Q4_K_M` - 4-bit quantization (K-quants, medium)
- `Q4_K_S` - 4-bit quantization (K-quants, small)
- `Q4_0` - 4-bit quantization (legacy)

## Performance Tips

1. **Run overnight** - Full benchmark suite takes several hours
2. **Monitor device** - Ensure device doesn't overheat
3. **Use debug mode** - Test one model first before full run
4. **Check disk space** - Models can be 100MB-5GB each
5. **Incremental saving** - Results saved after each model (safe to interrupt)

## Example Output

```
Progress: 5/25
================================================================================
Benchmarking: llama-3.2-1b (Q4_K_M)
================================================================================
Model already exists: Llama-3.2-1B-Instruct-Q4_K_M.gguf (638.42 MB)
Pushing Llama-3.2-1B-Instruct-Q4_K_M.gguf to device...
Successfully pushed Llama-3.2-1B-Instruct-Q4_K_M.gguf
Running benchmark for Llama-3.2-1B-Instruct-Q4_K_M.gguf...
Benchmark results - Prefill: 45.23 t/s, Decode: 12.34 t/s
Removing Llama-3.2-1B-Instruct-Q4_K_M.gguf from device...
✓ Completed llama-3.2-1b (Q4_K_M)
```

## Command Reference

```bash
# Show help
python benchmark_models.py --help

# Debug mode (smallest model only)
python benchmark_models.py --debug

# List available models in HF repos
python list_available_models.py

# Check specific model
python list_available_models.py --model gpt2
```

## Notes

- **Disk Space**: Ensure ~10GB free space for model downloads
- **Device Storage**: Each model needs space on device during benchmark
- **Time Estimate**: ~5-10 minutes per model variant
- **Network**: Requires stable internet for downloads
- **Results**: CSV is updated after each successful benchmark

## Support

If you encounter issues:
1. Check logs in `logs/benchmark_*.log`
2. Run with `--debug` flag first
3. Verify ADB connection: `adb devices`
4. Check HuggingFace CLI: `huggingface-cli --version`
