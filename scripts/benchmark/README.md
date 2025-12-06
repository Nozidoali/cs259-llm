# Benchmark Scripts

This directory contains scripts for benchmarking model performance and visualizing the results.

## Files

- **`benchmark_models.py`** - Main benchmarking script that tests model performance
- **`plot_benchmarks.py`** - Visualization script that creates plots from benchmark results
- **`benchmarks.csv`** - CSV file containing benchmark results
- **`benchmark_plot.png`** - Generated visualization plot
- **`QUICKSTART_BENCHMARK.md`** - Quick start guide for running benchmarks

## Usage

### Running Benchmarks

To run benchmarks on models:

```bash
cd scripts/benchmark
python benchmark_models.py
```

### Generating Plots

To visualize the benchmark results:

```bash
cd scripts/benchmark
python plot_benchmarks.py
```

This will:
- Read data from `benchmarks.csv`
- Generate a 2D scatter plot with:
  - X-axis: Geometric mean of prefill and decode tokens per second
  - Y-axis: Model size in MB
  - Blue reference lines for Llama 3.2 1B Q8_0 baseline
  - Square markers colored by model family
  - Automatic congestion detection to avoid overlapping labels
- Save the output to `benchmark_plot.png`

### Plot Features

- **Automatic label filtering**: Points that are too close together have their labels automatically hidden to prevent overlap
- **Legend with parameter sizes**: Model families are shown in the legend with their parameter sizes
- **Blue baseline**: The reference model (Llama 3.2 1B Q8_0) has blue text and reference lines
- **Wide format**: 20x8 inch figure optimized for presentations

## Output

The benchmark results are saved to `benchmarks.csv` with the following columns:
- `model_name` - Name of the model
- `param_size` - Parameter size (e.g., "1B", "3B")
- `quantization` - Quantization method used
- `layers` - Number of layers
- `model_size_mb` - Model size in megabytes
- `prefill_tokens_per_sec` - Prefill speed
- `decode_tokens_per_sec` - Decode speed
- `hf_repo` - Hugging Face repository
- `filename` - Model filename
- `timestamp` - Benchmark timestamp
