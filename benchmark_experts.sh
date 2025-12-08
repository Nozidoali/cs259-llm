#!/bin/bash
set -e

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cs259-llm

# Configuration
WORKSPACE="/home/alice/cs259-proj/cs259-llm/workspace/20251206_195554"
LLAMA_CPP_DIR="/home/alice/cs259-proj/llama.cpp"
PYTHON_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
LLAMA_BENCH="$LLAMA_CPP_DIR/build/bin/llama-bench"
QUANTIZE_TYPE="q8_0"

# Expert configurations to test
EXPERT_CONFIGS=(1 2)  # Testing 1, 2 experts per token (3 would need more experts)

# Results file
RESULTS_FILE="$WORKSPACE/benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

echo "================================================================"
echo "MoE Expert Benchmarking Script"
echo "================================================================"
echo "Workspace: $WORKSPACE"
echo "Testing num_experts_per_tok: ${EXPERT_CONFIGS[@]}"
echo "Results will be saved to: $RESULTS_FILE"
echo "================================================================"
echo ""

# Initialize results file
{
    echo "MoE Expert Performance Benchmark"
    echo "================================="
    echo "Date: $(date)"
    echo "Model: Qwen2MoE"
    echo "Quantization: $QUANTIZE_TYPE"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo ""
} > "$RESULTS_FILE"

# Function to modify config and convert
process_expert_config() {
    local num_experts=$1
    echo "================================================================"
    echo "Processing: num_experts_per_tok = $num_experts"
    echo "================================================================"
    
    # Create working directory
    local work_dir="$WORKSPACE/rmoe_qwen3_format_exp${num_experts}"
    
    # Copy the base model directory
    echo "→ Copying model files..."
    if [ -d "$work_dir" ]; then
        rm -rf "$work_dir"
    fi
    cp -r "$WORKSPACE/rmoe_qwen3_format" "$work_dir"
    
    # Modify config.json to set num_experts_per_tok
    echo "→ Modifying config.json (num_experts_per_tok = $num_experts)..."
    python3 <<EOF
import json
config_file = "$work_dir/config.json"
with open(config_file, 'r') as f:
    config = json.load(f)

config['num_experts_per_tok'] = $num_experts

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
    
print(f"✓ Updated num_experts_per_tok to {config['num_experts_per_tok']}")
EOF
    
    # Convert to GGUF
    local gguf_file="$WORKSPACE/rmoe_model_exp${num_experts}_${QUANTIZE_TYPE}.gguf"
    echo "→ Converting to GGUF: $gguf_file"
    
    if [ -f "$gguf_file" ]; then
        echo "  (Removing existing file)"
        rm -f "$gguf_file"
    fi
    
    cd "$LLAMA_CPP_DIR"
    if ! python3 "$PYTHON_SCRIPT" "$work_dir" --outfile "$gguf_file" --outtype "$QUANTIZE_TYPE" 2>&1 | tail -20; then
        echo "✗ GGUF conversion failed!"
        return 1
    fi
    
    if [ ! -f "$gguf_file" ]; then
        echo "✗ GGUF file not created!"
        return 1
    fi
    
    local gguf_size=$(du -h "$gguf_file" | cut -f1)
    echo "✓ GGUF created: $gguf_size"
    
    # Run benchmark
    echo "→ Running llama-bench..."
    echo ""
    
    local bench_output_file="$WORKSPACE/bench_exp${num_experts}_$(date +%Y%m%d_%H%M%S).txt"
    
    # Run benchmark with GPU offloading
    "$LLAMA_BENCH" \
        -m "$gguf_file" \
        -ngl 999 \
        -p 512 \
        -n 128 \
        -r 3 \
        2>&1 | tee "$bench_output_file"
    
    # Extract and save key metrics
    echo ""
    echo "→ Extracting results..."
    
    # Parse the benchmark output for performance metrics
    local pp_tokens=$(grep -A 20 "test.*pp" "$bench_output_file" | grep "avg" | awk '{print $NF}' | head -1)
    local tg_tokens=$(grep -A 20 "test.*tg" "$bench_output_file" | grep "avg" | awk '{print $NF}' | head -1)
    
    {
        echo ""
        echo "----------------------------------------------------------------"
        echo "num_experts_per_tok = $num_experts"
        echo "----------------------------------------------------------------"
        echo "GGUF file: $gguf_file"
        echo "Size: $gguf_size"
        echo ""
        echo "Performance Metrics:"
        echo "  Prompt Processing: $pp_tokens tokens/s"
        echo "  Text Generation:   $tg_tokens tokens/s"
        echo ""
        cat "$bench_output_file" | grep -A 30 "model"
        echo ""
    } >> "$RESULTS_FILE"
    
    echo "✓ Benchmark complete for num_experts_per_tok = $num_experts"
    echo ""
}

# Process each configuration
for num_experts in "${EXPERT_CONFIGS[@]}"; do
    process_expert_config $num_experts
    echo ""
done

# Display summary
echo "================================================================"
echo "BENCHMARK SUMMARY"
echo "================================================================"
cat "$RESULTS_FILE"
echo "================================================================"
echo "Full results saved to: $RESULTS_FILE"
echo "================================================================"

# Generate comparison table
echo ""
echo "Generating comparison table..."
python3 <<'PYTHON_EOF'
import re
import sys

results_file = sys.argv[1]

with open(results_file, 'r') as f:
    content = f.read()

# Extract results for each configuration
configs = []
for match in re.finditer(r'num_experts_per_tok = (\d+).*?Prompt Processing:\s+([\d.]+).*?Text Generation:\s+([\d.]+)', content, re.DOTALL):
    num_experts = int(match.group(1))
    pp_speed = float(match.group(2))
    tg_speed = float(match.group(3))
    configs.append((num_experts, pp_speed, tg_speed))

if configs:
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    print(f"{'Experts/Token':<15} {'PP Speed (t/s)':<20} {'TG Speed (t/s)':<20} {'TG Δ%':<15}")
    print("-"*70)
    
    baseline_tg = configs[0][2] if configs else None
    
    for num_exp, pp, tg in configs:
        delta = ((tg - baseline_tg) / baseline_tg * 100) if baseline_tg else 0
        delta_str = f"{delta:+.1f}%" if num_exp != configs[0][0] else "baseline"
        print(f"{num_exp:<15} {pp:<20.2f} {tg:<20.2f} {delta_str:<15}")
    
    print("="*70)
    print("\nPP = Prompt Processing, TG = Text Generation")
    print("="*70)
else:
    print("No results found in output file")
PYTHON_EOF "$RESULTS_FILE"

echo ""
echo "Done!"

