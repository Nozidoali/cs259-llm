#!/bin/bash
set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/alice/cs259-proj/llama.cpp}"
LLAMA_BENCH="$LLAMA_CPP_DIR/build/bin/llama-bench"
PYTHON_SCRIPT="$SCRIPT_DIR/train.py"

# Default options
CONVERSION_MODE="preserve_moe"
SKIP_PIPELINE=false
BENCH_ONLY=false
WORKSPACE_TIMESTAMP=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --conversion-mode)
            CONVERSION_MODE="$2"
            shift 2
            ;;
        --skip-pipeline)
            SKIP_PIPELINE=true
            shift
            ;;
        --bench-only)
            BENCH_ONLY=true
            SKIP_PIPELINE=true
            shift
            ;;
        --workspace-timestamp)
            WORKSPACE_TIMESTAMP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --conversion-mode MODE    Conversion mode: 'preserve_moe' or 'average' (default: preserve_moe)"
            echo "  --skip-pipeline          Skip running the pipeline, only run benchmarks on existing workspaces"
            echo "  --bench-only             Only run benchmarks (same as --skip-pipeline)"
            echo "  --workspace-timestamp TS Use existing workspace with timestamp TS (e.g., 20251206_081921)"
            echo "                            When set, train.py will use experts from this workspace when --skip-experts is used"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "This script runs the full pipeline for train_qwen2_1.json through train_qwen2_5.json"
            echo "and profiles each with llama-bench for prefill and decode performance."
            echo ""
            echo "Examples:"
            echo "  # Use existing experts from workspace 20251206_081921"
            echo "  $0 --workspace-timestamp 20251206_081921"
            echo ""
            echo "  # Use existing workspace and only run benchmarks"
            echo "  $0 --workspace-timestamp 20251206_081921 --bench-only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if llama-bench exists
if [ ! -f "$LLAMA_BENCH" ]; then
    echo "Error: llama-bench not found at $LLAMA_BENCH"
    echo "Please build llama.cpp or set LLAMA_CPP_DIR environment variable"
    exit 1
fi

# Check if train.py exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: train.py not found at $PYTHON_SCRIPT"
    exit 1
fi

# Results directory
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/pipeline_benchmark_${TIMESTAMP}.txt"
SUMMARY_JSON="$RESULTS_DIR/pipeline_benchmark_${TIMESTAMP}.json"

# Initialize results
{
    echo "================================================================"
    echo "Full Pipeline Benchmark Results"
    echo "================================================================"
    echo "Date: $(date)"
    echo "Configs: train_qwen2_1.json through train_qwen2_5.json"
    echo "================================================================"
    echo ""
} > "$RESULTS_FILE"

# Initialize JSON summary
echo "[]" > "$SUMMARY_JSON"

# Function to extract workspace timestamp from train.py output
extract_workspace_timestamp() {
    local log_file="$1"
    # Look for "Work directory: .../workspace/YYYYMMDD_HHMMSS" pattern
    local timestamp=$(grep -oP "Work directory:.*workspace/\K[0-9]{8}_[0-9]{6}" "$log_file" | head -1)
    if [ -z "$timestamp" ]; then
        # Try alternative pattern: "Using timestamp from WORKSPACE_TIMESTAMP"
        timestamp=$(grep -oP "Using timestamp.*:\s*\K[0-9]{8}_[0-9]{6}" "$log_file" | head -1)
    fi
    echo "$timestamp"
}

# Function to find GGUF file in workspace
find_gguf_file() {
    local workspace_dir="$1"
    local conversion_mode="$2"
    
    # Look for GGUF files in the workspace directory
    # Priority based on conversion mode
    local gguf_file=""
    
    if [ "$conversion_mode" = "preserve_moe" ]; then
        # Look for preserve_moe format first
        gguf_file=$(find "$workspace_dir" -name "rmoe_model_moe_*.gguf" -type f | head -1)
    fi
    
    # Fallback to any rmoe_model GGUF file
    if [ -z "$gguf_file" ]; then
        gguf_file=$(find "$workspace_dir" -name "rmoe_model_*.gguf" -type f | head -1)
    fi
    
    # Last resort: any GGUF file in workspace
    if [ -z "$gguf_file" ]; then
        gguf_file=$(find "$workspace_dir" -name "*.gguf" -type f | head -1)
    fi
    
    echo "$gguf_file"
}

# Function to run llama-bench and extract results
run_benchmark() {
    local gguf_file="$1"
    local config_name="$2"
    local bench_output_file="$RESULTS_DIR/llama_bench_${config_name}_${TIMESTAMP}.txt"
    
    echo "  → Running llama-bench on: $(basename "$gguf_file")" >&2
    
    # Run llama-bench with GPU offloading
    # -m: model file
    # -ngl: number of GPU layers (999 = all layers)
    # -p: prompt tokens (prefill test)
    # -n: generation tokens (decode test)
    # -r: number of repetitions
    # -o: output format (json for easier parsing)
    # Save output directly to log file (no tee, no stripping)
    "$LLAMA_BENCH" \
        -m "$gguf_file" \
        -ngl 999 \
        -p 512 \
        -n 128 \
        -r 3 \
        -o json \
        > "$bench_output_file" 2>&1
    
    # Return the output file path
    echo "$bench_output_file"
}

# Function to process one config
process_config() {
    local config_file="$1"
    local config_name=$(basename "$config_file" .json)
    
    echo ""
    echo "================================================================"
    echo "Processing: $config_name"
    echo "================================================================"
    echo "Config: $config_file"
    echo ""
    
    # Create a temporary log file for this run
    local temp_log="$RESULTS_DIR/${config_name}_${TIMESTAMP}.log"
    
    # Determine source workspace for experts and create new workspace for output
    local source_workspace_dir=""
    local new_workspace_timestamp=""
    
    if [ -n "$WORKSPACE_TIMESTAMP" ]; then
        source_workspace_dir="$SCRIPT_DIR/workspace/$WORKSPACE_TIMESTAMP"
        if [ ! -d "$source_workspace_dir" ]; then
            echo "✗ Source workspace not found: $source_workspace_dir"
            return 1
        fi
        echo "→ Using experts from workspace: $WORKSPACE_TIMESTAMP"
        echo "  Source workspace: $source_workspace_dir"
        
        # Create a new workspace for this config (don't set WORKSPACE_TIMESTAMP)
        new_workspace_timestamp=$(date +%Y%m%d_%H%M%S)_${config_name}
        local new_workspace_dir="$SCRIPT_DIR/workspace/$new_workspace_timestamp"
        mkdir -p "$new_workspace_dir"
        
        # Create symlinks to experts from source workspace
        echo "→ Creating symlinks to experts in new workspace..."
        mkdir -p "$new_workspace_dir/experts"
        for expert_dir in "$source_workspace_dir/experts"/*; do
            if [ -d "$expert_dir" ]; then
                local expert_name=$(basename "$expert_dir")
                ln -sf "$expert_dir" "$new_workspace_dir/experts/$expert_name"
                echo "  Linked expert: $expert_name"
            fi
        done
        
        # Set WORKSPACE_TIMESTAMP to the new workspace so train.py uses it
        export WORKSPACE_TIMESTAMP="$new_workspace_timestamp"
        echo "  New workspace: $new_workspace_timestamp"
    fi
    
    # Run train.py with skip-experts flag (experts are symlinked in new workspace)
    echo "→ Running train.py with --skip-experts..."
    echo "  Conversion mode: $CONVERSION_MODE"
    
    if ! python3 "$PYTHON_SCRIPT" "$config_file" --skip-experts --conversion-mode "$CONVERSION_MODE" 2>&1 | tee "$temp_log"; then
        echo "✗ Pipeline failed for $config_name"
        {
            echo ""
            echo "----------------------------------------------------------------"
            echo "Config: $config_name"
            echo "Status: FAILED"
            echo "----------------------------------------------------------------"
            echo ""
        } >> "$RESULTS_FILE"
        return 1
    fi
    
    echo ""
    echo "→ Pipeline completed, extracting workspace information..."
    
    # Determine workspace directory
    local workspace_dir=""
    
    if [ -n "$new_workspace_timestamp" ]; then
        # Use the new workspace we created
        workspace_dir="$SCRIPT_DIR/workspace/$new_workspace_timestamp"
    else
        # Extract from log or use WORKSPACE_TIMESTAMP
        local extracted_dir=$(grep -oP "Work directory:\s*\K[^\s]+" "$temp_log" | tail -1)
        if [ -n "$extracted_dir" ] && [ -d "$extracted_dir" ]; then
            workspace_dir="$extracted_dir"
            echo "  Found workspace from log: $workspace_dir"
        elif [ -n "$WORKSPACE_TIMESTAMP" ]; then
            workspace_dir="$SCRIPT_DIR/workspace/$WORKSPACE_TIMESTAMP"
        else
            # Fallback: extract workspace timestamp
            local workspace_timestamp=$(extract_workspace_timestamp "$temp_log")
            if [ -z "$workspace_timestamp" ]; then
                echo "  Warning: Could not extract workspace timestamp, trying to find latest workspace..."
                local latest_workspace=$(find "$SCRIPT_DIR/workspace" -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]*" | sort | tail -1)
                if [ -n "$latest_workspace" ]; then
                    workspace_dir="$latest_workspace"
                    echo "  Using latest workspace: $workspace_dir"
                else
                    echo "✗ Could not find workspace directory"
                    return 1
                fi
            else
                workspace_dir="$SCRIPT_DIR/workspace/$workspace_timestamp"
            fi
        fi
    fi
    if [ ! -d "$workspace_dir" ]; then
        echo "✗ Workspace directory not found: $workspace_dir"
        return 1
    fi
    
    echo "  Workspace: $workspace_dir"
    
    # Get number of experts from config for reporting
    local num_experts=$(python3 <<EOF
import json
try:
    with open("$config_file", 'r') as f:
        config = json.load(f)
    datasets = config.get("datasets", [])
    print(len(datasets))
except:
    print("N/A")
EOF
)
    
    # Find GGUF file
    local gguf_file=$(find_gguf_file "$workspace_dir" "$CONVERSION_MODE")
    if [ -z "$gguf_file" ] || [ ! -f "$gguf_file" ]; then
        echo "✗ GGUF file not found in workspace: $workspace_dir"
        echo "  Searched for: rmoe_model_moe_*.gguf, rmoe_model_*.gguf, *.gguf"
        {
            echo ""
            echo "----------------------------------------------------------------"
            echo "Config: $config_name"
            echo "Status: GGUF FILE NOT FOUND"
            echo "Workspace: $workspace_dir"
            echo "Conversion mode: $CONVERSION_MODE"
            echo "----------------------------------------------------------------"
            echo ""
        } >> "$RESULTS_FILE"
        return 1
    fi
    
    local gguf_size=$(du -h "$gguf_file" | cut -f1)
    echo "  ✓ Found GGUF file: $(basename "$gguf_file") ($gguf_size)"
    
    # Run benchmark
    echo ""
    echo "→ Running llama-bench..."
    local bench_output_file=$(run_benchmark "$gguf_file" "$config_name")
    
    # Save results
    {
        echo ""
        echo "----------------------------------------------------------------"
        echo "Config: $config_name"
        echo "Status: SUCCESS"
        echo "----------------------------------------------------------------"
        echo "Workspace: $workspace_dir"
        echo "Number of Experts: $num_experts"
        echo "GGUF File: $gguf_file"
        echo "Size: $gguf_size"
        echo "Benchmark Output: $bench_output_file"
        echo ""
    } >> "$RESULTS_FILE"
    
    # Update JSON summary
    python3 <<EOF
import json
import sys

summary_file = "$SUMMARY_JSON"
config_name = "$config_name"
gguf_file = "$gguf_file"
workspace_dir = "$workspace_dir"
bench_output_file = "$bench_output_file"

try:
    with open(summary_file, 'r') as f:
        results = json.load(f)
except:
    results = []

result_entry = {
    "config": config_name,
    "workspace": workspace_dir,
    "num_experts": int("$num_experts") if "$num_experts" != "N/A" else None,
    "gguf_file": gguf_file,
    "benchmark_output_file": bench_output_file,
    "status": "success"
}

results.append(result_entry)

with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)
EOF
    
    echo "✓ Benchmark complete for $config_name"
    echo ""
}

# Main execution
echo "================================================================"
echo "Full Pipeline Benchmark Script"
echo "================================================================"
echo "Data directory: $DATA_DIR"
echo "Results directory: $RESULTS_DIR"
echo "llama-bench: $LLAMA_BENCH"
echo "Conversion mode: $CONVERSION_MODE"
if [ -n "$WORKSPACE_TIMESTAMP" ]; then
    echo "Workspace timestamp: $WORKSPACE_TIMESTAMP (using existing experts from this workspace)"
fi
if [ "$SKIP_PIPELINE" = true ]; then
    echo "Mode: BENCHMARK ONLY (skipping pipeline)"
fi
echo "================================================================"
echo ""

# Process each config file
for i in {1..5}; do
    config_file="$DATA_DIR/train_qwen2_${i}.json"
    if [ ! -f "$config_file" ]; then
        echo "Warning: Config file not found: $config_file"
        continue
    fi
    process_config "$config_file"
done

# Generate summary
echo ""
echo "================================================================"
echo "BENCHMARK SUMMARY"
echo "================================================================"
cat "$RESULTS_FILE"
echo ""
echo "================================================================"
echo "JSON Summary: $SUMMARY_JSON"
echo "Full Results: $RESULTS_FILE"
echo "================================================================"

# Generate comparison table
echo ""
echo "Generating comparison table..."
python3 <<'PYTHON_EOF'
import json
import sys

summary_file = sys.argv[1]

try:
    with open(summary_file, 'r') as f:
        results = json.load(f)
    
    if results:
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY TABLE")
        print("="*80)
        print(f"{'Config':<20} {'Experts':<8} {'Benchmark Output File':<50} {'Status':<10}")
        print("-"*80)
        
        for result in results:
            config = result.get('config', 'unknown')
            num_experts = result.get('num_experts', 'N/A')
            bench_file = result.get('benchmark_output_file', 'N/A')
            status = result.get('status', 'unknown')
            
            # Truncate benchmark file path for display
            bench_display = bench_file.split('/')[-1] if bench_file != 'N/A' else 'N/A'
            experts_str = str(num_experts) if num_experts != 'N/A' else "N/A"
            
            print(f"{config:<20} {experts_str:<8} {bench_display:<50} {status:<10}")
        
        print("="*80)
        print("\nBenchmark output files contain full llama-bench results in JSON format")
        print("="*80)
    else:
        print("No results found")
except Exception as e:
    print(f"Error generating table: {e}")
PYTHON_EOF
"$SUMMARY_JSON"

echo ""
echo "Done!"

