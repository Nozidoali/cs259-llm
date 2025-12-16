#!/bin/bash
#
# Script to run llama-cli on NVIDIA RTX 3090 with CUDA support

# Base directory - adjust if your llama.cpp is in a different location
# If llama-cli is in your PATH, you can leave this empty
basedir="${LLAMA_CPP_DIR:-}"
[ -z "$basedir" ] && basedir="/home/alice/cs259-proj/llama.cpp/build"

cli_opts=

# Model path - adjust to your model location
modeldir="${MODEL_DIR:-$(dirname "$0")/../models/gguf}"
model="qmsum_1626-q4_0.gguf"
[ "$M" != "" ] && model="$M"

# Device: use "cuda" for CUDA GPU, "cuda:0" for specific GPU, or "none" for CPU
device="CUDA0"
[ "$D" != "" ] && device="$D"

# Number of GPU layers to offload (99 = offload as many as possible)
ngl=0
[ "$NGL" != "" ] && ngl="$NGL"

# Threads (CPU threads, typically number of physical cores)
threads=8
[ "$T" != "" ] && threads="$T"

# Verbose output
verbose=
[ "$V" != "" ] && verbose="-v"

set -x

# Build the command
cmd="llama-cli"

# If basedir is set and llama-cli exists there, use it
if [ -n "$basedir" ] && [ -f "$basedir/bin/llama-cli" ]; then
    cmd="$basedir/bin/llama-cli"
    export LD_LIBRARY_PATH="$basedir/lib:${LD_LIBRARY_PATH:-}"
elif [ -n "$basedir" ] && [ -f "$basedir/llama-cli" ]; then
    cmd="$basedir/llama-cli"
fi

# Run llama-cli locally
$cmd \
    -m "$modeldir/$model" \
    -t "$threads" \
    --mlock \
    --ctx-size 1024 \
    --batch-size 128 \
    -ctk q8_0 -ctv q8_0 \
    --temp 0.5 \
    -n 128 \
    --ignore-eos \
    --seed 42 \
    --no-display-prompt \
    -fa on \
    -ngl "$ngl" \
    --device "$device" \
    $verbose \
    $cli_opts \
    "$@"

# $cmd \
#     -m "$modeldir/$model" \
#     -t "$threads" \
#     --mlock \
#     --ctx-size 32768 \
#     --temp 0.5 \
#     --top-p 0.7 \
#     --seed 42 \
#     --no-display-prompt \
#     -fa on \
#     -ngl "$ngl" \
#     --device "$device" \
#     $verbose \
#     $cli_opts \
#     "$@"

# $cmd \
#   -m "$modeldir/$model" \
#   -t 8 \
#   --ctx-size 4096 \
#   --temp 0.5 \
#   --top-p 0.7 \
#   --seed 42 \
#   -n 20 \
#   -fa off \
#   --device "$device" \
#   "$@"