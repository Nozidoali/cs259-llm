#!/bin/bash
# Script to help locate relevant files in llama.cpp for shared expert modifications

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$(pwd)/external/llama.cpp}"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Error: llama.cpp directory not found at: $LLAMA_CPP_DIR"
    echo "Set LLAMA_CPP_DIR environment variable or ensure external/llama.cpp exists"
    exit 1
fi

echo "Searching for relevant files in: $LLAMA_CPP_DIR"
echo ""

echo "=== Conversion Script (Python) ==="
find "$LLAMA_CPP_DIR" -name "convert_hf_to_gguf.py" -o -name "*convert*.py" | head -5
echo ""

echo "=== Files containing 'shared_expert' ==="
grep -r "shared_expert" "$LLAMA_CPP_DIR" --include="*.py" --include="*.cpp" --include="*.h" | head -10
echo ""

echo "=== Files containing 'moe' (MoE related) ==="
grep -r "\bmoe\b" "$LLAMA_CPP_DIR" --include="*.cpp" --include="*.h" -l | head -10
echo ""

echo "=== GGUF metadata files ==="
find "$LLAMA_CPP_DIR" -name "*gguf*" -type f | head -10
echo ""

echo "=== Main inference files ==="
find "$LLAMA_CPP_DIR" -name "llama.cpp" -o -name "llama.h" | head -5
echo ""

echo "Done! Use these files to locate where to make modifications."

