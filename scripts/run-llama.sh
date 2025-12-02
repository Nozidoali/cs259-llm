#!/system/bin/sh
# Base directories
BASEDIR="/data/local/tmp/llama.cpp"
BINDIR="$BASEDIR/bin"
LIBDIR="$BASEDIR/lib"
MODELDIR="$BASEDIR/../gguf"
MODEL="Llama-3.2-1B-Instruct-f16.gguf"

THREADS=4
CTX_SIZE=8192
BATCH_SIZE=128

# Backend selector: cpu | gpu | dsp
BACKEND="$1"

# Clear old Snapdragon env vars
unset D M GGML_HEXAGON_NDEV GGML_HEXAGON_NHVX GGML_HEXAGON_HOSTBUF GGML_HEXAGON_VERBOSE GGML_HEXAGON_PROFILE GGML_HEXAGON_OPMASK

# Set LD paths
export LD_LIBRARY_PATH="$LIBDIR"
export ADSP_LIBRARY_PATH="$LIBDIR"

# Enable core dumps
ulimit -c unlimited

# Decide backend device
case "$BACKEND" in
  cpu)
    DEVICE="none"
    ;;
  gpu)
    DEVICE="GPUOpenCL"
    ;;
  dsp)
    DEVICE="HTP0"
    ;;
  *)
    echo "Usage: $0 [cpu|gpu|dsp]"
    exit 1
    ;;
esac

# Run llama-cli
cd "$BINDIR" || exit 1
./llama-cli \
    -m "$MODELDIR/$MODEL" \
    -t "$THREADS" \
    --ctx-size "$CTX_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --no-mmap \
    -ctk q8_0 -ctv q8_0 -fa on @ \