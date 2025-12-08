#!/system/bin/sh
#

# Basedir on device
BASEDIR="/data/local/tmp/llama.cpp"
BINDIR="$BASEDIR/bin"
LIBDIR="$BASEDIR/lib"
MODELDIR="$BASEDIR/../gguf"
MODEL="epoch_100_truthfulqa_q8_0.gguf"

THREADS=6
BATCH_SIZE=128

# Backend selector: cpu | gpu | dsp (optional, defaults to dsp)
BACKEND="cpu"
if [ "$1" = "cpu" ] || [ "$1" = "gpu" ] || [ "$1" = "dsp" ]; then
    BACKEND="$1"
    shift
fi

unset D M GGML_HEXAGON_NDEV GGML_HEXAGON_NHVX GGML_HEXAGON_HOSTBUF GGML_HEXAGON_VERBOSE GGML_HEXAGON_PROFILE GGML_HEXAGON_OPMASK

export LD_LIBRARY_PATH="$LIBDIR"
export ADSP_LIBRARY_PATH="$LIBDIR"

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
esac

cd "$BINDIR" || exit 1
./llama-bench \
    -m "$MODELDIR/$MODEL" \
    -t "$THREADS" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --mmap 0 \
    --cpu-mask 0xfc --cpu-strict 1 \
    --poll 1000 \
    -ngl 99 "$@" \
