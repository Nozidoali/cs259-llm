#!/bin/bash
# Check if the benchmarking environment is properly set up

echo "========================================"
echo "Benchmark Setup Verification"
echo "========================================"
echo ""

# Check Python
echo "1. Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✓ Python found: $PYTHON_VERSION"
else
    echo "   ✗ Python not found"
    exit 1
fi
echo ""

# Check huggingface-cli
echo "2. Checking huggingface-cli..."
if command -v huggingface-cli &> /dev/null; then
    HF_VERSION=$(huggingface-cli --version 2>&1 | head -n1)
    echo "   ✓ huggingface-cli found: $HF_VERSION"
else
    echo "   ✗ huggingface-cli not found"
    echo "   Install with: pip install huggingface_hub[cli]"
fi
echo ""

# Check ADB
echo "3. Checking ADB..."
if command -v adb &> /dev/null; then
    ADB_VERSION=$(adb --version | head -n1)
    echo "   ✓ ADB found: $ADB_VERSION"
    
    # Check for connected devices
    echo ""
    echo "   Connected devices:"
    DEVICES=$(adb devices | grep -v "List" | grep "device$" | wc -l | tr -d ' ')
    if [ "$DEVICES" -gt 0 ]; then
        adb devices | grep "device$" | sed 's/^/     /'
        echo "   ✓ $DEVICES device(s) connected"
    else
        echo "     ⚠ No devices connected"
        echo "     Connect device with: adb connect <ip>:5555"
    fi
else
    echo "   ✗ ADB not found"
    echo "   Install Android SDK platform-tools"
fi
echo ""

# Check required scripts
echo "4. Checking required scripts..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/scripts/push-model.sh" ]; then
    echo "   ✓ push-model.sh found"
else
    echo "   ✗ push-model.sh not found"
fi

if [ -f "$SCRIPT_DIR/scripts/run-bench.sh" ]; then
    echo "   ✓ run-bench.sh found"
else
    echo "   ✗ run-bench.sh not found"
fi

if [ -f "$SCRIPT_DIR/benchmark_models.py" ]; then
    echo "   ✓ benchmark_models.py found"
else
    echo "   ✗ benchmark_models.py not found"
fi
echo ""

# Check disk space
echo "5. Checking disk space..."
if command -v df &> /dev/null; then
    AVAILABLE=$(df -h . | tail -n1 | awk '{print $4}')
    echo "   Available space: $AVAILABLE"
    echo "   (Recommended: >10GB for model downloads)"
fi
echo ""

# Check if device has space
if command -v adb &> /dev/null && adb devices | grep -q "device$"; then
    echo "6. Checking device storage..."
    DEVICE_SPACE=$(adb shell df -h /data/local/tmp 2>/dev/null | tail -n1 | awk '{print $4}')
    if [ -n "$DEVICE_SPACE" ]; then
        echo "   Device available space: $DEVICE_SPACE"
        echo "   (Each model needs 100MB-5GB during benchmark)"
    else
        echo "   ⚠ Could not check device space"
    fi
    echo ""
fi

echo "========================================"
echo "Setup Check Complete"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Install missing dependencies if any"
echo "  2. Connect Android device via ADB"
echo "  3. Run debug test: python3 benchmark_models.py --debug"
echo "  4. Run full benchmark: python3 benchmark_models.py"
echo ""
echo "For detailed instructions, see: BENCHMARK_README.md"
