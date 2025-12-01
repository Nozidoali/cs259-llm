#!/bin/sh
# ADB push script for pushing GGUF models to Android device

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <local_gguf_file> <device_path> [--serial <adb_serial>]"
    echo "Example: $0 models/gguf/qwen2-0.5b-Q4_0.gguf /data/local/tmp/gguf/"
    exit 1
fi

LOCAL_FILE="$1"
DEVICE_PATH="$2"
ADB_SERIAL=""

# Parse optional serial argument
shift 2
while [ $# -gt 0 ]; do
    case "$1" in
        --serial)
            ADB_SERIAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate local file exists
if [ ! -f "$LOCAL_FILE" ]; then
    echo "Error: Local file not found: $LOCAL_FILE"
    exit 1
fi

# Build adb command
ADB_CMD="adb"
if [ -n "$ADB_SERIAL" ]; then
    ADB_CMD="$ADB_CMD -s $ADB_SERIAL"
fi

# Check if device is connected
if ! $ADB_CMD devices | grep -q "device$"; then
    echo "Error: No Android device connected"
    if [ -n "$ADB_SERIAL" ]; then
        echo "  (Looking for device with serial: $ADB_SERIAL)"
    fi
    exit 1
fi

# Create directory on device if it doesn't exist
echo "Creating directory on device: $DEVICE_PATH"
$ADB_CMD shell "mkdir -p $DEVICE_PATH" || {
    echo "Error: Failed to create directory on device"
    exit 1
}

# Get filename from path
FILENAME=$(basename "$LOCAL_FILE")

# Push file
echo "Pushing $FILENAME to device..."
echo "  From: $LOCAL_FILE"
echo "  To: $DEVICE_PATH$FILENAME"

$ADB_CMD push "$LOCAL_FILE" "$DEVICE_PATH$FILENAME" || {
    echo "Error: Failed to push file to device"
    exit 1
}

# Verify file was pushed
echo "Verifying file on device..."
DEVICE_SIZE=$($ADB_CMD shell "stat -c%s $DEVICE_PATH$FILENAME" 2>/dev/null | tr -d '\r')
LOCAL_SIZE=$(stat -f%z "$LOCAL_FILE" 2>/dev/null || stat -c%s "$LOCAL_FILE" 2>/dev/null)

if [ -z "$DEVICE_SIZE" ] || [ -z "$LOCAL_SIZE" ]; then
    echo "Warning: Could not verify file size"
else
    if [ "$DEVICE_SIZE" = "$LOCAL_SIZE" ]; then
        echo "✓ File verified: $DEVICE_SIZE bytes"
    else
        echo "Warning: File size mismatch (local: $LOCAL_SIZE, device: $DEVICE_SIZE)"
    fi
fi

echo "✓ Successfully pushed $FILENAME to $DEVICE_PATH"


