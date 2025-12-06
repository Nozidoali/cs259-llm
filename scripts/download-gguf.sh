#!/bin/bash

# Script to download GGUF files from RunPod using SCP
# Configure the SSH connection and date string in config.sh

# Source shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    source "$SCRIPT_DIR/config.sh"
else
    echo "Warning: config.sh not found. Using defaults."
    SSH_CMD="${SSH_CMD:-ssh root@69.19.136.225 -p 37178 -i ~/.ssh/id_ed25519}"
    DATE_STR="${DATE_STR:-20251205_021041}"
    MOBILE_DEST="${MOBILE_DEST:-/data/local/tmp/gguf}"
fi

# DATE_STR is required for this script
if [ -z "$DATE_STR" ] || [ "$DATE_STR" == "" ]; then
    echo "Error: DATE_STR is required. Please set it in config.sh or as an environment variable."
    echo "Example: DATE_STR=20251205_021041"
    exit 1
fi

# Extract user@host, port, and identity file from SSH command
SSH_USER_HOST=""
SSH_PORT=""
SSH_KEY=""

# Try to match: ssh user@host -p PORT -i KEY
if [[ $SSH_CMD =~ ssh[[:space:]]+([^[:space:]]+)[[:space:]]+-p[[:space:]]+([0-9]+)[[:space:]]+-i[[:space:]]+([^[:space:]]+) ]]; then
    SSH_USER_HOST="${BASH_REMATCH[1]}"
    SSH_PORT="${BASH_REMATCH[2]}"
    SSH_KEY="${BASH_REMATCH[3]}"
# Try to match: ssh user@host -p PORT
elif [[ $SSH_CMD =~ ssh[[:space:]]+([^[:space:]]+)[[:space:]]+-p[[:space:]]+([0-9]+) ]]; then
    SSH_USER_HOST="${BASH_REMATCH[1]}"
    SSH_PORT="${BASH_REMATCH[2]}"
# Try to match: ssh user@host -i KEY
elif [[ $SSH_CMD =~ ssh[[:space:]]+([^[:space:]]+)[[:space:]]+-i[[:space:]]+([^[:space:]]+) ]]; then
    SSH_USER_HOST="${BASH_REMATCH[1]}"
    SSH_KEY="${BASH_REMATCH[2]}"
# Try to match: ssh user@host
elif [[ $SSH_CMD =~ ssh[[:space:]]+([^[:space:]]+) ]]; then
    SSH_USER_HOST="${BASH_REMATCH[1]}"
else
    echo "Error: Could not parse SSH connection string"
    echo "Expected format: ssh user@host [-p PORT] [-i ~/.ssh/key]"
    exit 1
fi

# Expand tilde in SSH key path
SSH_KEY="${SSH_KEY/#\~/$HOME}"

echo "========================================="
echo "RunPod GGUF Downloader"
echo "========================================="
echo "SSH Host: $SSH_USER_HOST"
if [ -n "$SSH_PORT" ]; then
    echo "SSH Port: $SSH_PORT"
fi
if [ -n "$SSH_KEY" ]; then
    echo "SSH Key: $SSH_KEY"
fi
echo "Date String: $DATE_STR"
echo "========================================="
echo ""

# Local output directory
LOCAL_DIR="./downloaded_models/$DATE_STR"

# Remote workspace path with date string
REMOTE_WORK_DIR="${REMOTE_WORK_DIR:-/workspace/cs259-llm}"
REMOTE_DIR="$REMOTE_WORK_DIR/workspace/$DATE_STR"

# Build commands
SSH_OPTS="-tt -o LogLevel=ERROR"
SCP_CMD="scp -r"

if [ -n "$SSH_PORT" ]; then
    SSH_OPTS="-p $SSH_PORT $SSH_OPTS"
    SCP_CMD="$SCP_CMD -P $SSH_PORT"
fi

if [ -n "$SSH_KEY" ]; then
    SSH_OPTS="-i $SSH_KEY $SSH_OPTS"
    SCP_CMD="$SCP_CMD -i $SSH_KEY"
fi

# Create local directory
mkdir -p "$LOCAL_DIR"

# Check if GGUF files already exist
EXISTING_FILES=$(ls -1 "$LOCAL_DIR"/*.gguf 2>/dev/null | wc -l | tr -d ' ')

if [ "$EXISTING_FILES" -gt 0 ]; then
    echo "Found $EXISTING_FILES existing GGUF file(s) in $LOCAL_DIR"
    echo "Skipping download..."
    ls -lh "$LOCAL_DIR"/*.gguf
    echo ""
else
    # Build the SCP command and execute it
    REMOTE_PATTERN="$SSH_USER_HOST:$REMOTE_DIR/*.gguf"

    echo "Remote directory: $REMOTE_DIR"
    echo "Local directory: $LOCAL_DIR"
    echo ""
    echo "Downloading GGUF files using SCP..."
    echo "Command: $SCP_CMD $REMOTE_PATTERN $LOCAL_DIR/"
    echo ""

    # Execute SCP command
    if $SCP_CMD "$REMOTE_PATTERN" "$LOCAL_DIR/"; then
        echo ""
        echo "Download completed successfully!"
    else
        echo ""
        echo "Error: SCP download failed"
        echo "This might be due to PTY issues with RunPod."
        echo "You may need to use SSH with tar instead."
        exit 1
    fi
fi

# Verify files exist before pushing
GGUF_FILES=$(ls -1 "$LOCAL_DIR"/*.gguf 2>/dev/null)
if [ -z "$GGUF_FILES" ]; then
    echo "Error: No GGUF files found in $LOCAL_DIR"
    exit 1
fi

echo "========================================="
echo "Download Summary"
echo "========================================="
echo "Files in: $LOCAL_DIR"
COUNT=$(ls -1 "$LOCAL_DIR"/*.gguf 2>/dev/null | wc -l | tr -d ' ')
echo "Total GGUF files: $COUNT"
ls -lh "$LOCAL_DIR"/*.gguf
echo "========================================="
echo ""

# Push to mobile using adb
echo "Pushing GGUF files to mobile device..."
echo "Destination: $MOBILE_DEST"
echo ""

# Check if adb is available
if ! command -v adb &> /dev/null; then
    echo "Warning: adb command not found. Skipping mobile push."
    echo "Install Android SDK platform-tools to enable adb push."
    exit 0
fi

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Warning: No Android device connected. Skipping mobile push."
    echo "Connect a device via USB or enable ADB over network."
    exit 0
fi

# Create destination directory on mobile
adb shell "mkdir -p $MOBILE_DEST" 2>/dev/null

# Push each GGUF file
PUSHED_COUNT=0
for GGUF_FILE in "$LOCAL_DIR"/*.gguf; do
    if [ -f "$GGUF_FILE" ]; then
        FILENAME=$(basename "$GGUF_FILE")
        echo "Pushing $FILENAME..."
        if adb push "$GGUF_FILE" "$MOBILE_DEST/$FILENAME"; then
            echo "  ✓ Pushed: $MOBILE_DEST/$FILENAME"
            PUSHED_COUNT=$((PUSHED_COUNT + 1))
        else
            echo "  ✗ Failed to push: $FILENAME"
        fi
    fi
done

echo ""
echo "========================================="
echo "Mobile Push Summary"
echo "========================================="
echo "Pushed $PUSHED_COUNT file(s) to: $MOBILE_DEST"
echo "========================================="
