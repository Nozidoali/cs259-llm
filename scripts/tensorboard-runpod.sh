#!/bin/bash

# TensorBoard Launcher for RunPod
# This script starts TensorBoard to view training logs via RunPod HTTP Service

# Default port
PORT="${TENSORBOARD_PORT:-6006}"

# Try to find WORK_DIR from environment or use default
# Check if we're in the project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Try to get WORK_DIR from .env file if it exists
if [ -f "$PROJECT_DIR/.env" ]; then
    WORK_DIR=$(grep "^WORK_DIR=" "$PROJECT_DIR/.env" | cut -d '=' -f2 | tr -d '"' | tr -d "'" | xargs)
fi

# Use WORK_DIR from environment, .env, or default
WORK_DIR="${WORK_DIR:-${WORKSPACE_DIR:-$PROJECT_DIR}}"
WORKSPACE_DIR="$WORK_DIR/workspace"

LOG_DIR=""

# Find the largest date string in the workspace folder (most recent timestamp)
# Date format: YYYYMMDD_HHMMSS (e.g., 20251204_225232)
# Since date strings are in YYYYMMDD_HHMMSS format, sorting them will naturally
# put the largest (most recent) date first
if [ -d "$WORKSPACE_DIR" ]; then
    # Find all directories starting with "20" (year 2000+), sort in reverse to get largest date string
    LATEST_DIR=$(find "$WORKSPACE_DIR" -maxdepth 1 -type d -name "20*" -printf "%f\n" 2>/dev/null | sort -r | head -1)
    
    # If printf is not available (macOS), use alternative method
    if [ -z "$LATEST_DIR" ]; then
        LATEST_DIR=$(find "$WORKSPACE_DIR" -maxdepth 1 -type d -name "20*" 2>/dev/null | xargs -n1 basename | sort -r | head -1)
    fi
    
    if [ -n "$LATEST_DIR" ]; then
        LOG_DIR="$WORKSPACE_DIR/$LATEST_DIR"
        if [ -d "$LOG_DIR" ]; then
            echo "Found workspace with largest date string: $LOG_DIR"
        else
            LOG_DIR=""
        fi
    fi
fi

# If no timestamped directory found, try to find any directory with TensorBoard event files
if [ -z "$LOG_DIR" ] || [ ! -d "$LOG_DIR" ]; then
    # Search for directories containing TensorBoard event files
    if [ -d "$WORKSPACE_DIR" ]; then
        EVENT_FILE=$(find "$WORKSPACE_DIR" -type f -name "events.out.tfevents.*" 2>/dev/null | head -1)
        if [ -n "$EVENT_FILE" ]; then
            EVENT_DIR=$(dirname "$EVENT_FILE")
            # Get the parent workspace directory (go up from logs/experts/... or logs/rmoe_model...)
            LOG_DIR="$EVENT_DIR"
            # Try to find the workspace root by going up until we find a timestamped directory or workspace root
            while [ "$LOG_DIR" != "$WORKSPACE_DIR" ] && [ "$LOG_DIR" != "/" ]; do
                PARENT=$(dirname "$LOG_DIR")
                if [[ "$(basename "$PARENT")" =~ ^20[0-9]{6} ]]; then
                    LOG_DIR="$PARENT"
                    break
                fi
                LOG_DIR="$PARENT"
            done
            if [ -d "$LOG_DIR" ]; then
                echo "Found workspace with TensorBoard logs: $LOG_DIR"
            fi
        fi
    fi
fi

# Verify that the log directory actually contains event files
if [ -n "$LOG_DIR" ] && [ -d "$LOG_DIR" ]; then
    EVENT_COUNT=$(find "$LOG_DIR" -type f -name "events.out.tfevents.*" 2>/dev/null | wc -l)
    if [ "$EVENT_COUNT" -eq 0 ]; then
        echo "Warning: No TensorBoard event files found in $LOG_DIR"
        echo "This might mean:"
        echo "  1. Training hasn't started yet"
        echo "  2. Training hasn't reached logging_steps yet"
        echo "  3. Logs are in a different location"
        echo ""
        echo "Searching for event files in subdirectories..."
        find "$LOG_DIR" -type f -name "events.out.tfevents.*" 2>/dev/null | head -5
        if [ $? -ne 0 ] || [ -z "$(find "$LOG_DIR" -type f -name "events.out.tfevents.*" 2>/dev/null | head -1)" ]; then
            echo "No event files found. TensorBoard will start but may show no data until training progresses."
        fi
    else
        echo "Found $EVENT_COUNT TensorBoard event file(s)"
    fi
fi

# Fallback to workspace directory or explicit TENSORBOARD_LOG_DIR
if [ -z "$LOG_DIR" ] || [ ! -d "$LOG_DIR" ]; then
    if [ -d "$WORKSPACE_DIR" ]; then
        LOG_DIR="$WORKSPACE_DIR"
        echo "Using workspace directory: $LOG_DIR"
    else
        LOG_DIR="${TENSORBOARD_LOG_DIR:-$PROJECT_DIR}"
        echo "Using fallback directory: $LOG_DIR"
    fi
fi

echo "========================================="
echo "Starting TensorBoard"
echo "========================================="
echo "WORK_DIR: $WORK_DIR"
echo "WORKSPACE_DIR: $WORKSPACE_DIR"
echo "Log directory: $LOG_DIR"
echo "Port: $PORT"
echo "========================================="
if [ -d "$LOG_DIR" ]; then
    echo ""
    echo "Contents of log directory:"
    ls -la "$LOG_DIR" | head -10
    echo ""
    echo "Searching for log subdirectories..."
    find "$LOG_DIR" -type d -name "logs" 2>/dev/null | head -5
    echo ""
fi
echo "To access TensorBoard:"
echo "1. Go to RunPod dashboard"
echo "2. Click 'Connect' → 'HTTP Service'"
echo "3. RunPod will provide a URL to access TensorBoard"
echo "========================================="
echo ""

# Start TensorBoard
tensorboard --logdir="$LOG_DIR" --port="$PORT" --host=0.0.0.0 --reload_interval=30

