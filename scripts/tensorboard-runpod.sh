#!/bin/bash

# TensorBoard Launcher for RunPod
# This script starts TensorBoard to view training logs via RunPod HTTP Service

# Default port
PORT="${TENSORBOARD_PORT:-6006}"

# Find the most recent training log directory
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/cs259-llm/workspace}"

if [ -d "$WORKSPACE_DIR" ]; then
    # Find the most recent timestamped directory
    LATEST_DIR=$(find "$WORKSPACE_DIR" -type d -name "20*" | sort -r | head -1)
    if [ -n "$LATEST_DIR" ]; then
        LOG_DIR="$LATEST_DIR"
        echo "Using log directory: $LOG_DIR"
    else
        LOG_DIR="$WORKSPACE_DIR"
        echo "Using workspace directory: $LOG_DIR"
    fi
else
    LOG_DIR="${TENSORBOARD_LOG_DIR:-/workspace/cs259-llm/workspace}"
    echo "Using default log directory: $LOG_DIR"
fi

echo "========================================="
echo "Starting TensorBoard"
echo "========================================="
echo "Log directory: $LOG_DIR"
echo "Port: $PORT"
echo "========================================="
echo ""
echo "To access TensorBoard:"
echo "1. Go to RunPod dashboard"
echo "2. Click 'Connect' → 'HTTP Service'"
echo "3. RunPod will provide a URL to access TensorBoard"
echo "========================================="
echo ""

# Start TensorBoard
tensorboard --logdir="$LOG_DIR" --port="$PORT" --host=0.0.0.0 --reload_interval=30
