#!/bin/bash

# TensorBoard Launcher for RunPod via SSH
# This script starts TensorBoard on RunPod and sets up port forwarding
# Run this from your local terminal to access TensorBoard remotely

# Source shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    source "$SCRIPT_DIR/config.sh"
else
    echo "Warning: config.sh not found. Using defaults."
    SSH_CMD="${SSH_CMD:-ssh root@69.19.136.225 -p 37178 -i ~/.ssh/id_ed25519}"
    REMOTE_WORK_DIR="${REMOTE_WORK_DIR:-/workspace/cs259-llm}"
    TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
fi

# WORKSPACE_TIMESTAMP can be set in config.sh or as environment variable
WORKSPACE_TIMESTAMP="${WORKSPACE_TIMESTAMP:-}"

PORT="${TENSORBOARD_PORT:-6006}"
REMOTE_WORKSPACE_DIR="$REMOTE_WORK_DIR/workspace"

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
    echo "Or set SSH_CMD environment variable"
    exit 1
fi

# Expand tilde in SSH key path
SSH_KEY="${SSH_KEY/#\~/$HOME}"

# Build SSH command options
SSH_OPTS="-tt -o LogLevel=ERROR"
if [ -n "$SSH_PORT" ]; then
    SSH_OPTS="-p $SSH_PORT $SSH_OPTS"
fi
if [ -n "$SSH_KEY" ]; then
    SSH_OPTS="-i $SSH_KEY $SSH_OPTS"
fi

echo "========================================="
echo "TensorBoard Launcher for RunPod"
echo "========================================="
echo "SSH Host: $SSH_USER_HOST"
if [ -n "$SSH_PORT" ]; then
    echo "SSH Port: $SSH_PORT"
fi
if [ -n "$SSH_KEY" ]; then
    echo "SSH Key: $SSH_KEY"
fi
echo "TensorBoard Port: $PORT"
echo "Remote Work Dir: $REMOTE_WORK_DIR"
if [ -n "$WORKSPACE_TIMESTAMP" ] && [ "$WORKSPACE_TIMESTAMP" != "" ]; then
    echo "Workspace Timestamp: $WORKSPACE_TIMESTAMP"
elif [ -n "$DATE_STR" ] && [ "$DATE_STR" != "" ]; then
    echo "Date String: $DATE_STR"
fi
echo "========================================="
echo ""

# Find the log directory on remote machine
echo "Searching for TensorBoard logs on remote machine..."
REMOTE_LOG_DIR=""

# Check for WORKSPACE_TIMESTAMP first (set in start-command.sh), then DATE_STR
if [ -n "$WORKSPACE_TIMESTAMP" ] && [ "$WORKSPACE_TIMESTAMP" != "" ]; then
    REMOTE_LOG_DIR="$REMOTE_WORKSPACE_DIR/$WORKSPACE_TIMESTAMP/logs"
    echo "Using workspace timestamp from config: $WORKSPACE_TIMESTAMP"
    echo "TensorBoard log directory: $REMOTE_LOG_DIR"
    
    # Verify the directory exists
    CHECK_DIR_CMD="test -d '$REMOTE_LOG_DIR' && echo 'exists' || echo 'not found'"
    DIR_EXISTS=$(ssh $SSH_OPTS $SSH_USER_HOST "$CHECK_DIR_CMD" 2>/dev/null)
    if [ "$DIR_EXISTS" != "exists" ]; then
        echo "Warning: Directory $REMOTE_LOG_DIR does not exist on remote machine"
        echo "Falling back to DATE_STR or auto-detection..."
        REMOTE_LOG_DIR=""
    fi
elif [ -n "$DATE_STR" ] && [ "$DATE_STR" != "" ]; then
    REMOTE_LOG_DIR="$REMOTE_WORKSPACE_DIR/$DATE_STR/logs"
    echo "Using specified date string: $DATE_STR"
    echo "TensorBoard log directory: $REMOTE_LOG_DIR"
    
    # Verify the directory exists
    CHECK_DIR_CMD="test -d '$REMOTE_LOG_DIR' && echo 'exists' || echo 'not found'"
    DIR_EXISTS=$(ssh $SSH_OPTS $SSH_USER_HOST "$CHECK_DIR_CMD" 2>/dev/null)
    if [ "$DIR_EXISTS" != "exists" ]; then
        echo "Warning: Directory $REMOTE_LOG_DIR does not exist on remote machine"
        echo "Falling back to auto-detection..."
        REMOTE_LOG_DIR=""
    fi
fi

# If WORKSPACE_TIMESTAMP/DATE_STR not provided or directory doesn't exist, auto-detect
if [ -z "$REMOTE_LOG_DIR" ] || [ "$REMOTE_LOG_DIR" == "" ]; then
    # Try to find the latest timestamped directory with logs subdirectory
    FIND_CMD="find $REMOTE_WORKSPACE_DIR -maxdepth 1 -type d -name '20*' 2>/dev/null | sort -r | head -1"
    LATEST_DIR=$(ssh $SSH_OPTS $SSH_USER_HOST "$FIND_CMD" 2>/dev/null)
    
    if [ -n "$LATEST_DIR" ] && [ "$LATEST_DIR" != "" ]; then
        # Check if logs subdirectory exists
        CHECK_LOGS_CMD="test -d '$LATEST_DIR/logs' && echo '$LATEST_DIR/logs' || echo ''"
        LOGS_DIR=$(ssh $SSH_OPTS $SSH_USER_HOST "$CHECK_LOGS_CMD" 2>/dev/null)
        if [ -n "$LOGS_DIR" ] && [ "$LOGS_DIR" != "" ]; then
            REMOTE_LOG_DIR="$LOGS_DIR"
            echo "Found latest workspace with logs: $REMOTE_LOG_DIR"
        else
            REMOTE_LOG_DIR="$LATEST_DIR"
            echo "Found latest workspace (no logs subdirectory): $REMOTE_LOG_DIR"
        fi
    else
        # Try to find any directory with TensorBoard event files
        FIND_EVENT_CMD="find $REMOTE_WORKSPACE_DIR -type f -name 'events.out.tfevents.*' 2>/dev/null | head -1"
        EVENT_FILE=$(ssh $SSH_OPTS $SSH_USER_HOST "$FIND_EVENT_CMD" 2>/dev/null)
        
        if [ -n "$EVENT_FILE" ] && [ "$EVENT_FILE" != "" ]; then
            # Get directory containing the event file
            GET_DIR_CMD="dirname '$EVENT_FILE'"
            EVENT_DIR=$(ssh $SSH_OPTS $SSH_USER_HOST "$GET_DIR_CMD" 2>/dev/null)
            REMOTE_LOG_DIR="$EVENT_DIR"
            echo "Found workspace with TensorBoard logs: $REMOTE_LOG_DIR"
        fi
    fi
fi

# Fallback to workspace directory
if [ -z "$REMOTE_LOG_DIR" ] || [ "$REMOTE_LOG_DIR" == "" ]; then
    REMOTE_LOG_DIR="$REMOTE_WORKSPACE_DIR"
    echo "Using workspace directory: $REMOTE_LOG_DIR"
fi

# Verify event files exist
echo ""
echo "Checking for TensorBoard event files..."
EVENT_COUNT_CMD="find '$REMOTE_LOG_DIR' -type f -name 'events.out.tfevents.*' 2>/dev/null | wc -l"
EVENT_COUNT=$(ssh $SSH_OPTS $SSH_USER_HOST "$EVENT_COUNT_CMD" 2>/dev/null | tr -d ' ')

if [ "$EVENT_COUNT" -eq "0" ] || [ -z "$EVENT_COUNT" ]; then
    echo "Warning: No TensorBoard event files found in $REMOTE_LOG_DIR"
    echo "TensorBoard will start but may show no data until training progresses."
else
    echo "Found $EVENT_COUNT TensorBoard event file(s)"
fi

echo ""
echo "========================================="
echo "Starting TensorBoard on RunPod"
echo "========================================="
echo "Log directory: $REMOTE_LOG_DIR"
echo "Port: $PORT"
echo ""
echo "TensorBoard will be accessible at:"
echo "  http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop TensorBoard and port forwarding"
echo "========================================="
echo ""

# Check if TensorBoard is already running on the remote machine
CHECK_TB_CMD="pgrep -f 'tensorboard.*--port=$PORT' > /dev/null 2>&1"
if ssh $SSH_OPTS $SSH_USER_HOST "$CHECK_TB_CMD"; then
    echo "Warning: TensorBoard appears to be already running on port $PORT"
    echo "Killing existing TensorBoard process..."
    KILL_TB_CMD="pkill -f 'tensorboard.*--port=$PORT'"
    ssh $SSH_OPTS $SSH_USER_HOST "$KILL_TB_CMD" 2>/dev/null
    sleep 2
fi

# Start TensorBoard on remote machine in background and set up port forwarding
echo "Starting TensorBoard on remote machine..."
echo ""

# Start TensorBoard in background on remote
# We use nohup and redirect output to keep it running
REMOTE_SCRIPT="cd $REMOTE_WORK_DIR && source venv/bin/activate 2>/dev/null || true && nohup tensorboard --logdir='$REMOTE_LOG_DIR' --port=$PORT --host=0.0.0.0 --reload_interval=30 > /tmp/tensorboard.log 2>&1 &"

# Start TensorBoard
if ! ssh $SSH_OPTS $SSH_USER_HOST "$REMOTE_SCRIPT"; then
    echo "Error: Failed to start TensorBoard on remote machine"
    echo "Check if TensorBoard is installed: ssh $SSH_OPTS $SSH_USER_HOST 'which tensorboard'"
    exit 1
fi

echo "TensorBoard started on remote machine"
echo "Waiting for TensorBoard to initialize..."
sleep 3

# Verify TensorBoard is running
CHECK_TB_CMD="pgrep -f 'tensorboard.*--port=$PORT' > /dev/null 2>&1"
if ! ssh $SSH_OPTS $SSH_USER_HOST "$CHECK_TB_CMD"; then
    echo "Warning: TensorBoard may not have started correctly"
    echo "Check logs: ssh $SSH_OPTS $SSH_USER_HOST 'tail -20 /tmp/tensorboard.log'"
fi

# Set up SSH port forwarding
echo ""
echo "Setting up SSH port forwarding..."
echo "Forwarding localhost:$PORT -> $SSH_USER_HOST:$PORT"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping TensorBoard on remote machine..."
    ssh $SSH_OPTS $SSH_USER_HOST "pkill -f 'tensorboard.*--port=$PORT'" 2>/dev/null
    echo "TensorBoard stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start port forwarding in foreground (this will block until Ctrl+C)
# -N: don't execute remote command, just forward ports
# -L: local port forwarding
ssh $SSH_OPTS -N -L $PORT:localhost:$PORT $SSH_USER_HOST
