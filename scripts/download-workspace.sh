#!/bin/bash

# Download workspace folder from RunPod to local machine
# Usage: ./download-workspace.sh [workspace_timestamp] [local_destination]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Parse arguments and options
TIMESTAMP=""
LOCAL_DEST=""
INCLUDE_CHECKPOINTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --include-checkpoints)
            INCLUDE_CHECKPOINTS="yes"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [workspace_timestamp] [local_destination] [options]"
            echo ""
            echo "Arguments:"
            echo "  workspace_timestamp    Timestamp of workspace to download (default: from DATE_STR in config.sh)"
            echo "  local_destination      Local directory to save workspace (default: ./workspace)"
            echo ""
            echo "Options:"
            echo "  --include-checkpoints  Include checkpoint directories (default: exclude to save space)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Download using DATE_STR from config.sh"
            echo "  $0 20251206_081921              # Download specific workspace"
            echo "  $0 20251206_081921 ./my_models  # Download to custom location"
            echo "  $0 --include-checkpoints        # Include all checkpoints"
            exit 0
            ;;
        *)
            if [ -z "$TIMESTAMP" ]; then
                TIMESTAMP="$1"
            elif [ -z "$LOCAL_DEST" ]; then
                LOCAL_DEST="$1"
            else
                echo "Unknown argument: $1"
                echo "Run '$0 --help' for usage information"
                exit 1
            fi
            shift
            ;;
    esac
done

# Set defaults
TIMESTAMP="${TIMESTAMP:-$DATE_STR}"
LOCAL_DEST="${LOCAL_DEST:-./workspace}"

# Validate timestamp
if [ -z "$TIMESTAMP" ]; then
    echo "Error: No workspace timestamp provided"
    echo "Usage: $0 [workspace_timestamp] [local_destination] [options]"
    echo ""
    echo "Options:"
    echo "  1. Set DATE_STR in config.sh"
    echo "  2. Pass timestamp as first argument: $0 20251206_081921"
    echo "  3. Set DATE_STR environment variable"
    echo ""
    echo "Run '$0 --help' for more information"
    exit 1
fi

# Extract SSH connection details
SSH_HOST=$(echo "$SSH_CMD" | grep -oE '[^ ]+@[^ ]+' | head -1 | cut -d'@' -f2)
SSH_USER=$(echo "$SSH_CMD" | grep -oE '[^ ]+@[^ ]+' | head -1 | cut -d'@' -f1)
SSH_PORT=$(echo "$SSH_CMD" | grep -oE '\-p [0-9]+' | awk '{print $2}')
SSH_KEY=$(echo "$SSH_CMD" | grep -oE '\-i [^ ]+' | awk '{print $2}')

# Expand tilde in SSH_KEY
if [ -n "$SSH_KEY" ]; then
    SSH_KEY="${SSH_KEY/#\~/$HOME}"
fi

REMOTE_PATH="$REMOTE_WORK_DIR/workspace/$TIMESTAMP"
LOCAL_PATH="$LOCAL_DEST/$TIMESTAMP"

echo "========================================"
echo "Downloading Workspace from RunPod"
echo "========================================"
echo "Remote: $SSH_USER@$SSH_HOST:$REMOTE_PATH"
echo "Local:  $LOCAL_PATH"
echo "SSH Port: ${SSH_PORT:-default}"
echo "SSH Key: ${SSH_KEY:-default}"
echo ""

# Check if remote directory exists
echo "Checking remote directory..."
if ! $SSH_CMD "test -d $REMOTE_PATH"; then
    echo "Error: Remote workspace directory does not exist: $REMOTE_PATH"
    echo ""
    echo "Available workspaces on remote:"
    $SSH_CMD "ls -1 $REMOTE_WORK_DIR/workspace/ 2>/dev/null || echo 'No workspaces found'"
    exit 1
fi

# Create local destination directory
mkdir -p "$LOCAL_DEST"

# Download with rsync
echo "Downloading workspace folder..."
echo "This may take a while depending on the size..."
echo ""

# Build exclusion patterns
EXCLUDE_OPTS=""
if [ -z "$INCLUDE_CHECKPOINTS" ]; then
    echo "Note: Excluding checkpoint directories (only downloading final models)"
    echo "      Use --include-checkpoints to download all checkpoints"
    echo ""
    # Exclude checkpoint directories to save space and time
    EXCLUDE_OPTS="--exclude=checkpoint-*/ --exclude=*/checkpoint-*/ --exclude=checkpoints/ --exclude=*/checkpoints/"
else
    echo "Note: Including ALL checkpoint directories"
    echo ""
fi

# Build SSH command for rsync
SSH_OPTIONS=""
if [ -n "$SSH_PORT" ]; then
    SSH_OPTIONS="$SSH_OPTIONS -p $SSH_PORT"
fi
if [ -n "$SSH_KEY" ]; then
    SSH_OPTIONS="$SSH_OPTIONS -i $SSH_KEY"
fi

# Run rsync with proper SSH options and exclusions
if [ -n "$SSH_OPTIONS" ]; then
    rsync -avz --progress $EXCLUDE_OPTS -e "ssh $SSH_OPTIONS" "$SSH_USER@$SSH_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
else
    rsync -avz --progress $EXCLUDE_OPTS "$SSH_USER@$SSH_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Download Complete!"
    echo "========================================"
    echo "Workspace saved to: $LOCAL_PATH"
    echo ""
    echo "Contents:"
    ls -lh "$LOCAL_PATH" | tail -n +2
    echo ""
    
    # Show directory size
    if command -v du >/dev/null 2>&1; then
        TOTAL_SIZE=$(du -sh "$LOCAL_PATH" | cut -f1)
        echo "Total size: $TOTAL_SIZE"
    fi
    
    echo ""
    echo "Files downloaded:"
    find "$LOCAL_PATH" -type f | wc -l | xargs echo "  -"
else
    echo ""
    echo "Error: Download failed"
    exit 1
fi
