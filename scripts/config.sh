#!/bin/bash

# Shared configuration for RunPod scripts
# Edit these values to match your RunPod setup

# SSH connection command
# Format: ssh user@host [-p PORT] [-i ~/.ssh/key]
SSH_CMD="${SSH_CMD:-ssh root@69.19.136.225 -p 21871 -i ~/.ssh/id_ed25519}"

# Date string for workspace directory (YYYYMMDD_HHMMSS format)
# Leave empty to auto-detect the latest workspace
# Example: "20251205_021041"
DATE_STR="${DATE_STR:-20251205_221142}"

# Workspace timestamp (YYYYMMDD_HHMMSS format)
# Used to coordinate between TensorBoard and Python training
# If not set, will be generated automatically in start-command.sh
# Example: "20251205_221142"
WORKSPACE_TIMESTAMP="${WORKSPACE_TIMESTAMP:-}"

# Remote work directory
REMOTE_WORK_DIR="${REMOTE_WORK_DIR:-/workspace/cs259-llm}"

# TensorBoard port
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"

# Mobile destination path for adb push
MOBILE_DEST="${MOBILE_DEST:-/data/local/tmp/gguf}"
