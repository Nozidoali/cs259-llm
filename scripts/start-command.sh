bash -c "
set -e

cd /workspace

###############################
# SSH SERVER SETUP (EARLY)
###############################

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server

# Host keys
mkdir -p /run/sshd
ssh-keygen -A

# Authorized keys
mkdir -p /root/.ssh
chmod 700 /root/.ssh

# Add SSH public key from environment variable
if [ -n "\$SSH_PUBKEY" ]; then
  echo "\$SSH_PUBKEY" >> /root/.ssh/authorized_keys
  echo 'SSH public key added from SSH_PUBKEY environment variable.'
else
  echo 'Warning: SSH_PUBKEY not set. No SSH key added.'
fi

chmod 600 /root/.ssh/authorized_keys

# Start SSH
service ssh start
echo 'SSH server ready.'

# Clone repo if missing
if [ ! -d cs259-llm ]; then
  git clone --recurse-submodules https://github.com/Nozidoali/cs259-llm.git cs259-llm
fi

cd /workspace/cs259-llm

# Sync branch
git fetch origin
git checkout alice/rmoe
git reset --hard origin/alice/rmoe
git pull
git submodule update --init --recursive

# Python venv + deps
python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# .env
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
fi

# Hugging Face login (using HF_TOKEN env variable from RunPod)
if [ -n \"\$HF_TOKEN\" ]; then
  echo 'Logging into Hugging Face...'
  huggingface-cli login --token \"\$HF_TOKEN\" --add-to-git-credential
  export HUGGING_FACE_HUB_TOKEN=\"\$HF_TOKEN\"
  echo 'Hugging Face login complete.'
else
  echo 'Warning: HF_TOKEN not set. Hugging Face login skipped.'
fi

# Runtime envs
export WORK_DIR=/workspace/cs259-llm
export LLAMA_CPP_DIR=/workspace/cs259-llm/external/llama.cpp
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set workspace timestamp (shared between TensorBoard and Python training)
# If not set, generate current date string
if [ -z \"\$WORKSPACE_TIMESTAMP\" ]; then
  export WORKSPACE_TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
  echo \"WORKSPACE_TIMESTAMP not set, generated current timestamp: \$WORKSPACE_TIMESTAMP\"
else
  echo \"Using workspace timestamp from environment: \$WORKSPACE_TIMESTAMP\"
fi

echo 'Setup complete.'

###############################
# Start TensorBoard in background
###############################

TENSORBOARD_PORT=6006
# TensorBoard logs are stored at: $WORK_DIR/workspace/$WORKSPACE_TIMESTAMP/experts/{dataset}/logs/
# Monitor the workspace directory so it can find logs in all expert subdirectories
TENSORBOARD_LOG_DIR=\$WORK_DIR/workspace/\$WORKSPACE_TIMESTAMP

# Create workspace directory if it doesn't exist
mkdir -p \"\$TENSORBOARD_LOG_DIR\"

echo 'Starting TensorBoard in background...'
echo \"Workspace timestamp: \$WORKSPACE_TIMESTAMP\"
echo \"TensorBoard will monitor: \$TENSORBOARD_LOG_DIR (will find logs in experts/*/logs/ subdirectories)\"
echo \"TensorBoard will be available at: http://localhost:\$TENSORBOARD_PORT\"

# Start TensorBoard in background
nohup tensorboard --logdir=\"\$TENSORBOARD_LOG_DIR\" --port=\"\$TENSORBOARD_PORT\" --host=0.0.0.0 --reload_interval=30 > /tmp/tensorboard.log 2>&1 &
TENSORBOARD_PID=\$!
echo \"TensorBoard started (PID: \$TENSORBOARD_PID)\"
sleep 2

# Verify TensorBoard is running
if ps -p \$TENSORBOARD_PID > /dev/null; then
  echo 'TensorBoard is running in background.'
  echo \"Monitoring logs at: \$TENSORBOARD_LOG_DIR\"
else
  echo 'Warning: TensorBoard may not have started correctly.'
  echo 'Check logs: tail -20 /tmp/tensorboard.log'
fi

###############################
# Training
###############################

python train.py data/train_rmoe.json
echo 'Training completed.'

# Keep container alive
sleep infinity
"
