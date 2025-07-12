#!/bin/bash

# Ensure tmux is installed

if ! command -v tmux &> /dev/null
then
    echo "tmux could not be found, installing..."
    apt-get update && apt-get install -y tmux
fi

# Check and install dependencies if needed
echo "Checking Python dependencies..."

# List of required packages
PACKAGES=("vllm" "openai" "datasets" "tqdm" "transformers" "accelerate" "python-dotenv")

MISSING_PACKAGES=()

# Check each package
for package in "${PACKAGES[@]}"; do
    if ! python3 -c "import ${package}" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

# Install missing packages
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "Installing missing packages: ${MISSING_PACKAGES[*]}"
    pip install "${MISSING_PACKAGES[@]}"
else
    echo "All required packages are already installed."
fi

# --- Configuration (Restored from your original script) ---
# The name for your tmux session.
SESSION_NAME="completer"

# The model you want to serve with vLLM.
MODEL="Qwen/Qwen3-30B-A3B"

# The number of GPUs to use.
NUM_GPUS=4

# vLLM API server configuration
HOST="0.0.0.0"
PORT="8000"

# --- Parallelism Configuration ---
TENSOR_PARALLEL_SIZE=4
DATA_PARALLEL_SIZE=$((NUM_GPUS / TENSOR_PARALLEL_SIZE))


# --- Pre-flight Check ---
# Check if a session with the same name already exists.
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "A tmux session named '${SESSION_NAME}' already exists."
    echo "You can attach to it with: tmux attach -t ${SESSION_NAME}"
    echo "Or kill it with: tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi


# --- Script Execution ---

echo "Starting new tmux session: ${SESSION_NAME}"

# 1. Create a new detached tmux session and name the first window "Server | Client".
tmux new-session -d -s "${SESSION_NAME}" -n "Server | Client"

# 2. Start the vLLM server in the first pane (pane 0).
#    Restoring the `vllm serve` command and parameters from your original script.
#    Note: Some flags may be specific to your vLLM version or a custom build.
SERVER_CMD="vllm serve \"${MODEL}\" \
    --host ${HOST} \
    --port ${PORT} \
    --api-key 'key' \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --data-parallel-size ${DATA_PARALLEL_SIZE} \
    --enable-prefix-caching \
    --enable_expert_parallel \
    --generation_config 'auto' \
    --dtype 'bfloat16'"

tmux send-keys -t "${SESSION_NAME}:0.0" "${SERVER_CMD}" C-m
echo "vLLM server starting in the left pane..."

# 3. Split the window horizontally to create a second pane on the right.
tmux split-window -h -t "${SESSION_NAME}:0"

# 4. Run the client script in the new pane (pane 1).
#    This part includes a robust check to wait for the server to be ready.
#    It polls the server's health endpoint instead of using a fixed sleep time.
CLIENT_CMD="
echo 'Waiting for vLLM server to become available at http://${HOST}:${PORT} ...';
while ! curl -s --fail http://${HOST}:${PORT}/health > /dev/null; do
    printf '.';
    sleep 5;
done;
echo -e '\nServer is ready!';
echo 'Running main.py...';
python3 main.py --mode 'eval'
"

tmux send-keys -t "${SESSION_NAME}:0.1" "${CLIENT_CMD}" C-m

echo "Client script will run in the right pane as soon as the server is ready."
echo "To attach to the session and monitor, run: tmux attach -t ${SESSION_NAME}"
