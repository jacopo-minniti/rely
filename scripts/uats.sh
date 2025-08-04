#!/bin/bash

# UATS Parallel Execution Script with tmux
# Usage: ./uats.sh [num_gpus] [start_question] [end_question] [other_args...]

set -e

# Default values
NUM_GPUS=8
START_QUESTION=0
END_QUESTION=70
BASE_SAVE_DIR="uats_results"

# Shift arguments so remaining args are passed to uats.py
shift 4 || true

# Get the total number of questions to process
TOTAL_QUESTIONS=$((END_QUESTION - START_QUESTION))

# Calculate questions per GPU
QUESTIONS_PER_GPU=$((TOTAL_QUESTIONS / NUM_GPUS))
REMAINDER=$((TOTAL_QUESTIONS % NUM_GPUS))

echo "=== UATS Parallel Execution with tmux ==="
echo "Number of GPUs: $NUM_GPUS"
echo "Questions range: $START_QUESTION to $((END_QUESTION - 1)) (total: $TOTAL_QUESTIONS questions)"
echo "Questions per GPU: $QUESTIONS_PER_GPU"
echo "Remainder: $REMAINDER"
echo "Base save directory: $BASE_SAVE_DIR"
echo "Additional args: $@"
echo "================================"

# Create base save directory
mkdir -p "$BASE_SAVE_DIR"

# Create a new tmux session for UATS
SESSION_NAME="uats_parallel_$(date +%s)"
echo "Creating tmux session: $SESSION_NAME"

# Create the tmux session
tmux new-session -d -s "$SESSION_NAME"

# Function to run UATS on a specific GPU with a question range
run_uats_on_gpu() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    local save_dir=$4
    shift 4
    
    echo "Starting GPU $gpu_id: questions $start_idx to $((end_idx - 1))"
    
    # Create a new window for this GPU
    if [ $gpu_id -eq 0 ]; then
        # Use the first window (already created)
        tmux send-keys -t "$SESSION_NAME:0" "echo 'GPU $gpu_id: Processing questions $start_idx to $((end_idx - 1))'" Enter
        tmux send-keys -t "$SESSION_NAME:0" "CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/uats.py --start_idx $start_idx --end_idx $end_idx --base_save_dir \"$save_dir\" $@" Enter
    else
        # Create new window for additional GPUs
        tmux new-window -t "$SESSION_NAME" -n "GPU$gpu_id"
        tmux send-keys -t "$SESSION_NAME:GPU$gpu_id" "echo 'GPU $gpu_id: Processing questions $start_idx to $((end_idx - 1))'" Enter
        tmux send-keys -t "$SESSION_NAME:GPU$gpu_id" "CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/uats.py --start_idx $start_idx --end_idx $end_idx --base_save_dir \"$save_dir\" $@" Enter
    fi
    
    echo "GPU $gpu_id process started in tmux window"
}

# Launch processes for each GPU
current_start=$START_QUESTION
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    # Calculate end index for this GPU
    if [ $gpu -lt $REMAINDER ]; then
        # Distribute remainder questions among first few GPUs
        current_end=$((current_start + QUESTIONS_PER_GPU + 1))
    else
        current_end=$((current_start + QUESTIONS_PER_GPU))
    fi
    
    # Ensure we don't exceed the total number of questions
    if [ $current_end -gt $END_QUESTION ]; then
        current_end=$END_QUESTION
    fi
    
    # Run UATS on this GPU
    run_uats_on_gpu $gpu $current_start $current_end "$BASE_SAVE_DIR" "$@"
    
    # Update start index for next GPU
    current_start=$current_end
    
    # Stop if we've processed all questions
    if [ $current_start -ge $END_QUESTION ]; then
        break
    fi
done

echo "All processes started in tmux session: $SESSION_NAME"
echo ""
echo "To monitor progress:"
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "To detach from session (without killing processes):"
echo "  Press Ctrl+B, then D"
echo ""
echo "To kill the session and all processes:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "To list all sessions:"
echo "  tmux list-sessions"
echo ""
echo "Waiting for all processes to complete..."

# Wait for all tmux windows to complete
# This is a simple approach - in practice, you might want to monitor the processes more carefully
sleep 2

# Check if all processes are still running
while true; do
    # Count how many tmux windows are still running python processes
    running_count=$(tmux list-windows -t "$SESSION_NAME" -F "#{window_index}" | wc -l)
    python_count=$(tmux list-windows -t "$SESSION_NAME" -F "#{pane_current_command}" | grep -c python3 || true)
    
    if [ $python_count -eq 0 ]; then
        echo "All UATS processes completed!"
        break
    fi
    
    echo "Still running: $python_count processes"
    sleep 30  # Check every 30 seconds
done

echo "All UATS processes completed!"
echo "Results saved in: $BASE_SAVE_DIR"
echo ""
echo "To view the session:"
echo "  tmux attach-session -t $SESSION_NAME"
