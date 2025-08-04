#!/bin/bash

# Check if the input dataset exists
if [ ! -f "generations-mmlu-qwen3-1.7B.jsonl" ]; then
    echo "Error: Input dataset 'generations-mmlu-qwen3-1.7B.jsonl' not found!"
    exit 1
fi

# Count total lines in the dataset
total_lines=$(wc -l < "generations-mmlu-qwen3-1.7B.jsonl")
echo "Total dataset size: $total_lines lines"

# Calculate chunk size (divide dataset into 8 equal parts)
chunk_size=$((total_lines / 8))
echo "Chunk size: $chunk_size lines per GPU"

# Kill any existing tmux sessions with the same name
tmux kill-session -t fork_parallel 2>/dev/null || true

# Create a new tmux session
tmux new-session -d -s fork_parallel

# Function to create a window and run the process
create_gpu_process() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    
    # Create a new window for this GPU
    if [ $gpu_id -eq 0 ]; then
        # Use the first window (session already created)
        tmux rename-window -t fork_parallel:0 "GPU_${gpu_id}"
    else
        # Create new windows for other GPUs
        tmux new-window -t fork_parallel -n "GPU_${gpu_id}"
    fi
    
    # Run the Python script in this window
    tmux send-keys -t fork_parallel:GPU_${gpu_id} "CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m rely.extract.fork --start_idx ${start_idx} --end_idx ${end_idx} --output_path forks-mmlu-qwen3-1.7B_${gpu_id}.pt --gpu_id 0" C-m
}

# Launch 8 parallel processes
for i in {0..7}; do
    start_idx=$((i * chunk_size))
    
    # For the last GPU, make sure to include any remaining lines
    if [ $i -eq 7 ]; then
        end_idx=$total_lines
    else
        end_idx=$(((i + 1) * chunk_size))
    fi
    
    echo "Launching GPU $i: processing lines $start_idx to $((end_idx-1))"
    create_gpu_process $i $start_idx $end_idx
    
    # Small delay to avoid overwhelming the system
    sleep 1
done

echo "All 8 GPU processes launched in tmux session 'fork_parallel'"
echo "To monitor progress:"
echo "  tmux attach-session -t fork_parallel"
echo ""
echo "To list all windows:"
echo "  tmux list-windows -t fork_parallel"
echo ""
echo "To switch between windows:"
echo "  tmux select-window -t fork_parallel:GPU_X (where X is 0-7)"
echo ""
echo "To kill the session when done:"
echo "  tmux kill-session -t fork_parallel" 