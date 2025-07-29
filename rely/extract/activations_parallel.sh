#!/bin/bash

# Configuration
INPUT_FILE="100-short-completions-mmlu-qwen3-1.7B-v2.jsonl"  # Change this to your actual input file
MODEL_NAME="unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
NUM_GPUS=8
SESSION_NAME="activations_extraction"

# Get total number of lines in the input file
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
echo "Total lines in $INPUT_FILE: $TOTAL_LINES"

# Calculate lines per GPU
LINES_PER_GPU=$((TOTAL_LINES / NUM_GPUS))
echo "Lines per GPU: $LINES_PER_GPU"

# Create tmux session
tmux new-session -d -s "$SESSION_NAME"

# Create windows for each GPU
for gpu in $(seq 0 $((NUM_GPUS-1))); do
    # Calculate start and end indices for this GPU
    start_idx=$((gpu * LINES_PER_GPU))
    
    # For the last GPU, include any remaining lines
    if [ $gpu -eq $((NUM_GPUS-1)) ]; then
        end_idx=$TOTAL_LINES
    else
        end_idx=$(((gpu + 1) * LINES_PER_GPU))
    fi
    
    # Create a new window for this GPU
    if [ $gpu -eq 0 ]; then
        # First GPU uses the default window
        tmux send-keys -t "$SESSION_NAME" "echo \"Starting extraction on GPU $gpu (indices $start_idx to $end_idx)\" && CUDA_VISIBLE_DEVICES=$gpu python3 -m rely.extract.activations --device \"cuda:0\" --start-index \"$start_idx\" --end-index \"$end_idx\" --model-name \"$MODEL_NAME\" --output-file \"nn-short-100-v2-${gpu}.pt\" --input-file \"$INPUT_FILE\"" Enter
    else
        # Create new window for other GPUs
        tmux new-window -t "$SESSION_NAME" -n "gpu$gpu"
        tmux send-keys -t "$SESSION_NAME:gpu$gpu" "echo \"Starting extraction on GPU $gpu (indices $start_idx to $end_idx)\" && CUDA_VISIBLE_DEVICES=$gpu python3 -m rely.extract.activations --device \"cuda:0\" --start-index \"$start_idx\" --end-index \"$end_idx\" --model-name \"$MODEL_NAME\" --output-file \"nn-short-100-v2-${gpu}.pt\" --input-file \"$INPUT_FILE\"" Enter
    fi
    
    echo "Created window for GPU $gpu (indices $start_idx to $end_idx)"
done

echo "All extraction jobs started in tmux session '$SESSION_NAME'"
echo "To monitor progress:"
echo "  tmux attach-session -t '$SESSION_NAME'"
echo "  tmux list-windows -t '$SESSION_NAME'"
echo ""
echo "To kill all jobs:"
echo "  tmux kill-session -t '$SESSION_NAME'"
echo ""
echo "Output files will be:"
for gpu in $(seq 0 $((NUM_GPUS-1))); do
    echo "  nn-short-100-v2-${gpu}.pt"
done 