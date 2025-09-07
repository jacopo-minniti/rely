#!/usr/bin/env bash

# --- run_tmux_parallel.sh ---
# Launch four parallel runs of entropy-threshold.py in a tmux session, one per GPU.
# Each pane processes a distinct quarter of the dataset and writes its entropies
# to entropy_outputs/entropies_<shard>.npy.

set -euo pipefail

DATASET="generations-mmlu-qwen3-8B.jsonl"
OUTPUT_DIR="entropy_outputs"
SCRIPT="entropy-threshold.py"
SESSION="entropy_run"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Determine total number of prompts (lines)
TOTAL_PROMPTS=$(wc -l < "$DATASET")
CHUNK_SIZE=$(( (TOTAL_PROMPTS + 3) / 4 )) # round-up division into 4 shards

# Start (or kill and restart) tmux session
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n entropy

for SHARD in {0..7}; do
  START=$(( SHARD * CHUNK_SIZE ))
  END=$(( (SHARD + 1) * CHUNK_SIZE ))
  if [ "$END" -gt "$TOTAL_PROMPTS" ]; then END=$TOTAL_PROMPTS; fi

  CMD="CUDA_VISIBLE_DEVICES=$SHARD python $SCRIPT --start_idx $START --end_idx $END --output $OUTPUT_DIR/entropies_$SHARD.npy"

  if [ "$SHARD" -eq 0 ]; then
    tmux send-keys -t "$SESSION" "$CMD" C-m
  else
    tmux split-window -t "$SESSION" -h
    tmux select-layout -t "$SESSION" tiled
    tmux send-keys -t "$SESSION" "$CMD" C-m
  fi

done

tmux select-layout -t "$SESSION" tiled

echo "All shards started. Attach with: tmux attach -t $SESSION" 