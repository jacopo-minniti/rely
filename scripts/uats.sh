NUM_GPUS=3
SESSION_NAME="uats_parallel"

tmux new-session -d -s "$SESSION_NAME"

for ((i=0; i<$NUM_GPUS; i++)); do
    if [ $i -ne 0 ]; then
        tmux split-window -t "$SESSION_NAME":0 -h
    fi
    tmux select-pane -t "$SESSION_NAME":0.$i
    tmux send-keys -t "$SESSION_NAME":0.$i "CUDA_VISIBLE_DEVICES=$i python3 uats.py --device cuda:0 --uncertainty_device cuda:0 --value_device cuda:0" C-m
done

# Arrange panes evenly
tmux select-layout -t "$SESSION_NAME":0 tiled

echo "Started $NUM_GPUS parallel runs in tmux session '$SESSION_NAME'. Attach with: tmux attach-session -t $SESSION_NAME"
