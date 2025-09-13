#!/bin/bash
#SBATCH --job-name=sbs
#SBATCH --account=aip-rudner
#SBATCH --time=05:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=sbs-%j.out
#SBATCH --error=sbs-%j.err
#SBATCH -D /scratch/jacopo04

export HF_HOME="/scratch/jacopo04/.cache"

# --- Robust Cleanup ---
# This function will be called when the script exits for any reason.
cleanup() {
    echo "Script finished or was interrupted. Cleaning up..."
    # Check if the VLLM_PID variable is set and kill the process
    if [ -n "$VLLM_PID" ]; then
        echo "Killing vLLM server with PID: $VLLM_PID"
        kill $VLLM_PID
    fi
}

# The 'trap' command ensures the 'cleanup' function runs on exit, interruption, or termination.
trap cleanup EXIT SIGHUP SIGINT SIGTERM

# --- Environment Setup ---
# Load any necessary modules
module --force purge
module load StdEnv/2020
module load StdEnv/2023
module load python/3.11.5
module load gcc opencv arrow

# Activate your venv
source /scratch/jacopo04/.venv/bin/activate

# --- vLLM Server Launch ---
echo "Starting vLLM server in the background..."
export CUDA_VISIBLE_DEVICES="0"

# Start the server and add '&' to run it in the background
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4000 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --port 8000 &

# Store the Process ID (PID) of the last background command ($!)
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# --- Active Wait for Server ---
echo "⏳ Waiting for server to become available at http://localhost:8000 ..."
while ! curl -s http://localhost:8000/health > /dev/null; do
    echo "   ...server not ready yet. Retrying in 5 seconds."
    sleep 10
done
echo "Server is ready!"

# --- Main Script Execution ---
echo "Running main Python inference script..."

# vLLM server runs on GPU 0. Worker jobs run on GPUs 1, 2, ...
# We subtract 1 from the total number of GPUs to get the number of worker jobs.
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
NUM_JOBS=$((NUM_GPUS - 1))

if [ "$NUM_JOBS" -lt 1 ]; then
    echo "Error: At least 2 GPUs are required: 1 for vLLM and at least 1 for a worker."
    exit 1
fi

TOTAL_SAMPLES=500
SAMPLES_PER_JOB=$(( (TOTAL_SAMPLES + NUM_JOBS - 1) / NUM_JOBS ))

for i in $(seq 0 $((NUM_JOBS - 1))); do
    GPU_ID=$((i + 1)) # Worker GPUs start from 1
    START_IDX=$((i * SAMPLES_PER_JOB))
    END_IDX=$(( (i + 1) * SAMPLES_PER_JOB ))
    if [ $i -eq $((NUM_JOBS - 1)) ]; then
        END_IDX=$TOTAL_SAMPLES
    fi

    echo "Starting job $i on GPU $GPU_ID with indices $START_IDX to $END_IDX"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python /scratch/jacopo04/rely/rely/inference/sbs.py \
        --dataset nlile/hendrycks-MATH-benchmark \
        --output_dir ./flop_results/sbs_max_3_20 \
        --value_model_path Qwen/Qwen2.5-Math-PRM-7B \
        --num_workers 10 \
        --value_model_gpu 0 \
        --value_method product \
        --beam_width 3 \
        --n_samples 20 \
        --max_depth 200 \
        --idx_start $START_IDX \
        --idx_end $END_IDX &
done

wait # Wait for all background jobs to finish

echo "🎉 SBS run successfully!"

