import argparse
from datasets import load_dataset

from rely.inference.uats import UATSConfig, run_uats
from rely.utils import MATH_SYSTEM_PROMPT

def parse_args():
    parser = argparse.ArgumentParser(description="Run UATS with configurable arguments.")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument('--uncertainty_model_path', type=str, default="jacopo-minniti/Qwen2.5-Math-7B-PUM")
    parser.add_argument('--value_model_path', type=str, default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--uncertainty_device', type=str, default="cuda:0")
    parser.add_argument('--value_device', type=str, default="cuda:0")
    parser.add_argument('--beam_width', type=int, default=4)
    parser.add_argument('--budget', type=int, default=4000)
    parser.add_argument('--uncertainty_threshold', type=float, default=0.9)
    parser.add_argument('--uncertainty_scoring_method', type=str, default="last_step")
    parser.add_argument('--value_scoring_method', type=str, default="product")
    parser.add_argument('--max_step_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    # Dataset name and split are now constants below
    parser.add_argument('--num_gpus', type=int, default=1, help='Total number of GPUs running in parallel')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of this GPU (0-based)')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = UATSConfig(
        model_name=args.model_name,
        uncertainty_model_path=args.uncertainty_model_path,
        value_model_path=args.value_model_path,
        device=args.device,
        uncertainty_device=args.uncertainty_device,
        value_device=args.value_device,
        beam_width=args.beam_width,
        budget=args.budget,
        uncertainty_threshold=args.uncertainty_threshold,
        uncertainty_scoring_method=args.uncertainty_scoring_method,
        value_scoring_method=args.value_scoring_method,
        max_step_tokens=args.max_step_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    DATASET_NAME = "nlile/hendrycks-MATH-benchmark"
    DATASET_SPLIT = "test"
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

    # Split dataset among GPUs
    num_gpus = args.num_gpus
    gpu_index = args.gpu_index
    samples_per_gpu = len(dataset) // num_gpus
    start_idx = gpu_index * samples_per_gpu
    # Last GPU takes the remainder
    end_idx = (gpu_index + 1) * samples_per_gpu if gpu_index < num_gpus - 1 else len(dataset)
    sub_dataset = dataset.select(range(start_idx, end_idx))

    for local_idx, item in enumerate(sub_dataset):
        global_idx = start_idx + local_idx
        question = item["problem"]
        answer = item["answer"]
        run_uats(
            user_question=question,
            correct_answer=answer,
            system_prompt=MATH_SYSTEM_PROMPT,
            config=config,
            save_dir=f"uats_results/question_{global_idx}"
        )
