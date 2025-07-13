import argparse
import os
import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from rely.utils import check_answer, parse_decomposed_cot, convert_to_discrete_cot
from datasets import load_dataset
import random

'''
pip install vllm openai datasets tqdm transformers accelerate dotenv "ray[data]" "vllm[runai]"
engine_kwargs={
    "tensor_parallel_size": args.num_gpus,
    "data_parallel_size": 1,
    "enable_prefix_caching": True,
    "dtype": "bfloat16",
    "enable_prefix_caching": True,
    "enable_expert_parallel": True,
    "generation_config": "auto",
    "load_format": "runai_streamer"
},
'''


def create_full_prompt(problem, previous_steps=""):
    """
    Creates the full prompt for the model based on the problem and previous steps.
    """
    system_prompt = "You are a helpful assistant. Please solve the following math problem by thinking step-by-step enclosed in <think> tags. Do not use <step> tags, just newlines between steps."
    cot = "<think>"
    if previous_steps:
        cot += previous_steps
    
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n{cot}"
    )
    return prompt

def evaluate_with_ray(args):
    """
    Runs the evaluation process using Ray for offline batch inference.
    """
    print("🚀 Initializing Ray for offline evaluation...")
    ray.init()

    # 1. Load the source dataset
    print(f"📂 Loading source dataset: {args.source_dataset_name}")
    source_dataset = load_dataset(args.source_dataset_name, split="train")
    ds = ray.data.from_huggingface(source_dataset)

    # 2. Configure the vLLM engine with parameters from your serve.sh
    print("⚙️  Configuring the vLLM engine...")
    config = vLLMEngineProcessorConfig(
        model_source=args.inference_model_name,
        engine_kwargs={
            "tensor_parallel_size": args.num_gpus,
            "data_parallel_size": 1,
            "enable_prefix_caching": True,
            "dtype": "bfloat16",
            "enable_prefix_caching": True,
            "enable_expert_parallel": True,
            "generation_config": "auto",
            "load_format": "runai_streamer"
        },
        # Assuming one model replica that spans all GPUs
        concurrency=1, 
        # The batch size for feeding data into the vLLM engine
        batch_size=args.batch_size  
    )

    # 3. Build the LLM processor
    processor = build_llm_processor(
        config,
        # Preprocess each row of the dataset to create the prompt
        preprocess=lambda row: {
            "messages": create_full_prompt(
                row["question"],
                # Process a random step from the decomposed CoT
                "\n".join(
                    s.replace("<step>", "").replace("</step>", "").strip() 
                    for s in parse_decomposed_cot(row['decomposed_cot'])[:random.randint(0, len(parse_decomposed_cot(row['decomposed_cot'])))]
                )
            ),
            "sampling_params": {
                "temperature": 1.0,
                "max_tokens": args.max_answer_tokens + args.max_reasoning_tokens,
                "n": args.n,
                "stop": "</think>"
            },
            # Pass through original columns needed for postprocessing
            "original_data": row
        },
        # Postprocess the results to calculate the score
        postprocess=lambda result: {
            "problem": result["original_data"]["question"],
            "ground_truth": result["original_data"]["cleaned_solution"],
            "generated_answers": [choice["text"] for choice in result["raw_response"]["choices"]],
            "prm_score": sum(
                1 for choice in result["raw_response"]["choices"] 
                if check_answer(choice["text"], result["original_data"]["cleaned_solution"])
            ) / args.n
        }
    )

    # 4. Run the batch inference
    print("⚡️ Starting batch inference...")
    evaluated_ds = processor(ds)

    # 5. Show and save the results
    print("📊 Evaluation Results:")
    evaluated_ds.show(limit=10)

    output_path = os.path.join(args.output_dataset_path, "ray_evaluation_results")
    print(f"💾 Saving results to {output_path}")
    evaluated_ds.write_parquet(output_path)

    print("\n🎉 Ray evaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline evaluation with Ray.")
    
    # Arguments from your main.py and serve.sh
    parser.add_argument("--inference_model_name", type=str, default="Qwen/Qwen3-30B-A3B", help="Name of the inference model.")
    parser.add_argument("--source_dataset_name", type=str, default="jacopo-minniti/s1k-deepseek-base", help="Name of the source dataset.")
    parser.add_argument("--max_answer_tokens", type=int, default=500, help="Maximum number of tokens for the answer.")
    parser.add_argument("--max_reasoning_tokens", type=int, default=1200, help="Maximum number of tokens for reasoning.")
    parser.add_argument("--output_dataset_path", type=str, default="./value-head-s1-ray", help="Path to save the output dataset files.")
    parser.add_argument("--n", type=int, default=4, help="Number of completions to generate for evaluation.")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use (matches tensor_parallel_size).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for Ray Data processing.")

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dataset_path):
        os.makedirs(args.output_dataset_path)
        
    evaluate_with_ray(args)