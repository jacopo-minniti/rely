import argparse
import time
import random
import os
import hashlib

import torch
from dotenv import load_dotenv

from rely.inference import VLLMInference, APIInference
from rely.extract import ActivationsExtractor
from rely.evaluate import Evaluator
from rely.utils import load_data, parse_decomposed_cot, convert_to_discrete_cot

load_dotenv(override=True)

def math_shepherd(args):
    print("🚀 Starting dataset creation process...")

    # 1. Initialize utilities
    if args.inference_mode == "vllm":
        inference_engine = VLLMInference(args.inference_model_name)
    elif args.inference_mode == "api":
        inference_engine = APIInference(args.inference_model_name)
    else:
        raise ValueError(f"Invalid INFERENCE_MODE: {args.inference_mode}")

    # load model only if needed
    if args.mode in ['extract', 'both']:
        activation_extractor = ActivationsExtractor(args.extractor_model_name)
    
    evaluator = Evaluator(inference_engine)
    
    # 2. Load source data
    print(f"Loading source dataset: {args.source_dataset_name}/")
    source_dataset = load_data(args.source_dataset_name)
    
    output_dir = args.output_dataset_path
    os.makedirs(output_dir, exist_ok=True)

    print("Loading existing processed data...")
    processed_data = {}
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith(".pt"):
                try:
                    file_path = os.path.join(output_dir, filename)
                    batch = torch.load(file_path)
                    # Handle both list of dicts and single dict for backward compatibility
                    if isinstance(batch, dict):
                        batch = [batch]
                    for data_point in batch:
                        if 'problem' in data_point:
                            problem_hash = hashlib.sha256(data_point['problem'].encode()).hexdigest()
                            processed_data[problem_hash] = data_point
                except Exception as e:
                    print(f"⚠️  Warning: Could not load or process file {filename}. Error: {e}")
    print(f"Found {len(processed_data)} existing data points.")
    
    # 3. Iterate through the source dataset
    new_data_points = []
    for i, example in enumerate(source_dataset):
        try:
            problem = example['question']
            
            # Generate a unique hash for the problem
            problem_hash = hashlib.sha256(problem.encode()).hexdigest()

            # Initialize variables
            data_point = {}
            score = -1
            activations = torch.Tensor()

            # Load existing data point if it exists in our cache
            if problem_hash in processed_data:
                print(f"Loading existing data point for problem {i+1}/{len(source_dataset)}")
                data_point = processed_data[problem_hash]
                score = data_point.get('prm_score', -1)
                activations = data_point.get('activations', torch.Tensor())
            
            ground_truth = example['cleaned_solution']
            decomposed_cot = example['decomposed_cot']
            
            steps = parse_decomposed_cot(decomposed_cot)
            
            if not steps:
                print(f"    -> ⚠️ No steps found for example {i}. Skipping.")
                continue

            # Use existing step index or randomly sample one
            step_idx = data_point.get('step_index')
            if step_idx is None:
                step_idx = random.randint(0, len(steps) - 1)
                data_point['step_index'] = step_idx

            print(f"\nProcessing example {i+1}/{len(source_dataset)}: '{problem[:50]}...'\tStep {step_idx + 1}/{len(steps)}")
            
            # Reconstruct the continuous CoT prompt up to the sampled step
            continuous_cot_prompt = ""
            for j in range(step_idx + 1):
                raw_step_content = steps[j].replace("<step>", "").replace("</step>", "").strip()
                continuous_cot_prompt += ("\n" if continuous_cot_prompt else "") + raw_step_content

            # --- Evaluation Logic ---
            if args.mode in ['eval', 'both'] and score == -1:
                print(f"  - Evaluating Step {step_idx + 1}/{len(steps)}...")
                start = time.time()
                score = evaluator.evaluate(
                    problem,
                    continuous_cot_prompt,
                    ground_truth,
                    args.max_answer_tokens,
                    args.max_reasoning_tokens,
                    n=args.n,
                    hard=False
                )
                print(f"    -> Soft Score: {score:.2f} ({int(time.time() - start)} seconds)")
            elif score != -1:
                 print(f"  - Skipping evaluation, score already exists: {score:.2f}")


            # --- Extraction Logic ---
            if args.mode in ['extract', 'both'] and activations.numel() == 0:
                print(f"  - Extracting activations for Step {step_idx + 1}...")
                discrete_cot_prompt = convert_to_discrete_cot(continuous_cot_prompt)
                full_step_prompt_for_extraction = inference_engine._generate_full_prompt(
                    problem, 
                    discrete_cot_prompt
                )
                activations = activation_extractor.get_step_activations(full_step_prompt_for_extraction)
                
                if activations is None:
                    print(f"    -> ⚠️ Could not extract activations. Storing empty tensor.")
                    activations = torch.Tensor()
                else:
                    print(f"    -> Activations Tensor Shape: {activations.shape}")
            elif activations.numel() > 0:
                print(f"  - Skipping extraction, activations already exist. Shape: {activations.shape}")


            # Update data_point dictionary
            data_point.update({
                "problem": problem,
                "step_content": continuous_cot_prompt,
                "activations": activations,
                "prm_score": score
            })

            # Add the data point to the batch to be saved
            new_data_points.append(data_point)
            processed_data[problem_hash] = data_point # Update cache to avoid reprocessing in same run

            # Save the batch of data points periodically
            if len(new_data_points) >= args.save_interval:
                file_path = os.path.join(output_dir, f"batch_{int(time.time())}.pt")
                torch.save(new_data_points, file_path)
                print(f"💾 Saved batch of {len(new_data_points)} data points to {file_path}")
                new_data_points = []

        except Exception as e:
            print(f"❌ Error processing example {i+1}. Skipping. Error: {e}")
            continue

    # Save any remaining data points
    if new_data_points:
        file_path = os.path.join(output_dir, f"batch_{int(time.time())}.pt")
        torch.save(new_data_points, file_path)
        print(f"💾 Saved final batch of {len(new_data_points)} data points to {file_path}")
        
    print("\n🎉 Dataset creation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dataset creation process.")
    parser.add_argument("--inference_model_name", type=str, default="Qwen/Qwen3-30B-A3B", help="Name of the inference model.")
    parser.add_argument("--extractor_model_name", type=str, default="jacopo-minniti/Qwen2.5-7B-base", help="Name of the extractor model.")
    parser.add_argument("--source_dataset_name", type=str, default="jacopo-minniti/s1k-deepseek-base", help="Name of the source dataset.")
    parser.add_argument("--max_answer_tokens", type=int, default=500, help="Maximum number of tokens for the answer.")
    parser.add_argument("--max_reasoning_tokens", type=int, default=1000, help="Maximum number of tokens for reasoning.")
    parser.add_argument("--output_dataset_path", type=str, default="./value-head-s1", help="Path to save the output dataset files.")
    parser.add_argument("--inference_mode", type=str, default="vllm", choices=["vllm", "api"], help="Inference mode to use.")
    parser.add_argument("--n", type=int, default=4, help="Number of completions to generate for evaluation.")
    parser.add_argument("--save_interval", type=int, default=5, help="How many data points to batch before saving to a new file.")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "eval", "extract"], help="Operation mode: 'both', 'eval', or 'extract'.")
    
    args = parser.parse_args()
    math_shepherd(args)
