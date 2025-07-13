import argparse
import time
import random
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from dotenv import load_dotenv
from tqdm import tqdm

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
    else:
        activation_extractor = None
    
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
    all_examples = list(source_dataset)
    new_data_points = []

    def process_example(example):
        start_time = time.time()
        is_newly_processed = False
        try:
            problem = example['question']
            problem_hash = hashlib.sha256(problem.encode()).hexdigest()

            data_point = {}
            score = -1
            activations = torch.Tensor()

            if problem_hash in processed_data:
                data_point = processed_data[problem_hash]
                score = data_point.get('prm_score', -1)
                activations = data_point.get('activations', torch.Tensor())
            else:
                is_newly_processed = True

            ground_truth = example['cleaned_solution']
            decomposed_cot = example['decomposed_cot']
            steps = parse_decomposed_cot(decomposed_cot)
            
            if not steps:
                return None

            step_idx = data_point.get('step_index')
            if step_idx is None:
                step_idx = random.randint(0, len(steps) - 1)
                data_point['step_index'] = step_idx

            continuous_cot_prompt = ""
            for j in range(step_idx + 1):
                raw_step_content = steps[j].replace("<step>", "").replace("</step>", "").strip()
                continuous_cot_prompt += ("\n" if continuous_cot_prompt else "") + raw_step_content

            discrete_cot_prompt = convert_to_discrete_cot(continuous_cot_prompt)

            # --- Evaluation Logic ---
            if args.mode in ['eval', 'both'] and score == -1:
                score = evaluator.evaluate(
                    problem,
                    continuous_cot_prompt,
                    ground_truth,
                    args.max_answer_tokens,
                    args.max_reasoning_tokens,
                    n=args.n,
                    hard=False
                )

            # --- Extraction Logic ---
            if args.mode in ['extract', 'both'] and activations.numel() == 0 and activation_extractor:
                full_step_prompt_for_extraction = inference_engine._generate_full_prompt(
                    problem, 
                    discrete_cot_prompt
                )
                activations = activation_extractor.get_step_activations(full_step_prompt_for_extraction)
                if activations is None:
                    print("⚠️ Skipping activation due to empty tensor")
                    activations = torch.Tensor()

            # Update data_point dictionary
            data_point.update({
                "problem": problem,
                "step_content": discrete_cot_prompt, # Save the discrete CoT
                "activations": activations,
                "prm_score": score,
                "step_index": step_idx
            })
            duration = time.time() - start_time
            return data_point, duration, is_newly_processed

        except Exception as e:
            print(f"❌ Error processing an example. Skipping. Error: {e}")
            duration = time.time() - start_time
            return None, duration, False

    start_processing_time = time.time()
    completed_steps = 0
    newly_processed_count = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_example, example) for example in all_examples]
        
        for future in tqdm(as_completed(futures), total=len(all_examples), desc="Processing dataset"):
            result, duration, is_new = future.result()
            completed_steps += 1
            
            if is_new:
                newly_processed_count += 1

            total_elapsed_time = time.time() - start_processing_time
            steps_per_minute = (newly_processed_count / total_elapsed_time) * 60 if total_elapsed_time > 0 else 0
            
            tqdm.write(f"Step processed in {duration:.2f} seconds. Rate: {steps_per_minute:.2f} steps/min.")
            if result:
                new_data_points.append(result)
                
                problem_hash = hashlib.sha256(result['problem'].encode()).hexdigest()
                processed_data[problem_hash] = result

                # Save the batch of data points periodically
                if len(new_data_points) >= args.save_interval:
                    file_path = os.path.join(output_dir, f"batch_{int(time.time())}.pt")
                    torch.save(new_data_points, file_path)
                    print(f"💾 Saved batch of {len(new_data_points)} data points to {file_path}")
                    new_data_points = []

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
    parser.add_argument("--max_reasoning_tokens", type=int, default=1200, help="Maximum number of tokens for reasoning.")
    parser.add_argument("--output_dataset_path", type=str, default="./value-head-s1", help="Path to save the output dataset files.")
    parser.add_argument("--inference_mode", type=str, default="vllm", choices=["vllm", "api"], help="Inference mode to use.")
    parser.add_argument("--n", type=int, default=4, help="Number of completions to generate for evaluation.")
    parser.add_argument("--save_interval", type=int, default=50, help="How many data points to batch before saving to a new file.")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "eval", "extract"], help="Operation mode: 'both', 'eval', or 'extract'.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of parallel workers for processing the dataset.")
    
    args = parser.parse_args()
    math_shepherd(args)
