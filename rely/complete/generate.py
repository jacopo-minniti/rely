"""
Generate completions from Hugging Face datasets using vLLM.

This module provides functions for generating model completions from datasets
with support for data parallelism and distributed inference.
"""

import argparse
import json
import os
import logging
from multiprocessing import Process
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
from datasets import load_dataset
from rely.utils.load import load_dataset, save_dataset
from rely.utils.text_utils import format_prompt, MMLU_SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_from_dataset(
    dataset_name: str,
    output_file: str,
    model: str = "Qwen/Qwen3-8B",
    dataset_split: str = "train",
    dataset_subset: Optional[str] = None,
    num_samples: Optional[int] = None,
    question_field: str = "question",
    options_field: str = "options", 
    answer_index_field: str = "answer_index",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tp_size: int = 1,
    dp_size: int = 8,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 512,
    dtype: str = "bfloat16",
    system_prompt: str = MMLU_SYSTEM_PROMPT,
    prompt_formatter: Optional[callable] = None,
) -> None:
    """
    Generate completions from a Hugging Face dataset using vLLM with data parallelism.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        output_file: Path to the output JSONL file
        model: The model identifier for vLLM
        dataset_split: The dataset split to use (e.g., 'train', 'validation')
        dataset_subset: The dataset subset or configuration (if applicable)
        num_samples: Number of samples to process from the dataset
        question_field: The field in the dataset for the question text
        options_field: The field in the dataset for the multiple-choice options
        answer_index_field: The field in the dataset for the correct answer's index
        temperature: Sampling temperature for the model
        max_tokens: Maximum number of tokens to generate
        tp_size: Number of tensor parallel replicas PER data parallel rank
        dp_size: Total data parallel size (number of ranks across all nodes)
        gpu_memory_utilization: GPU memory utilization fraction
        max_num_seqs: Maximum number of sequences per batch
        dtype: Model data type
        system_prompt: The system prompt to use for formatting prompts.
        prompt_formatter: A callable that formats the user prompt with the system prompt.
    """
    # Create argument namespace for compatibility with existing code
    class Args:
        def __init__(self):
            self.dataset = dataset_name
            self.output_file = output_file
            self.model = model
            self.dataset_split = dataset_split
            self.dataset_subset = dataset_subset
            self.num_samples = num_samples
            self.question_field = question_field
            self.options_field = options_field
            self.answer_index_field = answer_index_field
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.tp = tp_size
            self.gpu_memory_utilization = gpu_memory_utilization
            self.max_num_seqs = max_num_seqs
            self.dtype = dtype
            self.system_prompt = system_prompt
            self.prompt_formatter = prompt_formatter or (lambda user_prompt: format_prompt(user_prompt, system_prompt))

    args = Args()

    # Multiprocessing setup for data parallelism
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    procs = []
    
    for i in range(dp_size):
        proc = Process(target=_generate_worker,
                      args=(args, dp_size, i, dp_master_ip, dp_master_port))
        proc.start()
        procs.append(proc)
    
    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            logging.error(f"Process {proc.pid} exited with code {proc.exitcode}")
            exit_code = proc.exitcode

    if exit_code != 0:
        raise RuntimeError(f"Generation failed with exit code {exit_code}")


def _generate_worker(args, dp_size: int, dp_rank: int, dp_master_ip: str, dp_master_port: int) -> None:
    """
    Worker function for data parallel generation.
    
    Args:
        args: Arguments namespace
        dp_size: Total data parallel size
        dp_rank: Current data parallel rank
        dp_master_ip: Master node IP address
        dp_master_port: Master node port
    """
    # Set Environment Variables for Data Parallelism
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # Initialize vLLM
    logging.info(f"[Rank {dp_rank}] Initializing vLLM...")
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            dtype=args.dtype
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    except Exception as e:
        logging.error(f"[Rank {dp_rank}] Failed to initialize vLLM: {e}")
        return

    # Load Hugging Face Dataset
    logging.info(f"[Rank {dp_rank}] Loading dataset '{args.dataset}'...")
    try:
        from datasets import load_dataset as hf_load_dataset
        if args.dataset_subset is not None:
            dataset = hf_load_dataset(args.dataset, name=args.dataset_subset, split=args.dataset_split)
        else:
            dataset = hf_load_dataset(args.dataset, split=args.dataset_split)
    except Exception as e:
        logging.error(f"[Rank {dp_rank}] Failed to load dataset: {e}")
        return

    # Determine Data Slice for this Rank
    if args.num_samples:
        dataset = dataset.select(range(args.num_samples))
    
    total_samples = len(dataset)
    samples_per_rank = total_samples // dp_size
    start_idx = dp_rank * samples_per_rank
    end_idx = (dp_rank + 1) * samples_per_rank
    if dp_rank == dp_size - 1:
        end_idx = total_samples  # Ensure the last rank gets all remaining samples

    # Extract and Format Prompts for this Rank's Slice
    full_prompts = []
    formatted_llm_prompts = []
    original_records = [] 
    
    logging.info(f"[Rank {dp_rank}] Preparing prompts from dataset slice (indices {start_idx} to {end_idx-1})...")
    
    # Create a list of indices for this rank to process
    rank_indices = list(range(start_idx, end_idx))
    
    for i in tqdm(rank_indices, desc=f"[Rank {dp_rank}] Preparing Prompts"):
        example = dataset[i]
        
        question = example.get(args.question_field)
        options = example.get(args.options_field)
        
        if not all([question, isinstance(question, str), options, isinstance(options, list)]):
            logging.warning(f"\n[Rank {dp_rank}] Skipping sample {i} due to missing/invalid question or options fields.")
            continue
            
        # Build the user-facing part of the prompt
        options_str = []
        for j, opt_list in enumerate(options):
            letter = chr(ord('A') + j)
            option_text = ", ".join(opt_list)
            options_str.append(f"({letter}) {option_text}")
            
        full_prompt = f"Question: {question}\n\nOptions:\n" + "\n".join(options_str)
        
        full_prompts.append(full_prompt)
        formatted_llm_prompts.append(args.prompt_formatter(full_prompt))
        original_records.append(example)

    if not formatted_llm_prompts:
        logging.error(f"[Rank {dp_rank}] No valid prompts were generated for this rank. Exiting.")
        return
        
    logging.info(f"[Rank {dp_rank}] Generated {len(formatted_llm_prompts)} prompts to process.")

    # Run vLLM Inference
    logging.info(f"[Rank {dp_rank}] Starting vLLM inference...")
    outputs = llm.generate(formatted_llm_prompts, sampling_params)
    logging.info(f"[Rank {dp_rank}] Inference complete.")

    # Save Results
    base, ext = os.path.splitext(args.output_file)
    rank_output_file = f"{base}_{dp_rank}{ext}"

    with open(rank_output_file, 'w', encoding='utf-8') as outfile:
        sorted_outputs = sorted(outputs, key=lambda x: int(x.request_id))
        
        for i, output in enumerate(tqdm(sorted_outputs, desc=f"[Rank {dp_rank}] Saving Results")):
            assistant_response = output.outputs[0].text.strip()
            
            original_record = original_records[i]
            full_question_prompt = full_prompts[i]
            
            answer_idx = original_record.get(args.answer_index_field)
            options_list = original_record.get(args.options_field)
            
            correct_letter = None
            if answer_idx is not None and options_list and 0 <= answer_idx < len(options_list):
                correct_letter = chr(ord('A') + answer_idx)

            result = {
                "question": full_question_prompt,
                "attempt": assistant_response,
                "solution": correct_letter
            }
            outfile.write(json.dumps(result) + '\n')

    logging.info(f"[Rank {dp_rank}] Processing complete. Results saved to {rank_output_file}")


def main():
    """Command-line interface for generate_from_dataset."""
    parser = argparse.ArgumentParser(description="Query a local vLLM model with a Hugging Face dataset using Data Parallelism.")

    # Dataset Arguments
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--dataset-split", type=str, default="train", help="The dataset split to use (e.g., 'train', 'validation').")
    parser.add_argument("--dataset-subset", type=str, default=None, help="The dataset subset or configuration (if applicable).")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process from the dataset. Processes all by default.")
    
    parser.add_argument("--question-field", type=str, default="question", help="The field in the dataset for the question text.")
    parser.add_argument("--options-field", type=str, default="options", help="The field in the dataset for the multiple-choice options.")
    parser.add_argument("--answer-index-field", type=str, default="answer_index", help="The field in the dataset for the correct answer's index.")

    # vLLM & Model Arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="The model identifier for vLLM.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output JSONL file. Rank will be appended.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for the model.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum number of tokens to generate.")
    parser.add_argument("--tp", type=int, default=1, help="Number of tensor parallel replicas PER data parallel rank.")
    parser.add_argument("--dp-size", type=int, default=8, help="Total data parallel size.")

    args = parser.parse_args()

    generate_from_dataset(
        dataset_name=args.dataset,
        output_file=args.output_file,
        model=args.model,
        dataset_split=args.dataset_split,
        dataset_subset=args.dataset_subset,
        num_samples=args.num_samples,
        question_field=args.question_field,
        options_field=args.options_field,
        answer_index_field=args.answer_index_field,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tp_size=args.tp,
        dp_size=args.dp_size
    )


if __name__ == "__main__":
    main() 