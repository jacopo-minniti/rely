"""
Generate completions from fork data using vLLM.

This module provides functions for generating multiple completions from fork data
with support for data parallelism and distributed inference.
"""

import os
import argparse
from multiprocessing import Process
from time import sleep
from typing import Optional, List, Dict, Any

import torch
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


SYSTEM_PROMPT = """The following are multiple choice questions (with answers) about general knowledge. Think step by step and then finish your answer with 'The correct answer is (X)' where X is the correct letter choice.

EXAMPLE

Question: The quantum efficiency of a photon detector is 0.1. If 100 photons are sent into the detector, one after the other, the detector will detect photons
Options:
(A) an average of 10 times, with an rms deviation of about 4
(B) an average of 10 times, with an rms deviation of about 3
(C) an average of 10 times, with an rms deviation of about 1
(D) an average of 10 times, with an rms deviation of about 0.1

## Final Answer
[...Brief explanation of your answer...] The correct answer is (B).
"""


def complete_from_forks(
    input_file: str,
    output_file: str,
    n_completions_per_item: int = 100,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    model: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    trust_remote_code: bool = False,
    dp_size: int = 4,
    tp_size: int = 2,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 512,
    node_size: int = 1,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 13345
) -> None:
    """
    Generate completions from fork data using vLLM with data parallelism.

    Args:
        input_file: Input PT file path (fork dataset)
        output_file: Output PT file path
        n_completions_per_item: Number of completions to generate per item
        max_new_tokens: Max new tokens to generate
        temperature: Sampling temperature
        model: Model name or path
        trust_remote_code: Trust remote code (required for some models)
        dp_size: Total data parallel size (number of ranks across all nodes)
        tp_size: Tensor parallel size (number of GPUs per process)
        gpu_memory_utilization: GPU memory utilization fraction
        max_num_seqs: Maximum number of sequences per batch
        node_size: Total number of nodes for distributed run
        node_rank: Rank of the current node (0 to node_size-1)
        master_addr: Master node IP address for communication
        master_port: Master node port for communication
    """
    # Create argument namespace for compatibility with existing code
    class Args:
        def __init__(self):
            self.input = input_file
            self.output = output_file
            self.n_completions_per_item = n_completions_per_item
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            self.model = model
            self.trust_remote_code = trust_remote_code
            self.dp_size = dp_size
            self.tp_size = tp_size
            self.gpu_memory_utilization = gpu_memory_utilization
            self.max_num_seqs = max_num_seqs
            self.node_size = node_size
            self.node_rank = node_rank
            self.master_addr = master_addr
            self.master_port = master_port

    args = Args()

    if args.dp_size > 1:
        # Distributed Setup
        if args.node_size > 1:
            master_addr = args.master_addr
            master_port = args.master_port
        else:
            master_addr = "127.0.0.1"
            master_port = get_open_port()
            print(f"Single node run. Using master address {master_addr}:{master_port}")

        if args.dp_size % args.node_size != 0:
            raise ValueError("dp_size must be divisible by node_size.")
        
        dp_per_node = args.dp_size // args.node_size
        
        # Launch Worker Processes
        processes = []
        for local_dp_rank in range(dp_per_node):
            global_dp_rank = args.node_rank * dp_per_node + local_dp_rank
            
            proc = Process(
                target=_complete_worker,
                args=(args, global_dp_rank, local_dp_rank, master_addr, master_port),
            )
            proc.start()
            processes.append(proc)
        
        # Wait for Processes and Handle Exit
        exit_code = 0
        for proc in processes:
            proc.join(timeout=15000) # 30-minute timeout per process
            if proc.exitcode is None:
                print(f"Process {proc.pid} timed out. Killing...")
                proc.kill()
                exit_code = 1
            elif proc.exitcode != 0:
                print(f"Process {proc.pid} exited with error code {proc.exitcode}.")
                exit_code = proc.exitcode

        if exit_code == 0:
            _merge_output_files(args)
        else:
            print("One or more processes failed. Skipping file merge.")
            
        if exit_code != 0:
            raise RuntimeError(f"Completion failed with exit code {exit_code}")

    else:
        # Single Process Run (dp_size=1)
        print("Running in single-process mode (dp_size=1).")
        _complete_worker(
            args=args, 
            global_dp_rank=0, 
            local_dp_rank=0, 
            master_addr='127.0.0.1', 
            master_port=get_open_port()
        )
        base, ext = os.path.splitext(args.output)
        os.rename(f"{base}.rank0.pt", args.output)
        print(f"Processing complete! Output written to {args.output}")


def _complete_worker(args, global_dp_rank: int, local_dp_rank: int, master_addr: str, master_port: int) -> None:
    """
    Worker function for data parallel completion generation.
    
    Args:
        args: Arguments namespace
        global_dp_rank: Global data parallel rank
        local_dp_rank: Local data parallel rank
        master_addr: Master node IP address
        master_port: Master node port
    """
    print(f"Starting worker: global_rank={global_dp_rank}, local_rank={local_dp_rank}")
    
    # Set Environment Variables for vLLM
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_addr
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    # Data Parallel Splitting
    try:
        all_data = torch.load(args.input)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {args.input}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load PT file at {args.input}: {e}")
        return

    total_items = len(all_data)
    chunk_size = total_items // args.dp_size
    remainder = total_items % args.dp_size
    
    def get_start_index(rank):
        return rank * chunk_size + min(rank, remainder)

    start_idx = get_start_index(global_dp_rank)
    end_idx = get_start_index(global_dp_rank + 1)
    data_chunk = all_data[start_idx:end_idx]

    if not data_chunk:
        print(f"DP rank {global_dp_rank} has no items to process. Exiting worker.")
        return

    print(f"DP rank {global_dp_rank} preparing {len(data_chunk)} prompts (from index {start_idx} to {end_idx-1}).")

    # Prepare all prompts in a batch
    prompts_to_process = []
    source_metadata = []
    for item in data_chunk:
        try:
            question = item["question"]
            cut_cot = item["cut_cot"]
            solution = item["solution"]
            
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{cut_cot}\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.</think>\n## Final Answer\n"
            
            prompts_to_process.append(prompt)
            source_metadata.append({
                "question": question,
                "cut_cot": cut_cot,
                "solution": solution,
            })
        except Exception as e:
            print(f"Error preparing item on rank {global_dp_rank}: {e}")
            continue

    if not prompts_to_process:
        print(f"DP rank {global_dp_rank} has no valid prompts to process after filtering. Exiting.")
        return

    # Initialize vLLM
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=args.trust_remote_code,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True
    )

    # Generate completions for the entire batch
    print(f"DP rank {global_dp_rank} starting generation for {len(prompts_to_process)} prompts...")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        n=args.n_completions_per_item, 
    )
    all_outputs = llm.generate(prompts_to_process, sampling_params)

    # Process results and write to file
    base, ext = os.path.splitext(args.output)
    output_file_path = f"{base}.rank{global_dp_rank}.pt"
    processed_count = 0
    output_data = []
    for source_info, output_group in zip(source_metadata, all_outputs):
        completions = [out.text for out in output_group.outputs]
        out_data = {
            "question": source_info["question"],
            "cut_cot": source_info["cut_cot"],
            "solution": source_info["solution"],
            "completions": completions,
            "cut_cot_activations": None,
        }
        # Find the original item in data_chunk to get cut_cot_activations
        for item in data_chunk:
            if (
                item["question"] == source_info["question"]
                and item["cut_cot"] == source_info["cut_cot"]
                and item["solution"] == source_info["solution"]
            ):
                out_data["cut_cot_activations"] = item.get("cut_cot_activations", None)
                break
        output_data.append(out_data)
        processed_count += 1
    torch.save(output_data, output_file_path)
    print(f"DP rank {global_dp_rank} finished. Processed {processed_count} examples.")
    print(f"Output for rank {global_dp_rank} written to {output_file_path}")
    sleep(1)


def _merge_output_files(args) -> None:
    """Merges the output files from each rank into a single .pt file."""
    print("Merging output files...")
    base, ext = os.path.splitext(args.output)
    merged_data = []
    for rank in range(args.dp_size):
        rank_file = f"{base}.rank{rank}.pt"
        try:
            part_data = torch.load(rank_file)
            if isinstance(part_data, list):
                merged_data.extend(part_data)
            else:
                print(f"Warning: {rank_file} does not contain a list, skipping.")
                continue
            os.remove(rank_file) # Clean up the temporary file
            print(f"Merged and removed {rank_file}")
        except FileNotFoundError:
            print(f"Warning: Output file for rank {rank} not found at {rank_file}. It might have had no data to process.")
    torch.save(merged_data, args.output)
    print(f"Final merged output written to {args.output}")


def main():
    """Command-line interface for complete_from_forks."""
    parser = argparse.ArgumentParser(description="vLLM Data Parallel Inference for Uncertainty Probe Data")
    # Application Arguments
    parser.add_argument("--input", type=str, default="forks-mmlu-qwen3-1.7B.pt", help="Input PT file path (fork dataset)")
    parser.add_argument("--output", type=str, default="short-completions-mmlu-qwen3-1.7B.pt", help="Output PT file path")
    parser.add_argument("--n-completions-per-item", type=int, default=100, help="Number of completions to generate per item")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    # vLLM & Distributed Arguments
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit", help="Model name or path")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code (required for some models)")
    parser.add_argument("--dp-size", type=int, default=4, help="Total data parallel size (number of ranks across all nodes)")
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size (number of GPUs per process)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilization fraction")
    parser.add_argument("--max-num-seqs", type=int, default=512, help="Maximum number of sequences per batch")
    
    # Multi-Node Arguments
    parser.add_argument("--node-size", type=int, default=1, help="Total number of nodes for distributed run")
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of the current node (0 to node_size-1)")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="Master node IP address for communication")
    parser.add_argument("--master-port", type=int, default=13345, help="Master node port for communication")
    
    args = parser.parse_args()

    complete_from_forks(
        input_file=args.input,
        output_file=args.output,
        n_completions_per_item=args.n_completions_per_item,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        node_size=args.node_size,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port
    )


if __name__ == "__main__":
    main() 