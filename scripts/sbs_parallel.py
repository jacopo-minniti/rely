import re
import logging
import time
import os
import json
from collections import Counter
from multiprocessing import Process

import torch
import torch.nn as nn
from unsloth import FastModel
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
from tqdm import tqdm
import argparse

# --- Configuration ---

# Disable torch compilation to prevent recompilation issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """The following are multiple choice questions (with answers) about science. Think step by step and then finish your answer with 'The correct answer is (X)' where X is the correct letter choice.

EXAMPLE

Question: The quantum efficiency of a photon detector is 0.1. If 100 photons are sent into the detector, one after the other, the detector will detect photons
Options:
(A) an average of 10 times, with an rms deviation of about 4
(B) an average of 10 times, with an rms deviation of about 3
(C) an average of 10 times, with an rms deviation of about 1
(D) an average of 10 times, with an rms deviation of about 0.1

## Your Example Answer
[...Explanation...] The correct answer is (B).
"""

# --- Core SBS Implementation (Logic Unchanged) ---

@dataclass
class SBSConfig:
    """Configuration for Step-level Beam Search"""
    step_beam_width: int = 3
    n_generate_sample: int = 5
    max_depth: int = 10
    temperature: float = 0.6
    max_tokens: int = 512
    need_value_func: bool = True
    remove_duplicate: bool = True
    verbose: bool = True
    generation_batch_size: int = 4

class ValueHead(nn.Module):
    """A multi-layer perceptron for value estimation."""
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout_p=0.3):
        super(ValueHead, self).__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            current_dim = h_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)

class SBSNode:
    """Node in the Step-level Beam Search tree"""
    def __init__(self, parent: Optional['SBSNode'] = None, text: str = "", depth: int = 0):
        self.parent = parent
        self.children: List['SBSNode'] = []
        self.text = text
        self.depth = depth
        self.full_text: str = (parent.full_text if parent else "") + text
        self.value: float = -100.0
        self.activation: Optional[torch.Tensor] = None
        self.is_terminal = False
        self.final_answer = ""
        self.tag = f"node_{id(self)}"
        self.state = {"text": text, "action": "", "action_input": "", "final_answer": ""}

    def has_children(self) -> bool:
        return len(self.children) > 0

    def add_child(self, child_text: str) -> 'SBSNode':
        cleaned = child_text.lstrip('\n')
        child = SBSNode(parent=self, text=cleaned, depth=self.depth + 1)
        self.children.append(child)
        return child

class StepBeamSearch:
    """
    Step-level Beam Search implementation with vLLM for generation and Unsloth for value estimation.
    MODIFIED to support placing the value model on a dedicated GPU.
    """
    def __init__(self,
                 inference_model: str,
                 activations_model: str,
                 config: SBSConfig,
                 vllm_params: Dict[str, Any],
                 value_head_path: Optional[str] = None,
                 value_model_device: str = "cuda:0"):
        self.config = config
        self.inference_model = inference_model
        self.activations_model = activations_model

        # --- 1. Load vLLM for Generation ---
        # vLLM will use the GPUs assigned to it by the distributed engine.
        logger.info(f"Loading vLLM model: {inference_model} with params: {vllm_params}")
        self.vllm_model = LLM(
            model=self.inference_model,
            **vllm_params
        )
        
        # --- 2. Load Unsloth Model for Value Estimation on a Dedicated GPU ---
        self.value_device = torch.device(value_model_device)
        logger.info(f"Loading Unsloth value model on dedicated device: {self.value_device}")
        
        self.unsloth_model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.activations_model,
            max_seq_length=24_000,
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )
        self.unsloth_model.to(self.value_device) # Move model to specified device
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- 3. Load Value Head on the Same Dedicated GPU ---
        self.value_head = ValueHead(input_dim=self.unsloth_model.config.hidden_size)
        if value_head_path:
            logger.info(f"Loading value head from '{value_head_path}' to {self.value_device}")
            # Load state dict directly to the target device
            self.value_head.load_state_dict(torch.load(value_head_path, map_location=self.value_device))
        
        self.value_head.to(device=self.value_device, dtype=torch.bfloat16)
        self.value_head.eval()
        
        self.root: Optional[SBSNode] = None
        self.active_beams: List[SBSNode] = []
        self.completed_beams: List[SBSNode] = []
        self.current_beam_width: int = config.step_beam_width
        self._prompt_cache = {}

    def clear_cache(self):
        self._prompt_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def create_prompt(self, question: str, partial_solution: str = "") -> str:
        cache_key = (question, partial_solution)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{question} \\think<|im_end|>\n<|im_start|>assistant\n{partial_solution}"
        self._prompt_cache[cache_key] = prompt
        return prompt

    def _extract_final_answer(self, text: str) -> Optional[str]:
        patterns = [
            r'The correct answer is \(([A-J])\)', r'The answer is \(([A-J])\)',
            r'So the answer is \(([A-J])\)', r'Answer: \(([A-J])\)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m: return m.group(1).upper()
        return None

    @torch.no_grad()
    def _get_activations_and_values(self, prompts: List[str], generated_texts: List[str]) -> List[Tuple[torch.Tensor, float]]:
        results = []
        for prompt, gen_text in zip(prompts, generated_texts):
            full_text = prompt + gen_text + "\n\n"
            # Tokenize and ensure tensors are on the correct device for the value model
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=False).to(self.value_device)
            
            outputs = self.unsloth_model(**inputs, output_hidden_states=True)
            
            activation = outputs.hidden_states[-2][0, -1, :]
            value = torch.sigmoid(self.value_head(activation.unsqueeze(0))).item()
            
            results.append((activation, value))
        return results

    def _generate_and_score_candidates(self, question: str) -> List[SBSNode]:
        if not self.active_beams: return []

        prompts = [self.create_prompt(question, node.full_text) for node in self.active_beams]
        sampling_params = SamplingParams(
            temperature=self.config.temperature, max_tokens=self.config.max_tokens,
            stop=["\n\n"], n=self.config.n_generate_sample,
        )
        vllm_outputs = self.vllm_model.generate(prompts, sampling_params)
        
        all_candidates, seen_full_texts = [], set()
        for i, output in enumerate(vllm_outputs):
            parent_node, parent_prompt = self.active_beams[i], prompts[i]
            generated_texts = [gen.text for gen in output.outputs]
            activations_and_values = self._get_activations_and_values([parent_prompt] * len(generated_texts), generated_texts)

            for j, gen_text in enumerate(generated_texts):
                activation, value = activations_and_values[j]
                snippet = gen_text.rstrip() + '\n\n'
                child_node = parent_node.add_child(snippet)
                child_node.activation, child_node.value = activation, value

                if ans := self._extract_final_answer(gen_text):
                    child_node.is_terminal, child_node.final_answer = True, ans
                
                if self.config.remove_duplicate:
                    if child_node.full_text in seen_full_texts: continue
                    seen_full_texts.add(child_node.full_text)
                
                all_candidates.append(child_node)

        if self.config.verbose and all_candidates:
            logger.info(f"Generated {len(all_candidates)} unique candidates this step.")
        return all_candidates

    def _update_beams(self, candidates: List[SBSNode]) -> int:
        if not candidates:
            self.active_beams, self.current_beam_width = [], 0
            return 0
        candidates.sort(key=lambda x: x.value, reverse=True)
        new_active_beams, newly_completed = [], 0
        
        for cand in candidates[:self.current_beam_width]:
            if cand.is_terminal or cand.depth >= self.config.max_depth:
                self.completed_beams.append(cand)
                newly_completed += 1
            else:
                new_active_beams.append(cand)
        
        self.active_beams = new_active_beams
        self.current_beam_width = len(self.active_beams)
        return newly_completed

    def _create_summary(self, solutions: List[Dict[str, Any]], question: str, ground_truth: Optional[str]) -> Dict[str, Any]:
        answers = [s['final_answer'] for s in solutions if s['final_answer'] != "Not found"]
        majority_vote = Counter(answers).most_common(1)[0][0] if answers else "N/A"
        accuracy = "N/A"
        if ground_truth and answers:
            correct_answers = sum(1 for ans in answers if ans.strip().upper() == ground_truth.strip().upper())
            accuracy = f"{correct_answers / len(answers):.2%}" if answers else "0.00%"
        return {"question": question, "ground_truth": ground_truth, "majority_vote": majority_vote, "accuracy": accuracy, "solutions": solutions}

    def _save_results(self, final_beams: List[SBSNode], base_path: str, question: str, ground_truth: Optional[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        solutions, saved_files = [], []
        os.makedirs(base_path, exist_ok=True)

        for idx, node in enumerate(final_beams):
            if not node.final_answer: node.final_answer = self._extract_final_answer(node.full_text) or ""
            termination_reason = "answer_found" if node.final_answer else "max_depth"
            
            solution_data = {"beam_index": idx + 1, "value": node.value, "final_answer": node.final_answer or "Not found",
                             "depth": node.depth, "termination_reason": termination_reason, "solution_path": node.full_text}
            solutions.append(solution_data)
            
            # Save individual beam file
            filename = f"beam_{idx+1:02d}.txt"
            file_path = os.path.join(base_path, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({k: v for k, v in solution_data.items() if k != 'solution_path'}, f, indent=4)
                f.write("\n\n---\n\n" + node.full_text)
            saved_files.append(file_path)

        # Save summary file
        summary_path = os.path.join(base_path, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self._create_summary(solutions, question, ground_truth), f, indent=4)
        saved_files.append(summary_path)
        return solutions, saved_files

    def run(self, question: str, ground_truth: Optional[str] = None, base_path: Optional[str] = None) -> Dict[str, Any]:
        """Main execution loop for a single question."""
        logger.info(f"Starting SBS for question: {question[:100]}...")
        self.clear_cache()
        self.root, self.active_beams = SBSNode(text="", depth=0), [self.root]
        self.completed_beams, self.current_beam_width = [], self.config.step_beam_width
        
        step = 0
        while self.active_beams and step < self.config.max_depth:
            step += 1
            if self.config.verbose: logger.info(f"\n--- SBS Step {step} | Active Beams: {len(self.active_beams)} ---")
            
            candidates = self._generate_and_score_candidates(question)
            if not candidates:
                logger.warning("No new candidates generated. Stopping search.")
                break
            self._update_beams(candidates)

        self.completed_beams.extend(self.active_beams)
        final_beams = sorted(self.completed_beams, key=lambda x: x.value, reverse=True)[:self.config.step_beam_width]
        
        solutions, saved_files = [], []
        if base_path:
            solutions, saved_files = self._save_results(final_beams, base_path, question, ground_truth)
        
        if self.config.verbose and solutions:
            best = solutions[0]
            logger.info(f"--- SBS Complete --- Best solution value: {best['value']:.4f}, Answer: {best['final_answer']}")

        return {"question": question, "ground_truth": ground_truth, "solutions": solutions}

# --- Data Parallel Execution Harness ---

def _sbs_worker(
    args: argparse.Namespace,
    dataset_slice: List[Dict[str, Any]],
    dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
):
    """Worker function for data-parallel Step-level Beam Search."""
    # 1. Set Environment for vLLM Data Parallelism
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    
    # Set GPUs visible to this process
    all_gpus = args.vllm_gpus.split(',') + [str(args.value_model_gpu)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(all_gpus)
    
    logger.info(f"[Rank {dp_rank}] Worker started. Visible GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # 2. Initialize SBS Instance (loads models once per worker)
    sbs_config = SBSConfig(
        step_beam_width=args.beam_width,
        n_generate_sample=args.n_samples,
        max_depth=args.max_depth,
        temperature=args.temperature,
        verbose=args.verbose,
    )
    vllm_params = {
        "tensor_parallel_size": args.tp_size,
        "gpu_memory_utilization": 0.9,
        "dtype": "bfloat16",
        "max_model_len": 24000,
        "enable_prefix_caching": True,
    }
    value_model_device = f"cuda:{args.value_model_gpu}"
    
    sbs_instance = StepBeamSearch(
        inference_model=args.inference_model,
        activations_model=args.activations_model,
        config=sbs_config,
        vllm_params=vllm_params,
        value_head_path=args.value_head_path,
        value_model_device=value_model_device,
    )

    # 3. Process Assigned Data Slice
    logger.info(f"[Rank {dp_rank}] Processing {len(dataset_slice)} items.")
    for item in tqdm(dataset_slice, desc=f"[Rank {dp_rank}] Processing"):
        question = item['question']
        ground_truth = item.get('answer') # Use .get for optional field
        original_index = item['original_index']
        
        # Create a unique output directory for each question
        output_path = os.path.join(args.output_dir, f"q_{original_index:04d}")
        
        try:
            sbs_instance.run(
                question=question,
                ground_truth=ground_truth,
                base_path=output_path
            )
        except Exception as e:
            logger.error(f"[Rank {dp_rank}] Error processing item {original_index}: {e}", exc_info=True)
            # Optionally save error info
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "error.log"), 'w') as f:
                f.write(f"Error on question index {original_index}:\n{e}")

    logger.info(f"[Rank {dp_rank}] Worker finished.")

def run_sbs_on_dataset(args: argparse.Namespace):
    """Main function to set up and run parallel SBS on a dataset."""
    # 1. Load and prepare dataset
    with open(args.input_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    # Add original index to each item for unique output paths
    for i, item in enumerate(dataset):
        item['original_index'] = i
    
    # 2. Setup for multiprocessing
    dp_size = args.dp_size
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    procs = []

    logger.info(f"Starting {dp_size} data-parallel workers.")
    
    # 3. Distribute data and launch workers
    total_samples = len(dataset)
    samples_per_rank = total_samples // dp_size
    
    for i in range(dp_size):
        start_idx = i * samples_per_rank
        end_idx = (i + 1) * samples_per_rank if i != dp_size - 1 else total_samples
        dataset_slice = dataset[start_idx:end_idx]
        
        if not dataset_slice:
            logger.warning(f"Rank {i} received an empty data slice. Skipping.")
            continue
            
        proc = Process(target=_sbs_worker, args=(args, dataset_slice, i, dp_master_ip, dp_master_port))
        proc.start()
        procs.append(proc)
        
    # 4. Wait for all processes to complete
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            logger.error(f"Process {proc.pid} exited with code {proc.exitcode}. Check logs for errors.")
            
    logger.info("All workers have completed.")

# --- Command-Line Interface ---

def main():
    parser = argparse.ArgumentParser(description="Run Step-level Beam Search on a dataset with Data Parallelism.")
    
    # --- I/O Arguments ---
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output results.")
    parser.add_argument("--inference_model", type=str, default="Qwen/Qwen3-1.7B", help="Model name for both vLLM and Unsloth.")
    parser.add_argument("--activations_model", type=str, default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit", help="Model name for both vLLM and Unsloth.")
    parser.add_argument("--value_head_path", type=str, required=True, help="Path to the pretrained value head state_dict.")

    # --- Parallelism Arguments ---
    parser.add_argument("--dp_size", type=int, default=1, help="Number of data parallel workers.")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM within each worker.")
    parser.add_argument("--vllm_gpus", type=str, default="0", help="Comma-separated list of GPU IDs for vLLM (e.g., '0,1,2,3').")
    parser.add_argument("--value_model_gpu", type=int, default=0, help="The single GPU ID for the value model (e.g., 4).")
    
    # --- SBS Algorithm Arguments ---
    parser.add_argument("--beam_width", type=int, default=3, help="Step-level beam width.")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to generate at each step.")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum search depth.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging for SBS steps.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Basic validation
    if str(args.value_model_gpu) in args.vllm_gpus.split(','):
        logger.warning(f"GPU {args.value_model_gpu} is assigned to both vLLM and the value model. This may cause OOM errors. It's recommended to use separate GPUs.")

    run_sbs_on_dataset(args)


if __name__ == "__main__":
    main()