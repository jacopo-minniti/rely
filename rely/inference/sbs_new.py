# sbs.py
import re
import logging
import time
import os
import json
import traceback
from collections import Counter
from multiprocessing import Process, Queue, set_start_method
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import random

import torch
import torch.nn as nn
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import argparse
from datasets import load_dataset

from rely.utils import MATH_SYSTEM_PROMPT, extract_final_answer, normalize_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- OpenAI Client Configuration ---
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"


@dataclass
class SBSConfig:
    """Configuration for Step-level Beam Search"""
    step_beam_width: int = 3
    n_generate_sample: int = 5
    max_depth: Optional[int] = None
    budget: Optional[int] = None
    temperature: float = 0.6
    max_tokens: int = 512
    need_value_func: bool = True
    remove_duplicate: bool = True
    verbose: bool = True
    generation_batch_size: int = 4
    value_method: str = "last_step"


class SBSNode:
    """Node in the Step-level Beam Search tree"""
    def __init__(self, parent: Optional['SBSNode'] = None, text: str = "", depth: int = 0):
        self.parent = parent
        self.children: List['SBSNode'] = []
        self.text = text
        self.depth = depth
        self.full_text: str = (parent.full_text if parent else "") + text
        self.value: float = -100.0
        self.is_terminal = False
        self.final_answer = ""

    def add_child(self, child_text: str) -> 'SBSNode':
        cleaned = child_text.lstrip('\n')
        child = SBSNode(parent=self, text=cleaned, depth=self.depth + 1)
        self.children.append(child)
        return child


class StepBeamSearch:
    """
    MODIFIED: Step-level Beam Search implementation.
    This class now acts as a client to a separate Value Server process
    and the vLLM OpenAI API server.
    """
    def __init__(self,
                 inference_model_name: str,
                 config: SBSConfig,
                 value_task_queue: Queue,
                 value_result_queue: Queue,
                 worker_rank: int):
        self.config = config
        if self.config.max_depth is None and self.config.budget is None:
            raise ValueError("Either max_depth or budget must be specified for the search.")
        self.inference_model_name = inference_model_name
        self.worker_rank = worker_rank
        
        self.value_task_queue = value_task_queue
        self.value_result_queue = value_result_queue

        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        logger.info(f"[Rank {self.worker_rank}] Client initialized to connect to {OPENAI_API_BASE}")
        
        self.root: Optional[SBSNode] = None
        self.active_beams: List[SBSNode] = []
        self.completed_beams: List[SBSNode] = []
        self.current_beam_width: int = config.step_beam_width
        self._prompt_cache = {}

        self.tokenizer = AutoTokenizer.from_pretrained(self.inference_model_name, trust_remote_code=True)
        self.total_generated_tokens = 0

    def clear_cache(self):
        self._prompt_cache.clear()
        self.total_generated_tokens = 0

    def create_prompt(self, question: str, partial_solution: str = "") -> str:
        cache_key = (question, partial_solution)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        prompt = f"<|im_start|>system\n{MATH_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{partial_solution}"
        self._prompt_cache[cache_key] = prompt
        return prompt

    def _get_values_from_server(self, prompts: List[str], generated_texts: List[str]) -> List[float]:
        if not prompts:
            return []
        request_id = str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "worker_rank": self.worker_rank,
            "prompts": prompts,
            "generated_texts": generated_texts
        }
        self.value_task_queue.put(payload)
        
        while True:
            response = self.value_result_queue.get()
            if response.get("request_id") == request_id:
                return response["values"]
            else:
                logger.warning(f"[Rank {self.worker_rank}] Received unexpected value result, requeuing.")
                self.value_result_queue.put(response)
                time.sleep(0.01)

    def _make_api_request(self, prompt: str, n_samples: int) -> List[str]:
        """Helper function to make a single API call for n_samples."""
        if n_samples <= 0:
            return []
        try:
            completion = self.client.completions.create(
                model=self.inference_model_name,
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stop=["\n\n"],
                n=n_samples,
            )
            generations = [choice.text for choice in completion.choices]
            try:
                for gen in generations:
                    self.total_generated_tokens += len(self.tokenizer.encode(gen, add_special_tokens=False))
            except Exception as e:
                logger.warning(f"[Rank {self.worker_rank}] Could not count tokens: {e}")
            return generations
        except Exception as e:
            logger.error(f"[Rank {self.worker_rank}] Error during API call: {e}")
            return []

    def _generate_and_score_candidates(self, question: str) -> List[SBSNode]:
        if not self.active_beams:
            return []

        num_active_beams = len(self.active_beams)
        total_samples_budget = self.config.n_generate_sample

        # Distribute the total sample budget among active beams
        base_samples = total_samples_budget // num_active_beams
        remainder = total_samples_budget % num_active_beams
        
        # Randomly distribute remainder samples instead of giving them to the first beams
        samples_per_beam = [base_samples] * num_active_beams
        if remainder > 0:
            indices_for_extra = random.sample(range(num_active_beams), remainder)
            for idx in indices_for_extra:
                samples_per_beam[idx] += 1

        all_candidates, seen_full_texts = [], set()
        prompts = [self.create_prompt(question, node.full_text) for node in self.active_beams]
        
        generated_outputs = [[] for _ in self.active_beams]
        with ThreadPoolExecutor(max_workers=num_active_beams) as executor:
            future_to_index = {
                executor.submit(self._make_api_request, prompts[i], samples_per_beam[i]): i
                for i in range(num_active_beams)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    generated_outputs[index] = future.result()
                except Exception as exc:
                    logger.error(f"[Rank {self.worker_rank}] API request generated an exception: {exc}")

        value_request_prompts = []
        value_request_texts = []
        parent_nodes = []

        for i, gen_texts in enumerate(generated_outputs):
            parent_node = self.active_beams[i]
            for gen_text in gen_texts:
                value_request_prompts.append(prompts[i])
                value_request_texts.append(gen_text)
                parent_nodes.append(parent_node)

        values = self._get_values_from_server(value_request_prompts, value_request_texts)

        candidate_idx = 0
        for i, gen_texts in enumerate(generated_outputs):
            for gen_text in gen_texts:
                parent_node = self.active_beams[i]
                new_step_value = values[candidate_idx]
                snippet = gen_text.rstrip() + '\n\n'
                child_node = parent_node.add_child(snippet)
                
                if self.config.value_method == 'product':
                    child_node.value = parent_node.value * new_step_value
                else: 
                    child_node.value = new_step_value
                
                if ans := extract_final_answer(gen_text):
                    child_node.is_terminal, child_node.final_answer = True, ans
                    
                if self.config.remove_duplicate:
                    if child_node.full_text in seen_full_texts:
                        candidate_idx += 1
                        continue
                    seen_full_texts.add(child_node.full_text)
                    
                all_candidates.append(child_node)
                candidate_idx += 1
        
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
            max_depth_reached = self.config.max_depth is not None and cand.depth >= self.config.max_depth
            if cand.is_terminal or max_depth_reached:
                self.completed_beams.append(cand)
                newly_completed += 1
            else:
                new_active_beams.append(cand)
        self.active_beams = new_active_beams
        self.current_beam_width = len(self.active_beams)
        return newly_completed

    def _create_summary(self, solutions: List[Dict[str, Any]], question: str, ground_truth: Optional[str]) -> Dict[str, Any]:
        ground_truth = normalize_answer(ground_truth)
        answers = [normalize_answer(s['final_answer']) for s in solutions if s['final_answer'] != "Not found"]
        majority_vote = Counter(answers).most_common(1)[0][0] if answers else "N/A"
        accuracy = "N/A"
        if ground_truth and answers:
            correct_answers = sum(1 for ans in answers if ans == ground_truth)
            accuracy = f"{correct_answers / len(answers):.2%}" if answers else "0.00%"
        return {
            "question": question,
            "ground_truth": ground_truth,
            "majority_vote": majority_vote,
            "accuracy": accuracy,
            "solutions": solutions,
            "total_tokens": self.total_generated_tokens
        }

    def _save_results(self, final_beams: List[SBSNode], base_path: str, question: str, ground_truth: Optional[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        solutions, saved_files = [], []
        os.makedirs(base_path, exist_ok=True)

        for idx, node in enumerate(final_beams):
            if not node.final_answer:
                node.final_answer = extract_final_answer(node.full_text) or ""
            termination_reason = "answer_found" if node.final_answer else "max_depth"
            solution_data = {
                "beam_index": idx + 1,
                "value": node.value,
                "final_answer": node.final_answer or "Not found",
                "depth": node.depth,
                "termination_reason": termination_reason,
                "solution_path": node.full_text
            }
            solutions.append(solution_data)

        summary_path = os.path.join(base_path, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self._create_summary(solutions, question, ground_truth), f, indent=4)
        saved_files.append(summary_path)
        
        return solutions, saved_files
        
    def _force_final_answers(self, question: str, beams: List[SBSNode]) -> None:
        """Force generation of final answers for beams that haven't terminated naturally."""
        beams_needing_answers = [beam for beam in beams if not beam.is_terminal and not beam.final_answer]
        
        if not beams_needing_answers:
            return
            
        logger.info(f"[Rank {self.worker_rank}] Forcing final answers for {len(beams_needing_answers)} beams")
        
        force_prompts = []
        for beam in beams_needing_answers:
            base_prompt = self.create_prompt(question, beam.full_text)
            force_prompt = base_prompt + "\n\n# Final Answer\n\\boxed{"
            force_prompts.append(force_prompt)
        
        forced_outputs = []
        with ThreadPoolExecutor(max_workers=len(force_prompts)) as executor:
            def make_forced_request(prompt):
                try:
                    completion = self.client.completions.create(
                        model=self.inference_model_name,
                        prompt=prompt,
                        max_tokens=50,
                        temperature=0.1,
                        stop=["}"],
                        n=1,
                    )
                    return completion.choices[0].text+"}" if completion.choices else ""
                except Exception as e:
                    logger.error(f"[Rank {self.worker_rank}] Error during forced final answer generation: {e}")
                    return ""
            
            future_to_index = {executor.submit(make_forced_request, prompt): i for i, prompt in enumerate(force_prompts)}
            forced_outputs = [""] * len(force_prompts)
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    forced_outputs[index] = future.result()
                except Exception as exc:
                    logger.error(f"[Rank {self.worker_rank}] Forced generation exception: {exc}")
        
        for i, beam in enumerate(beams_needing_answers):
            forced_text = forced_outputs[i]
            if forced_text:
                full_forced_text = "\n\n# Final Answer\n\\boxed{" + forced_text
                beam.text += full_forced_text
                beam.full_text += full_forced_text
                
                if ans := extract_final_answer(full_forced_text):
                    beam.final_answer = ans
                    beam.is_terminal = True
                    logger.info(f"[Rank {self.worker_rank}] Successfully forced final answer: {ans}")
                else:
                    clean_forced = forced_text.strip().rstrip('}')
                    if clean_forced:
                        beam.final_answer = clean_forced
                        beam.is_terminal = True
                        logger.info(f"[Rank {self.worker_rank}] Used raw forced text as answer: {clean_forced}")

    def run(self, question: str, ground_truth: Optional[str] = None, base_path: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"[Rank {self.worker_rank}] Starting SBS for question: {question[:100]}...")
        self.clear_cache()
        self.root = SBSNode(text="", depth=0)
        if self.config.value_method == "product":
            self.root.value = 1.0
        self.active_beams = [self.root]
        self.completed_beams, self.current_beam_width = [], self.config.step_beam_width
        step = 0
        while self.active_beams:
            if self.config.max_depth is not None and step >= self.config.max_depth:
                logger.info(f"[Rank {self.worker_rank}] Max depth of {self.config.max_depth} reached, forcing final answers.")
                self._force_final_answers(question, self.active_beams)
                break
            
            if self.config.budget is not None and self.total_generated_tokens >= self.config.budget:
                logger.info(f"[Rank {self.worker_rank}] Token budget of {self.config.budget} reached, forcing final answers.")
                self._force_final_answers(question, self.active_beams)
                break

            step += 1
            if self.config.verbose: logger.info(f"\n--- [Rank {self.worker_rank}] SBS Step {step} | Active Beams: {len(self.active_beams)} ---")
            candidates = self._generate_and_score_candidates(question)
            if not candidates:
                logger.warning(f"[Rank {self.worker_rank}] No new candidates generated. Stopping search.")
                break
            self._update_beams(candidates)

        self.completed_beams.extend(self.active_beams)
        final_beams = sorted(self.completed_beams, key=lambda x: x.value, reverse=True)[:self.config.step_beam_width]
        solutions, saved_files = [], []
        if base_path:
            solutions, saved_files = self._save_results(final_beams, base_path, question, ground_truth)
        if self.config.verbose and solutions:
            best = solutions[0]
            logger.info(f"--- [Rank {self.worker_rank}] SBS Complete --- Best solution value: {best['value']:.4f}, Answer: {best['final_answer']}")

        return {"question": question, "ground_truth": ground_truth, "solutions": solutions}

# --- Data Parallel Execution Harness ---

def _value_model_server(args: argparse.Namespace, task_queue: Queue, result_queues: List[Queue]):
    """
    This process loads the value model on a dedicated GPU and serves scoring requests
    from the client workers according to the official PRM documentation.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.value_model_gpu)
    value_device = torch.device("cuda:0")
    logger.info(f"[ValueServer] Starting on device {value_device}")

    tokenizer = AutoTokenizer.from_pretrained(args.value_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.value_model_path,
        num_labels=2,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # max_model_len=4000,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    logger.info("[ValueServer] Value model loaded. Waiting for tasks.")

    prompt_pattern = re.compile(
        r"<\|im_start\|>system\n(.*?)"
        r"<\|im_end\|>\n<\|im_start\|>user\n(.*?)"
        r"<\|im_end\|>\n<\|im_start|>assistant\n(.*)",
        re.DOTALL
    )

    @torch.no_grad()
    def get_values(prompts: List[str], generated_texts: List[str]) -> List[float]:
        conversation_strs = []
        for prompt, gen_text in zip(prompts, generated_texts):
            match = prompt_pattern.match(prompt)
            if not match:
                logger.error(f"Prompt did not match expected format. Cannot score. Prompt: {prompt[:200]}")
                continue

            system_msg, user_msg, partial_solution = match.groups()
            partial_solution = partial_solution or ""
            full_assistant_response = (partial_solution.strip() + '\n\n' + gen_text.strip()).strip()

            steps = [s.strip() for s in full_assistant_response.split('\n\n') if s.strip()]
            assistant_content = "<extra_0>".join(steps) + "<extra_0>"
            messages = [
                {"role": "system", "content": system_msg.strip()},
                {"role": "user", "content": user_msg.strip()},
                {"role": "assistant", "content": assistant_content},
            ]
            conv_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            conversation_strs.append(conv_str)

        if not conversation_strs:
            return [0.0] * len(prompts)

        inputs = tokenizer(
            conversation_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=5000,
        ).to(value_device)

        base_model_output = model.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)
        logits = model.score(base_model_output.last_hidden_state)
        probabilities = torch.softmax(logits, dim=-1)
        step_sep_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]
        token_masks = (inputs.input_ids == step_sep_id)
        batch_rewards = []
        for i in range(logits.size(0)):
            sample_probs = probabilities[i]
            sample_mask = token_masks[i]
            step_scores = sample_probs[sample_mask][:, 1]
            if len(step_scores) > 0:
                last_step_score = step_scores[-1].item()
                batch_rewards.append(last_step_score)
            else:
                logger.warning("No <extra_0> token found in a processed sample, returning reward of 0.0. This might be due to truncation.")
                batch_rewards.append(0.0)
        return batch_rewards

    while True:
        try:
            task = task_queue.get()
            if task == "STOP":
                logger.info("[ValueServer] Received STOP signal. Shutting down.")
                break
            request_id, worker_rank = task["request_id"], task["worker_rank"]
            values = get_values(task["prompts"], task["generated_texts"])
            result_queues[worker_rank].put({"request_id": request_id, "values": values})
        except Exception as e:
            logger.error(f"[ValueServer] Error processing task: {e}", exc_info=True)


def _sbs_worker(args: argparse.Namespace,
                dataset_slice: List[Dict[str, Any]],
                rank: int,
                task_queue: Queue,
                result_queue: Queue):
    logger.info(f"[Rank {rank}] Worker started.")
    
    sbs_config = SBSConfig(
        step_beam_width=args.beam_width,
        n_generate_sample=args.n_samples,
        max_depth=args.max_depth,
        budget=args.budget,
        temperature=args.temperature,
        verbose=args.verbose,
        value_method=args.value_method,
    )
    
    sbs_instance = StepBeamSearch(
        inference_model_name=args.inference_model,
        config=sbs_config,
        value_task_queue=task_queue,
        value_result_queue=result_queue,
        worker_rank=rank,
    )

    for item in tqdm(dataset_slice, desc=f"Rank {rank} Processing"):
        original_index = None
        output_path = None
        try:
            question = item['problem']
            ground_truth = item.get('answer')
            original_index = item['original_index']
            output_path = os.path.join(args.output_dir, f"q_{original_index:04d}")
            
            if os.path.exists(os.path.join(output_path, "summary.json")):
                logger.info(f"[Rank {rank}] Skipping item {original_index} as it is already completed.")
                continue
                
            sbs_instance.run(
                question=question,
                ground_truth=ground_truth,
                base_path=output_path
            )
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"[Rank {rank}] FATAL ERROR processing item {original_index}: {e}\nTRACEBACK:\n{error_traceback}")
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                with open(os.path.join(output_path, "error.log"), 'w') as f:
                    f.write(f"Error on question index {original_index}:\n{str(e)}\n\n{error_traceback}")

    logger.info(f"[Rank {rank}] Worker finished.")


def run_sbs_on_dataset(args: argparse.Namespace):
    set_start_method('spawn', force=True)
    
    ds = load_dataset(args.dataset, split='test')
    ds = ds.shuffle(seed=42)
    dataset = [dict(item) for item in ds]
    for i, item in enumerate(dataset):
        item['original_index'] = i

    # --- TEMPORARY: Filter out already processed questions ---
    # existing_results_dir = "results/sbs_max_4_20"
    # if os.path.exists(existing_results_dir):
    #     processed_indices = set()
    #     for folder_name in os.listdir(existing_results_dir):
    #         if folder_name.startswith("q_"):
    #             try:
    #                 idx = int(folder_name.split("_")[1])
    #                 processed_indices.add(idx)
    #             except (ValueError, IndexError):
    #                 continue
        
    #     original_count = len(dataset)
    #     dataset = [item for item in dataset if item['original_index'] not in processed_indices]
    #     logger.info(f"Filtered out {original_count - len(dataset)} already processed questions. Remaining: {len(dataset)}")
    # --- END TEMPORARY BLOCK ---

    num_workers = args.num_workers
    value_task_queue = Queue()
    value_result_queues = [Queue() for _ in range(num_workers)]
    procs = []

    logger.info("Starting Value Model Server process...")
    value_server_proc = Process(
        target=_value_model_server,
        args=(args, value_task_queue, value_result_queues)
    )
    value_server_proc.start()
    procs.append(value_server_proc)
    time.sleep(30)

    total_samples = len(dataset)
    samples_per_worker = (total_samples + num_workers - 1) // num_workers

    logger.info(f"Starting {num_workers} SBS client workers.")
    for i in range(num_workers):
        start_idx = i * samples_per_worker
        end_idx = min(start_idx + samples_per_worker, total_samples)
        dataset_slice = dataset[start_idx:end_idx]
        if not dataset_slice:
            continue
            
        proc = Process(
            target=_sbs_worker,
            args=(args, dataset_slice, i, value_task_queue, value_result_queues[i])
        )
        proc.start()
        procs.append(proc)
        
    for proc in procs[1:]:
        proc.join()

    logger.info("All SBS workers finished. Shutting down Value Server...")
    value_task_queue.put("STOP")
    value_server_proc.join()
    logger.info("All processes have completed.")


def main():
    parser = argparse.ArgumentParser(description="Run SBS using a vLLM server.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--inference_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name served by vLLM.")
    parser.add_argument("--value_model_path", type=str, required=True, help="Path to pretrained value model (AutoModelForSequenceClassification).")
    
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel client worker processes.")
    parser.add_argument("--value_model_gpu", type=int, default=0, help="The single GPU ID for the value model server.")
    
    parser.add_argument("--beam_width", type=int, default=4, help="Step-level beam width.")
    parser.add_argument("--n_samples", type=int, default=5, help="Total samples to generate at each step, distributed among active beams.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum search depth.")
    parser.add_argument("--budget", type=int, default=None, help="Maximum number of tokens to generate across all steps.")
    parser.add_argument("--temperature", type=float, default=1, help="Generation temperature.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging.")
    
    parser.add_argument(
        "--value_method", 
        type=str, 
        default="last_step", 
        choices=["last_step", "product"], 
        help="Method to calculate node value: 'last_step' (default) or 'product' of path."
    )
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_sbs_on_dataset(args)

if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES="1" vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 5000 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --port 8000

python rely/inference/sbs.py \
    --dataset nlile/hendrycks-MATH-benchmark \
    --output_dir sbs_results_b1_4_b2_5_product// \
    --value_model_path Qwen/Qwen2.5-Math-PRM-7B \
    --num_workers 5 \
    --value_model_gpu 0 \
    --value_method product \
    --budget 20000
'''