# rely/inference/sbs/main.py

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
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from datasets import load_dataset

from rely.utils import MATH_SYSTEM_PROMPT, extract_final_answer, normalize_answer, prompt_pattern
from rely.inference.sbs.strategies import SamplingStrategy, UniformStrategy, PumStrategy, TokenEntropyStrategy, PumPerValueStrategy, UCBStrategy
from rely.inference.sbs.utils import SBSConfig, SBSNode, _uncertainty_model_server, _value_model_server

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- OpenAI Client Configuration ---
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"



class StepBeamSearch:
    """Step-level Beam Search implementation with pluggable sampling strategies."""
    def __init__(self,
                 inference_model_name: str,
                 config: SBSConfig,
                 strategy: SamplingStrategy,
                 value_task_queue: Queue,
                 value_result_queue: Queue,
                 worker_rank: int):
        self.config = config
        if self.config.max_depth is None and self.config.budget is None:
            raise ValueError("Either max_depth or budget must be specified for the search.")
        
        self.inference_model_name = inference_model_name
        self.strategy = strategy
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
        payload = {"request_id": request_id, "worker_rank": self.worker_rank, "prompts": prompts, "generated_texts": generated_texts}
        self.value_task_queue.put(payload)
        
        while True:
            response = self.value_result_queue.get()
            if response.get("request_id") == request_id:
                return response["values"]
            self.value_result_queue.put(response)
            time.sleep(0.01)

    def _make_api_request_with_samples(self, prompt: str, n_samples: int) -> List[Dict[str, Any]]:
        if n_samples <= 0:
            return []
        
        request_params = {
            "model": self.inference_model_name,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stop": ["\n\n"],
            "n": n_samples,
        }
        if self.strategy.requires_logprobs():
            request_params["logprobs"] = self.config.entropy_k

        try:
            completion = self.client.completions.create(**request_params)
            results = []
            for choice in completion.choices:
                results.append({
                    'text': choice.text,
                    'logprobs': choice.logprobs
                })
                self.total_generated_tokens += len(self.tokenizer.encode(choice.text, add_special_tokens=False))
            return results
        except Exception as e:
            logger.error(f"[Rank {self.worker_rank}] Error during API call: {e}")
            return []

    def _generate_and_score_candidates(self, question: str) -> List[SBSNode]:
        if not self.active_beams:
            return []

        sample_distribution = self.strategy.distribute_samples(self, question)

        all_candidates, seen_full_texts = [], set()
        prompts = [self.create_prompt(question, node.full_text) for node in self.active_beams]
        
        generated_outputs = [[] for _ in self.active_beams]
        with ThreadPoolExecutor(max_workers=len(self.active_beams)) as executor:
            future_to_index = {executor.submit(self._make_api_request_with_samples, prompt, n_samples): i 
                               for i, (prompt, n_samples) in enumerate(zip(prompts, sample_distribution))}
                
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    generated_outputs[index] = future.result()
                except Exception as exc:
                    logger.error(f"[Rank {self.worker_rank}] API request generated an exception: {exc}")

        value_request_prompts, value_request_texts, candidate_data = [], [], []
        for i, gen_results in enumerate(generated_outputs):
            parent_node = self.active_beams[i]
            for gen_result in gen_results:
                value_request_prompts.append(prompts[i])
                value_request_texts.append(gen_result['text'])
                candidate_data.append({
                    'parent_node': parent_node,
                    'generation_result': gen_result
                })

        values = self._get_values_from_server(value_request_prompts, value_request_texts)

        candidate_idx = 0
        for data in candidate_data:
            parent_node = data['parent_node']
            gen_result = data['generation_result']
            gen_text = gen_result['text']
            
            new_step_value = values[candidate_idx]
            snippet = gen_text.rstrip() + '\n\n'
            child_node = parent_node.add_child(snippet)
            
            child_node.value = parent_node.value * new_step_value if self.config.value_method == 'product' else new_step_value
            
            # Let the strategy update the uncertainty if it needs to
            gen_result['entropy_k'] = self.config.entropy_k
            self.strategy.update_candidate_uncertainty(child_node, gen_result)

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
        
        # Use UCB scores for sorting if UCB strategy is being used
        if isinstance(self.strategy, UCBStrategy):
            candidates.sort(key=lambda x: getattr(x, 'ucb_score', x.value), reverse=True)
        else:
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
            accuracy = f"{correct_answers / len(answers):.2%}"
        return {
            "question": question, "ground_truth": ground_truth, "majority_vote": majority_vote,
            "accuracy": accuracy, "solutions": solutions, "total_tokens": self.total_generated_tokens
        }

    def _save_results(self, final_beams: List[SBSNode], base_path: str, question: str, ground_truth: Optional[str]):
        solutions = []
        os.makedirs(base_path, exist_ok=True)

        for idx, node in enumerate(final_beams):
            if not node.final_answer:
                node.final_answer = extract_final_answer(node.full_text) or "Not found"
            
            solution_data = {
                "beam_index": idx + 1, "value": node.value, "uncertainty": node.uncertainty,
                "final_answer": node.final_answer, "depth": node.depth, "solution_path": node.full_text
            }
            solutions.append(solution_data)

        summary_path = os.path.join(base_path, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self._create_summary(solutions, question, ground_truth), f, indent=4)
        
        return solutions

    def _force_final_answers(self, question: str, beams: List[SBSNode]) -> None:
        beams_needing_answers = [beam for beam in beams if not beam.is_terminal and not beam.final_answer]
        if not beams_needing_answers:
            return
            
        logger.info(f"[Rank {self.worker_rank}] Forcing final answers for {len(beams_needing_answers)} beams")        
        force_prompts = [self.create_prompt(question, beam.full_text) + "\n\n# Final Answer\n\\boxed{"
                         for beam in beams_needing_answers]
        forced_outputs = [""] * len(force_prompts)
        with ThreadPoolExecutor(max_workers=len(force_prompts)) as executor:
            def make_forced_request(prompt):
                try:
                    completion = self.client.completions.create(
                        model=self.inference_model_name, prompt=prompt, max_tokens=50,
                        temperature=0.1, stop=["}"], n=1,
                    )
                    return completion.choices[0].text + "}" if completion.choices else ""
                except Exception as e:
                    logger.error(f"[Rank {self.worker_rank}] Error during forced generation: {e}")
                    return ""
            
            future_to_index = {executor.submit(make_forced_request, prompt): i for i, prompt in enumerate(force_prompts)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    forced_outputs[index] = future.result()
                except Exception as exc:
                    logger.error(f"[Rank {self.worker_rank}] Forced generation exception: {exc}")
        
        for i, beam in enumerate(beams_needing_answers):
            if forced_outputs[i]:
                full_forced_text = "\n\n# Final Answer\n\\boxed{"
                beam.text += full_forced_text
                beam.full_text += full_forced_text
                beam.is_terminal = True
                beam.final_answer = extract_final_answer(full_forced_text) or forced_outputs[i].strip().rstrip('}')

    def run(self, question: str, ground_truth: Optional[str] = None, base_path: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"[Rank {self.worker_rank}] Starting SBS for question: {question[:100]}...")
        self.clear_cache()
        self.root = SBSNode(text="", depth=0)
        if self.config.value_method == "product":
            self.root.value = 1.0
            
        self.active_beams = [self.root]
        self.completed_beams = []
        self.current_beam_width = self.config.step_beam_width
        
        step = 0
        while self.active_beams:
            if (self.config.max_depth and step >= self.config.max_depth) or \
               (self.config.budget and self.total_generated_tokens >= self.config.budget):
                logger.info(f"[Rank {self.worker_rank}] Search limit reached. Forcing final answers.")
                self._force_final_answers(question, self.active_beams)
                break

            step += 1
            if self.config.verbose: 
                logger.info(f"\n--- [Rank {self.worker_rank}] SBS Step {step} | Active Beams: {len(self.active_beams)} ---")
            
            candidates = self._generate_and_score_candidates(question)
            if not candidates:
                logger.warning(f"[Rank {self.worker_rank}] No new candidates generated. Stopping search.")
                break
            
            # Calculate UCB scores if using UCB strategy
            if isinstance(self.strategy, UCBStrategy):
                self.strategy.calculate_ucb_scores(self, question)
                
            self._update_beams(candidates)

        self.completed_beams.extend(self.active_beams)
        # Sort final beams using UCB scores if UCB strategy is being used
        if isinstance(self.strategy, UCBStrategy):
            final_beams = sorted(self.completed_beams, key=lambda x: getattr(x, 'ucb_score', x.value), reverse=True)[:self.config.step_beam_width]
        else:
            final_beams = sorted(self.completed_beams, key=lambda x: x.value, reverse=True)[:self.config.step_beam_width]
        
        solutions = []
        if base_path:
            solutions = self._save_results(final_beams, base_path, question, ground_truth)
        
        if self.config.verbose and solutions:
            best = solutions[0]
            logger.info(f"--- [Rank {self.worker_rank}] SBS Complete --- Best solution value: {best['value']:.4f}, Answer: {best['final_answer']}")

        return {"question": question, "ground_truth": ground_truth, "solutions": solutions}




def _sbs_worker(args: argparse.Namespace, dataset_slice: List[Dict[str, Any]], rank: int, strategy: SamplingStrategy, value_task_queue: Queue, value_result_queue: Queue):
    logger.info(f"[Rank {rank}] Worker started with strategy: {args.strategy}")
    sbs_config = SBSConfig(
        strategy=args.strategy,
        step_beam_width=args.beam_width, n_total_samples=args.n_samples, max_depth=args.max_depth,
        budget=args.budget, temperature=args.temperature, verbose=args.verbose,
        value_method=args.value_method, uncertainty_method=args.uncertainty_method,
        entropy_k=args.entropy_k, remove_duplicate=not args.keep_duplicates
    )
    sbs_instance = StepBeamSearch(
        inference_model_name=args.inference_model, config=sbs_config, strategy=strategy,
        value_task_queue=value_task_queue, value_result_queue=value_result_queue, worker_rank=rank,
    )

    for item in tqdm(dataset_slice, desc=f"Rank {rank} Processing"):
        output_path = None
        try:
            original_index = item['original_index']
            output_path = os.path.join(args.output_dir, f"q_{original_index:04d}")
            if os.path.exists(os.path.join(output_path, "summary.json")):
                logger.info(f"[Rank {rank}] Skipping item {original_index} as it is already completed.")
                continue
            sbs_instance.run(question=item['problem'], ground_truth=item.get('answer'), base_path=output_path)
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"[Rank {rank}] FATAL ERROR processing item {item.get('original_index')}: {e}\n{error_traceback}")
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                with open(os.path.join(output_path, "error.log"), 'w') as f:
                    f.write(f"Error on question index {item.get('original_index')}:\n{e}\n\n{error_traceback}")
    logger.info(f"[Rank {rank}] Worker finished.")


def run_sbs_on_dataset(args: argparse.Namespace):
    set_start_method('spawn', force=True)
    
    ds = load_dataset(args.dataset, split='test')
    ds = ds.shuffle(seed=42)
    dataset = [{'original_index': i, **item} for i, item in enumerate(ds)]

    if args.idx_start is not None and args.idx_end is not None:
        dataset = dataset[args.idx_start:args.idx_end]
        logger.info(f"Processing dataset slice from {args.idx_start} to {args.idx_end}. Total items: {len(dataset)}")

    if os.path.exists(args.output_dir):
        processed_indices = set()
        for folder_name in os.listdir(args.output_dir):
            if folder_name.startswith("q_"):
                try:
                    idx = int(folder_name.split("_")[1])
                    processed_indices.add(idx)
                except (IndexError, ValueError):
                    continue
        
        original_count = len(dataset)
        dataset = [item for item in dataset if item['original_index'] not in processed_indices]
        logger.info(f"Excluded {original_count - len(dataset)} already processed questions. Remaining: {len(dataset)}")

    num_workers = args.num_workers
    value_task_queue = Queue()
    value_result_queues = [Queue() for _ in range(num_workers)]
    
    procs = []
    uncertainty_task_queue, uncertainty_result_queues = None, None
    
    if args.strategy in ['pum', 'pum_per_value', 'ucb']:
        uncertainty_task_queue = Queue()
        uncertainty_result_queues = [Queue() for _ in range(num_workers)]
        logger.info("Starting Uncertainty Model Server process...")
        uncertainty_server_proc = Process(target=_uncertainty_model_server, args=(args, uncertainty_task_queue, uncertainty_result_queues))
        uncertainty_server_proc.start()
        procs.append(uncertainty_server_proc)

    logger.info("Starting Value Model Server process...")
    value_server_proc = Process(target=_value_model_server, args=(args, value_task_queue, value_result_queues))
    value_server_proc.start()
    procs.append(value_server_proc)
    
    logger.info("Waiting for servers to load models...")
    time.sleep(30)

    total_samples = len(dataset)
    samples_per_worker = (total_samples + num_workers - 1) // num_workers
    
    logger.info(f"Starting {num_workers} SBS client workers.")
    for i in range(num_workers):
        start_idx = i * samples_per_worker
        end_idx = min(start_idx + samples_per_worker, total_samples)
        if not (dataset_slice := dataset[start_idx:end_idx]): continue
        
        # Instantiate strategy for each worker
        if args.strategy == 'pum':
            assert uncertainty_task_queue is not None and uncertainty_result_queues is not None
            strategy = PumStrategy(uncertainty_task_queue, uncertainty_result_queues[i])
        elif args.strategy == 'pum_per_value':
            assert uncertainty_task_queue is not None and uncertainty_result_queues is not None
            strategy = PumPerValueStrategy(uncertainty_task_queue, uncertainty_result_queues[i])
        elif args.strategy == 'ucb':
            assert uncertainty_task_queue is not None and uncertainty_result_queues is not None
            strategy = UCBStrategy(uncertainty_task_queue, uncertainty_result_queues[i], c=args.ucb_c)
        elif args.strategy == 'token_entropy':
            strategy = TokenEntropyStrategy()
        else: # uniform
            strategy = UniformStrategy()

        proc = Process(target=_sbs_worker, args=(args, dataset_slice, i, strategy, value_task_queue, value_result_queues[i]))
        proc.start()
        procs.append(proc)
        
    for proc in procs[len(procs)-num_workers:]:
        proc.join()

    logger.info("All SBS workers finished. Shutting down servers...")
    value_task_queue.put("STOP")
    if uncertainty_task_queue:
        uncertainty_task_queue.put("STOP")
    
    for proc in procs:
        if proc.is_alive():
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()

    logger.info("All processes have completed.")


def main():
    parser = argparse.ArgumentParser(description="Run Step-level Beam Search with various strategies.")
    # Common arguments
    parser.add_argument("--dataset", type=str, required=True, help="Path to input JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--inference_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name served by vLLM.")
    parser.add_argument("--value_model_path", type=str, required=True, help="Path to pretrained value model.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel client worker processes.")
    parser.add_argument("--value_model_gpu", type=int, default=0, help="GPU ID for the value model server.")
    parser.add_argument("--idx_start", type=int, default=None, help="Start index of the dataset split.")
    parser.add_argument("--idx_end", type=int, default=None, help="End index of the dataset split.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging.")
    parser.add_argument("--keep_duplicates", action='store_true', help="Keep duplicate nodes instead of deduplicating them.")

    # SBS parameters
    parser.add_argument("--beam_width", type=int, default=4, help="Step-level beam width.")
    parser.add_argument("--n_samples", type=int, default=12, help="Total samples per step.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum search depth.")
    parser.add_argument("--budget", type=int, default=None, help="Maximum total generated tokens.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature.")
    parser.add_argument("--value_method", type=str, default="last_step", choices=["last_step", "product"], help="Method to calculate node value.")

    # Strategy-specific arguments
    parser.add_argument("--strategy", type=str, default="uniform", choices=["uniform", "pum", "token_entropy", "pum_per_value", "ucb"], help="Sampling strategy to use.")
    
    # PUM-specific
    pum_group = parser.add_argument_group('PUM Strategy Arguments')
    pum_group.add_argument("--uncertainty_model_path", type=str, help="Path to pretrained PUM uncertainty model.")
    pum_group.add_argument("--uncertainty_model_gpu", type=int, help="GPU ID for the uncertainty model server.")
    pum_group.add_argument("--uncertainty_method", type=str, default="last_step", choices=["last_step", "product", "average", "minimum"], help="Method to aggregate PUM uncertainty scores.")

    # Token Entropy-specific
    entropy_group = parser.add_argument_group('Token Entropy Strategy Arguments')
    entropy_group.add_argument("--entropy_k", type=int, default=20, help="Top-k tokens to consider for entropy calculation.")
    
    # UCB-specific
    ucb_group = parser.add_argument_group('UCB Strategy Arguments')
    ucb_group.add_argument("--ucb_c", type=float, default=1.0, help="UCB exploration parameter c (default: 1.0).")
    
    args = parser.parse_args()

    if args.strategy in ['pum', 'pum_per_value', 'ucb']:
        if not args.uncertainty_model_path or args.uncertainty_model_gpu is None:
            parser.error("--uncertainty_model_path and --uncertainty_model_gpu are required for the selected strategy.")

    os.makedirs(args.output_dir, exist_ok=True)
    run_sbs_on_dataset(args)

if __name__ == "__main__":
    main()