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
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
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
    n_total_samples: int = 5
    max_depth: Optional[int] = None
    budget: Optional[int] = None
    temperature: float = 0.6
    max_tokens: int = 512
    need_value_func: bool = True
    need_uncertainty_func: bool = True
    remove_duplicate: bool = True
    verbose: bool = True
    generation_batch_size: int = 4
    value_method: str = "last_step"
    uncertainty_method: str = "last_step"


class SBSNode:
    """Node in the Step-level Beam Search tree"""
    def __init__(self, parent: Optional['SBSNode'] = None, text: str = "", depth: int = 0):
        self.parent = parent
        self.children: List['SBSNode'] = []
        self.text = text
        self.depth = depth
        self.full_text: str = (parent.full_text if parent else "") + text
        self.value: float = -100.0
        self.uncertainty: float = 0.5
        self.is_terminal = False
        self.final_answer = ""

    def add_child(self, child_text: str) -> 'SBSNode':
        cleaned = child_text.lstrip('\n')
        child = SBSNode(parent=self, text=cleaned, depth=self.depth + 1)
        self.children.append(child)
        return child


class StepBeamSearch:
    """Step-level Beam Search implementation with uncertainty-weighted sampling."""
    def __init__(self,
                 inference_model_name: str,
                 config: SBSConfig,
                 value_task_queue: Queue,
                 value_result_queue: Queue,
                 uncertainty_task_queue: Queue,
                 uncertainty_result_queue: Queue,
                 worker_rank: int):
        self.config = config
        if self.config.max_depth is None and self.config.budget is None:
            raise ValueError("Either max_depth or budget must be specified for the search.")
        
        self.inference_model_name = inference_model_name
        self.worker_rank = worker_rank
        
        self.value_task_queue = value_task_queue
        self.value_result_queue = value_result_queue
        self.uncertainty_task_queue = uncertainty_task_queue
        self.uncertainty_result_queue = uncertainty_result_queue

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

    def _get_uncertainties_from_server(self, prompts: List[str]) -> List[float]:
        if not prompts:
            return []
        request_id = str(uuid.uuid4())
        payload = {"request_id": request_id, "worker_rank": self.worker_rank, "prompts": prompts}
        self.uncertainty_task_queue.put(payload)
        
        while True:
            response = self.uncertainty_result_queue.get()
            if response.get("request_id") == request_id:
                return response["uncertainties"]
            self.uncertainty_result_queue.put(response)
            time.sleep(0.01)

    def _calculate_sample_distribution(self, uncertainty_scores: List[float]) -> List[int]:
        if not uncertainty_scores:
            return []

        num_beams = len(uncertainty_scores)
        total_samples = self.config.n_total_samples

        total_uncertainty = sum(uncertainty_scores)
        if total_uncertainty == 0:
            # Fallback to even distribution if all uncertainties are zero
            samples_per_beam = total_samples // num_beams
            remainder = total_samples % num_beams
            distribution = [samples_per_beam] * num_beams
            for i in range(remainder):
                distribution[i] += 1
            return distribution

        # Proportional allocation
        normalized_scores = [score / total_uncertainty for score in uncertainty_scores]
        sample_distribution = [int(norm_score * total_samples) for norm_score in normalized_scores]

        # Distribute remainder to most uncertain beams to stay within budget
        total_assigned = sum(sample_distribution)
        remainder = total_samples - total_assigned
        if remainder > 0:
            sorted_indices = sorted(range(num_beams), key=lambda k: uncertainty_scores[k], reverse=True)
            for i in range(remainder):
                sample_distribution[sorted_indices[i % num_beams]] += 1
        
        # "Robin Hood" redistribution: take from the richest, give to the poorest
        zero_indices = [i for i, s in enumerate(sample_distribution) if s == 0]
        if zero_indices:
            for i in zero_indices:
                # Find the current richest beam that can afford to give one sample
                # This needs to be re-evaluated in each iteration as the max might change
                donatable_beams = sorted([(s, j) for j, s in enumerate(sample_distribution) if s > 1], reverse=True)
                
                if donatable_beams:
                    # Index of the richest beam
                    max_index = donatable_beams[0][1]
                    
                    # Redistribute
                    sample_distribution[max_index] -= 1
                    sample_distribution[i] += 1
                else:
                    # This occurs if n_samples < num_beams, where no beam has more than 1 sample.
                    # Cannot redistribute without violating budget or the "min 1" rule.
                    break
        
        if self.config.verbose:
            logger.info(f"[Rank {self.worker_rank}] Normalized uncertainty scores: {[f'{s:.3f}' for s in normalized_scores]}")

        return sample_distribution

    def _make_api_request_with_samples(self, prompt: str, n_samples: int) -> List[str]:
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
            for gen in generations:
                self.total_generated_tokens += len(self.tokenizer.encode(gen, add_special_tokens=False))
            return generations
        except Exception as e:
            logger.error(f"[Rank {self.worker_rank}] Error during API call: {e}")
            return []

    def _generate_and_score_candidates(self, question: str) -> List[SBSNode]:
        if not self.active_beams:
            return []

        uncertainty_prompts = [self.create_prompt(question, beam.full_text) for beam in self.active_beams]
        uncertainty_scores = self._get_uncertainties_from_server(uncertainty_prompts)
        
        for i, beam in enumerate(self.active_beams):
            if i < len(uncertainty_scores):
                beam.uncertainty = uncertainty_scores[i]
        
        sample_distribution = self._calculate_sample_distribution(uncertainty_scores)
        
        if self.config.verbose:
            logger.info(f"[Rank {self.worker_rank}] Uncertainty scores: {[f'{s:.3f}' for s in uncertainty_scores]}")
            logger.info(f"[Rank {self.worker_rank}] Sample distribution: {sample_distribution}")

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

        value_request_prompts, value_request_texts, parent_nodes = [], [], []
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
                
                child_node.value = parent_node.value * new_step_value if self.config.value_method == 'product' else new_step_value
                
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
 + forced_outputs[i]
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
                
            self._update_beams(candidates)

        self.completed_beams.extend(self.active_beams)
        final_beams = sorted(self.completed_beams, key=lambda x: x.value, reverse=True)[:self.config.step_beam_width]
        
        solutions = []
        if base_path:
            solutions = self._save_results(final_beams, base_path, question, ground_truth)
        
        if self.config.verbose and solutions:
            best = solutions[0]
            logger.info(f"--- [Rank {self.worker_rank}] SBS Complete --- Best solution value: {best['value']:.4f}, Answer: {best['final_answer']}")

        return {"question": question, "ground_truth": ground_truth, "solutions": solutions}


def _uncertainty_model_server(args: argparse.Namespace, task_queue: Queue, result_queues: List[Queue]):
    uncertainty_device = torch.device(f"cuda:{args.uncertainty_model_gpu}")
    logger.info(f"[UncertaintyServer] Starting on device {uncertainty_device}")

    tokenizer = AutoTokenizer.from_pretrained(args.uncertainty_model_path, trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.uncertainty_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    logger.info("[UncertaintyServer] Uncertainty model loaded.")

    prompt_pattern = re.compile(r"<\|im_start\|>system\n(.*?)\|im_end\|>\n<\|im_start\|>user\n(.*?)\|im_end\|>\n<\|im_start\|>assistant\n(.*)", re.DOTALL)

    @torch.no_grad()
    def get_uncertainties(prompts: List[str]) -> List[float]:
        conversation_strs = []
        for prompt in prompts:
            match = prompt_pattern.match(prompt)
            if not match:
                logger.error(f"Prompt did not match expected format. Cannot score. Prompt: {prompt[:200]}")
                continue
            system_msg, user_msg, partial_solution = match.groups()
            steps = [s.strip() for s in (partial_solution or "").split('\n\n') if s.strip()]
            formatted_content = "<extra_0>".join(steps) + "<extra_0>" if steps else "<extra_0>"
            messages = [{"role": "system", "content": system_msg.strip()}, {"role": "user", "content": user_msg.strip()}, {"role": "assistant", "content": formatted_content}]
            conversation_strs.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

        if not conversation_strs:
            return [0.5] * len(prompts)

        batch_size = 8
        all_uncertainties = []
        step_sep_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]

        for i in range(0, len(conversation_strs), batch_size):
            batch_conversations = conversation_strs[i:i + batch_size]
            inputs = tokenizer(batch_conversations, return_tensors="pt", padding=True, truncation=True, max_length=7000).to(uncertainty_device)

            try:
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                token_masks = (inputs.input_ids == step_sep_id)
                
                uncertainty_probs = torch.softmax(outputs.logits, dim=-1)[:, :, 1]
                has_separator = token_masks.any(dim=1)
                default_uncertainty = torch.tensor(0.5, device=uncertainty_device)
                
                calculated_uncertainties = torch.zeros_like(has_separator, dtype=torch.float)

                if args.uncertainty_method == "product":
                    clamped_probs = uncertainty_probs.clamp(min=1e-8)
                    masked_probs = torch.where(token_masks, clamped_probs, 1.0)
                    calculated_uncertainties = masked_probs.prod(dim=1)
                elif args.uncertainty_method == "average":
                    masked_probs = uncertainty_probs * token_masks.float()
                    sums = masked_probs.sum(dim=1)
                    counts = token_masks.sum(dim=1).clamp(min=1)
                    calculated_uncertainties = sums / counts
                elif args.uncertainty_method == "minimum":
                    masked_probs = torch.where(token_masks, uncertainty_probs, 2.0)
                    minimums, _ = masked_probs.min(dim=1)
                    calculated_uncertainties = minimums
                else:  # "last_step" is default
                    seq_len = token_masks.size(1)
                    reverse_mask = torch.flip(token_masks, dims=[1])
                    last_indices = (seq_len - 1) - torch.argmax(reverse_mask.float(), dim=1)
                    calculated_uncertainties = torch.gather(uncertainty_probs, 1, last_indices.unsqueeze(-1)).squeeze(-1)
                
                final_uncertainties = torch.where(has_separator, calculated_uncertainties, default_uncertainty)
                all_uncertainties.extend(final_uncertainties.cpu().tolist())

                del inputs, outputs, token_masks, uncertainty_probs
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.error(f"[UncertaintyServer] CUDA OOM in batch, falling back to single processing.")
                torch.cuda.empty_cache()
                for single_conv in batch_conversations:
                    try:
                        single_inputs = tokenizer([single_conv], return_tensors="pt", padding=True, truncation=True, max_length=7000).to(uncertainty_device)
                        outputs = model(input_ids=single_inputs.input_ids, attention_mask=single_inputs.attention_mask)
                        token_mask = (single_inputs.input_ids[0] == step_sep_id)
                        
                        if not token_mask.any():
                            all_uncertainties.append(0.5)
                            continue

                        uncertainty_probs = torch.softmax(outputs.logits, dim=-1)[0, :, 1]
                        
                        if args.uncertainty_method == "product":
                            score = uncertainty_probs[token_mask].clamp(min=1e-8).prod().item()
                        elif args.uncertainty_method == "average":
                            score = uncertainty_probs[token_mask].mean().item()
                        elif args.uncertainty_method == "minimum":
                            score = uncertainty_probs[token_mask].min().item()
                        else: # "last_step"
                            last_idx = torch.where(token_mask)[0][-1]
                            score = uncertainty_probs[last_idx].item()
                        
                        all_uncertainties.append(score)

                        del single_inputs, outputs
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"[UncertaintyServer] Error processing single sample in fallback: {e}")
                        all_uncertainties.append(0.5)
        
        return all_uncertainties

    while True:
        try:
            task = task_queue.get()
            if task == "STOP": break
            request_id, worker_rank = task["request_id"], task["worker_rank"]
            uncertainties = get_uncertainties(task["prompts"])
            result_queues[worker_rank].put({"request_id": request_id, "uncertainties": uncertainties})
        except Exception as e:
            logger.error(f"[UncertaintyServer] Error processing task: {e}", exc_info=True)
    logger.info("[UncertaintyServer] Shutting down.")


def _value_model_server(args: argparse.Namespace, task_queue: Queue, result_queues: List[Queue]):
    value_device = torch.device(f"cuda:{args.value_model_gpu}")
    logger.info(f"[ValueServer] Starting on device {value_device}")

    tokenizer = AutoTokenizer.from_pretrained(args.value_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.value_model_path, num_labels=2, device_map="auto", trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    logger.info("[ValueServer] Value model loaded.")

    prompt_pattern = re.compile(r"<\|im_start\|>system\n(.*?)\|im_end\|>\n<\|im_start\|>user\n(.*?)\|im_end\|>\n<\|im_start\|>assistant\n(.*)", re.DOTALL)

    @torch.no_grad()
    def get_values(prompts: List[str], generated_texts: List[str]) -> List[float]:
        conversation_strs = []
        for prompt, gen_text in zip(prompts, generated_texts):
            match = prompt_pattern.match(prompt)
            if not match:
                logger.error(f"Prompt did not match expected format. Cannot score. Prompt: {prompt[:200]}")
                continue
            system_msg, user_msg, partial_solution = match.groups()
            full_assistant_response = ((partial_solution or "").strip() + '\n\n' + gen_text.strip()).strip()
            steps = [s.strip() for s in full_assistant_response.split('\n\n') if s.strip()]
            assistant_content = "<extra_0>".join(steps) + "<extra_0>"
            messages = [{"role": "system", "content": system_msg.strip()}, {"role": "user", "content": user_msg.strip()}, {"role": "assistant", "content": assistant_content}]
            conversation_strs.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

        if not conversation_strs:
            return [0.0] * len(prompts)

        batch_size = 8
        all_rewards = []
        step_sep_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]

        for i in range(0, len(conversation_strs), batch_size):
            batch_conversations = conversation_strs[i:i + batch_size]
            try:
                inputs = tokenizer(batch_conversations, return_tensors="pt", padding=True, truncation=True, max_length=7000).to(value_device)
                base_model_output = model.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)
                logits = model.score(base_model_output.last_hidden_state)
                
                probabilities = torch.softmax(logits, dim=-1)
                token_masks = (inputs.input_ids == step_sep_id)
                scores = probabilities[:, :, 1]
                
                seq_len = token_masks.size(1)
                reverse_mask = torch.flip(token_masks, dims=[1])
                last_indices = (seq_len - 1) - torch.argmax(reverse_mask.float(), dim=1)
                
                last_step_scores = torch.gather(scores, 1, last_indices.unsqueeze(-1)).squeeze(-1)
                has_separator = token_masks.any(dim=1)
                
                batch_rewards_tensor = torch.where(has_separator, last_step_scores, torch.tensor(0.0, device=value_device))
                all_rewards.extend(batch_rewards_tensor.cpu().tolist())

                del inputs, base_model_output, logits, probabilities
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.error(f"[ValueServer] CUDA OOM in batch, falling back to single processing.")
                torch.cuda.empty_cache()
                for single_conv in batch_conversations:
                    try:
                        single_inputs = tokenizer([single_conv], return_tensors="pt", padding=True, truncation=True, max_length=7000).to(value_device)
                        base_model_output = model.model(input_ids=single_inputs.input_ids, attention_mask=single_inputs.attention_mask, use_cache=False)
                        logits = model.score(base_model_output.last_hidden_state)
                        
                        probabilities = torch.softmax(logits, dim=-1)
                        token_mask = (single_inputs.input_ids[0] == step_sep_id)
                        
                        if token_mask.any():
                            scores = probabilities[0, :, 1]
                            last_idx = torch.where(token_mask)[0][-1]
                            score = scores[last_idx].item()
                            all_rewards.append(score)
                        else:
                            all_rewards.append(0.0)

                        del single_inputs, base_model_output, logits, probabilities
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"[ValueServer] Error processing single sample in fallback: {e}")
                        all_rewards.append(0.0)
        
        return all_rewards


    while True:
        try:
            task = task_queue.get()
            if task == "STOP": break
            request_id, worker_rank = task["request_id"], task["worker_rank"]
            values = get_values(task["prompts"], task["generated_texts"])
            result_queues[worker_rank].put({"request_id": request_id, "values": values})
        except Exception as e:
            logger.error(f"[ValueServer] Error processing task: {e}", exc_info=True)
    logger.info("[ValueServer] Shutting down.")


def _sbs_worker(args: argparse.Namespace, dataset_slice: List[Dict[str, Any]], rank: int, value_task_queue: Queue, value_result_queue: Queue, uncertainty_task_queue: Queue, uncertainty_result_queue: Queue):
    logger.info(f"[Rank {rank}] Worker started.")
    sbs_config = SBSConfig(
        step_beam_width=args.beam_width, n_total_samples=args.n_samples, max_depth=args.max_depth,
        budget=args.budget, temperature=args.temperature, verbose=args.verbose,
        value_method=args.value_method, uncertainty_method=args.uncertainty_method,
    )
    sbs_instance = StepBeamSearch(
        inference_model_name=args.inference_model, config=sbs_config, value_task_queue=value_task_queue,
        value_result_queue=value_result_queue, uncertainty_task_queue=uncertainty_task_queue,
        uncertainty_result_queue=uncertainty_result_queue, worker_rank=rank,
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

    # --- Check for already processed questions and exclude them ---
    results_dir = "results/sbs_uncertain_max_4_20"
    if os.path.exists(results_dir):
        processed_indices = set()
        for folder_name in os.listdir(results_dir):
            if folder_name.startswith("q_"):
                try:
                    idx = int(folder_name.split("_")[1])
                    processed_indices.add(idx)
                except (IndexError, ValueError):
                    continue
        
        original_count = len(dataset)
        dataset = [item for item in dataset if item['original_index'] not in processed_indices]
        logger.info(f"Excluded {original_count - len(dataset)} already processed questions. Remaining: {len(dataset)}")
    # --- End of exclusion block ---

    num_workers = args.num_workers
    value_task_queue = Queue()
    value_result_queues = [Queue() for _ in range(num_workers)]
    uncertainty_task_queue = Queue()
    uncertainty_result_queues = [Queue() for _ in range(num_workers)]
    
    logger.info("Starting Value Model Server process...")
    value_server_proc = Process(target=_value_model_server, args=(args, value_task_queue, value_result_queues))
    value_server_proc.start()
    
    logger.info("Starting Uncertainty Model Server process...")
    uncertainty_server_proc = Process(target=_uncertainty_model_server, args=(args, uncertainty_task_queue, uncertainty_result_queues))
    uncertainty_server_proc.start()
    
    time.sleep(30) # Allow time for models to load

    total_samples = len(dataset)
    samples_per_worker = (total_samples + num_workers - 1) // num_workers
    worker_procs = []

    logger.info(f"Starting {num_workers} SBS client workers.")
    for i in range(num_workers):
        start_idx = i * samples_per_worker
        end_idx = min(start_idx + samples_per_worker, total_samples)
        if not (dataset_slice := dataset[start_idx:end_idx]): continue
        proc = Process(target=_sbs_worker, args=(args, dataset_slice, i, value_task_queue, value_result_queues[i], uncertainty_task_queue, uncertainty_result_queues[i]))
        proc.start()
        worker_procs.append(proc)
        
    for proc in worker_procs:
        proc.join()

    logger.info("All SBS workers finished. Shutting down servers...")
    value_task_queue.put("STOP")
    uncertainty_task_queue.put("STOP")
    value_server_proc.join()
    uncertainty_server_proc.join()
    logger.info("All processes have completed.")


def main():
    parser = argparse.ArgumentParser(description="Run SBS using a vLLM server with uncertainty-weighted sampling.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--inference_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name served by vLLM.")
    parser.add_argument("--value_model_path", type=str, required=True, help="Path to pretrained value model.")
    parser.add_argument("--uncertainty_model_path", type=str, default="jacopo-minniti/Qwen2.5-Math-7B-PUM-half_entropy", help="Path to pretrained uncertainty model.")
    
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel client worker processes.")
    parser.add_argument("--value_model_gpu", type=int, default=0, help="GPU ID for the value model server.")
    parser.add_argument("--uncertainty_model_gpu", type=int, default=0, help="GPU ID for the uncertainty model server.")
    
    parser.add_argument("--beam_width", type=int, default=4, help="Step-level beam width.")
    parser.add_argument("--n_samples", type=int, default=12, help="Total samples per step, distributed by uncertainty.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum search depth.")
    parser.add_argument("--budget", type=int, default=None, help="Maximum total generated tokens.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging.")
    
    parser.add_argument("--value_method", type=str, default="last_step", choices=["last_step", "product"], help="Method to calculate node value.")
    parser.add_argument("--uncertainty_method", type=str, default="last_step", choices=["last_step", "product", "average", "minimum"], help="Method to aggregate uncertainty scores.")

    parser.add_argument("--idx_start", type=int, default=None, help="Start index of the dataset split.")
    parser.add_argument("--idx_end", type=int, default=None, help="End index of the dataset split.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_sbs_on_dataset(args)

if __name__ == "__main__":
    main()