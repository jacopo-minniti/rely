# rely/inference/sbs/utils.py

import argparse
import logging
from multiprocessing import Queue
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel

from rely.utils import prompt_pattern
from rely.train.soft_prm.model import SoftClassificationPRMModel

logger = logging.getLogger(__name__)

@dataclass
class SBSConfig:
    """Configuration for Step-level Beam Search"""
    strategy: str = "uniform"
    step_beam_width: int = 3
    n_total_samples: int = 5
    max_depth: Optional[int] = None
    budget: Optional[int] = None
    temperature: float = 0.6
    max_tokens: int = 512
    remove_duplicate: bool = True
    verbose: bool = True
    value_method: str = "last_step"
    uncertainty_method: str = "last_step" # For PUM
    entropy_k: int = 20 # For TokenEntropy

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

def _uncertainty_model_server(args: argparse.Namespace, task_queue: Queue, result_queues: List[Queue]):
    uncertainty_device = torch.device(f"cuda:{args.uncertainty_model_gpu}")
    logger.info(f"[UncertaintyServer] Starting on device {uncertainty_device}")

    tokenizer = AutoTokenizer.from_pretrained(args.uncertainty_model_path, trust_remote_code=True)
    # OLD AutoModelForTokenClassification model loading (commented out)
    # model = AutoModelForTokenClassification.from_pretrained(
    #     args.uncertainty_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    # )
    
    # NEW custom SoftClassificationPRMModel loading
    model = SoftClassificationPRMModel.from_pretrained(
        args.uncertainty_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    # Ensure all model parameters are in bfloat16 to avoid dtype mismatch
    model = model.to(dtype=torch.bfloat16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    logger.info("[UncertaintyServer] Uncertainty model loaded.")

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
            try:
                inputs = tokenizer(batch_conversations, return_tensors="pt", padding=True, truncation=True, max_length=7000).to(uncertainty_device)
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                token_masks = (inputs.input_ids == step_sep_id)
                
                # OLD AutoModelForTokenClassification logic (commented out)
                # uncertainty_probs = torch.softmax(outputs.logits, dim=-1)[:, :, 1]
                
                # NEW custom SoftClassificationPRMModel logic - direct sigmoid probabilities
                uncertainty_probs = outputs.logits
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

                        # OLD AutoModelForTokenClassification logic (commented out)
                        # uncertainty_probs = torch.softmax(outputs.logits, dim=-1)[0, :, 1]
                        
                        # NEW custom SoftClassificationPRMModel logic - direct sigmoid probabilities
                        uncertainty_probs = outputs.logits[0, :]
                        
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
    # NEW custom SoftClassificationPRMModel loading
    model = SoftClassificationPRMModel.from_pretrained(
        args.value_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    # Ensure all model parameters are in bfloat16 to avoid dtype mismatch
    model = model.to(dtype=torch.bfloat16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    logger.info("[ValueServer] Value model loaded.")

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
            assistant_content = "<extra_0>".join(steps) + "<extra_0>" if steps else "<extra_0>"
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
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                token_masks = (inputs.input_ids == step_sep_id)
                
                # NEW custom SoftClassificationPRMModel logic - direct sigmoid probabilities
                value_probs = outputs.logits
                has_separator = token_masks.any(dim=1)
                default_value = torch.tensor(1.0, device=value_device)
                
                calculated_values = torch.zeros_like(has_separator, dtype=torch.float)

                # Use "last_step" method (same as uncertainty model default)
                seq_len = token_masks.size(1)
                reverse_mask = torch.flip(token_masks, dims=[1])
                last_indices = (seq_len - 1) - torch.argmax(reverse_mask.float(), dim=1)
                calculated_values = torch.gather(value_probs, 1, last_indices.unsqueeze(-1)).squeeze(-1)
                
                # TEMPORARY CHANGE: Invert value scores (1 - score) - EASY ROLLBACK
                inverted_scores = 1.0 - calculated_values
                final_values = torch.where(has_separator, inverted_scores, default_value)
                # END TEMPORARY CHANGE
                all_rewards.extend(final_values.cpu().tolist())

                del inputs, outputs, token_masks, value_probs
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.error(f"[ValueServer] CUDA OOM in batch, falling back to single processing.")
                torch.cuda.empty_cache()
                for single_conv in batch_conversations:
                    try:
                        single_inputs = tokenizer([single_conv], return_tensors="pt", padding=True, truncation=True, max_length=7000).to(value_device)
                        outputs = model(input_ids=single_inputs.input_ids, attention_mask=single_inputs.attention_mask)
                        token_mask = (single_inputs.input_ids[0] == step_sep_id)
                        
                        if not token_mask.any():
                            # TEMPORARY CHANGE: Use 1.0 as default for inverted scores - EASY ROLLBACK
                            all_rewards.append(1.0)
                            # END TEMPORARY CHANGE
                            continue

                        # NEW custom SoftClassificationPRMModel logic - direct sigmoid probabilities
                        value_probs = outputs.logits[0, :]
                        
                        # Use "last_step" method
                        last_idx = torch.where(token_mask)[0][-1]
                        score = value_probs[last_idx].item()
                        
                        # TEMPORARY CHANGE: Invert value scores (1 - score) - EASY ROLLBACK
                        inverted_score = 1.0 - score
                        all_rewards.append(inverted_score)
                        # END TEMPORARY CHANGE

                        del single_inputs, outputs
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