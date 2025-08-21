"""
Step-level Beam Search (SBS) implementation using Qwen2.5-Math-PRM for value estimation.

Uses Process Reward Model (PRM) for step-wise value estimation:
- Direct text-to-step-rewards estimation using Process Reward Model
- Automatically inserts <extra_0> tokens between reasoning steps
- Supports multiple value scoring methods:
  * "product": Product of step rewards (default) - penalizes paths with any weak steps
  * "minimum": Minimum step reward - focuses on the single weakest link
  * "average": Average of step rewards - balanced approach considering all steps equally
  * "last_step": Only the last step reward - for comparing new generations
- Returns normalized step rewards between 0 and 1
- Expects final answers in \boxed{} format for optimal PRM alignment

Usage example:
config = SBSConfig(value_scoring_method="average")  # or "product", "minimum", "last_step"
sbs = StepBeamSearch("generation_model_name", config, value_model_path="Qwen/Qwen2.5-Math-PRM-7B")
"""

import re
import logging
import time
import torch
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os
import json
from collections import Counter

# Disable torch compilation to prevent recompilation issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """The following are questions about mathematics. Think step by step and provide your answer in the format '\\boxed{}' with inside your final answer."""


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison by:
    - Converting to lowercase
    - Removing all whitespace (spaces, tabs, newlines)
    - Removing common punctuation that doesn't affect mathematical meaning
    - Handling special cases like fractions, decimals, etc.
    """
    if not answer or answer == "?":
        return answer
    
    # Convert to string and lowercase
    normalized = str(answer).lower()
    
    # Remove all whitespace
    normalized = re.sub(r'\s+', '', normalized)
    
    # Remove common punctuation that doesn't affect meaning
    # Keep mathematical operators and decimal points
    normalized = re.sub(r'[,;:()[\]{}"]', '', normalized)
    
    # Handle common mathematical expressions
    # Convert fractions like "1/2" to consistent format
    normalized = re.sub(r'(\d+)/(\d+)', r'\1/\2', normalized)
    
    # Remove trailing zeros after decimal point
    if '.' in normalized:
        normalized = normalized.rstrip('0').rstrip('.')
    
    return normalized


def extract_final_answer(text: str) -> Optional[str]:
    """
    Finds the last \\boxed{} in a string and extracts its content,
    correctly handling nested braces.
    """
    # 1. Find the starting position of the last \boxed{
    start_marker = r'\boxed{'
    last_box_start_pos = text.rfind(start_marker)

    # If \boxed{ is not found, return None
    if last_box_start_pos == -1:
        return None

    # 2. The actual content starts right after the marker
    content_start_pos = last_box_start_pos + len(start_marker)

    # 3. Use a counter (brace_level) to find the matching closing brace
    brace_level = 1
    for i in range(content_start_pos, len(text)):
        char = text[i]
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1

        # 4. When the brace level is 0, we've found the matching brace
        if brace_level == 0:
            # The content is the substring between the start and this point
            return text[content_start_pos:i]

    # If the loop finishes, it means a matching closing brace was not found
    return None


def make_step_rewards(logits, token_masks):
    """Extract step-wise rewards from PRM model outputs."""
    
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        # Find non-zero elements (where tokens match <extra_0>)
        non_zero_mask = sample.sum(dim=-1) != 0
        if non_zero_mask.any():
            valid_probs = sample[non_zero_mask] # valid_tokens, num_labels
            if valid_probs.size(1) >= 2:
                # Extract positive class probabilities (index 1)
                positive_probs = valid_probs[:, 1]
                non_zero_elements_list = positive_probs.cpu().tolist()
            else:
                # Fallback if unexpected dimensions
                non_zero_elements_list = valid_probs.flatten().cpu().tolist()
        else:
            # No valid tokens found
            non_zero_elements_list = []
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

@dataclass
class SBSConfig:
    """Configuration for Step-level Beam Search with PRM"""
    step_beam_width: int = 3
    n_generate_sample: int = 5
    max_depth: int = 10
    temperature: float = 0.6
    max_tokens: int = 512
    remove_duplicate: bool = True
    verbose: bool = True
    generation_batch_size: int = 4
    value_scoring_method: str = "product"  # "product", "minimum", "average", or "last_step"

class SBSNode:
    """Node in the Step-level Beam Search tree"""
    def __init__(self, parent: Optional['SBSNode'] = None, text: str = "", depth: int = 0):
        self.parent = parent
        self.children: List['SBSNode'] = []
        self.text = text
        self.depth = depth
        # Efficiently cache the full text path from the root
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
        # Strip excessive leading newlines to reduce repetition accumulation
        cleaned = child_text.lstrip('\n')
        child = SBSNode(parent=self, text=cleaned, depth=self.depth + 1)
        self.children.append(child)
        return child

class StepBeamSearch:
    """Step-level Beam Search implementation with vLLM for generation and PRM for value estimation."""
    def __init__(self, model_name: str, config: SBSConfig, value_model_path: str):
        self.config = config
        self.model_name = model_name
        
        # Validate scoring method
        valid_methods = ["product", "minimum", "average", "last_step"]
        if config.value_scoring_method not in valid_methods:
            raise ValueError(f"Invalid value_scoring_method '{config.value_scoring_method}'. Must be one of: {valid_methods}")
        
        # solve cache issues with torch dynamo
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors = True
        
        logger.info(f"Loading vLLM model: {model_name} on GPU 0")
        vllm_load_start = time.time()
        self.vllm_model = LLM(
            model=self.model_name, 
            max_model_len=8_000,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            tensor_parallel_size=1,
            distributed_executor_backend="mp"
        )
        if self.config.verbose:
            vllm_load_time = time.time() - vllm_load_start
            logger.info(f"vLLM model loading took {vllm_load_time:.3f}s")
        
        # Load PRM value model on GPU 1
        logger.info(f"Loading PRM value model from: {value_model_path} on GPU 1")
        value_model_start = time.time()
        self.value_model = AutoModel.from_pretrained(
            value_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 1},  # Force model to GPU 1
            trust_remote_code=True,
            # use_cache=False  # Disable cache to avoid compatibility issues
        )
        self.value_tokenizer = AutoTokenizer.from_pretrained(value_model_path, trust_remote_code=True)
        if self.value_tokenizer.pad_token is None:
            self.value_tokenizer.pad_token = self.value_tokenizer.eos_token
        self.value_model.eval()
        if self.config.verbose:
            value_model_time = time.time() - value_model_start
            logger.info(f"PRM value model loading took {value_model_time:.3f}s")
            logger.info(f"Using value scoring method: {self.config.value_scoring_method}")
        
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
        """Extract a final answer from text, capturing everything inside \\boxed{}."""
        return extract_final_answer(text)

    @torch.no_grad()
    def _get_activations_and_values(self, prompts: List[str], full_reasoning_paths: List[str]) -> List[Tuple[Optional[torch.Tensor], float]]:
        """Get values using PRM value model.
        
        Args:
            prompts: The original prompts used for generation
            full_reasoning_paths: Complete reasoning chains (parent path + new generation)
        """
        results = []
        
        for prompt, full_reasoning_text in zip(prompts, full_reasoning_paths):
            # Prepare the text with <extra_0> tokens between steps
            # Split by double newlines and join with <extra_0>
            steps = full_reasoning_text.split('\n\n')
            # Filter out empty steps
            steps = [step.strip() for step in steps if step.strip()]
            if not steps:
                # If no steps, just use the original text
                formatted_text = full_reasoning_text.strip() + "<extra_0>"
            else:
                formatted_text = "<extra_0>".join(steps) + "<extra_0>"
            
            # Extract question from the prompt - it's after the user role in the template
            question_match = re.search(r'<\|im_start\|>user\n(.+?) \\think<\|im_end\|>', prompt, re.DOTALL)
            if question_match:
                question_text = question_match.group(1).strip()
            else:
                # Fallback - use a generic question
                question_text = "Please solve this step by step."
            
            # Create messages in the format expected by the PRM
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": formatted_text}
            ]
            
            # Apply chat template
            conversation_str = self.value_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # Tokenize and move to GPU 1 (where the value model is)
            input_ids = self.value_tokenizer.encode(
                conversation_str, 
                return_tensors="pt", 
                truncation=True,
                max_length=8_000
            ).to("cuda:1") 
            
            # Get model outputs
            try:
                outputs = self.value_model(
                    input_ids=input_ids,
                    use_cache=False,
                    return_dict=True
                )
            except (AttributeError, TypeError) as e:
                # Fallback for compatibility issues
                logger.warning(f"Cache issue encountered: {e}. Trying without cache.")
                outputs = self.value_model(input_ids=input_ids)
            
            # Extract step rewards
            step_sep_tokens = self.value_tokenizer.encode("<extra_0>", add_special_tokens=False)
            if step_sep_tokens:
                step_sep_id = step_sep_tokens[0]
            else:
                # Fallback if token not found
                step_sep_id = self.value_tokenizer.encode("<extra_0>")[0]
            token_masks = (input_ids == step_sep_id)
            step_rewards = make_step_rewards(outputs[0], token_masks)
            
            # Calculate value based on the configured scoring method
            if step_rewards and step_rewards[0]:
                # Adding small epsilon to avoid exact zeros that would make entire product zero
                epsilon = 1e-8
                safe_rewards = [max(r, epsilon) for r in step_rewards[0]]
                
                if self.config.value_scoring_method == "product":
                    # Use product of all step rewards - penalizes paths with weak steps
                    value = math.prod(safe_rewards)
                elif self.config.value_scoring_method == "minimum":
                    # Use minimum step reward - focuses on the weakest link
                    value = min(safe_rewards)
                elif self.config.value_scoring_method == "average":
                    # Use average of all step rewards - balanced approach
                    value = sum(safe_rewards) / len(safe_rewards)
                elif self.config.value_scoring_method == "last_step":
                    # Use only the last step reward - for comparing new generations
                    value = safe_rewards[-1]
                else:
                    # Fallback to product if invalid method specified
                    logger.warning(f"Invalid value_scoring_method '{self.config.value_scoring_method}', using 'product'")
                    value = math.prod(safe_rewards)
            else:
                # Fallback value if no rewards found - use low score rather than neutral
                value = 0.1
            
            # No activation for PRM approach
            results.append((None, value))
        
        return results

    def _generate_and_score_candidates(self, question: str) -> List[SBSNode]:
        """Generates new steps for active beams and scores them."""
        if not self.active_beams:
            return []

        prompts = [self.create_prompt(question, node.full_text) for node in self.active_beams]
        
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stop=["\n\n"],
            n=self.config.n_generate_sample,
        )
        
        vllm_outputs = self.vllm_model.generate(prompts, sampling_params)
        
        all_candidates = []
        seen_full_texts = set()  # for duplicate removal
        for i, output in enumerate(vllm_outputs):
            parent_node = self.active_beams[i]
            parent_prompt = prompts[i]
            
            generated_texts = [gen.text for gen in output.outputs]
            
            # Create full reasoning paths (parent's full text + new generated text)
            # The PRM needs to see the entire reasoning chain, not just the new step
            full_reasoning_paths = [parent_node.full_text + gen.text for gen in output.outputs]
            
            # Get activations and values in a single pass
            activations_and_values = self._get_activations_and_values([parent_prompt] * len(generated_texts), full_reasoning_paths)

            for j, gen_text in enumerate(generated_texts):
                activation, value = activations_and_values[j]
                # Ensure each snippet ends with exactly one blank line (align with stop token) for clean concatenation
                snippet = gen_text.rstrip() + '\n\n'
                child_node = parent_node.add_child(snippet)
                child_node.activation = activation
                # IMPORTANT: The value was computed on the full reasoning path (parent + new step)
                # This ensures the PRM saw the complete context when scoring this step
                child_node.value = value

                # Answer / terminal detection (no dependency on </think>)
                ans = self._extract_final_answer(gen_text)
                if ans:
                    child_node.is_terminal = True
                    child_node.final_answer = ans
                
                # Deduplicate full paths if enabled
                if self.config.remove_duplicate:
                    if child_node.full_text in seen_full_texts:
                        continue  # drop duplicate candidate
                    seen_full_texts.add(child_node.full_text)
                
                all_candidates.append(child_node)

        # Lightweight logging of top candidate values + snippets
        if self.config.verbose and all_candidates:
            logger.info(f"Generated {len(all_candidates)} unique candidates this step.")
            top_preview = []
            for cand in sorted(all_candidates, key=lambda c: c.value, reverse=True)[:3]:
                snippet = cand.text.strip().replace('\n', ' ')[:50]
                top_preview.append(f"{cand.value:.3f} | {snippet}")
            logger.info(f"Top candidates: " + " || ".join(top_preview))

        return all_candidates

    def _update_beams(self, candidates: List[SBSNode]) -> int:
        """Updates active and completed beams. Returns number of beams newly completed this step."""
        if not candidates:
            self.active_beams = []
            self.current_beam_width = 0
            return 0
        candidates.sort(key=lambda x: x.value, reverse=True)
        new_active_beams = []
        newly_completed = 0
        
        # Only consider top candidates up to the current beam width
        for cand in candidates[:self.current_beam_width]:
            if cand.is_terminal or cand.depth >= self.config.max_depth:
                self.completed_beams.append(cand)
                newly_completed += 1
            else:
                new_active_beams.append(cand)
        
        self.active_beams = new_active_beams
        # The new beam width is the number of active beams. It cannot increase.
        self.current_beam_width = len(self.active_beams)
        
        return newly_completed

    def _create_summary(self, solutions: List[Dict[str, Any]], question: str, ground_truth: Optional[str]) -> Dict[str, Any]:
        """Creates a summary dictionary of the results."""
        answers = [s['final_answer'] for s in solutions if s['final_answer'] != "Not found"]
        
        majority_vote = "N/A"
        if answers:
            answer_counts = Counter(answers)
            majority_vote = answer_counts.most_common(1)[0][0]

        accuracy = "N/A"
        if ground_truth and answers:
            correct_answers = sum(1 for ans in answers if ans.strip().lower() == ground_truth.strip().lower())
            accuracy = f"{correct_answers / len(answers):.2%}" if answers else "0.00%"

        summary = {
            "question": question,
            "ground_truth": ground_truth,
            "majority_vote": majority_vote,
            "accuracy": accuracy,
            "solutions": solutions
        }
        return summary

    def _save_results(self, final_beams: List[SBSNode], base_path: str, question: str, ground_truth: Optional[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Saves beam search results to files."""
        solutions = []
        saved_files = []
        os.makedirs(base_path, exist_ok=True)

        for idx, node in enumerate(final_beams):
            if not node.final_answer:
                node.final_answer = self._extract_final_answer(node.full_text) or ""
            
            termination_reason = "answer_found" if node.final_answer else ("max_depth" if node.depth >= self.config.max_depth else "partial")
            
            solution_data = {
                "beam_index": idx + 1,
                "value": node.value,
                "final_answer": node.final_answer or "Not found",
                "depth": node.depth,
                "termination_reason": termination_reason,
                "solution_path": node.full_text,
            }
            solutions.append(solution_data)

            filename = f"beam_{idx+1:02d}.txt"
            file_path = os.path.join(base_path, filename)
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("# Metadata\n")
                    f.write(f"value: {node.value:.6f}\n")
                    f.write(f"final_answer: {node.final_answer or 'Not found'}\n")
                    f.write(f"depth: {node.depth}\n")
                    f.write(f"termination_reason: {termination_reason}\n")
                    f.write("\n# Full generation (step-delimited by blank lines)\n\n")
                    f.write(node.full_text)
                saved_files.append(file_path)
            except Exception as e:
                logger.warning(f"Failed to save beam {idx+1} to {file_path}: {e}")

        # Create and save summary.json
        summary_data = self._create_summary(solutions, question, ground_truth)
        summary_path = os.path.join(base_path, "summary.json")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=4)
            saved_files.append(summary_path)
        except Exception as e:
            logger.warning(f"Failed to save summary.json to {summary_path}: {e}")

        return solutions, saved_files

    def run(self, question: str, ground_truth: Optional[str] = None, base_path: Optional[str] = None) -> Dict[str, Any]:
        """Main execution loop for the Step-level Beam Search.
        If base_path is provided, saves each completed beam (metadata + full generation) as a separate .txt file.
        """
        overall_start_time = time.time()
        logger.info(f"Starting SBS for question: {question[:100]}...")
        self.clear_cache()
        
        self.root = SBSNode(text="", depth=0)
        self.active_beams = [self.root]
        self.completed_beams = []
        self.current_beam_width = self.config.step_beam_width
        
        step = 0
        while self.active_beams and step < self.config.max_depth:
            step += 1
            logger.info(f"\n--- SBS Step {step} ---")
            logger.info(f"Active beams: {len(self.active_beams)}")
            candidates = self._generate_and_score_candidates(question)
            if not candidates:
                logger.warning("No new candidates generated. Stopping search.")
                break
            newly_completed = self._update_beams(candidates)
            logger.info(f"Beams completed this step: {newly_completed}")
            logger.info(f"Active beams remaining: {len(self.active_beams)}")
            # Early stop if all active beams are terminals (should not happen because terminals moved, but guard)
            if not self.active_beams:
                break
        # Add any remaining active beams to the completed list (max depth reached)
        if self.active_beams:
            self.completed_beams.extend(self.active_beams)
            self.active_beams.clear()

        ordered_completed = sorted(self.completed_beams, key=lambda x: x.value, reverse=True)
        
        # Trim to the desired beam width before saving
        final_beams = ordered_completed[:self.config.step_beam_width]
        
        solutions, saved_files = [], []
        if base_path:
            solutions, saved_files = self._save_results(final_beams, base_path, question, ground_truth)
        
        logger.info("\n--- SBS Complete ---")
        logger.info(f"Total runtime: {time.time() - overall_start_time:.2f}s")
        logger.info(f"Total completed beams found: {len(self.completed_beams)}")
        if solutions:
            best_solution = solutions[0]
            logger.info(f"Best solution value: {best_solution['value']:.4f}, Answer: {best_solution['final_answer']}")
        if saved_files:
            logger.info(f"Saved top {len(saved_files)} beam files to: {base_path}")

        return {"question": question, "ground_truth": ground_truth, "solutions": solutions, "saved_files": saved_files}