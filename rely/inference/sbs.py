import re
import logging
import time
from unsloth import FastModel
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from vllm import LLM, SamplingParams
import os
import json
from collections import Counter
import json
from collections import Counter
import json
from collections import Counter
import json
from collections import Counter
import json
from collections import Counter

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
    """Step-level Beam Search implementation with vLLM for generation and Unsloth for value estimation."""
    def __init__(self, model_name: str, config: SBSConfig, value_head_path: Optional[str] = None):
        self.config = config
        self.model_name = model_name
        
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors = True
        
        # Load vLLM model for generation
        logger.info(f"Loading vLLM model: {model_name}")
        vllm_load_start = time.time()
        self.vllm_model = LLM(
            model="Qwen/Qwen3-1.7B", 
            max_model_len=24_000,
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True
        )
        if self.config.verbose:
            vllm_load_time = time.time() - vllm_load_start
            logger.info(f"vLLM model loading took {vllm_load_time:.3f}s")
        
        # Load Unsloth model for value estimation (forward pass only)
        logger.info(f"Loading Unsloth model for value estimation: {model_name}")
        unsloth_load_start = time.time()
        self.unsloth_model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=24_000,
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )
        if self.config.verbose:
            unsloth_load_time = time.time() - unsloth_load_start
            logger.info(f"Unsloth model loading took {unsloth_load_time:.3f}s")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.value_head = ValueHead(input_dim=self.unsloth_model.config.hidden_size)
        if value_head_path:
            logger.info(f"Loading value head from: {value_head_path}")
            value_head_start = time.time()
            self.value_head.load_state_dict(torch.load(value_head_path, map_location=self.unsloth_model.device))
            if self.config.verbose:
                value_head_time = time.time() - value_head_start
                logger.info(f"Value head loading took {value_head_time:.3f}s")
        self.value_head.to(device=self.unsloth_model.device, dtype=torch.bfloat16)
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
        """Extract a final answer letter from text. Accept several phrasings."""
        patterns = [
            r'The correct answer is \(([A-J])\)',
            r'The answer is \(([A-J])\)',
            r'So the answer is \(([A-J])\)',
            r'Answer: \(([A-J])\)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        return None

    @torch.no_grad()
    def _get_activations_and_values(self, prompts: List[str], generated_texts: List[str]) -> List[Tuple[torch.Tensor, float]]:
        """Get activations and values from the Unsloth model and value head."""
        results = []
        for prompt, gen_text in zip(prompts, generated_texts):
            full_text = prompt + gen_text + "\n\n"
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=False).to(self.unsloth_model.device)
            
            outputs = self.unsloth_model(**inputs, output_hidden_states=True)
            
            # Get the hidden state of the last token from the second to last layer
            activation = outputs.hidden_states[-2][0, -1, :]
            value = torch.sigmoid(self.value_head(activation.unsqueeze(0))).item()
            
            results.append((activation, value))
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
            
            # Get activations and values in a single pass
            activations_and_values = self._get_activations_and_values([parent_prompt] * len(generated_texts), generated_texts)

            for j, gen_text in enumerate(generated_texts):
                activation, value = activations_and_values[j]
                # Ensure each snippet ends with exactly one blank line (align with stop token) for clean concatenation
                snippet = gen_text.rstrip() + '\n\n'
                child_node = parent_node.add_child(snippet)
                child_node.activation = activation
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
            correct_answers = sum(1 for ans in answers if ans.strip().upper() == ground_truth.strip().upper())
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