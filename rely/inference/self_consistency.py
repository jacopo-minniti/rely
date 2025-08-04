from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Literal
import json
import logging
import re
from pathlib import Path
from datetime import datetime

import torch
from vllm import LLM, SamplingParams

from rely.utils.text_utils import (
    format_system_prompt,
    ensure_think_ending,
    MMLU_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency inference."""

    # Model / sampling
    model_name: str = "Qwen/Qwen3-8B"
    num_samples: int = 20
    max_new_tokens: int = 500
    # A very brief follow-up generation to let the model output the final answer
    follow_up_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.95

    # Special tokens / patterns
    end_think_token: str = "</think>"
    answer_pattern: str = r"\(([A-Z])\)"  # Captures single letter inside parentheses

    # Hardware
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


@dataclass
class SelfConsistencyResult:
    """Holds results from self-consistency inference."""

    answers: List[str]
    generated_texts: List[str]  # Full generated texts for each sample
    distribution: Dict[str, int]
    most_consistent_answer: Optional[str]
    config: SelfConsistencyConfig


class SelfConsistencyInference:
    """Simple self-consistency inference implementation."""

    def __init__(self, config: SelfConsistencyConfig):
        self.config = config
        self.llm: Optional[LLM] = None
        self._initialize_model()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _initialize_model(self):
        """Load the model once so we can reuse across generations."""
        try:
            logger.info(f"Loading model {self.config.model_name} with vLLM …")
            self.llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text from the model with the provided prompt."""
        if self.llm is None:
            raise ValueError("LLM not initialized")

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer letter using the configured regex pattern."""
        matches = list(re.finditer(self.config.answer_pattern, text))
        if not matches:
            return None
        return matches[-1].group(1)  # Take the last match

    def _single_sample(self, system_prompt: str, user_question: str) -> tuple[str, str]:
        """Generate a single reasoning sample and return the extracted answer and full generated text."""
        # Build the initial prompt with an open <think> tag so the model reasons.
        base_prompt = format_system_prompt(system_prompt, user_question) + "<think>\n"

        # First (full budget) generation
        generated = self._generate(base_prompt, self.config.max_new_tokens)
        current_text = base_prompt + generated

        # If the model did not decide to close the thinking section, force it to do so
        if self.config.end_think_token not in generated:
            logger.debug("End-of-thinking token not found, issuing follow-up generation …")
            # Ensure prompt ends with </think> so the model realises reasoning is done.
            current_text = ensure_think_ending(current_text)
            follow_up = self._generate(current_text, self.config.follow_up_tokens)
            current_text += follow_up

        answer = self._extract_answer(current_text)
        if answer is None:
            logger.debug("No answer extracted from sample.")
        return (answer or "?", current_text)  # Return both answer and full generated text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_inference(self, system_prompt: str, user_question: str) -> SelfConsistencyResult:
        """Run self-consistency inference and return aggregated results."""
        answers: List[str] = []
        generated_texts: List[str] = []
        logger.info(f"Running self-consistency for {self.config.num_samples} samples …")
        for i in range(self.config.num_samples):
            answer, full_text = self._single_sample(system_prompt, user_question)
            answers.append(answer)
            generated_texts.append(full_text)
            logger.debug(f"Sample {i + 1}/{self.config.num_samples} ⇒ {answer}")

        # Compute distribution
        distribution: Dict[str, int] = {}
        for ans in answers:
            distribution[ans] = distribution.get(ans, 0) + 1

        # Identify the most frequent answer (ties broken arbitrarily)
        most_consistent: Optional[str] = None
        if distribution:
            most_consistent = max(distribution.items(), key=lambda kv: kv[1])[0]

        return SelfConsistencyResult(
            answers=answers,
            generated_texts=generated_texts,
            distribution=distribution,
            most_consistent_answer=most_consistent,
            config=self.config,
        )


def create_self_consistency_inference(config: SelfConsistencyConfig) -> SelfConsistencyInference:
    """Factory helper to build the inference object."""
    return SelfConsistencyInference(config)


def save_self_consistency_result(
    result: SelfConsistencyResult,
    system_prompt: str,
    user_question: str,
    output_path: Union[str, Path],
    correct_answer: Optional[str] = None,
) -> None:
    """Persist results to disk for later inspection.
    
    Saves:
    1. Individual generation files (generation_0.txt, generation_1.txt, etc.)
    2. Summary file (summary.json) with results and distribution
    
    Creates a timestamped subdirectory: run_{datetime}/
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save individual generation files
    for i, (answer, generated_text) in enumerate(zip(result.answers, result.generated_texts)):
        generation_file = run_dir / f"generation_{i}.txt"
        with open(generation_file, "w") as f:
            f.write("# System Prompt\n")
            f.write(system_prompt)
            f.write("\n\n# Question\n")
            f.write(user_question)
            f.write("\n\n# Generated Text\n")
            f.write(generated_text)
            f.write(f"\n\n# Extracted Answer\n")
            f.write(answer)
        
        logger.debug(f"Saved generation {i} to {generation_file}")

    # Save summary file as JSON
    summary_file = run_dir / "summary.json"
    
    # Prepare summary data
    summary_data = {
        "model": result.config.model_name,
        "num_samples": result.config.num_samples,
        "temperature": result.config.temperature,
        "top_p": result.config.top_p,
        "max_new_tokens": result.config.max_new_tokens,
        "most_consistent_answer": result.most_consistent_answer,
        "run_timestamp": timestamp,
        "run_directory": f"run_{timestamp}",
        "answer_distribution": {
            ans: {
                "count": count,
                "percentage": (count / result.config.num_samples) * 100
            }
            for ans, count in result.distribution.items()
        },
        "all_answers": result.answers,
        "individual_generation_files": [
            f"generation_{i}.txt" for i in range(result.config.num_samples)
        ]
    }
    
    # Add evaluation metrics if correct_answer is provided
    if correct_answer is not None:
        correct_count = sum(1 for answer in result.answers if answer.strip().upper() == correct_answer.strip().upper())
        hard_label = 1 if correct_count > 0 else 0
        soft_label = correct_count / len(result.answers) if result.answers else 0.0
        
        summary_data["correct_answer"] = correct_answer
        summary_data["hard_label"] = hard_label
        summary_data["soft_label"] = soft_label
        summary_data["correct_count"] = correct_count
        summary_data["total_answers"] = len(result.answers)
    
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Saved self-consistency results to {run_dir}")
    logger.info(f"Summary: {summary_file}")
    logger.info(f"Individual generations: {len(result.answers)} files")


def run_self_consistency_inference(
    system_prompt: str,
    user_question: str,
    config: Optional[SelfConsistencyConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    correct_answer: Optional[str] = None,
) -> SelfConsistencyResult:
    """Run self-consistency inference with the given (or default) configuration.
    
    Args:
        system_prompt: The system prompt to use for inference
        user_question: The user question to answer
        config: Configuration for self-consistency inference (uses default if None)
        save_path: Optional path to save results to disk (if provided)
    
    Returns:
        SelfConsistencyResult containing the inference results
    """
    if config is None:
        config = SelfConsistencyConfig()
    inference = create_self_consistency_inference(config)
    result = inference.run_inference(system_prompt, user_question)
    
    # Save results if path is provided
    if save_path is not None:
        save_self_consistency_result(result, system_prompt, user_question, save_path, correct_answer)
    
    return result