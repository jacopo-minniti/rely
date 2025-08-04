from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import json
import logging
import re
from pathlib import Path
from datetime import datetime

from vllm import LLM, SamplingParams

from rely.utils.text_utils import (
    format_system_prompt,
    ensure_think_ending,
    MMLU_SYSTEM_PROMPT,
)

# Dummy implementations for standalone execution
def format_system_prompt(system_prompt: str, user_question: str) -> str:
    return f"{system_prompt}\n\nUSER: {user_question}\nASSISTANT: "

def ensure_think_ending(text: str) -> str:
    end_tag = "</think>"
    if text.rstrip().endswith(end_tag):
        return text
    return text.rstrip() + f"\n{end_tag}\n"

logger = logging.getLogger(__name__)


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency inference."""
    # Model / sampling
    model_name: str = "Qwen/Qwen2-7B-Instruct"
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


@dataclass
class BatchSelfConsistencyResult:
    """Holds results from batch self-consistency inference."""
    results: List[SelfConsistencyResult]  # One result per question
    questions: List[str]  # The questions that were processed
    config: SelfConsistencyConfig


class SelfConsistencyInference:
    """Efficient self-consistency inference with batch follow-up processing."""

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
            logger.info(f"Loading model {self.config.model_name} with vLLM…")
            self.llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                enable_prefix_caching=True,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _generate(self, prompt: str, max_tokens: int, n: int = 1) -> Union[str, List[str]]:
        """Generate text from the model. Returns a list for n > 1."""
        if self.llm is None:
            raise ValueError("LLM not initialized")

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens,
            n=n,
        )
        # vLLM generate expects a list of prompts
        outputs = self.llm.generate([prompt], sampling_params)
        
        # When n>1, outputs[0] contains all n samples for the single prompt
        generated = [output.text for output in outputs[0].outputs]
        return generated if n > 1 else generated[0]

    def _generate_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        """Generate text for a batch of *different* prompts."""
        if self.llm is None:
            raise ValueError("LLM not initialized")

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer letter using the configured regex pattern."""
        matches = list(re.finditer(self.config.answer_pattern, text))
        if not matches:
            return None
        return matches[-1].group(1)  # Take the last match

    # ------------------------------------------------------------------
    # Core Logic (Refactored for Efficiency)
    # ------------------------------------------------------------------
    def _get_consistent_samples(self, system_prompt: str, user_question: str) -> List[tuple[str, str]]:
        """
        Generates N samples and efficiently handles follow-ups for incomplete ones in a single batch.

        Returns:
            A list of tuples, where each tuple is (answer, full_generated_text).
        """
        base_prompt = format_system_prompt(system_prompt, user_question) + "<think>\n"

        # 1. Generate all N samples in a single efficient call
        initial_samples = self._generate(base_prompt, self.config.max_new_tokens, n=self.config.num_samples)

        complete_results = []
        follow_up_prompts = []

        # 2. Partition samples into "complete" and "incomplete" (needs follow-up)
        for generated_text in initial_samples:
            full_text = base_prompt + generated_text
            if self.config.end_think_token in generated_text:
                answer = self._extract_answer(full_text)
                complete_results.append((answer or "?", full_text))
            else:
                follow_up_prompt = ensure_think_ending(full_text)
                follow_up_prompts.append(follow_up_prompt)

        # 3. Process the follow-up batch in a single call (if any)
        if follow_up_prompts:
            logger.debug(f"Issuing a batch follow-up for {len(follow_up_prompts)} incomplete samples...")
            follow_up_completions = self._generate_batch(follow_up_prompts, self.config.follow_up_tokens)

            for original_prompt, completion in zip(follow_up_prompts, follow_up_completions):
                final_text = original_prompt + completion
                answer = self._extract_answer(final_text)
                complete_results.append((answer or "?", final_text))

        return complete_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_inference(self, system_prompt: str, user_question: str) -> SelfConsistencyResult:
        """Run self-consistency for a single question and return aggregated results."""
        logger.info(f"Running self-consistency for '{user_question[:50]}...' with {self.config.num_samples} samples...")
        
        # The refactored helper does all the generation and follow-up work efficiently.
        results = self._get_consistent_samples(system_prompt, user_question)

        answers = [res[0] for res in results]
        generated_texts = [res[1] for res in results]

        for i, answer in enumerate(answers):
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

    def run_batch_inference(self, system_prompt: str, user_questions: List[str]) -> BatchSelfConsistencyResult:
        """Run self-consistency inference on multiple questions efficiently."""
        logger.info(f"Running batch self-consistency for {len(user_questions)} questions with {self.config.num_samples} samples each...")
        all_results = []

        # This now correctly loops over UNIQUE questions and uses the efficient single-question method.
        for i, user_question in enumerate(user_questions):
            logger.info(f"Processing question {i + 1}/{len(user_questions)}: {user_question[:50]}...")
            question_result = self.run_inference(system_prompt, user_question)
            all_results.append(question_result)

        return BatchSelfConsistencyResult(
            results=all_results,
            questions=user_questions,
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
    """Persist results to disk for later inspection."""
    output_path = Path(output_path)
    run_dir = output_path
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save individual generation files
    for i, (answer, text) in enumerate(zip(result.answers, result.generated_texts)):
        with open(run_dir / f"generation_{i}.txt", "w") as f:
            f.write(f"# System Prompt\n{system_prompt}\n\n")
            f.write(f"# Question\n{user_question}\n\n")
            f.write(f"# Generated Text\n{text}\n\n")
            f.write(f"# Extracted Answer\n{answer}\n")
    
    # Prepare and save summary file
    summary_data = {
        "model": result.config.model_name,
        "num_samples": result.config.num_samples,
        "temperature": result.config.temperature,
        "top_p": result.config.top_p,
        "most_consistent_answer": result.most_consistent_answer,
        "answer_distribution": {
            ans: {
                "count": count,
                "percentage": round((count / len(result.answers)) * 100, 2)
            } for ans, count in sorted(result.distribution.items(), key=lambda item: item[1], reverse=True)
        },
        "all_answers": result.answers,
    }
    
    if correct_answer is not None:
        correct_count = sum(1 for ans in result.answers if ans.strip().upper() == correct_answer.strip().upper())
        summary_data["evaluation"] = {
            "correct_answer": correct_answer,
            "is_most_consistent_correct": result.most_consistent_answer == correct_answer,
            "correct_count": correct_count,
            "accuracy": round(correct_count / len(result.answers), 4)
        }
        
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Saved self-consistency results to {run_dir}")


def run_self_consistency(
    system_prompt: str,
    user_question: Union[str, List[str]],
    config: Optional[SelfConsistencyConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    correct_answer: Optional[Union[str, List[str]]] = None,
) -> Union[SelfConsistencyResult, BatchSelfConsistencyResult]:
    """
    Main entry point to run self-consistency inference.

    Handles both single and batch questions and orchestrates saving results.
    """
    if config is None:
        config = SelfConsistencyConfig()
    inference = create_self_consistency_inference(config)

    is_batch = isinstance(user_question, list)
    questions = user_question if is_batch else [user_question]
    
    answers = None
    if correct_answer:
        answers = correct_answer if isinstance(correct_answer, list) else [correct_answer]
        if len(answers) != len(questions):
            raise ValueError("Number of correct answers must match the number of questions.")

    if not is_batch:
        result = inference.run_inference(system_prompt, questions[0])
        if save_path:
            save_self_consistency_result(result, system_prompt, questions[0], save_path, answers[0] if answers else None)
        return result
    else:
        batch_result = inference.run_batch_inference(system_prompt, questions)
        if save_path:
            # Create a single run directory for the entire batch
            run_dir = Path(save_path) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            for i, (res, q) in enumerate(zip(batch_result.results, batch_result.questions)):
                question_save_path = run_dir / f"question_{i}"
                save_self_consistency_result(res, system_prompt, q, question_save_path, answers[i] if answers else None)
        return batch_result