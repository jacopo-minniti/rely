from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import json
import logging
import re
from pathlib import Path
from collections import Counter

from vllm import LLM, SamplingParams, RequestOutput

# Import helpers from sbs.py
from rely.utils import MATH_SYSTEM_PROMPT, format_prompt, extract_final_answer, normalize_answer


logger = logging.getLogger(__name__)


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency inference."""
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    num_samples: int = 20
    max_new_tokens: int = 500
    temperature: float = 1.0
    top_p: float = 0.95
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


@dataclass
class SelfConsistencyResult:
    """Holds results from self-consistency inference."""
    answers: List[str]
    generated_texts: List[str]
    distribution: Dict[str, int]
    most_consistent_answer: Optional[str]
    config: SelfConsistencyConfig



class SelfConsistencyInference:
    """Efficient self-consistency inference with batch follow-up processing."""

    def __init__(self, config: SelfConsistencyConfig):
        self.config = config
        self.llm: Optional[LLM] = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            logger.info(f"Loading model {self.config.model_name} with vLLM…")
            self.llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                enable_prefix_caching=True,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _generate(self, prompt: str, max_tokens: int, n: int = 1) -> List[str]:
        """Generate text from the model."""
        if self.llm is None:
            raise ValueError("LLM not initialized")

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens,
            n=n,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        
        return [output.text for output in outputs[0].outputs]

    def _get_consistent_samples(self, system_prompt: str, user_question: str) -> List[tuple[str, str]]:
        """
        Generates N samples with single generation only.

        Returns:
            A list of tuples: (answer, full_generated_text).
        """
        base_prompt = format_prompt(user_question, system_prompt)

        # Generate all N samples
        generated_texts = self._generate(base_prompt, self.config.max_new_tokens, n=self.config.num_samples)

        results = []
        for generated_text in generated_texts:
            full_text = base_prompt + generated_text
            answer = extract_final_answer(full_text)
            results.append(("Not found" if answer is None else answer, full_text))

        return results

    def run_inference(self, system_prompt: str, user_question: str) -> SelfConsistencyResult:
        """Run self-consistency for a single question and return aggregated results."""
        logger.info(f"Running self-consistency for '{user_question[:50]}...' with {self.config.num_samples} samples...")
        
        # `results` now contains (answer, text)
        results = self._get_consistent_samples(system_prompt, user_question)

        # Unpack the results into separate lists
        answers = [res[0] for res in results]
        generated_texts = [res[1] for res in results]

        for i, answer in enumerate(answers):
            logger.debug(f"Sample {i + 1}/{self.config.num_samples} ⇒ {answer}")

        # Create distribution based on normalized answers for better grouping
        normalized_distribution: Dict[str, List[str]] = {}
        for ans in answers:
            normalized_ans = normalize_answer(str(ans))
            if normalized_ans not in normalized_distribution:
                normalized_distribution[normalized_ans] = []
            normalized_distribution[normalized_ans].append(ans)

        # Create final distribution with counts and select most frequent original answer for each group
        distribution: Dict[str, int] = {}
        for normalized_ans, original_answers in normalized_distribution.items():
            # For the distribution, use the most common original form of this normalized answer
            original_counts = {}
            for orig_ans in original_answers:
                original_counts[orig_ans] = original_counts.get(orig_ans, 0) + 1
            most_common_original = max(original_counts.items(), key=lambda kv: kv[1])[0]
            distribution[most_common_original] = len(original_answers)

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


def save_self_consistency_result(
    result: SelfConsistencyResult,
    system_prompt: str,
    user_question: str,
    output_path: Union[str, Path],
    correct_answer: Optional[Union[str, List[str]]] = None,
) -> None:
    """Persist results to disk for later inspection."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save individual generation files - name them beam_01.txt, beam_02.txt, etc.
    for i, (answer, text) in enumerate(zip(result.answers, result.generated_texts), 1):
        with open(output_path / f"beam_{i:02d}.txt", "w") as f:
            f.write(text)
    
    # Prepare solutions in the exact same format as SBS
    solutions = []
    for i, (answer, text) in enumerate(zip(result.answers, result.generated_texts)):
        solution_data = {
            "beam_index": i + 1,
            "value": 1.0,  # Self-consistency doesn't have value scores, use uniform value
            "final_answer": answer,
            "depth": 1,  # Self-consistency is single-step, so depth is always 1
            "termination_reason": "answer_found" if answer != "Not found" else "max_tokens",
            "solution_path": text.split("<|im_start|>assistant\n")[-1] if "<|im_start|>assistant\n" in text else text
        }
        solutions.append(solution_data)
    
    # Create summary in exact same format as SBS
    def create_summary(solutions: List[Dict[str, Any]], question: str, ground_truth: Optional[str]) -> Dict[str, Any]:
        ground_truth = normalize_answer(ground_truth) if ground_truth else ""
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
            "total_tokens": 0  # vLLM doesn't easily provide token counts, so we use 0
        }
    
    # Handle correct_answer input (string or list)
    correct_answer_str = ""
    if correct_answer:
        if isinstance(correct_answer, list):
            correct_answer_str = str(correct_answer[0]) if correct_answer else ""
        else:
            correct_answer_str = str(correct_answer)
    
    summary_data = create_summary(solutions, user_question, correct_answer_str)
        
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)  # Use indent=4 to match SBS format

    logger.info(f"Saved self-consistency results to {output_path}")


def run_self_consistency(
    user_question: Union[str, List[str]],
    config: Optional[SelfConsistencyConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    correct_answer: Optional[Union[str, List[str]]] = None,
    system_prompt: str = MATH_SYSTEM_PROMPT,
) -> Union[SelfConsistencyResult, List[SelfConsistencyResult]]:
    if config is None:
        config = SelfConsistencyConfig()
    inference = SelfConsistencyInference(config)

    # Handle single question case
    if isinstance(user_question, str):
        result = inference.run_inference(system_prompt, user_question)
        if save_path:
            save_self_consistency_result(result, system_prompt, user_question, save_path, correct_answer)
        return result
    
    # Handle list of questions case
    results = []
    questions = user_question
    answers = correct_answer if isinstance(correct_answer, list) else [correct_answer] * len(questions)
    
    for i, (question, answer) in enumerate(zip(questions, answers)):
        logger.info(f"Processing question {i+1}/{len(questions)}")
        result = inference.run_inference(system_prompt, question)
        
        if save_path:
            # Create individual directory for each question
            question_save_path = Path(save_path) / f"question_{i}"
            save_self_consistency_result(result, system_prompt, question, question_save_path, answer)
        
        results.append(result)
    
    return results


if __name__ == "__main__":
    from datasets import load_dataset
    logging.basicConfig(level=logging.INFO)

    dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")

    questions, answers = [], []
    for item in dataset:
        questions.append(item["problem"])
        answers.append(item["answer"])

    run_self_consistency(
        user_question=questions,
        correct_answer=answers,
        config=SelfConsistencyConfig(
            num_samples=4,
            max_new_tokens=6000,
            temperature=1.0,
            top_p=0.95,
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
        ),
        save_path="results/self_consistency_max_4",
    )