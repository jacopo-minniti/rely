from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import json
import logging
import re
import os
import argparse
import traceback
from pathlib import Path
from collections import Counter

from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Import helpers from sbs.py
from rely.utils import MATH_SYSTEM_PROMPT, format_prompt, extract_final_answer, normalize_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        self.tokenizer: Optional[AutoTokenizer] = None
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
            # Initialize tokenizer for token counting
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
            logger.info("Model and tokenizer loaded successfully.")
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

    def _count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        if self.tokenizer is None:
            # Fallback to rough word count if tokenizer is not available
            return len(text.split())
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            # Fallback to word count if tokenization fails
            return len(text.split())

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
            # Ensure we handle None and empty strings properly
            final_answer = answer if (answer is not None and answer.strip()) else "Not found"
            results.append((final_answer, full_text))

        return results

    def run_inference(self, 
                     user_question: str, 
                     ground_truth: Optional[str] = None, 
                     base_path: Optional[str] = None,
                     system_prompt: str = MATH_SYSTEM_PROMPT) -> Dict[str, Any]:
        """Run self-consistency for a single question and return aggregated results in SBS format."""
        logger.info(f"Running self-consistency for '{user_question[:50]}...' with {self.config.num_samples} samples...")
        
        # `results` now contains (answer, text)
        results = self._get_consistent_samples(system_prompt, user_question)

        # Unpack the results into separate lists
        answers = [res[0] for res in results]
        generated_texts = [res[1] for res in results]

        for i, answer in enumerate(answers):
            logger.debug(f"Sample {i + 1}/{self.config.num_samples} ⇒ {answer}")

        # Prepare solutions in the exact same format as SBS
        solutions = []
        total_tokens = 0
        for i, (answer, text) in enumerate(zip(answers, generated_texts)):
            # Extract just the assistant response part for solution_path (like SBS does)
            solution_path = text.split("<|im_start|>assistant\n")[-1] if "<|im_start|>assistant\n" in text else text
            # Count tokens for this solution
            solution_tokens = self._count_tokens(text)
            total_tokens += solution_tokens
            
            termination_reason = "answer_found" if answer != "Not found" else "max_tokens"
            solution_data = {
                "beam_index": i + 1,
                "value": 1.0,  # Self-consistency doesn't have value scores, use uniform value
                "final_answer": answer,
                "depth": 1,  # Self-consistency is single-step, so depth is always 1
                "termination_reason": termination_reason,
                "solution_path": solution_path
            }
            solutions.append(solution_data)

        # Save results if base_path is provided (following SBS pattern)
        saved_files = []
        if base_path:
            saved_files = save_self_consistency_result(solutions, base_path, user_question, ground_truth, total_tokens)

        return {"question": user_question, "ground_truth": ground_truth, "solutions": solutions, "total_tokens": total_tokens}


def create_self_consistency_summary(solutions: List[Dict[str, Any]], question: str, ground_truth: Optional[str], total_tokens: int = 0) -> Dict[str, Any]:
    """Create summary in exact same format as SBS."""
    normalized_ground_truth = normalize_answer(ground_truth) if ground_truth else ""
    
    # Filter out "Not found" and empty answers, then normalize
    valid_answers = [s['final_answer'] for s in solutions 
                    if s['final_answer'] and s['final_answer'] != "Not found" and s['final_answer'].strip()]
    normalized_answers = [normalize_answer(ans) for ans in valid_answers]
    # Remove any answers that became empty after normalization
    normalized_answers = [ans for ans in normalized_answers if ans.strip()]
    
    majority_vote = Counter(normalized_answers).most_common(1)[0][0] if normalized_answers else "N/A"
    accuracy = "N/A"
    if normalized_ground_truth and normalized_answers:
        correct_answers = sum(1 for ans in normalized_answers if ans == normalized_ground_truth)
        accuracy = f"{correct_answers / len(normalized_answers):.2%}" if normalized_answers else "0.00%"
    
    return {
        "question": question,
        "ground_truth": ground_truth,  # Keep original for compatibility
        "majority_vote": majority_vote,
        "accuracy": accuracy,
        "solutions": solutions,
        "total_tokens": total_tokens
    }


def save_self_consistency_result(final_solutions: List[Dict[str, Any]], base_path: str, question: str, ground_truth: Optional[str], total_tokens: int = 0) -> List[str]:
    """Save results in exact same format as SBS."""
    import os
    saved_files = []
    os.makedirs(base_path, exist_ok=True)

    summary_path = os.path.join(base_path, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(create_self_consistency_summary(final_solutions, question, ground_truth, total_tokens), f, indent=4)
    saved_files.append(summary_path)
    
    return saved_files


def run_self_consistency(
    user_questions: List[str],
    config: Optional[SelfConsistencyConfig] = None,
    ground_truths: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    system_prompt: str = MATH_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """Run self-consistency for a list of questions."""
    if config is None:
        config = SelfConsistencyConfig()

    inference = SelfConsistencyInference(config)
    results = []

    if ground_truths is None:
        ground_truths = [None] * len(user_questions)

    for i, (question, ground_truth) in enumerate(tqdm(zip(user_questions, ground_truths), total=len(user_questions), desc="Running Self-Consistency")):
        base_path = os.path.join(output_dir, f"q_{i:04d}") if output_dir else None
        result = inference.run_inference(
            user_question=question,
            ground_truth=ground_truth,
            base_path=base_path,
            system_prompt=system_prompt,
        )
        results.append(result)
    return results


def run_self_consistency_on_dataset(args):
    """Main function that follows the exact SBS pattern for dataset processing."""
    # Load and prepare dataset exactly like SBS
    ds = load_dataset(args.dataset, split='test')
    ds = ds.shuffle(seed=42)
    dataset = []
    for i, item in enumerate(ds):
        item_dict = dict(item)
        item_dict['original_index'] = i
        dataset.append(item_dict)

    if args.idx_start is not None and args.idx_end is not None:
        dataset = dataset[args.idx_start:args.idx_end]
        logger.info(f"Processing dataset slice from {args.idx_start} to {args.idx_end}. Total items: {len(dataset)}")

    logger.info(f"Starting self-consistency inference on {len(dataset)} questions.")
    
    # Initialize self-consistency with config
    config = SelfConsistencyConfig(
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    inference = SelfConsistencyInference(config)

    for item in tqdm(dataset, desc="Processing"):
        original_index = None
        output_path = None
        try:
            question = item['problem']
            ground_truth = item.get('answer')
            original_index = item['original_index']
            output_path = os.path.join(args.output_dir, f"q_{original_index:04d}")
            
            if os.path.exists(os.path.join(output_path, "summary.json")):
                logger.info(f"Skipping item {original_index} as it is already completed.")
                continue
                
            inference.run_inference(
                user_question=question,
                ground_truth=ground_truth,
                base_path=output_path
            )
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"FATAL ERROR processing item {original_index}: {e}\nTRACEBACK:\n{error_traceback}")
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                with open(os.path.join(output_path, "error.log"), 'w') as f:
                    f.write(f"Error on question index {original_index}:\n{str(e)}\n\n{error_traceback}")

    logger.info("Self-consistency inference completed.")


def main():
    parser = argparse.ArgumentParser(description="Run Self-Consistency inference on dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name for vLLM.")
    
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples for self-consistency.")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM.")
    
    parser.add_argument("--idx_start", type=int, default=None, help="Start index of the dataset split.")
    parser.add_argument("--idx_end", type=int, default=None, help="End index of the dataset split.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_self_consistency_on_dataset(args)


if __name__ == "__main__":
    main()