import os
import logging
import torch
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass
from pathlib import Path
import re

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from rely.utils.text_utils import (
    format_system_prompt,
    ensure_think_ending,
    MMLU_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)


@dataclass
class BudgetForcingConfig:
    """Configuration for Budget Forcing inference."""
    model_name: str = "Qwen/Qwen3-8B"
    mode: Literal["num_ignores", "max_budget_tokens"] = "num_ignores"
    num_ignores: int = 3  # Only used when mode == "num_ignores"
    max_budget_tokens: int = 1000  # Only used when mode == "max_budget_tokens"
    wait_string: str = "Wait, let me rethink."
    end_think_token: str = "</think>"
    max_new_tokens: int = 500
    temperature: float = 1.0
    top_p: float = 0.95
    device: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


@dataclass
class BudgetForcingResult:
    """Result from budget forcing inference."""
    final_text: str
    total_tokens: int
    num_replacements: int
    mode: str
    config: BudgetForcingConfig


class BudgetForcingInference:
    """
    Budget Forcing inference implementation.
    
    This class implements a test-time intervention that replaces the end-of-thinking
    token with a "wait" string to force the model to continue reasoning.
    """
    
    def __init__(self, config: BudgetForcingConfig):
        self.config = config
        self.llm = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the vLLM model and tokenizer."""
        try:
            logger.info(f"Loading model {self.config.model_name} with vLLM...")
            self.llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True
            )
            # Get tokenizer from the model
            self.tokenizer = self.llm.get_tokenizer()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return len(self.tokenizer.encode(text))
    
    def _replace_end_think_token(self, text: str) -> str:
        """Replace the end-of-thinking token with the wait string."""
        # Remove the end think token and add the wait string
        text = text.replace(self.config.end_think_token, "")
        text = text.rstrip() + "\n" + self.config.wait_string + "\n"
        return text
    
    def _generate_with_vllm(self, prompt: str) -> str:
        """Generate text using vLLM."""
        if self.llm is None:
            raise ValueError("LLM not initialized")
        
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def _inference_with_num_ignores(self, system_prompt: str, user_question: str) -> BudgetForcingResult:
        """Run inference with fixed number of ignores."""
        prompt = format_system_prompt(system_prompt, user_question)
        current_text = prompt
        num_replacements = 0
        
        for ignore_count in range(self.config.num_ignores + 1):  # +1 for the final generation
            # Generate text
            generated_text = self._generate_with_vllm(current_text)
            current_text += generated_text
            
            # Check if we should continue (if we haven't reached the limit yet)
            if ignore_count < self.config.num_ignores and self.config.end_think_token in generated_text:
                # Replace the end think token and continue
                current_text = self._replace_end_think_token(current_text)
                num_replacements += 1
                logger.info(f"Replaced end think token (ignore {ignore_count + 1}/{self.config.num_ignores})")
            else:
                # Either we've reached the limit or no end think token found
                break
        
        total_tokens = self._count_tokens(current_text)
        
        return BudgetForcingResult(
            final_text=current_text,
            total_tokens=total_tokens,
            num_replacements=num_replacements,
            mode="num_ignores",
            config=self.config
        )
    
    def _inference_with_max_budget(self, system_prompt: str, user_question: str) -> BudgetForcingResult:
        """Run inference with maximum token budget."""
        prompt = format_system_prompt(system_prompt, user_question)
        current_text = prompt
        num_replacements = 0
        
        while True:
            # Generate text
            generated_text = self._generate_with_vllm(current_text)
            current_text += generated_text
            
            # Count tokens so far
            total_tokens = self._count_tokens(current_text)
            
            # Check if we should continue
            if (self.config.end_think_token in generated_text and 
                total_tokens < self.config.max_budget_tokens):
                # Replace the end think token and continue
                current_text = self._replace_end_think_token(current_text)
                num_replacements += 1
                logger.info(f"Replaced end think token (tokens: {total_tokens}/{self.config.max_budget_tokens})")
            else:
                # Either no end think token found or budget reached
                break
        
        final_total_tokens = self._count_tokens(current_text)
        
        return BudgetForcingResult(
            final_text=current_text,
            total_tokens=final_total_tokens,
            num_replacements=num_replacements,
            mode="max_budget_tokens",
            config=self.config
        )
    
    def run_inference(self, system_prompt: str, user_question: str) -> BudgetForcingResult:
        """
        Run budget forcing inference.
        
        Args:
            system_prompt: System prompt to use
            user_question: User question to answer
            
        Returns:
            BudgetForcingResult containing the final text and metadata
        """
        logger.info(f"Starting budget forcing inference with mode: {self.config.mode}")
        
        if self.config.mode == "num_ignores":
            return self._inference_with_num_ignores(system_prompt, user_question)
        elif self.config.mode == "max_budget_tokens":
            return self._inference_with_max_budget(system_prompt, user_question)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")


def create_budget_forcing_inference(config: BudgetForcingConfig) -> BudgetForcingInference:
    """
    Create a BudgetForcingInference instance.
    
    Args:
        config: Configuration for the budget forcing inference
        
    Returns:
        BudgetForcingInference instance
    """
    return BudgetForcingInference(config)


def run_budget_forcing_inference(
    system_prompt: str,
    user_question: str,
    config: Optional[BudgetForcingConfig] = None
) -> BudgetForcingResult:
    """
    Run budget forcing inference with the given configuration.
    
    Args:
        system_prompt: System prompt to use
        user_question: User question to answer
        config: Configuration for the inference (uses default if None)
        
    Returns:
        BudgetForcingResult containing the final text and metadata
    """
    if config is None:
        config = BudgetForcingConfig()
    
    inference = create_budget_forcing_inference(config)
    return inference.run_inference(system_prompt, user_question)


def save_budget_forcing_result(
    result: BudgetForcingResult,
    output_path: Union[str, Path]
) -> None:
    """
    Save the budget forcing result to a file.
    
    Args:
        result: BudgetForcingResult to save
        output_path: Path to save the result
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"# Budget Forcing Result\n")
        f.write(f"Mode: {result.mode}\n")
        f.write(f"Total Tokens: {result.total_tokens}\n")
        f.write(f"Number of Replacements: {result.num_replacements}\n")
        f.write(f"Wait String: {result.config.wait_string}\n")
        f.write(f"End Think Token: {result.config.end_think_token}\n")
        f.write(f"\n# Generated Text\n")
        f.write(result.final_text)
    
    logger.info(f"Saved budget forcing result to {output_path}")


# Example usage and testing functions
def example_usage():
    """Example usage of the budget forcing inference."""
    
    # Example 1: Using num_ignores mode
    config_num_ignores = BudgetForcingConfig(
        model_name="Qwen/Qwen3-8B",
        mode="num_ignores",
        num_ignores=2,
        wait_string="Wait, let me think about this more carefully."
    )
    
    system_prompt = MMLU_SYSTEM_PROMPT
    user_question = "What is the capital of France?"
    
    result = run_budget_forcing_inference(system_prompt, user_question, config_num_ignores)
    print(f"Result with num_ignores mode:")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Replacements: {result.num_replacements}")
    print(f"Final text: {result.final_text}")
    
    # Example 2: Using max_budget_tokens mode
    config_max_budget = BudgetForcingConfig(
        model_name="Qwen/Qwen3-8B",
        mode="max_budget_tokens",
        max_budget_tokens=800,
        wait_string="Let me reconsider this step by step."
    )
    
    result2 = run_budget_forcing_inference(system_prompt, user_question, config_max_budget)
    print(f"\nResult with max_budget_tokens mode:")
    print(f"Total tokens: {result2.total_tokens}")
    print(f"Replacements: {result2.num_replacements}")
    print(f"Final text: {result2.final_text}")


if __name__ == "__main__":
    example_usage()