from typing import Tuple, Optional
import torch
from transformers import AutoTokenizer

# Default system prompt for MMLU-Pro style questions
MATH_SYSTEM_PROMPT = """The following are questions about mathematics. Think step by step and provide your answer in the format '\\boxed{}' with inside your final answer."""


MMLU_SYSTEM_PROMPT = """The following are multiple choice questions (with answers) about science. Think step by step and then finish your answer with 'The correct answer is (X)' where X is the correct letter choice.

EXAMPLE

Question: The quantum efficiency of a photon detector is 0.1. If 100 photons are sent into the detector, one after the other, the detector will detect photons
Options:
(A) an average of 10 times, with an rms deviation of about 4
(B) an average of 10 times, with an rms deviation of about 3
(C) an average of 10 times, with an rms deviation of about 1
(D) an average of 10 times, with an rms deviation of about 0.1

## Your Example Answer
[...Explanation...] The correct answer is (B)."""

def get_last_step_pos(text: str, tokenizer: AutoTokenizer) -> Tuple[int, str]:
    """
    Returns the token position after the last '\n\n' in the text.
    If not present, appends '\n\n' and returns the position after that.
    
    Args:
        text: Input text
        tokenizer: Tokenizer to use for tokenization
    
    Returns:
        Tuple of (token_position, processed_text)
    """
    if '\n\n' not in text:
        text = text + '\n\n'
    char_idx = text.rfind('\n\n') + 2
    prefix = text[:char_idx]
    prefix_token_ids = tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()
    return len(prefix_token_ids) - 1, text


def count_tokens_after_marker(text: str, tokenizer: AutoTokenizer, marker: str = "<|im_start|>assistant") -> int:
    """
    Count tokens after a specific marker in the text.
    
    Args:
        text: Input text
        tokenizer: Tokenizer to use
        marker: Marker to search for
    
    Returns:
        Number of tokens after the marker
    """
    marker_idx = text.find(marker)
    if marker_idx == -1:
        return len(tokenizer(text, return_tensors="pt").input_ids[0])
    after_marker_text = text[marker_idx + len(marker):]
    return len(tokenizer(after_marker_text, return_tensors="pt").input_ids[0])


def ensure_think_ending(text: str) -> str:
    """
    Ensure the text ends with the think closing tag.
    
    Args:
        text: Input text
    
    Returns:
        Text with proper think ending
    """
    if not text.strip().endswith("</think>") and not text.strip().endswith("</think>\n"):
        return text.rstrip() + "\n</think>\n## Final Answer\n"
    return text


def format_prompt(question: str, system_prompt: str = MMLU_SYSTEM_PROMPT, add_think: bool = False, cot="") -> str:
    """
    Format the question into a prompt for the model.

    Args:
        question: The question text to format.
        system_prompt: The system prompt to prepend to the question.

    Returns:
        A formatted prompt string.
    """

    user_question = question.strip()
    if add_think:
        user_question += " \\think"
        cot = "<think>\n" + cot.lstrip()

    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_question}<|im_end|>\n<|im_start|>assistant\n{cot}"


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