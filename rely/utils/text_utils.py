import re
from typing import Tuple, Optional
from transformers import AutoTokenizer

# Default system prompt for MMLU-Pro style questions
MATH_SYSTEM_PROMPT = """The following are questions about mathematics. Think step by step and provide your answer in the format '\\boxed{}' with inside your final answer. The final answers should either be a number (in digits) or a latex expression."""


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

prompt_pattern = re.compile(
    r"<\|im_start\|>system\n(.*?)"
    r"<\|im_end\|>\n<\|im_start\|>user\n(.*?)"
    r"<\|im_end\|>\n<\|im_start\|>assistant\n(.*)",
    re.DOTALL
)

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


def format_prompt(question: str, system_prompt: str = MATH_SYSTEM_PROMPT, add_think: bool = False, cot="") -> str:
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
    
    # 3. Find the matching closing brace by counting nested braces
    brace_count = 1
    pos = content_start_pos
    
    while pos < len(text) and brace_count > 0:
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1
    
    # 4. If we didn't find a matching closing brace, return None
    if brace_count > 0:
        return None
    
    # 5. Extract the content between the braces
    content_end_pos = pos - 1  # pos is one past the closing brace
    content = text[content_start_pos:content_end_pos]
    
    return content.strip()


import re

def normalize_answer(answer: str) -> str:
    """
    Normalizes a mathematical answer for robust comparison using safe string methods.
    This version does NOT evaluate mathematical expressions.
    """
    if not answer:
        return ""

    # 1. Initial text cleaning and equation handling
    normalized = str(answer).lower().strip()
    if '=' in normalized:
        normalized = normalized.split('=')[-1].strip()

    # 2. Remove LaTeX delimiters first
    normalized = re.sub(r'^(\$|\\\[|\\\(|\\\$)|(\$|\\\]|\\\)|\$$)$', '', normalized).strip()

    # 3. Handle LaTeX commands non-destructively
    # Remove sizing commands
    normalized = re.sub(r'\\left|\\right', '', normalized)
    # Keep the content of text-styling commands (FIXED)
    normalized = re.sub(r'\\(text|mathrm|mathbf|boldsymbol)\s*\{([^}]*)\}', r'\2', normalized)
    # Convert fractions, adding parentheses for safety
    normalized = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', normalized)
    # Convert LaTeX percent symbol
    normalized = re.sub(r'\\%', '%', normalized)

    # 4. Handle numerical conversions
    # Convert percentages to decimals
    if '%' in normalized:
        normalized = re.sub(r'(\d*\.?\d+)\s*%', lambda m: str(float(m.group(1)) / 100.0), normalized)
    # Remove thousands separators
    normalized = re.sub(r',(?=\d)', '', normalized)

    # 5. Standardize spacing (SAFER METHOD)
    # Replace multiple whitespace characters with a single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    # Remove spaces around common operators to standardize expressions like "2 + 1" to "2+1"
    normalized = re.sub(r'\s*([+\-*/=()^])\s*', r'\1', normalized)
    
    # 6. Final cleanup for numeric answers
    # Remove trailing zeros from decimals
    if '.' in normalized:
        normalized = normalized.rstrip('0').rstrip('.')
        if normalized == "": # Handles cases like "0.0" -> ""
            normalized = "0"
            
    return normalized