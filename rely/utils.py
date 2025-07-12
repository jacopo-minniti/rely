import re
import string
from datasets import load_dataset
import os
from openai import OpenAI
from pydantic import BaseModel, Field


def parse_decomposed_cot(cot_string):
    """Parses the XML-like CoT string into a list of steps."""
    # Find all content within <step>...</step> tags
    steps = re.findall(r"<step>(.*?)</step>", cot_string, re.DOTALL)
    # Re-wrap them in tags to preserve the original format for prompts
    return [f"<step>\n{step.strip()}\n</step>" for step in steps]

def load_data(dataset_name: str, subset="default", split="train"):
    return load_dataset(dataset_name, subset, split=split)


def check_answer(model_answer, ground_truth_answer) -> bool:
    """
    Checks if the model's answer is correct against a ground truth.
    It normalizes both answers by stripping whitespace, converting to lowercase,
    and removing punctuation. It also extracts answers from LaTeX-style \\boxed{} expressions.
    It checks for exact match or if the ground truth is a substring of the model's answer.
    """
    # Normalize ground truth
    norm_ground_truth = str(ground_truth_answer).strip().lower()
    translator = str.maketrans('', '', string.punctuation)
    norm_ground_truth = norm_ground_truth.translate(translator)

    # Normalize model answer
    norm_model_ans = str(model_answer).strip().lower()

    # Check for \boxed{...} and extract the content
    boxed_match = re.search(r'\\boxed{(.*?)}', norm_model_ans)
    if boxed_match:
        norm_model_ans = boxed_match.group(1).strip()

    # Remove punctuation for consistent comparison
    norm_model_ans = norm_model_ans.translate(translator)
    
    # print(f"{norm_ground_truth}\t|\t{norm_model_ans}")
    # Check for exact match or if ground truth is in the model's answer
    return norm_model_ans == norm_ground_truth or norm_ground_truth in norm_model_ans

def convert_to_discrete_cot(continuous_cot):
    """Converts a newline-separated CoT to a discrete <step>-based one."""
    # Split by one or more newlines
    steps = re.split(r'\n+', continuous_cot.strip())
    discrete_cot = ""
    for step in steps:
        if step:
            # Remove any existing <step> or </step> to avoid duplication
            step_clean = step.replace("<step>", "").replace("</step>", "").strip()
            if step_clean:
                discrete_cot += f"<step>\n{step_clean}\n</step>\n"
    return discrete_cot.strip()



def clean_solution(solution_text: str) -> str:
    """
    Uses an OpenAI model to extract only the final answer from a solution text.
    """

    MODEL_NAME = "gpt-4.1-mini"
    SYSTEM_PROMPT = """You are an expert at extracting the final answer from a piece of text containing a solution to a problem. Your task is to identify and return only the final answer, removing any preceding explanations, reasoning, or conversational text. For example, if the input is 'The answer is 25 because...', you should only output '25'. If the input is a proof, output the final statement. If the answer is a letter, like '(a)', just output 'a'."""

    # --- Pydantic Model for Structured Output ---
    class CleanedSolution(BaseModel):
        cleaned_answer: str = Field(description="The final, pure answer with all explanations and reasoning removed. For example, if the solution is 'The answer is 25 because...', this should be '25'.")

    # --- OpenAI API Client ---
    # Ensure the OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI()

    if not solution_text or not isinstance(solution_text, str):
        return ""
    try:
        # This appears to be a placeholder for a structured output feature.
        # The standard OpenAI library uses `client.chat.completions.create`
        # with `response_model` for structured output with some libraries like `instructor`.
        # Assuming `client.responses.parse` is a custom or future implementation.
        # For now, we will replicate the logic as provided.
        # If using a library like `instructor`, the call would be different.
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": solution_text},
            ],
            # The following line is conceptual. `instructor` uses `response_model`.
            # text_format=CleanedSolution, # This is not a standard parameter.
        )
        # A more standard way to get a simple text response:
        cleaned_answer = response.choices[0].message.content
        return cleaned_answer.strip()

    except Exception as e:
        print(f"An error occurred while cleaning solution: {e}")
        return "" # Return empty string on failure


