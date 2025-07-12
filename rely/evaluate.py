import re
from typing import Union
from rely.inference import VLLMInference, APIInference
from rely.utils import check_answer
from tqdm import tqdm


class Evaluator:
    """
    Handles the evaluation of reasoning steps by generating multiple completions
    and comparing them against a ground truth answer.
    """

    def __init__(self, engine: Union[VLLMInference, APIInference]):
        """
        Initializes the Evaluator.

        Args:
            engine (Union[VLLMInference, APIInference]): An instance of the VLLMInference or APIInference class
                                          for generating text.
        """
        self.engine = engine

    def evaluate(self, problem, step_prompt, ground_truth_answer, max_answer_tokens, max_reasoning_tokens, n, hard):
        """
        Calculates the quality score of a given reasoning step.

        Args:
            problem (str): The initial problem statement.
            step_prompt (str): The CoT generated so far.
            ground_truth_answer (str): The correct final answer.
            max_answer_tokens (int): The maximum number of tokens for the answer.
            max_reasoning_tokens (int): The maximum number of tokens for reasoning.
            n (int): The number of completions to generate.
            hard (bool): If True, returns 0 or 1. If False, returns a soft score [0, 1].

        Returns:
            float: The calculated score for the step.
        """
        success_count = 0

        generated_answers = self.engine.generate_from_step(
            problem,
            step_prompt,
            temperature=1.0,  # Using a single temperature for the batch
            max_answer_tokens=max_answer_tokens,
            max_thinking_tokens=max_reasoning_tokens,
            n=n
        )

        if not generated_answers:
            print("⚠️ Answer not generated. Automatically giving score of 0.")
            return 0.0

        for generated_answer in tqdm(generated_answers, total=n, desc="Evaluating generations"):
            if generated_answer and check_answer(generated_answer, ground_truth_answer):
                success_count += 1
                if hard:
                    # Early exit for hard scoring
                    return 1

        if hard:
            return 0

        return success_count / n
