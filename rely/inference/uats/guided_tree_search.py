import logging
import math
import uuid
from typing import List

import torch
from openai import OpenAI
from transformers import AutoTokenizer

from .config import Branch, UATSConfig
from rely.utils.text_utils import (
    count_tokens_after_marker,
    format_prompt,
    MATH_SYSTEM_PROMPT,
    extract_final_answer,
)

logger = logging.getLogger(__name__)

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"


class GuidedTreeSearch:
    """
    Guided Tree Search implementation for UATS (Uncertainty Aware Tree Search).
    """

    @classmethod
    def uncertainty_to_branches(cls, score: float, uncertainty_threshold: float, max_branches: int) -> int:
        """Convert an uncertainty score to the number of branches to explore."""
        if uncertainty_threshold:
            branches = max_branches if score >= uncertainty_threshold else 1
        else:
            branches = round(math.exp(score))
        return min(branches, max_branches)

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: UATSConfig,
        uncertainty_task_queue,
        uncertainty_result_queue,
        value_task_queue,
        value_result_queue,
        worker_rank: int,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        self.uncertainty_task_queue = uncertainty_task_queue
        self.uncertainty_result_queue = uncertainty_result_queue
        self.value_task_queue = value_task_queue
        self.value_result_queue = value_result_queue
        self.worker_rank = worker_rank

    def _get_score_from_server(self, queue_task, queue_result, text: str) -> float:
        """Send text to a model server and get the aggregated score."""
        request_id = str(uuid.uuid4())
        queue_task.put((request_id, self.worker_rank, text))
        while True:
            response_id, result = queue_result.get()
            if response_id == request_id:
                return result

    def _generate_step(self, ids: torch.Tensor, num_completions: int = 1) -> List[tuple[str, torch.Tensor]]:
        """Generate one or more reasoning steps."""
        prompt_text = self.tokenizer.decode(ids[0], skip_special_tokens=True)
        results = []
        try:
            completion = self.client.completions.create(
                model=self.config.model_name,
                prompt=prompt_text,
                max_tokens=self.config.max_step_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=["\n\n"],
                n=num_completions,
            )
            for choice in completion.choices:
                step_text = choice.text
                new_tokens = self.tokenizer.encode(step_text, add_special_tokens=False, return_tensors='pt')[0]
                results.append((step_text, new_tokens.to(ids.device)))
            return results
        except Exception as e:
            logger.error(f"Error during API call: {e}")
            return [("", torch.tensor([], dtype=torch.long, device=ids.device))]

    def _generate_final_answer(self, branch: Branch) -> str:
        """Generate a final answer for a completed branch."""
        prompt_addition = "\n\n## Final Answer\n"
        text = branch.text.strip() + prompt_addition
        try:
            completion = self.client.completions.create(
                model=self.config.model_name,
                prompt=text,
                max_tokens=self.config.max_step_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                n=1,
            )
            return prompt_addition + completion.choices[0].text
        except Exception as e:
            logger.error(f"Error during final answer API call: {e}")
            return ""

    def search(self, user_question: str, system_prompt: str = MATH_SYSTEM_PROMPT) -> tuple[list[Branch], list[Branch], int]:
        """Perform guided tree search."""
        prompt = format_prompt(user_question, system_prompt)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        initial_steps = self._generate_step(prompt_ids)
        if not initial_steps:
            return [], [], 0
        first_step_text, new_tokens = initial_steps[0]
        tokens_used = len(new_tokens)

        first_ids = torch.cat([prompt_ids[0], new_tokens]).long().unsqueeze(0)
        first_node_text = prompt + first_step_text
        total_tokens = count_tokens_after_marker(first_node_text, self.tokenizer)

        value_score = self._get_score_from_server(self.value_task_queue, self.value_result_queue, first_node_text)

        branch_id_counter = 0
        all_branches = []
        root_branch = Branch(
            text=first_node_text, ids=first_ids.cpu(), step_count=1, score=value_score,
            uncertainty=None, value=value_score, total_tokens=total_tokens,
            id=branch_id_counter, parent_id=None
        )
        all_branches.append(root_branch)
        branch_id_counter += 1

        # Beam contains non-finished branches to expand
        beam = [root_branch]

        while tokens_used < self.config.budget:
            if not beam:
                logger.info("No more branches to expand. Stopping search.")
                break

            all_new_candidates = []
            budget_exceeded = False
            
            # Expand branches in the current beam
            for branch in beam:
                uncertainty_score = self._get_score_from_server(self.uncertainty_task_queue, self.uncertainty_result_queue, branch.text)
                num_branches_to_gen = self.uncertainty_to_branches(uncertainty_score, self.config.uncertainty_threshold, self.config.max_branches)
                
                generated_steps = self._generate_step(branch.ids.to(self.device), num_completions=num_branches_to_gen)

                for step_text, new_tokens in generated_steps:
                    tokens_used += len(new_tokens)
                    if tokens_used >= self.config.budget:
                        budget_exceeded = True

                    new_text = branch.text + "\n\n" + step_text
                    value_score = self._get_score_from_server(self.value_task_queue, self.value_result_queue, new_text)
                    total_tokens = count_tokens_after_marker(new_text, self.tokenizer)

                    candidate = Branch(
                        text=new_text, ids=torch.cat([branch.ids[0].cpu(), new_tokens.cpu()]).long().unsqueeze(0),
                        step_count=branch.step_count + 1, score=value_score, uncertainty=uncertainty_score,
                        value=value_score, total_tokens=total_tokens, id=branch_id_counter, parent_id=branch.id
                    )
                    
                    if final_answer := extract_final_answer(candidate.text):
                        candidate.final_answer = final_answer

                    all_new_candidates.append(candidate)
                    branch_id_counter += 1
                    
                    if budget_exceeded:
                        break
                if budget_exceeded:
                    break
            
            all_branches.extend(all_new_candidates)

            if budget_exceeded:
                logger.info("Token budget reached. Stopping search.")
                break
            
            if not all_new_candidates:
                logger.info("No new candidates generated. Stopping search.")
                break

            # Sort all branches by value to determine the top branches
            all_branches.sort(key=lambda b: b.value, reverse=True)
            
            # These are the overall top branches
            top_branches = all_branches[:self.config.beam_width]
            
            # Check if all top branches are finished
            if len(top_branches) == self.config.beam_width and all(b.final_answer is not None for b in top_branches):
                logger.info("All top branches have final answers. Stopping search.")
                break

            # The new beam is the top non-finished branches from the entire pool
            non_finished_branches = [b for b in all_branches if b.final_answer is None]
            beam = non_finished_branches[:self.config.beam_width]

        # Final selection of branches
        all_branches.sort(key=lambda b: b.value, reverse=True)
        final_branches = all_branches[:self.config.beam_width]

        # Ensure final branches have a final answer
        for branch in final_branches:
            if not branch.final_answer:
                generated_text = self._generate_final_answer(branch)
                branch.text += generated_text
                branch.final_answer = extract_final_answer(branch.text)
            
        return final_branches, all_branches, tokens_used

