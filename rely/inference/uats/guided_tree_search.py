import logging
import math
import uuid
from typing import List, Optional, Union

import torch
from openai import OpenAI
from transformers import AutoTokenizer, StopStringCriteria

from .config import Branch, UATSConfig
from rely.utils.text_utils import (
    count_tokens_after_marker,
    format_prompt,
    MATH_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"


class GuidedTreeSearch:
    """
    Guided Tree Search implementation for UATS (Uncertainty Aware Tree Search).
    """

    @classmethod
    def uncertainty_to_branches(cls, score: float, uncertainty_threshold: Union[float, None]) -> int:
        """Convert an uncertainty score (expected in [0, 1]) to the number of branches that should be explored."""
        branches = 1
        # classification setting
        if uncertainty_threshold:
            branches = 2 if score >= uncertainty_threshold else 1
        # regression 
        else:
            # perplexity
            branches = round(math.exp(score))

        return branches

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: UATSConfig,
        uncertainty_task_queue,
        uncertainty_result_queue,
        value_task_queue,
        value_result_queue,
        worker_rank: int,
        question: str = "",
    ):
        # REMOVED: model, uncertainty_model, value_model
        # self.model = model
        self.tokenizer = tokenizer
        # self.uncertainty_model = uncertainty_model
        # self.value_model = value_model
        self.config = config
        self.question = question
        self.device = "cuda:0"

        # NEW: Add client for vLLM server
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        
        # NEW: Queues for communication with model servers
        self.uncertainty_task_queue = uncertainty_task_queue
        self.uncertainty_result_queue = uncertainty_result_queue
        self.value_task_queue = value_task_queue
        self.value_result_queue = value_result_queue
        self.worker_rank = worker_rank
    
    
    def _get_from_server(self, queue_task, queue_result, payload):
        """Generic function to send a task to a server and get the result."""
        request_id = str(uuid.uuid4())
        queue_task.put((request_id, self.worker_rank) + payload)
        while True:
            response_id, result = queue_result.get()
            if response_id == request_id:
                return result

    def _value_model_score(
        self,
        text: Optional[str] = None,
        ids: Optional[torch.Tensor] = None,
    ) -> tuple[float, torch.Tensor]:
        """Get value score for a given text using the autoregressive value model.

        Passing ``ids`` avoids the costly re-tokenisation of the full prompt at
        every step.  Exactly one of ``text`` or ``ids`` must be provided.
        """

        if ids is None:
            if text is None:
                raise ValueError("Either `text` or `ids` must be provided to _value_model_score")

            # Ensure the text ends with the step delimiter so that we have clean step boundaries
            if not text.endswith("\n\n"):
                text = text + "\n\n"
            
            # Tokenize for returning ids
            inputs = self.tokenizer(text, return_tensors="pt")  # type: ignore[call-arg]
            ids_tensor = inputs.input_ids.to(self.device)
            
            # Extract the reasoning part after the prompt for value model evaluation
            reasoning_text = self._extract_reasoning_from_full_text(text)
        else:
            ids_tensor = ids if ids.dim() == 2 else ids.unsqueeze(0)
            full_text = self.tokenizer.decode(ids_tensor[0], skip_special_tokens=True)
            reasoning_text = self._extract_reasoning_from_full_text(full_text)

        # Get value score from the value model server
        value_score = self._get_from_server(
            self.value_task_queue,
            self.value_result_queue,
            (self.question, reasoning_text)
        )
        return value_score, ids_tensor.cpu()

    
    def _extract_reasoning_from_full_text(self, full_text: str) -> str:
        """Extract only the reasoning part from the full prompt text."""
        # Look for the assistant's response start marker
        if "<|im_start|>assistant\n" in full_text:
            reasoning_part = full_text.split("<|im_start|>assistant\n", 1)[1]
            # Remove any trailing end markers
            reasoning_part = reasoning_part.replace("<|im_end|>", "").strip()
            return reasoning_part
        
        # Fallback: return the whole text
        return full_text

    
    def _generate_step(
        self, ids: torch.Tensor
    ) -> tuple[str, torch.Tensor]:
        # MODIFIED: to use the vLLM server
        
        # First, decode the input tensor to get the prompt text
        prompt_text = self.tokenizer.decode(ids[0], skip_special_tokens=True)

        try:
            completion = self.client.completions.create(
                model=self.config.model_name,
                prompt=prompt_text,
                max_tokens=self.config.max_step_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=["\n\n"],
                n=1, # In UATS, we generate one step at a time per branch expansion
            )
            step_text = completion.choices[0].text
            new_tokens = self.tokenizer.encode(step_text, add_special_tokens=False, return_tensors='pt')[0]
            
            return step_text, new_tokens.to(ids.device)

        except Exception as e:
            logger.error(f"[Rank {self.worker_rank}] Error during API call: {e}")
            return "", torch.tensor([], dtype=torch.long, device=ids.device)

    def _generate_final_answer(self, branch: Branch) -> str:
        # MODIFIED: to use the vLLM server
        text = branch.text.strip()
        header = "\n## Final Answer\n"
        final_text = text.rstrip() + header

        try:
            completion = self.client.completions.create(
                model=self.config.model_name,
                prompt=final_text,
                max_tokens=self.config.max_step_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                # No stop sequence here, let it finish.
                n=1,
            )
            generated_answer = completion.choices[0].text
            return header + generated_answer

        except Exception as e:
            logger.error(f"[Rank {self.worker_rank}] Error during final answer API call: {e}")
            return ""

    def _get_num_branches(self, ids: torch.Tensor) -> tuple[int, float]:
        # MODIFIED: to use the uncertainty model server
        ids_tensor = ids if ids.dim() == 2 else ids.unsqueeze(0)
        full_text = self.tokenizer.decode(ids_tensor[0], skip_special_tokens=True)
        reasoning_text = self._extract_reasoning_from_full_text(full_text)

        uncertainty_score = self._get_from_server(
            self.uncertainty_task_queue,
            self.uncertainty_result_queue,
            (self.question, reasoning_text)
        )
        u = self.uncertainty_to_branches(uncertainty_score, self.config.uncertainty_threshold)
        return u, uncertainty_score


    def search(self, user_question: str, system_prompt: str = MATH_SYSTEM_PROMPT) -> tuple[list[Branch], list[Branch], int]:
        """
        Perform guided tree search.

        Args:
            system_prompt: System prompt to use
            user_question: User question to answer

        Returns:
            List of all branches explored during search
        """
        # Store the question for value model calls
        self.question = user_question
        
        prompt = format_prompt(user_question, system_prompt)

        # ------------------------------------------------------------------
        # Initial prompt tokenisation (performed **once**).
        # ------------------------------------------------------------------
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")  # type: ignore[call-arg]
        prompt_ids = prompt_inputs.input_ids  # (1, L)

        # ------------------------------------------------------------------
        # Generate the very first reasoning step.
        # ------------------------------------------------------------------
        first_step_text, new_tokens = self._generate_step(prompt_ids.to(self.device))
        # Track overall token usage across all branches (reasoning phase only)
        tokens_used = len(new_tokens)

        # Concatenate the newly generated token IDs to obtain the full prompt
        # for the first branch.
        first_ids = torch.cat([prompt_ids[0].to(self.device), new_tokens]).unsqueeze(0)

        first_node_text = prompt + first_step_text
        total_tokens = count_tokens_after_marker(first_node_text, self.tokenizer)

        value_score, _ = self._value_model_score(ids=first_ids)

        # Normalise by sequence length to mitigate length bias.
        norm_score = value_score / max(total_tokens, 1)

        logger.info(f"[Branch 0 | Step 1] Generated step content:\n{first_step_text}\n{'='*40}")

        # Initialize beam with first branch
        branch_id_counter = 0
        all_branches = []
        root_branch = Branch(
            text=first_node_text,
            ids=first_ids.cpu(),
            step_count=1,
            score=norm_score,
            uncertainty=None,
            value=value_score,
            total_tokens=total_tokens,
            id=branch_id_counter,
            parent_id=None,
        )
        beam = [root_branch]
        all_branches.append(root_branch)
        branch_id_counter += 1
        step = 0
        # Keep track of branches that have finished (either by emitting the
        # closing tag or by exceeding the token budget).
        finished: List[Branch] = []

        while beam:
            # Stop if the global token budget has been exhausted
            if tokens_used >= self.config.budget:
                logger.info(
                    f"Token budget reached: {tokens_used} >= {self.config.budget}. Stopping search."
                )
                break

            logger.info(f"Step {step}: Expanding {len(beam)} branches.")
            all_candidates = []

            for branch_idx, branch in enumerate(beam):
                # Get uncertainty score and number of branches
                u, uncertainty_score = self._get_num_branches(branch.ids.to(self.device))
                logger.info(f"Branch {branch_idx} step {branch.step_count}: Uncertainty score={uncertainty_score:.4f}, u={u}")

                # Generate branches based on uncertainty
                budget_exceeded = False
                for _ in range(u):
                    # Sampling is already stochastic; no manual seeds needed.
                    step_text, new_tokens = self._generate_step(branch.ids.to(self.device))
                    # Update global token budget usage
                    tokens_used += len(new_tokens)

                    new_ids_1d = torch.cat(
                        [
                            branch.ids[0].to(self.device) if branch.ids.dim() == 2 else branch.ids.to(self.device),
                            new_tokens,
                        ]
                    )
                    new_ids = new_ids_1d.unsqueeze(0).to(dtype=torch.int)

                    new_text = branch.text + step_text
                    total_tokens = count_tokens_after_marker(new_text, self.tokenizer)

                    value_score, _ = self._value_model_score(ids=new_ids)

                    norm_score = value_score / max(total_tokens, 1)

                    logger.info(f"[Branch {branch_idx} | Step {branch.step_count+1}] Generated step content:\n{step_text}\n{'='*40}")
                    logger.info(f"Generated node: Value score={value_score:.4f}")

                    candidate = Branch(
                        text=new_text,
                        ids=new_ids.cpu(),
                        step_count=branch.step_count + 1,
                        score=norm_score,
                        uncertainty=uncertainty_score,
                        value=value_score,
                        total_tokens=total_tokens,
                        id=branch_id_counter,
                        parent_id=branch.id,
                    )
                    all_branches.append(candidate)
                    branch_id_counter += 1

                    # Check termination conditions
                    finished_here = False
                    # Check if EOS token was generated
                    if self.tokenizer.eos_token_id in new_tokens:
                        logger.info(f"Branch {branch_idx} finished with EOS token.")
                        finished_here = True
                    # Check if the global token budget has been exhausted after this generation
                    if tokens_used >= self.config.budget:
                        logger.info(
                            f"Global token budget exhausted ({tokens_used} >= {self.config.budget})."
                        )
                        finished_here = True
                        budget_exceeded = True

                    if finished_here:
                        finished.append(candidate)
                        break  # stop generating further continuations for this branch
                    else:
                        all_candidates.append(candidate)

                    if budget_exceeded:
                        break  # Exit inner branch-generation loop

                if budget_exceeded:
                    break  # Exit branch expansion loop

            # Prune to top-k by *normalised* score.
            if len(all_candidates) > self.config.beam_width:
                all_candidates = sorted(
                    all_candidates,
                    key=lambda x: x.score,
                    reverse=True,
                )[: self.config.beam_width]

            beam = all_candidates

            if tokens_used >= self.config.budget:
                logger.info("Global token budget reached during pruning. Ending search.")
                break

            step += 1

        logger.info("Search finished – all branches have terminated.")

        # --- Select the top beam_width branches by value score ---
        # Combine finished branches and the final beam of active branches
        all_potential_final_branches = finished + beam
        
        # Sort all potential final branches by their value score in descending order
        all_potential_final_branches.sort(key=lambda b: b.value, reverse=True)
        
        # Select the top `beam_width` branches
        final_branches = all_potential_final_branches[:self.config.beam_width]

        logger.info(f"Selected top {len(final_branches)} branches from {len(all_potential_final_branches)} potential candidates ({len(finished)} terminated, {len(beam)} in final beam).")

        # Generate final answers only for the selected top branches
        for branch in final_branches:
            branch.final_answer = self._generate_final_answer(branch)
            
        return final_branches, all_branches, tokens_used