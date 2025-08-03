import logging
import torch
from typing import List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StopStringCriteria

from rely.utils.probes import UncertaintyProbe, ValueProbe, load_probes
from rely.utils.text_utils import (
    count_tokens_after_marker,
    format_system_prompt,
    MMLU_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class UATSConfig:
    """Configuration for UATS inference."""
    model_name: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4"
    uncertainty_probe_path: Optional[str] = None
    value_probe_path: Optional[str] = None
    beam_width: int = 3
    max_branch_tokens: int = 1024
    max_step_tokens: int = 256
    device: str = "auto"
    probe_device: str = "cuda"
    temperature: float = 1.0
    top_p: float = 0.95


@dataclass
class Branch:
    """Represents a single branch in the search tree."""
    text: str
    ids: torch.Tensor
    step_count: int
    score: float
    uncertainty: Optional[float]
    value: float
    total_tokens: int
    final_answer: Optional[str] = None


class GuidedTreeSearch:
    """
    Guided Tree Search implementation for UATS (Uncertainty Aware Tree Search).
    """
    
    @classmethod
    def uncertainty_to_branches(cls, score: float) -> int:
        """Convert an uncertainty score (expected in [0, 1]) to the number of branches that should be explored."""
        if score >= 0.5:
            return 2
        return 1
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        uncertainty_probe: UncertaintyProbe,
        value_probe: ValueProbe,
        config: UATSConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.uncertainty_probe = uncertainty_probe
        self.value_probe = value_probe
        self.config = config
        self.device = next(model.parameters()).device

    def _value_probe(
        self,
        text: Optional[str] = None,
        ids: Optional[torch.Tensor] = None,
    ) -> tuple[float, torch.Tensor]:
        """Get value score for a given text or a pre-tokenised tensor.

        Passing ``ids`` avoids the costly re-tokenisation of the full prompt at
        every step.  Exactly one of ``text`` or ``ids`` must be provided.
        """

        if ids is None:
            if text is None:
                raise ValueError("Either `text` or `ids` must be provided to _value_probe")

            # Ensure the text ends with the step delimiter so that the probe
            # attends to the completion token.
            if not text.endswith("\n\n"):
                text = text + "\n\n"

            inputs = self.tokenizer(text, return_tensors="pt")  # type: ignore[call-arg]
            ids_tensor = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
        else:
            # ``ids`` already contains the full prompt – just create an
            # attention mask of ones.
            ids_tensor = ids if ids.dim() == 2 else ids.unsqueeze(0)
            ids_tensor = ids_tensor.to(self.device)
            attention_mask = torch.ones_like(ids_tensor, dtype=torch.long)

        with torch.no_grad():
            outputs = self.model(  # type: ignore[attr-defined]
                ids_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-2]
            probe_hidden_state = hidden_states[:, -1:, :].squeeze(1)
            value_logits = self.value_probe(probe_hidden_state)
            value_score = torch.sigmoid(value_logits).item()

        return value_score, ids_tensor.cpu()

    def _generate_step(
        self, ids: torch.Tensor
    ) -> tuple[str, torch.Tensor]:
        """Generate a single reasoning step given the *already tokenised* prompt.

        Returns the generated step **text** as well as the **token IDs** for the
        newly produced tokens.  This enables the caller to append the new IDs
        to the running prompt without having to re-tokenise the whole string.
        """

        model_input = ids if ids.dim() == 2 else ids.unsqueeze(0)
        model_input = model_input.to(self.device)

        stop_criteria = StopStringCriteria(self.tokenizer, ["\n\n"])
        with torch.no_grad():
            generated = self.model.generate(  # type: ignore[attr-defined]
                model_input,
                max_new_tokens=self.config.max_step_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,  # type: ignore[attr-defined]
                eos_token_id=self.tokenizer.eos_token_id,  # type: ignore[attr-defined]
                stopping_criteria=[stop_criteria],
                return_dict_in_generate=True,
            )

        new_tokens = generated.sequences[0][model_input.shape[1]:]
        step_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)  # type: ignore[attr-defined]
        return step_text, new_tokens

    def _generate_final_answer(self, branch: Branch) -> str:
        """Generate final answer for a completed branch."""
        text = branch.text.strip()
        # Determine which header should be added so that the output text explicitly
        # contains the </think> marker and the "## Final Answer" section.
        if not text.endswith("</think>") and not text.endswith("</think>\n"):
            header = "\n</think>\n## Final Answer\n"
        else:
            header = "\n## Final Answer\n"

        # Prompt fed to the language model consists of the reasoning trace plus the header.
        # Ensure consistent single newline formatting
        final_text = text.rstrip() + header
        
        gen_inputs = self.tokenizer(final_text, return_tensors="pt")
        gen_ids = gen_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                gen_ids,
                max_new_tokens=self.config.max_step_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
        
        new_tokens = generated.sequences[0][gen_ids.shape[1]:]
        generated_answer = self.tokenizer.decode(new_tokens, skip_special_tokens=False)  # type: ignore[attr-defined]
        # Return the header together with the generated answer so that callers will
        # see </think> and "## Final Answer" in the saved output.
        # Ensure consistent single newline formatting
        return header + generated_answer

    def _get_num_branches(self, ids: torch.Tensor) -> tuple[int, float]:
        """Compute the number of branches to sample based on the uncertainty probe.

        Accepts a pre-tokenised ``ids`` tensor to avoid repeated calls to the
        tokenizer.
        """

        ids_tensor = ids if ids.dim() == 2 else ids.unsqueeze(0)
        attention_mask = torch.ones_like(ids_tensor, dtype=torch.long)

        with torch.no_grad():
            outputs = self.model( 
                ids_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-2]
            probe_hidden_state = hidden_states[:, -1:, :].squeeze(1)
            uncertainty_logits = self.uncertainty_probe(probe_hidden_state)
            approx_I_score = torch.sigmoid(uncertainty_logits).item()

        u = min(self.uncertainty_to_branches(approx_I_score), self.config.beam_width)
        return u, approx_I_score

    def search(self, user_question: str, system_prompt: str = MMLU_SYSTEM_PROMPT) -> List[Branch]:
        """
        Perform guided tree search.
        
        Args:
            system_prompt: System prompt to use
            user_question: User question to answer
            
        Returns:
            List of all branches explored during search
        """
        prompt = format_system_prompt(system_prompt, user_question)

        # ------------------------------------------------------------------
        # Initial prompt tokenisation (performed **once**).
        # ------------------------------------------------------------------
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")  # type: ignore[call-arg]
        prompt_ids = prompt_inputs.input_ids  # (1, L)

        # ------------------------------------------------------------------
        # Generate the very first reasoning step.
        # ------------------------------------------------------------------
        first_step_text, new_tokens = self._generate_step(prompt_ids.to(self.device))

        # Concatenate the newly generated token IDs to obtain the full prompt
        # for the first branch.
        first_ids = torch.cat([prompt_ids[0].to(self.device), new_tokens]).unsqueeze(0)

        first_node_text = prompt + first_step_text
        total_tokens = count_tokens_after_marker(first_node_text, self.tokenizer)

        value_score, _ = self._value_probe(ids=first_ids)

        # Normalise by sequence length to mitigate length bias.
        norm_score = value_score / max(total_tokens, 1)
        
        logger.info(f"[Branch 0 | Step 1] Generated step content:\n{first_step_text}\n{'='*40}")
        
        # Initialize beam with first branch
        beam = [Branch(
            text=first_node_text,
            ids=first_ids.cpu(),
            step_count=1,
            score=norm_score,
            uncertainty=None,
            value=value_score,
            total_tokens=total_tokens
        )]

        step = 0
        # Keep track of branches that have finished (either by emitting the
        # closing tag or by exceeding the token budget).
        finished: List[Branch] = []
        
        while beam:
            logger.info(f"Step {step}: Expanding {len(beam)} branches.")
            all_candidates = []
            
            for branch_idx, branch in enumerate(beam):
                # Get uncertainty score and number of branches
                u, approx_I_score = self._get_num_branches(branch.ids.to(self.device))
                logger.info(f"Branch {branch_idx} step {branch.step_count}: Uncertainty score={approx_I_score:.4f}, u={u}")
                
                # Generate branches based on uncertainty
                for _ in range(u):
                    # Sampling is already stochastic; no manual seeds needed.
                    step_text, new_tokens = self._generate_step(branch.ids.to(self.device))

                    new_ids_1d = torch.cat([
                        branch.ids[0].to(self.device) if branch.ids.dim() == 2 else branch.ids.to(self.device),
                        new_tokens,
                    ])
                    new_ids = new_ids_1d.unsqueeze(0)

                    new_text = branch.text + step_text
                    total_tokens = count_tokens_after_marker(new_text, self.tokenizer)

                    value_score, _ = self._value_probe(ids=new_ids)

                    norm_score = value_score / max(total_tokens, 1)
                    
                    logger.info(f"[Branch {branch_idx} | Step {branch.step_count+1}] Generated step content:\n{step_text}\n{'='*40}")
                    logger.info(f"Generated node: Value score={value_score:.4f}")
                    
                    candidate = Branch(
                        text=new_text,
                        ids=new_ids.cpu(),
                        step_count=branch.step_count + 1,
                        score=norm_score,
                        uncertainty=approx_I_score,
                        value=value_score,
                        total_tokens=total_tokens
                    )
                    
                    # Check termination conditions
                    finished_here = False
                    if "</think>" in step_text:
                        logger.info(
                            f"Branch {branch_idx} finished with '</think>' token.")
                        finished_here = True
                    elif total_tokens >= self.config.max_branch_tokens:
                        logger.info(
                            f"Branch {branch_idx} reached max_branch_tokens ({total_tokens} >= {self.config.max_branch_tokens})")
                        finished_here = True

                    if finished_here:
                        finished.append(candidate)
                        break  # stop generating further continuations for this branch
                    else:
                        all_candidates.append(candidate)
            
            # --------------------------------------------------------------
            # Prune to top-k by *normalised* score.
            # --------------------------------------------------------------
            if len(all_candidates) > self.config.beam_width:
                all_candidates = sorted(
                    all_candidates,
                    key=lambda x: x.score,
                    reverse=True,
                )[: self.config.beam_width]

            beam = all_candidates
            logger.info(f"After pruning: {len(beam)} active branches so far.")

            step += 1
            
        logger.info("Search finished – all branches have terminated.")

        # Final branches comprise both those naturally finished *and* any that
        # were still in the beam when the optional safety net triggered.
        final_branches = finished if not beam else finished + beam

        for branch in final_branches:
            branch.final_answer = self._generate_final_answer(branch)

        return final_branches



# ------------------------------------------------------------
# UATS utilities
# ------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 4096,
    dtype: str = "bfloat16",
    load_in_4bit: bool = True,
) -> tuple:
    """
    Load Unsloth model and tokenizer for fast inference.

    Args:
        model_name: Name of the model to load
        device: Device to load model on (unused, Unsloth handles device)
        max_seq_length: Maximum sequence length for the model
        dtype: Data type for model weights
        load_in_4bit: Whether to load model in 4-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    return model, tokenizer

def create_uats_searcher(config: UATSConfig) -> GuidedTreeSearch:
    """
    Create a UATS searcher with the given configuration.
    
    Args:
        config: Configuration for UATS
        
    Returns:
        Configured GuidedTreeSearch instance
    """
    logger.info(f"Loading model: {config.model_name}")
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    logger.info("Model and tokenizer loaded successfully")
    
    # Get model properties
    hidden_size = model.config.hidden_size
    model_dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype
    
    logger.info(f"Loading probes (hidden_size={hidden_size}, dtype={model_dtype})")
    # Load probes
    try:
        uncertainty_probe, value_probe = load_probes(
            hidden_size=hidden_size,
            model_dtype=model_dtype,
            uncertainty_probe_path=config.uncertainty_probe_path,
            value_probe_path=config.value_probe_path,
            device=config.probe_device
        )
        logger.info("Probes loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Probe file not found: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Failed to load probe: {e}")
        raise
    
    # Create searcher
    searcher = GuidedTreeSearch(
        model=model,
        tokenizer=tokenizer,
        uncertainty_probe=uncertainty_probe,
        value_probe=value_probe,
        config=config
    )
    logger.info("UATS searcher created successfully")
    
    return searcher

def save_branches(
    branches: List[Branch],
    output_dir: Union[str, Path],
    max_branch_tokens: int,
    beam_width: Optional[int] | None = None,
) -> None:
    """
    Save the *active* branches at the end of the search to disk.

    Active branches correspond to those that remain in the beam after the final
    expansion.  In this implementation they are identified as the branches
    that have the **maximum `step_count`** among all recorded branches, because
    the search appends the current beam to the results after every expansion.

    If more than `beam_width` active branches somehow exist (should not happen
    because the beam is already capped), the highest-scoring ones are kept.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Identify the active branches (those with the greatest step_count).
    # ------------------------------------------------------------------
    if not branches:
        return

    max_step_count = max(b.step_count for b in branches)
    final_branches = [b for b in branches if b.step_count == max_step_count]

    if beam_width is not None and len(final_branches) > beam_width:
        final_branches = sorted(final_branches, key=lambda b: b.score, reverse=True)[:beam_width]

    logger.debug(f"Saving {len(final_branches)} final branches")
    for i, branch in enumerate(final_branches):
        filepath = output_dir / f"branch_{i}.txt"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Step count: {branch.step_count}\n")
            f.write(f"Score: {branch.score:.4f}\n")
            f.write(f"Uncertainty: {branch.uncertainty}\n")
            f.write(f"Value: {branch.value:.4f}\n")
            f.write(f"Total tokens: {branch.total_tokens}\n")
            f.write("\n--- Branch Text ---\n")
            # Ensure consistent single newline formatting for the end of thinking token
            branch_text = branch.text
            if branch.final_answer:
                final_answer_text = branch.final_answer.strip()
                # Ensure proper formatting: STEP\n</think>\n## Final Answer\nANSWER
                f.write(branch_text)
                f.write(final_answer_text)
            else:
                f.write(branch_text)

def run_uats(
    user_question: str,
    system_prompt: str = MMLU_SYSTEM_PROMPT,
    config: Optional[UATSConfig] = None,
    save_dir: Optional[Union[str, Path]] = None
) -> List[Branch]:
    """
    Run UATS search with a simple API and optionally save branches.

    Args:
        system_prompt: System prompt to use
        user_question: User question to answer
        config: Optional configuration (uses default if None)
        save_dir: Optional directory to save branches (if provided)

    Returns:
        List of all branches explored during search
    """
    if config is None:
        config = UATSConfig()

    logger.info("Creating UATS searcher...")
    searcher = create_uats_searcher(config)
    logger.info("Starting UATS search...")
    branches = searcher.search(user_question, system_prompt)
    logger.info(f"UATS search completed with {len(branches)} branches explored")
    
    if save_dir is not None:
        logger.info(f"Saving branches to {save_dir}")
        save_branches(
            branches,
            save_dir,
            max_branch_tokens=config.max_branch_tokens,
            beam_width=config.beam_width,
        )
    
    return branches