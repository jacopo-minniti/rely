import logging
import torch
from typing import List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StopStringCriteria

from rely.utils.probes import UncertaintyProbe, ValueProbe, load_probes
from rely.utils.text_utils import (
    get_last_step_pos,
    count_tokens_after_marker,
    format_system_prompt,
    ensure_think_ending,
    MMLU_SYSTEM_PROMPT,
)


def uncertainty_to_branches(score: float) -> int:
    """Convert an uncertainty score (expected in [0, 1]) to the number of branches that should be explored. """
    if score >= 0.5:
        return 2
    return 1

logger = logging.getLogger(__name__)


@dataclass
class UATSConfig:
    """Configuration for UATS inference."""
    model_name: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4"
    uncertainty_probe_path: Optional[str] = None
    value_probe_path: Optional[str] = None
    beam_width: int = 3
    n_approx: int = 10
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

    def _value_probe(self, text: str) -> tuple[float, torch.Tensor]:
        """Get value score for a given text."""
        last_step_pos, probe_text = get_last_step_pos(text, self.tokenizer)
        inputs = self.tokenizer(probe_text, return_tensors="pt")
        ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-2]
            # Extract the hidden state of the first token *after* the last "\n\n"
            # marker, i.e. the first token generated in the most recent step. This
            # follows the original design where probes look at the activation that
            # corresponds exactly to the start of the new reasoning node.
            probe_hidden_state = hidden_states[:, last_step_pos:last_step_pos + 1, :].squeeze(1)
            value_score = self.value_probe(probe_hidden_state).item()
        
        # Move ids to CPU before returning to avoid keeping many full prompts
        # resident in GPU memory for every explored node.
        return value_score, ids.cpu()

    def _generate_step(self, text: str) -> str:
        """Generate a single reasoning step."""
        gen_inputs = self.tokenizer(text, return_tensors="pt")
        gen_ids = gen_inputs.input_ids.to(self.device)
        # Stop generation when the model emits the double-newline delimiter.
        # NOTE: Transformers expects the stop string list *first*.
        stop_criteria = StopStringCriteria(["\n\n"], self.tokenizer)
        
        with torch.no_grad():
            generated = self.model.generate(
                gen_ids,
                max_new_tokens=self.config.max_step_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=[stop_criteria],
                return_dict_in_generate=True
            )
        
        new_tokens = generated.sequences[0][gen_ids.shape[1]:]
        gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return gen_text

    def _generate_final_answer(self, branch: Branch) -> str:
        """Generate final answer for a completed branch."""
        final_text = ensure_think_ending(branch.text)
        gen_inputs = self.tokenizer(final_text, return_tensors="pt")
        gen_ids = gen_inputs.input_ids.to(self.device)
        stop_criteria = StopStringCriteria(["\n\n"], self.tokenizer)
        
        with torch.no_grad():
            generated = self.model.generate(
                gen_ids,
                max_new_tokens=self.config.max_step_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=[stop_criteria],
                return_dict_in_generate=True
            )
        
        new_tokens = generated.sequences[0][gen_ids.shape[1]:]
        final_answer = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return final_answer

    def _get_num_branches(self, ids: torch.Tensor, attention_mask: torch.Tensor) -> int:
        """Compute the number of branches to sample based on uncertainty probe."""
        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-2]
            # Extract the hidden state of the first token after the last "\n\n" marker
            # This assumes ids and attention_mask correspond to the probe_text
            # The caller must ensure correct slicing if needed
            # For now, we use the last token
            probe_hidden_state = hidden_states[:, -1:, :].squeeze(1)
            approx_I_score = self.uncertainty_probe(probe_hidden_state).item()
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
        initial_text = prompt
        
        # Generate first step
        first_step_text = self._generate_step(initial_text)
        first_node_text = initial_text + first_step_text
        total_tokens = count_tokens_after_marker(first_node_text, self.tokenizer)
        value_score, ids_val = self._value_probe(first_node_text)
        
        logger.info(f"[Branch 0 | Step 1] Generated step content:\n{first_step_text}\n{'='*40}")
        
        # Initialize beam with first branch
        beam = [Branch(
            text=first_node_text,
            ids=ids_val.cpu(),
            step_count=1,
            score=value_score,
            uncertainty=None,
            value=value_score,
            total_tokens=total_tokens
        )]

        all_branches = []
        step = 0
        triggered_branch_idx = None
        triggered_reason = None
        
        while beam:
            logger.info(f"Step {step}: Expanding {len(beam)} branches.")
            all_candidates = []
            
            for branch_idx, branch in enumerate(beam):
                # Get uncertainty score and number of branches
                last_step_pos, probe_text = get_last_step_pos(branch.text, self.tokenizer)
                inputs = self.tokenizer(probe_text, return_tensors="pt")
                ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                u, approx_I_score = self._get_num_branches(ids, attention_mask)
                logger.info(f"Branch {branch_idx} step {branch.step_count}: Uncertainty score={approx_I_score:.4f}, u={u}")
                
                # Generate branches based on uncertainty
                for _ in range(u):
                    # Sampling is already stochastic; no manual seeds needed.
                    step_text = self._generate_step(branch.text)
                    new_text = branch.text + step_text
                    total_tokens = count_tokens_after_marker(new_text, self.tokenizer)
                    value_score, ids_val = self._value_probe(new_text)
                    
                    logger.info(f"[Branch {branch_idx} | Step {branch.step_count+1}] Generated step content:\n{step_text}\n{'='*40}")
                    logger.info(f"Generated node: Value score={value_score:.4f}")
                    
                    candidate = Branch(
                        text=new_text,
                        ids=ids_val.cpu(),
                        step_count=branch.step_count + 1,
                        score=value_score,
                        uncertainty=approx_I_score,
                        value=value_score,
                        total_tokens=total_tokens
                    )
                    
                    # Check termination conditions
                    if "</think>" in step_text:
                        logger.info(f"Branch {branch_idx} triggered finish condition with '</think>' token.")
                        if triggered_branch_idx is None:  # keep the first that triggers
                            triggered_branch_idx = len(all_branches) + len(all_candidates)
                            triggered_reason = "</think>"
                        all_candidates.append(candidate)
                        break  # stop generating more continuations for *this* branch
                    elif total_tokens >= self.config.max_branch_tokens:
                        logger.info(f"Branch {branch_idx} triggered finish condition: max_branch_tokens reached ({total_tokens} >= {self.config.max_branch_tokens})")
                        if triggered_branch_idx is None:
                            triggered_branch_idx = len(all_branches) + len(all_candidates)
                            triggered_reason = "max_branch_tokens"
                        all_candidates.append(candidate)
                        break  # stop generating more continuations for this branch
                    else:
                        all_candidates.append(candidate)
            
            # Prune to top branches
            if len(all_candidates) > self.config.beam_width:
                all_candidates = sorted(all_candidates, key=lambda x: x.score, reverse=True)[:self.config.beam_width]
                logger.info(f"Pruned to top {self.config.beam_width} branches by value score.")
            
            all_branches.extend(beam)
            beam = all_candidates
            logger.info(f"After pruning: {len(beam)} active, {len(all_branches)} total branches so far.")
            step += 1
            
            if triggered_branch_idx is not None:
                break
        
        all_branches.extend(beam)
        logger.info(f"Search finished. Branch {triggered_branch_idx} triggered finish condition: {triggered_reason}")
        
        # Generate final answers for all branches
        for branch in all_branches:
            branch.final_answer = self._generate_final_answer(branch)
        
        return all_branches



# ------------------------------------------------------------
# UATS utilities
# ------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: str = "auto") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer.
    
    Args:
        model_name: Name of the model to load
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device
    )
    return model, tokenizer

def create_uats_searcher(config: UATSConfig) -> GuidedTreeSearch:
    """
    Create a UATS searcher with the given configuration.
    
    Args:
        config: Configuration for UATS
        
    Returns:
        Configured GuidedTreeSearch instance
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model_name, config.device)
    
    # Get model properties
    hidden_size = model.config.hidden_size
    model_dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype
    
    # Load probes
    uncertainty_probe, value_probe = load_probes(
        hidden_size=hidden_size,
        model_dtype=model_dtype,
        uncertainty_probe_path=config.uncertainty_probe_path,
        value_probe_path=config.value_probe_path,
        device=config.probe_device
    )
    
    # Create searcher
    searcher = GuidedTreeSearch(
        model=model,
        tokenizer=tokenizer,
        uncertainty_probe=uncertainty_probe,
        value_probe=value_probe,
        config=config
    )
    
    return searcher

def save_branches(branches: List[Branch], output_dir: Union[str, Path], max_branch_tokens: int) -> None:
    """
    Save branches to files.
    
    Args:
        branches: List of branches to save
        output_dir: Directory to save branches in
        max_branch_tokens: Maximum branch tokens threshold
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    def is_final_branch(branch: Branch) -> bool:
        text = branch.text
        ends_with_think = text.strip().endswith("</think>") or text.strip().endswith("</think>\n")
        hit_max_tokens = branch.total_tokens >= max_branch_tokens
        return ends_with_think or hit_max_tokens
    
    final_branches = [b for b in branches if is_final_branch(b)]
    
    for i, branch in enumerate(final_branches):
        filename = f"branch_{i}.txt"
        filepath = output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Step count: {branch.step_count}\n")
            f.write(f"Score: {branch.score:.4f}\n")
            f.write(f"Uncertainty: {branch.uncertainty}\n")
            f.write(f"Value: {branch.value:.4f}\n")
            f.write(f"Total tokens: {branch.total_tokens}\n")
            f.write("\n--- Branch Text ---\n")
            f.write(branch.text)
            if branch.final_answer:
                f.write(branch.final_answer) 

def run_uats_search(
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

    searcher = create_uats_searcher(config)
    branches = searcher.search(user_question, system_prompt)
    if save_dir is not None:
        save_branches(branches, save_dir, config.max_branch_tokens)
    return branches
