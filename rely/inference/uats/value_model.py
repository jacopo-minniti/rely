"""
Value model for UATS using autoregressive LLM with classification head.
Similar to the PRM approach used in SBS but integrated into UATS.
"""

import math
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging

from rely.utils import MATH_SYSTEM_PROMPT as SYSTEM_PROMPT

# Disable torch compilation to prevent recompilation issues
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# torch._dynamo.config.suppress_errors = True

logger = logging.getLogger(__name__)


def make_step_rewards(logits, token_masks):
    """Extract step-wise rewards from PRM model outputs."""
    
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        # Find non-zero elements (where tokens match <extra_0>)
        non_zero_mask = sample.sum(dim=-1) != 0
        if non_zero_mask.any():
            valid_probs = sample[non_zero_mask]  # valid_tokens, num_labels
            if valid_probs.size(1) >= 2:
                # Extract positive class probabilities (index 1)
                positive_probs = valid_probs[:, 1]
                non_zero_elements_list = positive_probs.cpu().tolist()
            else:
                # Fallback if unexpected dimensions
                non_zero_elements_list = valid_probs.flatten().cpu().tolist()
        else:
            # No valid tokens found
            non_zero_elements_list = []
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


class UATSValueModel:
    """Value model for UATS using autoregressive LLM with classification head."""
    
    def __init__(self, model_path: str, device: str = "cuda:1", scoring_method: str = "product"):
        """Initialize the value model.
        
        Args:
            model_path: Path to HuggingFace model with classification head
            device: Device to load the model on
            scoring_method: How to combine step rewards ("product", "minimum", "average", "last_step")
        """
        self.device = device
        self.scoring_method = scoring_method
        
        # Validate scoring method
        valid_methods = ["product", "minimum", "average", "last_step"]
        if scoring_method not in valid_methods:
            raise ValueError(f"Invalid scoring_method '{scoring_method}'. Must be one of: {valid_methods}")
        
        logger.info(f"Loading UATS value model from: {model_path} on {device}")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        
        logger.info(f"UATS value model loaded with scoring method: {scoring_method}")
    
    @torch.no_grad()
    def get_value(self, question: str, reasoning_text: str) -> float:
        """Get value score for a reasoning text given a question.
        
        Args:
            question: The original question being solved
            reasoning_text: The reasoning chain to evaluate
            
        Returns:
            Value score between 0 and 1
        """
        # Prepare the text with <extra_0> tokens between steps
        # Split by double newlines and join with <extra_0>
        steps = reasoning_text.split('\n\n')
        # Filter out empty steps
        steps = [step.strip() for step in steps if step.strip()]
        if not steps:
            # If no steps, just use the original text
            formatted_text = reasoning_text.strip() + "<extra_0>"
        else:
            formatted_text = "<extra_0>".join(steps) + "<extra_0>"
        
        # Create messages in the format expected by the PRM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": formatted_text}
        ]
        
        # Apply chat template
        conversation_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Tokenize and move to the value model device
        input_ids = self.tokenizer.encode(
            conversation_str, 
            return_tensors="pt", 
            truncation=True,
            max_length=8_000
        ).to(self.device)
        
        # Get model outputs
        try:
            outputs = self.model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True
            )
        except (AttributeError, TypeError) as e:
            # Fallback for compatibility issues
            logger.warning(f"Cache issue encountered: {e}. Trying without cache.")
            outputs = self.model(input_ids=input_ids)
        
        # Extract step rewards
        step_sep_tokens = self.tokenizer.encode("<extra_0>", add_special_tokens=False)
        if step_sep_tokens:
            step_sep_id = step_sep_tokens[0]
        else:
            # Fallback if token not found
            step_sep_id = self.tokenizer.encode("<extra_0>")[0]
        
        token_masks = (input_ids == step_sep_id)
        step_rewards = make_step_rewards(outputs[0], token_masks)
        
        # Calculate value based on the configured scoring method
        if step_rewards and step_rewards[0]:
            # Adding small epsilon to avoid exact zeros that would make entire product zero
            epsilon = 1e-8
            safe_rewards = [max(r, epsilon) for r in step_rewards[0]]
            
            if self.scoring_method == "product":
                # Use product of all step rewards - penalizes paths with weak steps
                value = math.prod(safe_rewards)
            elif self.scoring_method == "minimum":
                # Use minimum step reward - focuses on the weakest link
                value = min(safe_rewards)
            elif self.scoring_method == "average":
                # Use average of all step rewards - balanced approach
                value = sum(safe_rewards) / len(safe_rewards)
            elif self.scoring_method == "last_step":
                # Use only the last step reward - for comparing new generations
                value = safe_rewards[-1]
            else:
                # Fallback to product if invalid method specified
                logger.warning(f"Invalid scoring_method '{self.scoring_method}', using 'product'")
                value = math.prod(safe_rewards)
        else:
            # Fallback value if no rewards found - use low score rather than neutral
            value = 0.1
        
        return value
    
    def clear_cache(self):
        """Clear model cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
