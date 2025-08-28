import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List
import numpy as np

class Scorer:
    def __init__(self, model, tokenizer: AutoTokenizer, device: str, scoring_method: str, model_type: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.scoring_method = scoring_method
        self.model_type = model_type  # 'value' or 'uncertainty'

    def _get_step_scores(self, text: str) -> List[float]:
        """
        Processes text to get per-step positive class probabilities.
        This is the core logic that aligns with the user's reference code.
        """
        # The special token is used to mark the end of each reasoning step.
        separator_token = "<extra_0>"
        
        # For value model (PRM), the entire reasoning chain is the input.
        # For uncertainty model (PUM), it's the same.
        messages = [
            {"role": "assistant", "content": text},
        ]
        
        # We need to manually split and join to ensure the separator is correctly placed.
        steps = [s.strip() for s in text.split('\n\n') if s.strip()]
        if not steps:
            return []
            
        formatted_content = separator_token.join(steps) + separator_token
        
        # Re-apply the template with the correctly formatted content
        messages[0]['content'] = formatted_content
        conversation_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        inputs = self.tokenizer(conversation_str, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits  # Shape: [1, seq_len, num_classes]

        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Find the exact indices of the separator token
        separator_id = self.tokenizer.encode(separator_token, add_special_tokens=False)[0]
        separator_indices = (input_ids[0] == separator_id).nonzero(as_tuple=True)[0]

        if separator_indices.nelement() == 0:
            return []

        # Extract probabilities ONLY at the separator token locations
        step_probs = probabilities[0, separator_indices, :]  # Shape: [num_steps, num_classes]
        
        # The score is the probability of the positive class (index 1)
        positive_probs = step_probs[:, 1]
        
        return positive_probs.cpu().tolist()

    def score(self, text: str) -> float:
        """
        Scores a given text by calculating per-step probabilities and aggregating them.
        """
        step_scores = self._get_step_scores(text)

        if not step_scores:
            # Return a neutral score if no steps are found
            return 0.0 if self.scoring_method != "product" else 1.0

        if self.scoring_method == "last_step":
            return step_scores[-1]
        elif self.scoring_method == "product":
            # Use log probabilities for numerical stability, then exp to return to prob scale if needed,
            # but in our search, log scale is better. The sum of logs is monotonic with the product.
            return np.sum(np.log(step_scores))
        elif self.scoring_method == "average":
            return np.mean(step_scores)
        elif self.scoring_method == "minimum":
            return np.min(step_scores)
        
        # Fallback for an unknown method
        return 0.0