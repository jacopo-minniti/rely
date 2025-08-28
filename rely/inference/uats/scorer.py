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
        self.model_type = model_type

    def _get_step_scores(self, text: str) -> List[float]:
        """
        Processes text to get per-step positive class probabilities, handling
        different model output structures.
        """
        separator_token = "<extra_0>"
        
        steps = [s.strip() for s in text.split('\n\n') if s.strip()]
        if not steps:
            return []
            
        formatted_content = separator_token.join(steps) + separator_token
        
        messages = [{"role": "assistant", "content": formatted_content}]
        conversation_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        inputs = self.tokenizer(conversation_str, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if self.model_type == "value":
                base_model_output = self.model.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)
                logits = self.model.score(base_model_output.last_hidden_state)
            else:
                outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                logits = outputs.logits

        probabilities = F.softmax(logits, dim=-1)

        separator_id = self.tokenizer.encode(separator_token, add_special_tokens=False)[0]
        separator_indices = (inputs.input_ids[0] == separator_id).nonzero(as_tuple=True)[0]

        if separator_indices.nelement() == 0:
            return []

        step_probs = probabilities[0, separator_indices, :]
        positive_probs = step_probs[:, 1]
        
        return positive_probs.cpu().tolist()

    def score(self, text: str) -> float:
        """
        Scores a given text by calculating and aggregating per-step probabilities.
        """
        step_scores = self._get_step_scores(text)

        if not step_scores:
            return 0.0

        if len(step_scores) == 1:
            return step_scores[0]

        if self.scoring_method == "last_step":
            return step_scores[-1]
        elif self.scoring_method == "product":
            return np.prod(step_scores)
        elif self.scoring_method == "average":
            return np.mean(step_scores)
        elif self.scoring_method == "minimum":
            return np.min(step_scores)
        
        return 0.0
