import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)

class Scorer:
    def __init__(self, model, tokenizer: AutoTokenizer, device: str, scoring_method: str, model_type: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.scoring_method = scoring_method
        self.model_type = model_type

    def _aggregate_scores(self, step_scores: List[float]) -> float:
        """Aggregate step scores into a single score based on the configured method."""
        if not step_scores:
            return 0.0
        if len(step_scores) == 1:
            return step_scores[0]

        if self.scoring_method == "last_step":
            return step_scores[-1]
        elif self.scoring_method == "product":
            return float(np.prod(step_scores))
        elif self.scoring_method == "average":
            return float(np.mean(step_scores))
        elif self.scoring_method == "minimum":
            return float(np.min(step_scores))
        
        return 0.0

    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> List[float]:
        """
        Scores a batch of texts by calculating and aggregating per-step probabilities for each.
        """
        if not texts:
            return []

        conversation_strs = []
        separator_token = "<extra_0>"
        from rely.utils.text_utils import MATH_SYSTEM_PROMPT

        for text in texts:
            if "<|im_start|>assistant\n" in text:
                parts = text.split("<|im_start|>assistant\n")
                header, assistant_content = parts[0], parts[1]
                if assistant_content.endswith("<|im_end|>"):
                    assistant_content = assistant_content[:-len("<|im_end|>")]
                
                # Extract system and user prompts from header
                system_content = MATH_SYSTEM_PROMPT # default
                user_content = ""
                try:
                    sys_match = re.search(r"<\|im_start\|>system\n(.*?)(<\|im_end|>)", header, re.DOTALL)
                    if sys_match: system_content = sys_match.group(1).strip()
                    
                    usr_match = re.search(r"<\|im_start\|>user\n(.*?)(<\|im_end|>)", header, re.DOTALL)
                    if usr_match: user_content = usr_match.group(1).strip()
                except Exception:
                    pass # Keep defaults if regex fails

            else: # Fallback for plain text
                assistant_content = text
                system_content = MATH_SYSTEM_PROMPT
                user_content = "Solve the problem."

            steps = [s.strip() for s in assistant_content.split('\n\n') if s.strip()]
            formatted_content = separator_token.join(steps) + separator_token
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": formatted_content}
            ]
            conv_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            conversation_strs.append(conv_str)

        all_aggregated_scores = []
        batch_size = 16

        for i in range(0, len(conversation_strs), batch_size):
            batch_conversations = conversation_strs[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_conversations,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.device)

            try:
                if self.model_type == "value":
                    base_model_output = self.model.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)
                    logits = self.model.score(base_model_output.last_hidden_state)
                else:
                    outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                    logits = outputs.logits

                probabilities = F.softmax(logits, dim=-1)
                step_sep_id = self.tokenizer.encode(separator_token, add_special_tokens=False)[0]
                token_masks = (inputs.input_ids == step_sep_id)
                
                for j in range(logits.size(0)):
                    sample_probs = probabilities[j]
                    sample_mask = token_masks[j]
                    
                    if sample_mask.any():
                        step_scores = sample_probs[sample_mask][:, 1].cpu().tolist()
                        aggregated_score = self._aggregate_scores(step_scores)
                        all_aggregated_scores.append(aggregated_score)
                    else:
                        all_aggregated_scores.append(0.0)
                
                del inputs, logits, probabilities
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"[Scorer] Error processing batch: {e}")
                # Add default scores for the failed batch items
                all_aggregated_scores.extend([0.0] * len(batch_conversations))
        
        return all_aggregated_scores

    def score(self, text: str) -> float:
        """Scores a single text. For compatibility, use score_batch for efficiency."""
        return self.score_batch([text])[0]

