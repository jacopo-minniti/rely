# model.py

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Union


class SoftClassificationPRMModel(PreTrainedModel):
    config_class = AutoConfig
    _supports_sdpa = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    """
    A soft classification model that wraps a base transformer model with a linear head.
    
    This model outputs continuous values between 0 and 1 for each token, suitable for 
    Process Reward Model (PRM) tasks with normalized scores.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.transformer = AutoModel.from_config(
            config,
            dtype=torch.bfloat16, # getattr(config, 'torch_dtype', torch.bfloat16),
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
            
        self.hidden_size = self.transformer.config.hidden_size
            
        self.classification_head = nn.Linear(self.hidden_size, 1)
        self.classification_head = self.classification_head.float()
        
        # Default loss type
        self.loss_type = "bce"
        
        self.post_init()
    
    @classmethod
    def from_base_model(cls, base_model_name: str, **kwargs):
        """
        Load a SoftClassificationPRMModel from a pretrained base model for initial training.
        """
        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        
        config.base_model_name = base_model_name
        
        model = cls(config)
        
        model.transformer = AutoModel.from_pretrained(
            base_model_name, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            **kwargs
        )
        
        # Initialize with default loss type
        model.loss_type = "bce"
        
        return model
    
    def resize_token_embeddings(self, new_num_tokens: int):
        if self.transformer:
            self.transformer.resize_token_embeddings(new_num_tokens)
    
    def set_loss_type(self, loss_type: str):
        """Set the loss type for training."""
        if loss_type not in ["bce", "mse"]:
            raise ValueError(f"loss_type must be either 'bce' or 'mse', got '{loss_type}'")
        self.loss_type = loss_type
    
    def _set_gradient_checkpointing(self, module, value=False):
        if module is self.transformer:
            if hasattr(module, 'gradient_checkpointing_enable'):
                if value:
                    module.gradient_checkpointing_enable()
                else:
                    module.gradient_checkpointing_disable()
            elif hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = value
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            if gradient_checkpointing_kwargs is not None:
                self.transformer.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
            else:
                self.transformer.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        if hasattr(self.transformer, 'gradient_checkpointing_disable'):
            self.transformer.gradient_checkpointing_disable()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> TokenClassifierOutput:
        """
        Forward pass optimized for bfloat16 inference with a stable float32 loss calculation.
        """
        # This part of the model runs in bfloat16, assuming the model
        # was loaded with torch_dtype=torch.bfloat16
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Logits are produced and kept in bfloat16
        logits = self.classification_head(hidden_states)
        logits = logits.squeeze(-1)
        
        loss = None
        # The loss calculation block is entered only during training/evaluation with labels
        if labels is not None:
            # Filtering operations are performed in the native bfloat16 dtype
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss_mask = active_labels != -100.0
            
            if loss_mask.sum() > 0:
                filtered_logits = active_logits[loss_mask]
                filtered_labels = active_labels[loss_mask]
                
                # --- MODIFICATION ---
                # Inputs are explicitly cast to float32 ONLY for the loss function,
                # ensuring numerical stability.
                if self.loss_type == "bce":
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(filtered_logits.float(), filtered_labels.float())
                elif self.loss_type == "mse":
                    loss_fct = nn.MSELoss()
                    # For MSE, apply sigmoid to the upcasted float32 logits
                    filtered_probs = torch.sigmoid(filtered_logits.float())
                    loss = loss_fct(filtered_probs, filtered_labels.float())
                else:
                    raise ValueError(f"Unsupported loss type: {self.loss_type}")
            else:
                # If no labels are present, create a zero loss tensor
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # --- MODIFICATION ---
        # The final output (probabilities) is calculated from the original bfloat16 logits,
        # ensuring the inference path is fully bfloat16.
        final_outputs = torch.sigmoid(logits)
        
        return TokenClassifierOutput(
            loss=loss, # Loss is a float32 scalar (which is fine and expected)
            logits=final_outputs, # The returned logits (probabilities) are bfloat16
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )