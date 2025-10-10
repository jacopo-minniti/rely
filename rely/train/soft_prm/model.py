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
            dtype=getattr(config, 'torch_dtype', torch.bfloat16),
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
            
        self.hidden_size = self.transformer.config.hidden_size
            
        self.classification_head = nn.Linear(self.hidden_size, 1)
        self.classification_head = self.classification_head.float()
        
        # Default loss type
        self.loss_type = "bce"
        self.bce_pos_weight = None
        self.bce_label_weight = None
        
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
    
    def set_bce_pos_weight(self, weight: float):
        """Set the positive weight for BCE loss."""
        self.bce_pos_weight = weight
    
    def set_bce_label_weight(self, weight: float):
        """Set the label-based weight for BCE loss."""
        self.bce_label_weight = weight
    
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
        Forward pass of the soft classification model.
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        
        logits = self.classification_head(hidden_states)
        logits = logits.squeeze(-1)
        
        if logits.dtype != torch.float32:
            logits = logits.float()
        
        loss = None
        if labels is not None:
            if labels.dtype != torch.float32:
                labels = labels.float()
            
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1)[active_loss]
            
            active_labels = labels.view(-1)[active_loss]
            loss_mask = active_labels != -100.0
            
            if loss_mask.sum() > 0:
                filtered_logits = active_logits[loss_mask]
                filtered_labels = active_labels[loss_mask]
                
                if self.loss_type == "bce":
                    pos_weight = torch.tensor(self.bce_pos_weight, device=filtered_logits.device) if self.bce_pos_weight is not None else None
                    
                    weights = None
                    if self.bce_label_weight is not None:
                        # Create weights based on labels. Higher labels get more weight.
                        # Weights are 1 for label 0, and 1 + bce_label_weight for label 1.
                        weights = 1 + filtered_labels * self.bce_label_weight
                        
                    loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight, weight=weights)
                    loss = loss_fct(filtered_logits, filtered_labels)
                elif self.loss_type == "mse":
                    loss_fct = nn.MSELoss()
                    # For MSE loss, we need to apply sigmoid to logits first to get probabilities
                    filtered_probs = torch.sigmoid(filtered_logits)
                    loss = loss_fct(filtered_probs, filtered_labels)
                else:
                    raise ValueError(f"Unsupported loss type: {self.loss_type}")
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        final_outputs = torch.sigmoid(logits)
        
        return TokenClassifierOutput(
            loss=loss,
            logits=final_outputs,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )