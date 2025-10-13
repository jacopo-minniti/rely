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
        self.mask_zeros = False
        
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
    
    def set_mask_zeros(self, mask_zeros: bool):
        """Set whether to mask labels with values close to zero."""
        self.mask_zeros = mask_zeros
    
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

            if self.mask_zeros:
                # When mask_zeros is True, samples where all labels are effectively zero are ignored.
                # A label is considered zero if it's < 0.001.
                # We identify samples that have labels, but all of them are zero, and set their labels to -100.
                non_zero_label_exists = torch.any((labels >= 0.001) & (attention_mask == 1), dim=1)
                any_label_exists = torch.any((labels != -100.0) & (attention_mask == 1), dim=1)
                samples_to_ignore = ~non_zero_label_exists & any_label_exists
                
                if samples_to_ignore.any():
                    labels[samples_to_ignore] = -100.0
            
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            # Standard mask for labels marked with -100.0
            loss_mask = active_labels != -100.0
            
            # If mask_zeros is enabled, also mask labels with values < 0.001.
            # This is to prevent the model from being penalized for predicting small values for steps
            # that are effectively incorrect or have a score of 0. The 0.001 threshold is used
            # to account for floating-point inaccuracies.
            if self.mask_zeros:
                loss_mask = loss_mask & (active_labels >= 0.001)

            if loss_mask.sum() > 0:
                filtered_logits = active_logits[loss_mask]
                filtered_labels = active_labels[loss_mask]
                
                if self.loss_type == "bce":
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(filtered_logits, filtered_labels)
                elif self.loss_type == "mse":
                    loss_fct = nn.MSELoss()
                    # For MSE loss, we need to apply sigmoid to logits first to get probabilities
                    filtered_probs = torch.sigmoid(filtered_logits)
                    loss = loss_fct(filtered_probs, filtered_labels)
                else:
                    raise ValueError(f"Unsupported loss type: {self.loss_type}")
            else:
                # Return a zero loss that is connected to the graph to avoid issues with gradient accumulation
                loss = (logits.sum() * 0.0)
        
        final_outputs = torch.sigmoid(logits)
        
        return TokenClassifierOutput(
            loss=loss,
            logits=final_outputs,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )