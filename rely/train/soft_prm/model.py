# model.py

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Union


class RegressionPRMModel(PreTrainedModel):
    config_class = AutoConfig
    _supports_sdpa = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    """
    A regression model that wraps a base transformer model with a linear head.
    
    This model outputs continuous values for each token, suitable for 
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
            
        self.regression_head = nn.Linear(self.hidden_size, 1)
        self.regression_head = self.regression_head.float()
        
        self.mask_zeros = False
        
        self.post_init()
    
    @classmethod
    def from_base_model(cls, base_model_name: str, **kwargs):
        """
        Load a RegressionPRMModel from a pretrained base model for initial training.
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
        
        return model
    
    def resize_token_embeddings(self, new_num_tokens: int):
        if self.transformer:
            self.transformer.resize_token_embeddings(new_num_tokens)
    
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
        Forward pass of the regression model.
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        
        logits = self.regression_head(hidden_states)
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
            
            # Standard mask for labels marked with -100.0
            loss_mask = active_labels != -100.0
            
            # If mask_zeros is enabled, also mask labels with values < 0.001.
            # This prevents the model from being penalized for predicting small values for steps
            # that are effectively incorrect or have a score of 0. The 0.001 threshold is used
            # to account for floating-point inaccuracies.
            if self.mask_zeros:
                loss_mask = loss_mask & (active_labels >= 0.001)

            if loss_mask.sum() > 0:
                filtered_logits = active_logits[loss_mask]
                filtered_labels = active_labels[loss_mask]
                
                loss_fct = nn.MSELoss()
                loss = loss_fct(filtered_logits, filtered_labels)
            else:
                # Return a zero loss that is connected to the graph to avoid issues with gradient accumulation
                loss = (logits.sum() * 0.0)
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
