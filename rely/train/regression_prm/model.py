# model.py

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Union


class RegressionPRMModel(PreTrainedModel):
    config_class = AutoConfig
    _supports_sdpa = True
    """
    A regression model that wraps a base transformer model with a linear regression head.
    
    This model outputs continuous values for each token, suitable for Process Reward Model (PRM)
    regression tasks.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Instantiate the base transformer using the config from the checkpoint.
        # This creates the correct architecture without loading any weights yet.
        self.transformer = AutoModel.from_config(
            config,
            torch_dtype=getattr(config, 'torch_dtype', torch.bfloat16),
            trust_remote_code=True,
            attn_implementation="eager"  # or your preferred implementation
        )
            
        # Get hidden size from transformer config
        self.hidden_size = self.transformer.config.hidden_size
            
        # Regression head - single linear layer outputting 1 value per token
        self.regression_head = nn.Linear(self.hidden_size, 1)
        
        # Ensure regression head uses float32
        self.regression_head = self.regression_head.float()
        
        # Initialize weights
        self.post_init()
    
    @classmethod
    def from_base_model(cls, base_model_name: str, **kwargs):
        """
        Load a RegressionPRMModel from a pretrained base model for initial training.
        """
        # Load the base model config first
        config = AutoConfig.from_pretrained(base_model_name, **kwargs)
        
        # Add our custom attributes to config for saving
        config.base_model_name = base_model_name
        
        # Create our custom model shell using the config
        model = cls(config)
        
        # Now, load the pretrained weights for the transformer part
        model.transformer = AutoModel.from_pretrained(base_model_name, **kwargs)
        
        return model
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings when new tokens are added."""
        if self.transformer:
            self.transformer.resize_token_embeddings(new_num_tokens)
    
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
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Apply regression head
        logits = self.regression_head(hidden_states)  # (batch_size, seq_len, 1)
        logits = logits.squeeze(-1)  # (batch_size, seq_len)
        
        # Ensure logits are float for regression
        if logits.dtype != torch.float32:
            logits = logits.float()
        
        loss = None
        if labels is not None:
            # Ensure labels are float type for regression
            if labels.dtype != torch.float32:
                labels = labels.float()
                
            # Use MSE loss for regression
            loss_fct = nn.MSELoss()
            
            # Only compute loss on non-ignored tokens (labels != -100.0)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1)[active_loss]
            
            # Filter out -100.0 labels
            active_labels = labels.view(-1)[active_loss]
            loss_mask = active_labels != -100.0
            
            if loss_mask.sum() > 0:
                filtered_logits = active_logits[loss_mask]
                filtered_labels = active_labels[loss_mask]
                loss = loss_fct(filtered_logits, filtered_labels)
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )