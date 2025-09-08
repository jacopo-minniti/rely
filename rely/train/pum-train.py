import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from process_dataset import load_cached_dataset, process_and_cache_dataset


class ModelConfig:
    # Model and Tokenizer
    base_model = "Qwen/Qwen2.5-Math-7B"
    model_type = AutoModelForTokenClassification
    tokenizer_type = AutoTokenizer
    
    # Quantization - Set to False as per your config
    load_in_8bit = False
    load_in_4bit = False
    
    # Dataset
    dataset_path = "jacopo-minniti/MATH-PUM-qwen2.5-1.5B"
    dataset_name = "regression"
    train_split = "train"
    test_split = "test"
    
    # Special Tokens
    step_separator_token = "<extra_0>"
    special_tokens = [step_separator_token]
    
    # System Prompt for Chat Template
    MATH_SYSTEM_PROMPT = """The following are questions about mathematics. Think step by step and provide your answer in the format '\\boxed{}' with inside your final answer. The final answers should either be a number (in digits) or a latex expression."""
    
    # Training Hyperparameters
    output_dir = "./outputs/out"
    sequence_len = 4096
    gradient_accumulation_steps = 4
    micro_batch_size = 1
    eval_batch_size = 1
    num_epochs = 7
    optimizer = "adamw_torch_fused"
    lr_scheduler = "cosine"
    learning_rate = 1e-4
    weight_decay = 0.01
    warmup_ratio = 0.3
    
    # Hardware & Performance
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tf32 = torch.cuda.is_available() and torch.cuda.is_tf32_supported()
    gradient_checkpointing = True
    
    # Logging and Saving
    logging_steps = 5
    
    # Cache settings
    cache_dir = "/scratch/jacopo04/.cache/datasets"
    processed_dataset_name = "processed_pum_regression"
    
    # Hugging Face Hub
    hub_model_id = "jacopo-minniti/Qwen2.5-Math-7B-PUM-regression"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Computes regression metrics for token classification, ignoring -100 labels.
    """
    predictions, labels = eval_pred
    # Predictions are logits, shape (batch_size, seq_len, 1). Squeeze the last dim.
    predictions = predictions.squeeze(-1)
    
    # Filter out ignored indices (-100)
    true_predictions = predictions[labels != -100]
    true_labels = labels[labels != -100]

    # If there are no true labels in the batch, return 0 for all metrics
    if len(true_labels) == 0:
        return {"mse": 0, "mae": 0, "r2": 0}

    mse = mean_squared_error(true_labels, true_predictions)
    mae = mean_absolute_error(true_labels, true_predictions)
    r2 = r2_score(true_labels, true_predictions)
    
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }


if __name__ == "__main__":
    print("Starting Token Classification Reward Model Training for Regression...")

    # 1. Load Tokenizer
    print(f"Loading tokenizer: {ModelConfig.base_model}")
    tokenizer = ModelConfig.tokenizer_type.from_pretrained(ModelConfig.base_model)
    
    if ModelConfig.special_tokens:
        print(f"Adding special tokens: {ModelConfig.special_tokens}")
        tokenizer.add_special_tokens({"additional_special_tokens": ModelConfig.special_tokens})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Preprocessed Dataset from Cache
    print("Loading preprocessed dataset from cache...")
    try:
        tokenized_dataset = load_cached_dataset()
        print("Successfully loaded cached dataset!")
    except FileNotFoundError:
        print("Cached dataset not found. Processing dataset for the first time...")
        tokenized_dataset = process_and_cache_dataset()
    
    train_dataset = tokenized_dataset[ModelConfig.train_split]
    eval_dataset = tokenized_dataset[ModelConfig.test_split]
    
    print(f"Dataset loaded. Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # 3. Load Model
    print(f"Loading base model for Token Classification: {ModelConfig.base_model}")

    model = ModelConfig.model_type.from_pretrained(
        ModelConfig.base_model,
        num_labels=1,  # Critical for regression!
        torch_dtype=torch.bfloat16 if ModelConfig.bf16 else torch.float32,
    )
    
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # 4. Configure Training Arguments
    print("Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=ModelConfig.output_dir,
        num_train_epochs=ModelConfig.num_epochs,
        per_device_train_batch_size=ModelConfig.micro_batch_size,
        per_device_eval_batch_size=ModelConfig.eval_batch_size,
        gradient_accumulation_steps=ModelConfig.gradient_accumulation_steps,
        gradient_checkpointing=ModelConfig.gradient_checkpointing,
        optim=ModelConfig.optimizer,
        learning_rate=ModelConfig.learning_rate,
        lr_scheduler_type=ModelConfig.lr_scheduler,
        warmup_ratio=ModelConfig.warmup_ratio,
        weight_decay=ModelConfig.weight_decay,
        bf16=ModelConfig.bf16,
        tf32=ModelConfig.tf32,
        logging_dir=f"{ModelConfig.output_dir}/logs",
        logging_steps=ModelConfig.logging_steps,
        evaluation_strategy="steps",
        eval_steps=ModelConfig.logging_steps,
        save_strategy="steps",
        save_steps=ModelConfig.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        push_to_hub=True,
        hub_model_id=ModelConfig.hub_model_id,
        report_to="wandb",
    )
    
    # Data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 5. Initialize Trainer and Start Training
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Training finished. Saving model...")
    trainer.save_model(ModelConfig.output_dir)
    
    print("Pushing model to the Hugging Face Hub...")
    trainer.push_to_hub()
    
    print("Script finished successfully.")