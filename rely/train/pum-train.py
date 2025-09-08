import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification
)
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Configuration - Based on your provided parameters
# ---------------------------------------------------------------------------
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
    dataset_name = "half_entropy"
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
    micro_batch_size = 8
    eval_batch_size = 8
    num_epochs = 10
    optimizer = "adamw_torch_fused"
    lr_scheduler = "cosine"
    learning_rate = 5e-4
    weight_decay = 0.01
    warmup_ratio = 0.3
    
    # Hardware & Performance
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tf32 = torch.cuda.is_available() and torch.cuda.is_tf32_supported()
    gradient_checkpointing = True
    
    # Logging and Saving
    logging_steps = 5
    save_strategy = "epoch"
    
    # Hugging Face Hub
    hub_model_id = "jacopo-minniti/Qwen2.5-Math-7B-PUM-regression"
    hub_strategy = "end" # push at the end of training

# ---------------------------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------------------------

def process_and_tokenize(examples, tokenizer):
    """
    Processes each example into cumulative steps, tokenizes them, and creates
    token-level labels for regression. The label is applied only at the last
    <extra_0> token of each cumulative step.
    """
    # These will hold the processed data for the entire batch
    batch_input_ids, batch_attention_mask, batch_labels = [], [], []

    separator_token_id = tokenizer.convert_tokens_to_ids(ModelConfig.step_separator_token)
    
    prompts = examples["prompt"]
    completions = examples["completions"]
    scores = examples["labels"]

    for i in range(len(prompts)): # Iterate through each example in the batch
        user_prompt = prompts[i]
        
        for j in range(len(completions[i])): # Iterate through each step in the example
            # 1. Create the cumulative completion string
            cumulative_completion = ModelConfig.step_separator_token.join(completions[i][:j+1])

            # 2. Apply the chat template
            messages = [
                {"role": "system", "content": ModelConfig.MATH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": cumulative_completion}
            ]
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # 3. Tokenize the formatted text
            tokenized_inputs = tokenizer(
                formatted_text,
                truncation=True,
                padding="max_length",
                max_length=ModelConfig.sequence_len,
            )
            
            input_ids = tokenized_inputs["input_ids"]
            
            # 4. Create the token-level labels tensor
            # Initialize with -100 (ignore index for loss calculation)
            labels = np.full(len(input_ids), -100, dtype=np.float32)

            # Find the position of the *last* separator token
            # We search in reverse to find the last occurrence efficiently
            last_sep_idx = -1
            for k in range(len(input_ids) - 1, -1, -1):
                if input_ids[k] == separator_token_id:
                    last_sep_idx = k
                    break
            
            # If a separator token is found, assign the score to that position
            if last_sep_idx != -1:
                labels[last_sep_idx] = scores[i][j]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(tokenized_inputs["attention_mask"])
            batch_labels.append(labels)
            
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }

# ---------------------------------------------------------------------------
# Evaluation Metrics
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

# ---------------------------------------------------------------------------
# Main Training Script
# ---------------------------------------------------------------------------

def main():
    print("Starting Token Classification Reward Model Training for Regression...")

    # 1. Load Tokenizer
    print(f"Loading tokenizer: {ModelConfig.base_model}")
    tokenizer = ModelConfig.tokenizer_type.from_pretrained(ModelConfig.base_model)
    
    if ModelConfig.special_tokens:
        print(f"Adding special tokens: {ModelConfig.special_tokens}")
        tokenizer.add_special_tokens({"additional_special_tokens": ModelConfig.special_tokens})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and Preprocess Dataset
    print(f"Loading dataset from path: {ModelConfig.dataset_path}, name: {ModelConfig.dataset_name}")
    dataset = load_dataset(ModelConfig.dataset_path, ModelConfig.dataset_name)
    
    if ModelConfig.test_split not in dataset:
        print(f"'{ModelConfig.test_split}' split not found. Splitting 'train' into train/test (90/10).")
        train_test_split = dataset[ModelConfig.train_split].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({'train': train_test_split['train'], 'test': train_test_split['test']})

    print("Processing and tokenizing the dataset...")
    original_columns = dataset[ModelConfig.train_split].column_names
    tokenized_dataset = dataset.map(
        lambda examples: process_and_tokenize(examples, tokenizer),
        batched=True,
        remove_columns=original_columns,
        num_proc=os.cpu_count() # Use multiple processors for faster preprocessing
    )

    train_dataset = tokenized_dataset[ModelConfig.train_split]
    eval_dataset = tokenized_dataset[ModelConfig.test_split]
    
    print(f"Dataset prepared. Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
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
        save_strategy=ModelConfig.save_strategy,
        evaluation_strategy=ModelConfig.save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        push_to_hub=True,
        hub_model_id=ModelConfig.hub_model_id,
        hub_strategy=ModelConfig.hub_strategy,
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
        tokenizer=tokenizer,
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


if __name__ == "__main__":
    main()