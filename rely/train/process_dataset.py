import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np


class ModelConfig:
    # Model and Tokenizer
    base_model = "Qwen/Qwen2.5-Math-7B"
    tokenizer_type = AutoTokenizer
    
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
    sequence_len = 4096
    
    # Cache settings
    cache_dir = "/scratch/jacopo04/.cache/datasets"
    processed_dataset_name = "processed_pum_regression"


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


def process_and_cache_dataset():
    """
    Processes the dataset once and saves it to cache for future use.
    """
    print("Starting dataset preprocessing and caching...")
    
    # Create cache directory if it doesn't exist
    os.makedirs(ModelConfig.cache_dir, exist_ok=True)
    
    # Check if processed dataset already exists
    cache_path = os.path.join(ModelConfig.cache_dir, ModelConfig.processed_dataset_name)
    if os.path.exists(cache_path):
        print(f"Processed dataset already exists at {cache_path}")
        print("Loading cached dataset...")
        return load_dataset(cache_path)
    
    # 1. Load Tokenizer
    print(f"Loading tokenizer: {ModelConfig.base_model}")
    tokenizer = ModelConfig.tokenizer_type.from_pretrained(ModelConfig.base_model)
    
    if ModelConfig.special_tokens:
        print(f"Adding special tokens: {ModelConfig.special_tokens}")
        tokenizer.add_special_tokens({"additional_special_tokens": ModelConfig.special_tokens})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Raw Dataset
    print(f"Loading dataset from path: {ModelConfig.dataset_path}, name: {ModelConfig.dataset_name}")
    dataset = load_dataset(ModelConfig.dataset_path, ModelConfig.dataset_name)
    
    if ModelConfig.test_split not in dataset:
        print(f"'{ModelConfig.test_split}' split not found. Splitting 'train' into train/test (90/10).")
        train_test_split = dataset[ModelConfig.train_split].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({'train': train_test_split['train'], 'test': train_test_split['test']})

    # 3. Process and Tokenize Dataset
    print("Processing and tokenizing the dataset...")
    original_columns = dataset[ModelConfig.train_split].column_names
    tokenized_dataset = dataset.map(
        lambda examples: process_and_tokenize(examples, tokenizer),
        batched=True,
        remove_columns=original_columns,
    )

    print(f"Dataset processed. Train samples: {len(tokenized_dataset[ModelConfig.train_split])}, "
          f"Eval samples: {len(tokenized_dataset[ModelConfig.test_split])}")
    
    # 4. Save to cache
    print(f"Saving processed dataset to cache: {cache_path}")
    tokenized_dataset.save_to_disk(cache_path)
    
    print("Dataset preprocessing and caching completed successfully!")
    return tokenized_dataset


def load_cached_dataset():
    """
    Loads the cached processed dataset.
    """
    cache_path = os.path.join(ModelConfig.cache_dir, ModelConfig.processed_dataset_name)
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cached dataset not found at {cache_path}. "
            f"Please run process_and_cache_dataset() first."
        )
    
    print(f"Loading cached dataset from: {cache_path}")
    return load_dataset(cache_path)


if __name__ == "__main__":
    print("Processing and caching dataset...")
    processed_dataset = process_and_cache_dataset()
    
    print(f"Dataset cached successfully!")
    print(f"Train samples: {len(processed_dataset['train'])}")
    print(f"Test samples: {len(processed_dataset['test'])}")
    print(f"Cache location: {os.path.join(ModelConfig.cache_dir, ModelConfig.processed_dataset_name)}")
