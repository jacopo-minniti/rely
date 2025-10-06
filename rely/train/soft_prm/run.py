import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
from trainer import SoftClassificationPRMTrainer
from model import SoftClassificationPRMModel
from trl import PRMConfig
import random

def main():
    # --- 1. Load Model and Tokenizer ---
    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    step_separator_token = "<extra_0>"
    if step_separator_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [step_separator_token]})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - DeepSpeed will handle initialization
    print("Loading model...")
    model = SoftClassificationPRMModel.from_base_model(model_name, torch_dtype=torch.bfloat16)
    
    # Resize token embeddings if needed
    if step_separator_token not in tokenizer.get_vocab():
        model.resize_token_embeddings(len(tokenizer))

    # --- 2. Load Dataset ---
    print("Loading dataset...")
    train_dataset = load_dataset(
        "jacopo-minniti/MATH-PUM-qwen2.5-1.5B", 
        name="variance_v2", 
        split="train"
    )
    eval_dataset = load_dataset(
        "jacopo-minniti/MATH-PUM-qwen2.5-1.5B",
        name="variance_v2",
        split="test"
    )
    
    print(f"Original train dataset size: {len(train_dataset)}")
    print("Downsampling all-0.0 label trajectories...")

    # Split into all-0.0 and rest
    all_zero_indices = []
    rest_indices = []
    for i, example in enumerate(train_dataset):
        labels = example['labels']
        if set(labels) == {0.0} or set(labels) == {0}:
            all_zero_indices.append(i)
        else:
            rest_indices.append(i)

    # Downsample all-0.0 to match the number of rest (or use all if fewer available)
    random.seed(42)
    sample_size = min(len(all_zero_indices), len(rest_indices))
    downsampled_zero_indices = random.sample(all_zero_indices, sample_size)

    # Combine and sort indices for reproducibility
    final_indices = sorted(downsampled_zero_indices + rest_indices)
    train_dataset = train_dataset.select(final_indices)
    
    print(f"Filtered train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # --- 3. Configure Training Arguments ---
    print("Configuring training arguments...")
    training_args = PRMConfig(
        output_dir="./.cache/variance_downsample_model",
        hub_model_id="jacopo-minniti/Qwen2.5-Math-1.5B-PUM-variance",
        max_length=4096,
        train_on_last_step_only=False,
        step_separator=step_separator_token,
        num_train_epochs=3, 
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        weight_decay=0.1,
        optim="adamw_torch",  # Changed from adamw_torch_fused to avoid compilation issues
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4, 
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,    # Enable gradient checkpointing for memory savings
        bf16=True,
        logging_steps=10,  
        save_strategy="epoch",
        eval_strategy="epoch",
        push_to_hub=False,
        remove_unused_columns=False,
        # Memory optimization settings
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # Parallel data loading
        # Communication optimization
        ddp_find_unused_parameters=False,  # Faster training when all parameters are used
    )

    # --- 4. Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = SoftClassificationPRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        loss="bce",  # Can be "bce" (default) or "mse"
    )

    # --- 5. Start Training ---
    print("Starting training...")
    trainer.train()
    print("Training finished!")

    # --- 6. Push to Hub ---
    # print("Pushing final model to the Hub...")
    # trainer.push_to_hub("Qwen2.5-Math-1.5B-PUM-cwe", token="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")
    # print("Script finished successfully.")


if __name__ == "__main__":
    main()
