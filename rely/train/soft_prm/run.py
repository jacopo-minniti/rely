import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
from trainer import RegressionPRMTrainer
from model import RegressionPRMModel
from trl import PRMConfig
import random

def main():
    # --- 1. Load Model and Tokenizer ---
    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    step_separator_token = "<extra_0>"
    if step_separator_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [step_separator_token]})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - DeepSpeed will handle initialization
    print("Loading model...")
    model = RegressionPRMModel.from_base_model(model_name, torch_dtype=torch.bfloat16)
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    # --- 2. Load Dataset ---
    print("Loading dataset...")
    train_dataset = load_dataset("jacopo-minniti/MATH-PUM-qwen2.5-1.5B", "variance", split="train")
    eval_dataset = load_dataset("jacopo-minniti/MATH-PUM-qwen2.5-1.5B", "variance", split="test")

    # --- 3. Configure Training Arguments ---
    print("Configuring training arguments...")
    training_args = PRMConfig(
        output_dir="./.cache/variance-regression-1.5B",
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
        logging_steps=5,  
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
    trainer = RegressionPRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        mask_zeros=False,
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