import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from accelerate import Accelerator
import os

# Set environment variables for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Set NCCL timeout to handle slow communication
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes
os.environ["NCCL_BLOCKING_WAIT"] = "1"
# Reduce NCCL logging
os.environ["NCCL_DEBUG"] = "WARN"

from trainer import RegressionPRMTrainer
from model import RegressionPRMModel
from trl import PRMConfig

def main():
    accelerator = Accelerator()

    # --- 1. Load Model and Tokenizer ---
    with accelerator.main_process_first():
        if accelerator.is_main_process: print("Loading model and tokenizer...")
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")
        
        step_separator_token = "<extra_0>"
        if step_separator_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [step_separator_token]})

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - let DeepSpeed handle the initialization
    if accelerator.is_main_process: print("Loading model...")
    model = RegressionPRMModel.from_base_model("Qwen/Qwen2.5-Math-7B", dtype=torch.bfloat16)
    
    # Resize token embeddings if needed
    if step_separator_token not in tokenizer.get_vocab():
        model.resize_token_embeddings(len(tokenizer))

    # --- 2. Load Dataset ---
    with accelerator.main_process_first():
        if accelerator.is_main_process: print("Loading dataset...")
        train_dataset = load_dataset(
            "jacopo-minniti/MATH-PUM-qwen2.5-1.5B", 
            name="regression", 
            split="train"
        )
        eval_dataset = load_dataset(
            "jacopo-minniti/MATH-PUM-qwen2.5-1.5B",
            name="regression",
            split="test"
        )
        
        # For testing, limit dataset size to avoid memory and timeout issues
        if accelerator.is_main_process: 
            print(f"Original train dataset size: {len(train_dataset)}")
            print("Limiting dataset size for testing...")
        
        # Take only first 1000 samples for testing
        train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))
        
        if accelerator.is_main_process:
            print(f"Limited train dataset size: {len(train_dataset)}")
            print(f"Limited eval dataset size: {len(eval_dataset)}")

    # --- 3. Configure Training Arguments ---
    if accelerator.is_main_process: print("Configuring training arguments...")
    training_args = PRMConfig(
        output_dir="./outputs/out",
        hub_model_id="jacopo-minniti/Qwen2.5-Math-7B-PUM-regression",
        max_length=1024,  # Further reduced from 2048 to 1024 to save memory
        max_prompt_length=512,  # Reduced from 1024 to 512
        max_completion_length=512,  # Reduced from 1024 to 512
        train_on_last_step_only=False,  # Added missing parameter
        dataset_num_proc=4,  # Added missing parameter
        step_separator=step_separator_token,
        num_train_epochs=2,  # Reduced from 5 to 2 epochs to test
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        weight_decay=0.1,
        optim="adamw_torch_fused",
        per_device_train_batch_size=1,  # Keep at 1
        per_device_eval_batch_size=1,   # Keep at 1
        gradient_accumulation_steps=16,  # Increased from 8 to 16 to reduce communication frequency
        gradient_checkpointing=False,    # Disabled to avoid issues with custom model
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        tf32=torch.cuda.is_available() and torch.cuda.is_tf32_supported(),
        logging_steps=50,  # Increased from 10 to reduce logging overhead
        save_strategy="epoch",
        eval_strategy="epoch",
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,    # Disable pin memory to save GPU memory
        dataloader_num_workers=0,       # Reduce data loading overhead
        deepspeed="/scratch/jacopo04/rely/deepspeed_configs/zero3_bf16_optimized.json"  # Optimized DeepSpeed config
    )

    # --- 4. Initialize Trainer ---
    if accelerator.is_main_process: print("Initializing Trainer...")
    trainer = RegressionPRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # --- 5. Start Training ---
    if accelerator.is_main_process: print("Starting training...")
    trainer.train()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process: print("Training finished!")

    # --- 6. Push to Hub ---
    if accelerator.is_main_process:
        print("Pushing final model to the Hub...")
        trainer.push_to_hub("Qwen2.5-Math-7B-PUM-regression", token="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")
        print("Script finished successfully.")


if __name__ == "__main__":
    main()