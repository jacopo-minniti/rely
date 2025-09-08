import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trainer import RegressionPRMTrainer
from model import RegressionPRMModel
from trl import PRMConfig

def main():
    # --- 1. Load Model and Tokenizer ---
    print("Loading model and tokenizer...")
    
    # Load the custom regression model
    model = RegressionPRMModel.from_pretrained("Qwen/Qwen2.5-Math-7B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")
    
    # Add special tokens if they don't exist
    step_separator_token = "<extra_0>"
    if step_separator_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [step_separator_token]})
        # Resize model embeddings if a new token was added
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load Dataset ---
    print("Loading dataset...")
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

    # --- 3. Configure Training Arguments ---
    print("Configuring training arguments...")
    training_args = PRMConfig(
        # Model and Data
        output_dir="./outputs/out",
        hub_model_id="jacopo-minniti/Qwen2.5-Math-7B-PUM-regression",
        max_length=4096,
        step_separator=step_separator_token,
        
        # Training Hyperparameters
        num_train_epochs=7,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.3,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        
        # Batching and Gradient
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        
        # Hardware & Performance
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        tf32=torch.cuda.is_available() and torch.cuda.is_tf32_supported(),
        
        # Logging and Saving
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        push_to_hub=False,
        
        # Misc
        remove_unused_columns=False # Important for custom dataset structures
    )

    # --- 4. Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = RegressionPRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, # This will be used for tokenization inside the trainer
    )

    # --- 5. Start Training ---
    print("Starting training...")
    trainer.train()
    print("Training finished!")

    # --- 6. Push to Hub ---
    print("Pushing final model to the Hub...")
    trainer.push_to_hub(token="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")
    print("Script finished successfully.")


if __name__ == "__main__":
    main()