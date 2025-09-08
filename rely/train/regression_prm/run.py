import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from accelerate import Accelerator

from trainer import RegressionPRMTrainer
from model import RegressionPRMModel
from trl import PRMConfig

def main():
    accelerator = Accelerator()

    # --- 1. Load Model and Tokenizer ---
    with accelerator.main_process_first():
        if accelerator.is_main_process:
            print("Loading model and tokenizer...")
        
        model = RegressionPRMModel.from_base_model("Qwen/Qwen2.5-Math-1.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
        
        step_separator_token = "<extra_0>"
        if step_separator_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [step_separator_token]})
            model.resize_token_embeddings(len(tokenizer))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load Dataset ---
    with accelerator.main_process_first():
        if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        print("Configuring training arguments...")
    training_args = PRMConfig(
        output_dir="./outputs/out",
        hub_model_id="jacopo-minniti/Qwen2.5-Math-7B-PUM-regression",
        max_length=4096,
        step_separator=step_separator_token,
        num_train_epochs=5,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        weight_decay=0.1,
        optim="adamw_torch_fused",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        tf32=torch.cuda.is_available() and torch.cuda.is_tf32_supported(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        push_to_hub=False,
        remove_unused_columns=False
    )

    # --- 4. Initialize Trainer ---
    if accelerator.is_main_process:
        print("Initializing Trainer...")
    trainer = RegressionPRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # --- 5. Start Training ---
    if accelerator.is_main_process:
        print("Starting training...")
    trainer.train()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Training finished!")

    # --- 6. Push to Hub ---
    if accelerator.is_main_process:
        print("Pushing final model to the Hub...")
        trainer.push_to_hub("Qwen2.5-Math-7B-PUM-regression", token="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")
        print("Script finished successfully.")


if __name__ == "__main__":
    main()