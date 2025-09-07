# prm_train.py
import argparse
import os
import yaml
import torch
from datasets import load_dataset
from trl import PRMConfig, PRMTrainer
from transformers import AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig

def main(config_path: str):
    """
    Main function to run PRM training from a YAML config file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # 1. Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print("Configuration loaded successfully:")
    print(config)

    # 2. Configure Quantization (optional)
    quantization_config = None
    if config.get('load_in_4bit'):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif config.get('load_in_8bit'):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # 3. Load Model and Tokenizer
    model_name = config['base_model']
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=config['num_labels'],
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 4. Add and handle special tokens
    if 'special_tokens' in config and 'additional_special_tokens' in config['special_tokens']:
        special_tokens_dict = {"additional_special_tokens": config['special_tokens']['additional_special_tokens']}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added special tokens: {special_tokens_dict['additional_special_tokens']}")
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print("Pad token set to EOS token.")

    # 5. Load Datasets
    # The config expects a list, so we'll take the first entry for train/test
    train_ds_config = config['datasets'][0]
    test_ds_config = config['test_datasets'][0]

    train_dataset = load_dataset(
        train_ds_config['path'], 
        name=train_ds_config.get('name'), 
        split=train_ds_config['split']
    )
    eval_dataset = load_dataset(
        test_ds_config['path'], 
        name=test_ds_config.get('name'), 
        split=test_ds_config['split']
    )
    print(f"Loaded train dataset: {train_ds_config['path']} with {len(train_dataset)} samples.")
    print(f"Loaded eval dataset: {test_ds_config['path']} with {len(eval_dataset)} samples.")

    # 6. Configure WandB (Weights & Biases) if specified
    if config.get('wandb_project'):
        os.environ['WANDB_PROJECT'] = config['wandb_project']
        report_to = "wandb"
    else:
        report_to = "none"

    # 7. Map YAML config to PRMConfig arguments
    training_args = PRMConfig(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['micro_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=float(config['learning_rate']),
        lr_scheduler_type=config['lr_scheduler'],
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        logging_steps=config['logging_steps'],
        save_strategy=config['save_strategy'],
        # eval_strategy is inferred from eval_steps or evals_per_epoch in more complex setups
        # For simplicity, we'll set it directly if you add it to your yaml.
        # Otherwise, the default 'no' will be used unless save_strategy is also 'epoch'.
        eval_strategy="epoch" if config['evals_per_epoch'] > 0 else "no",
        max_length=config['sequence_len'],
        remove_unused_columns=config.get('remove_unused_columns', False),
        gradient_checkpointing=config['gradient_checkpointing'],
        bf16=True if config.get('bf16') == 'auto' else False,
        fp16=config.get('fp16', False),
        tf32=config.get('tf32', False),
        report_to=report_to,
        run_name=config.get('wandb_name'),
        push_to_hub=True if config.get('hub_model_id') else False,
        hub_model_id=config.get('hub_model_id'),
        hub_strategy=config.get('hub_strategy', 'every_save'),
        # PRM-specific arguments
        step_separator=train_ds_config['step_separator'],
        train_on_last_step_only=train_ds_config['train_on_last_step_only'],
    )
    
    # 8. Initialize and run the Trainer
    trainer = PRMTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # 9. Save final model and push to hub
    if training_args.push_to_hub:
        print(f"Pushing final model to Hub: {training_args.hub_model_id}")
        trainer.push_to_hub()
    else:
        print(f"Saving final model to {config['output_dir']}")
        trainer.save_model(config['output_dir'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Process Reward Model from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)