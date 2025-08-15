import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# accelerate launch prm-train.py --method sparse --dataset jacopo-minniti/uats-prm-nn-long-4 --dataset_subset sparse

os.environ["WANDB_PROJECT"] = "value-model"

class ModelConfig:
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    OUTPUT_DIR = "./prm-no-infiltration"


def preprocess_data(examples, method: str):
    """
    Transforms the raw dataset into classification examples based on the chosen method.
    """
    processed_examples = {"text": [], "label": []}

    if method == 'fill':
        # This is the original logic for the 'fill' (dense) dataset format.
        for i in range(len(examples["prompt"])):
            prompt = examples["prompt"][i]
            completions = examples["completions"][i]
            labels = examples["labels"][i]
            
            history = ""
            for step, label in zip(completions, labels):
                # Input is the prompt + all previous steps + current step.
                current_text = prompt + "\n\n" + history + step
                processed_examples["text"].append(current_text.strip())
                processed_examples["label"].append(int(label))
                
                # Update history with the specified separator.
                history += step + "\n\n"
    
    elif method == 'sparse':
        for i in range(len(examples["prompt"])):
            text = examples["prompt"][i].strip() + "\n" + examples["cut_cot"][i].strip()

            # The label is the single boolean value inside the list.
            label = int(examples["labels"][i][0])
            
            processed_examples["text"].append(text.strip())
            processed_examples["label"].append(label)
            
    return processed_examples

def tokenize_function(examples, tokenizer):
    """Tokenize the text data."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=8192,
    )

def main():
    """Main function to run the PRM training."""
    
    # ## NEW ## - Setup argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Train a PRM/Value Model.")
    parser.add_argument("--method", type=str, required=True, choices=['fill', 'sparse'],
                        help="The dataset processing method ('fill' or 'sparse').")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the local training dataset folder or file.")
    parser.add_argument("--dataset_subset", type=str, default=None,
                        help="Optional subset of the dataset to use")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    raw_dataset = load_dataset(args.dataset, args.dataset_subset)

    processed_train_dataset = raw_dataset['train'].map(
        lambda exs: preprocess_data(exs, method=args.method),
        batched=True,
        batch_size=10,
        remove_columns=raw_dataset['train'].column_names,
    )
    processed_test_dataset = raw_dataset['test'].map(
        lambda exs: preprocess_data(exs, method=args.method),
        batched=True,
        batch_size=10,
        remove_columns=raw_dataset['test'].column_names,
    )
    
    tokenized_train_dataset = processed_train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    tokenized_test_dataset = processed_test_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        ModelConfig.MODEL_NAME, num_labels=2
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=ModelConfig.OUTPUT_DIR,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
        max_grad_norm=1.0,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        report_to="wandb",
        push_to_hub=False,
        save_strategy="steps",
        save_steps=1500,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    trainer.train()

    trainer.save_model(ModelConfig.OUTPUT_DIR)
    print(f"Model saved to {ModelConfig.OUTPUT_DIR}")

if __name__ == "__main__":
    main()