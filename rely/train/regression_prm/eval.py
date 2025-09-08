import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import r2_score
from tqdm import tqdm
import argparse
from pathlib import Path

from model import RegressionPRMModel
from trainer import RegressionPRMTrainer
from trl import PRMConfig


def load_model_and_tokenizer(checkpoint_path: str):
    """Load the model and tokenizer from checkpoint."""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load tokenizer from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load the model from checkpoint
    model = RegressionPRMModel.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    
    return model, tokenizer


def tokenize_example(example, tokenizer, step_separator="<extra_0>", max_length=4096):
    """Tokenize a single example in the same format as training."""
    # Get the tokenization function from trainer
    tokenized = RegressionPRMTrainer.tokenize_row(
        features=example,
        tokenizer=tokenizer,
        step_separator=step_separator,
        max_length=max_length,
        max_prompt_length=None,
        max_completion_length=None,
        train_on_last_step_only=False,
        is_eval=True,
    )
    return tokenized


def evaluate_model(model, tokenizer, dataset, device, max_examples=None):
    """Evaluate the model on the dataset and return predictions and labels."""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    
    # Limit examples if specified
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} examples...")
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                # Tokenize the example
                tokenized = tokenize_example(example, tokenizer)
                
                # Convert to tensors and move to device
                input_ids = torch.tensor([tokenized["input_ids"]], device=device)
                labels_tensor = torch.tensor([tokenized["labels"]], device=device, dtype=torch.float32)
                
                # Create attention mask and cast it to the model's dtype
                attention_mask = (input_ids != tokenizer.pad_token_id).to(model.dtype)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = outputs.logits.float().cpu().numpy()[0]  # (seq_len,)
                labels = labels_tensor.cpu().numpy()[0]  # (seq_len,)
                
                # Extract only the non-ignored labels and corresponding predictions
                valid_mask = labels != -100.0
                if valid_mask.sum() > 0:
                    valid_predictions = predictions[valid_mask]
                    valid_labels = labels[valid_mask]
                    
                    all_predictions.extend(valid_predictions.tolist())
                    all_labels.extend(valid_labels.tolist())
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
    
    return np.array(all_predictions), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Regression PRM model")
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="/scratch/jacopo04/outputs/out/checkpoint-7650",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--max_examples", 
        type=int, 
        default=None,
        help="Maximum number of examples to evaluate (default: all)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    print("="*60)
    print("REGRESSION PRM MODEL EVALUATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Max examples: {args.max_examples or 'All'}")
    print("="*60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_dataset(
        "jacopo-minniti/MATH-PUM-qwen2.5-1.5B", 
        name="regression", 
        split="test"
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Evaluate model
    predictions, labels = evaluate_model(
        model=model,
        tokenizer=tokenizer, 
        dataset=test_dataset,
        device=args.device,
        max_examples=args.max_examples
    )
    
    # Calculate metrics
    if len(predictions) > 0:
        r2 = r2_score(labels, predictions)
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Number of predictions: {len(predictions)}")
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print("="*60)
        
        # Additional statistics
        print(f"\nLabel statistics:")
        print(f"  Mean: {np.mean(labels):.4f}")
        print(f"  Std:  {np.std(labels):.4f}")
        print(f"  Min:  {np.min(labels):.4f}")
        print(f"  Max:  {np.max(labels):.4f}")
        
        print(f"\nPrediction statistics:")
        print(f"  Mean: {np.mean(predictions):.4f}")
        print(f"  Std:  {np.std(predictions):.4f}")
        print(f"  Min:  {np.min(predictions):.4f}")
        print(f"  Max:  {np.max(predictions):.4f}")
        
    else:
        print("No valid predictions found!")


if __name__ == "__main__":
    main()
