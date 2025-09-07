import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from tqdm import tqdm

def preprocess_data_sparse(examples):
    """
    Preprocesses data using the sparse method (same as in training script).
    """
    processed_examples = {"text": [], "label": []}
    
    for i in range(len(examples["prompt"])):
        # The text to classify is the prompt + CoT up to the point of completion
        text = examples["prompt"][i] + examples["cut_cot"][i]
        
        # The label is the single boolean value inside the list
        label = int(examples["labels"][i][0])
        
        processed_examples["text"].append(text.strip())
        processed_examples["label"].append(label)
        
    return processed_examples

def evaluate_model_on_dataset(model, tokenizer, test_dataset, device):
    """
    Evaluate the model on the entire test dataset and return predictions and labels.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    # Process samples in batches for efficiency
    batch_size = 16
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
        batch_end = min(i + batch_size, len(test_dataset))
        batch_samples = test_dataset[i:batch_end]
        
        # Preprocess batch
        processed_batch = preprocess_data_sparse({
            "prompt": batch_samples["prompt"],
            "cut_cot": batch_samples["cut_cot"], 
            "labels": batch_samples["labels"]
        })
        
        texts = processed_batch["text"]
        labels = processed_batch["label"]
        
        # Tokenize batch
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predictions and probabilities
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels)
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def print_evaluation_statistics(predictions, labels, probabilities):
    """
    Print simplified evaluation statistics.
    """
    # Basic accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Precision, recall, F1 for each class and averages
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    # AUROC
    # Get probabilities for positive class (class 1)
    positive_probs = probabilities[:, 1]
    auroc = roc_auc_score(labels, positive_probs)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Confidence statistics
    max_probs = np.max(probabilities, axis=1)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Number of each class
    print(f"\nClass Distribution:")
    print(f"  Class 0 (Negative): {support[0]} samples")
    print(f"  Class 1 (Positive): {support[1]} samples")
    
    # Core metrics
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    print(f"AUROC: {auroc:.4f}")
    
    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              0     1")
    print(f"Actual    0  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"          1  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Confidence statistics
    print(f"\nConfidence Scores:")
    print(f"  Mean: {np.mean(max_probs):.4f}")
    print(f"  Min:  {np.min(max_probs):.4f}")
    print(f"  Max:  {np.max(max_probs):.4f}")

def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model and tokenizer from checkpoint
    model_path = "outputs/out"
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Move model to device
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("jacopo-minniti/uats-prm-nn-long-4", "sparse")
    test_dataset = dataset['test']
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Show a sample first
    first_sample = test_dataset[0]
    processed_sample = preprocess_data_sparse({
        "prompt": [first_sample["prompt"]],
        "cut_cot": [first_sample["cut_cot"]],
        "labels": [first_sample["labels"]]
    })

    print(f"\nSample input (first 30 chars):")
    print(f"{processed_sample['text'][0][:30]}...")
    print(f"Sample label: {processed_sample['label'][0]}")
    
    # Evaluate on entire test set
    predictions, labels, probabilities = evaluate_model_on_dataset(model, tokenizer, test_dataset, device)
    
    # Print comprehensive statistics
    print_evaluation_statistics(predictions, labels, probabilities)

if __name__ == "__main__":
    main()