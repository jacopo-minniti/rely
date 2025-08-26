import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from rely.utils import MATH_SYSTEM_PROMPT

def create_collate_fn(tokenizer, separator_token="<extra_0>"):
    """
    Creates a collate function to process batches of data.
    This function tokenizes the input, finds the exact positions of the separator tokens,
    and pads the sequences for batch processing. This is the robust method.
    """
    # Pre-tokenize the separator to get its ID
    separator_id = tokenizer.encode(separator_token, add_special_tokens=False)[0]

    def collate_fn(batch):
        all_input_ids = []
        all_labels = []
        all_score_indices = []

        for sample in batch:
            # 1. Apply the chat template to format the conversation
            messages = [
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": sample['prompt']},
                # Join completions with the separator token
                {"role": "assistant", "content": separator_token.join(sample['completions']) + separator_token},
            ]
            conversation_str = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 2. Tokenize the entire conversation string
            input_ids = tokenizer.encode(conversation_str, add_special_tokens=False)
            
            # 3. Find the exact indices of the separator token
            # This is the robust part: we find indices by ID, not by decoding strings
            score_indices = [i for i, token_id in enumerate(input_ids) if token_id == separator_id]

            # 4. Important Check: Ensure the number of found separators matches the number of labels
            # This check replaces the old "skipped" logic.
            if len(score_indices) == len(sample['labels']):
                all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                all_labels.append(sample['labels'])
                all_score_indices.append(score_indices)
            # Silently skip if there's a mismatch, though this should be rare now.
            else:
                print("Skipped one sample...")

        if not all_input_ids:
            return None

        # 5. Pad the sequences to the same length for batching
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        attention_mask = (padded_input_ids != tokenizer.pad_token_id)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': all_labels,
            'score_indices': all_score_indices
        }
        
    return collate_fn


def evaluate_dataset(dataset, tokenizer, model, batch_size=8):
    """
    Evaluate the model on the entire dataset using the robust batching method.
    """
    # Use the robust collate function with a DataLoader
    collate_fn = create_collate_fn(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print(f"Evaluating {len(dataset)} samples with batch size {batch_size}...")
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: # Skip empty batches that might result from filtering
                continue

            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits # Shape: [batch_size, seq_len, num_classes]

            # Get probabilities for the relevant tokens
            probabilities = F.softmax(logits, dim=-1)

            # Iterate through each sample in the batch
            for i in range(len(batch['labels'])):
                score_indices = batch['score_indices'][i]
                labels = batch['labels'][i]
                
                # Extract the probabilities ONLY at the separator token locations
                # This is precise because we pre-calculated the indices.
                step_probs = probabilities[i, score_indices, :] # Shape: [num_steps, num_classes]
                
                # Get the probability of the positive class (class 1)
                positive_probs = step_probs[:, 1].cpu().tolist()
                
                all_probabilities.extend(positive_probs)
                all_predictions.extend([1 if p > 0.5 else 0 for p in positive_probs])
                all_labels.extend([1 if label else 0 for label in labels])

    if not all_predictions:
        print("No valid predictions found!")
        return None
    
    # --- The rest of the function is for calculating and printing metrics, same as before ---
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auroc = roc_auc_score(all_labels, all_probabilities)
    
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    unique_predictions, pred_counts = np.unique(all_predictions, return_counts=True)
    
    label_distribution = dict(zip(unique_labels, label_counts))
    pred_distribution = dict(zip(unique_predictions, pred_counts))
    
    prob_mean = np.mean(all_probabilities)
    prob_std = np.std(all_probabilities)
    prob_min = np.min(all_probabilities)
    prob_max = np.max(all_probabilities)
    
    prob_distances = np.abs(np.array(all_probabilities) - 0.5)
    avg_confidence = np.mean(prob_distances)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'auroc': auroc,
        'num_samples': len(all_predictions),
        'label_distribution': label_distribution,
        'prediction_distribution': pred_distribution,
        'probability_stats': {
            'mean': prob_mean, 'std': prob_std, 'min': prob_min,
            'max': prob_max, 'avg_confidence': avg_confidence
        }
    }

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Model and Tokenizer
    model_name = "jacopo-minniti/Qwen2.5-Math-7B-PUM-nn"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Add a pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        device_map=device, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Load Dataset
    dataset = load_dataset("jacopo-minniti/MATH-PUM-qwen2.5-1.5B", "pp-1", split="test")
    dataset = dataset.shuffle(seed=42).select(range(1000))

    # 3. Evaluate the dataset with the new robust function
    results = evaluate_dataset(dataset, tokenizer, model, batch_size=24) # Adjust batch size based on VRAM

    # 4. Print Results
    if results:
        print(f"\nEvaluation Results:")
        print(f"Number of step predictions: {results['num_samples']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"AUROC: {results['auroc']:.4f}")
        
        print(f"\n--- Diagnostic Statistics ---")
        
        print(f"Ground Truth Label Distribution:")
        for label, count in results['label_distribution'].items():
            percentage = count / results['num_samples'] * 100
            print(f"  Label {label}: {count} ({percentage:.1f}%)")
        
        print(f"\nModel Prediction Distribution:")
        for pred, count in results['prediction_distribution'].items():
            percentage = count / results['num_samples'] * 100
            print(f"  Prediction {pred}: {count} ({percentage:.1f}%)")
        
        prob_stats = results['probability_stats']
        print(f"\nProbability Statistics:")
        print(f"  Mean probability: {prob_stats['mean']:.4f}")
        print(f"  Std deviation: {prob_stats['std']:.4f}")
        print(f"  Min probability: {prob_stats['min']:.4f}")
        print(f"  Max probability: {prob_stats['max']:.4f}")
        print(f"  Average confidence (distance from 0.5): {prob_stats['avg_confidence']:.4f}")
    else:
        print("Evaluation failed - no valid predictions found.")
