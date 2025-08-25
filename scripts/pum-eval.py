import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
import torch.nn.functional as F
from rely.utils import MATH_SYSTEM_PROMPT
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

skipped = 0

def make_step_rewards(logits, token_masks):
    # Get probabilities from logits
    probabilities = F.softmax(logits, dim=-1)  # shape: [batch, seq_len, 2] for binary classification
    
    # Apply token masks to only consider step separator positions
    masked_probs = probabilities * token_masks.unsqueeze(-1)  # [batch, seq_len, 2]
    
    all_scores_res = []
    for i in range(probabilities.size(0)):  # for each batch
        sample_probs = masked_probs[i]  # [seq_len, 2]
        
        # Find positions where we have step separators
        step_positions = torch.where(token_masks[i])[0]
        
        # For binary classification, extract the positive class probability (index 1)
        step_rewards = []
        for pos in step_positions:
            pos_probs = sample_probs[pos]  # [2]
            positive_prob = pos_probs[1].item()  # probability of positive class
            step_rewards.append(positive_prob)
        
        all_scores_res.append(step_rewards)
    
    return all_scores_res


model_name = "jacopo-minniti/PUM-Qwen2.5-Math-7B-PP-1"
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()


def process_sample(sample, tokenizer, model):
    """Process a single sample and return predictions and labels"""
    
    messages = [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": sample['prompt']},
        {"role": "assistant", "content": "\n\n".join(sample['completions']) + "\n\n"},
    ]
    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )

    input_ids = tokenizer.encode(
        conversation_str, 
        return_tensors="pt", 
    ).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # Find tokens that contain step separators (\n\n)
    token_masks = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, token_id in enumerate(input_ids[0]):
        decoded = tokenizer.decode([token_id])
        if '\n\n' in decoded:
            token_masks[0, i] = True

    step_rewards = make_step_rewards(outputs.logits, token_masks)
    
    # Get the predictions (probabilities for positive class)
    predictions = step_rewards[0] if step_rewards and step_rewards[0] else []
    
    # Get the labels
    labels = sample['labels']
    
    # Ensure we have the same number of predictions and labels
    min_len = min(len(predictions), len(labels))
    if len(predictions) != len(labels):
        global skipped
        skipped += 1
        return None, None
    
    predictions = predictions[:min_len]
    labels = labels[:min_len]
    
    return predictions, labels


def evaluate_dataset(dataset, tokenizer, model):
    """Evaluate the model on the entire dataset"""
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print(f"Evaluating {len(dataset)} samples...")
    
    for sample in tqdm(dataset):
        predictions, labels = process_sample(sample, tokenizer, model)
        
        if predictions and labels:  # Only add if we have valid predictions and labels
            all_probabilities.extend(predictions)  # Raw probabilities for AUROC
            all_predictions.extend([1 if p > 0.5 else 0 for p in predictions])  # Binary predictions
            all_labels.extend([1 if label else 0 for label in labels])  # Convert boolean to int
    
    if not all_predictions:
        print("No valid predictions found!")
        return None
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auroc = roc_auc_score(all_labels, all_probabilities)
    
    # Calculate diagnostic statistics
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    unique_predictions, pred_counts = np.unique(all_predictions, return_counts=True)
    
    # Label distribution
    label_distribution = dict(zip(unique_labels, label_counts))
    pred_distribution = dict(zip(unique_predictions, pred_counts))
    
    # Probability statistics
    prob_mean = np.mean(all_probabilities)
    prob_std = np.std(all_probabilities)
    prob_min = np.min(all_probabilities)
    prob_max = np.max(all_probabilities)
    
    # Prediction confidence (how far from 0.5 threshold)
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
            'mean': prob_mean,
            'std': prob_std,
            'min': prob_min,
            'max': prob_max,
            'avg_confidence': avg_confidence
        }
    }

# Load model
model_name = "jacopo-minniti/PUM-Qwen2.5-Math-7B-PP-1"
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

dataset = load_dataset("jacopo-minniti/MATH-PUM-qwen2.5-1.5B", "pp-1", split="test")
# seed=42
dataset = dataset.shuffle().select(range(1000))

# Evaluate the dataset
results = evaluate_dataset(dataset, tokenizer, model)

if results:
    print(f"\nEvaluation Results:")
    print(f"Number of step predictions: {results['num_samples']}")
    print(f"Skipped samples due to length mismatch: {skipped}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"AUROC: {results['auroc']:.4f}")
    
    print(f"\n--- Diagnostic Statistics ---")
    
    # Label distribution
    print(f"Ground Truth Label Distribution:")
    for label, count in results['label_distribution'].items():
        percentage = count / results['num_samples'] * 100
        print(f"  Label {label}: {count} ({percentage:.1f}%)")
    
    # Prediction distribution
    print(f"\nModel Prediction Distribution:")
    for pred, count in results['prediction_distribution'].items():
        percentage = count / results['num_samples'] * 100
        print(f"  Prediction {pred}: {count} ({percentage:.1f}%)")
    
    # Probability statistics
    prob_stats = results['probability_stats']
    print(f"\nProbability Statistics:")
    print(f"  Mean probability: {prob_stats['mean']:.4f}")
    print(f"  Std deviation: {prob_stats['std']:.4f}")
    print(f"  Min probability: {prob_stats['min']:.4f}")
    print(f"  Max probability: {prob_stats['max']:.4f}")
    print(f"  Average confidence (distance from 0.5): {prob_stats['avg_confidence']:.4f}")
    
    # Check for potential issues
    print(f"\n--- Potential Issues Check ---")
    
    # Check if model is predicting only one class
    if len(results['prediction_distribution']) == 1:
        only_class = list(results['prediction_distribution'].keys())[0]
        print(f"⚠️  WARNING: Model is predicting only class {only_class}!")
    else:
        print("✓ Model is predicting both classes")
    
    # Check for extreme probability bias
    if prob_stats['mean'] > 0.8:
        print(f"⚠️  WARNING: Mean probability is very high ({prob_stats['mean']:.3f}) - model may be overconfident in positive class")
    elif prob_stats['mean'] < 0.2:
        print(f"⚠️  WARNING: Mean probability is very low ({prob_stats['mean']:.3f}) - model may be overconfident in negative class")
    else:
        print(f"✓ Mean probability seems reasonable ({prob_stats['mean']:.3f})")
    
    # Check for low confidence/variance
    if prob_stats['std'] < 0.1:
        print(f"⚠️  WARNING: Very low probability variance ({prob_stats['std']:.3f}) - model may not be well calibrated")
    else:
        print(f"✓ Probability variance seems reasonable ({prob_stats['std']:.3f})")
    
    # Check if predictions are too close to threshold
    if prob_stats['avg_confidence'] < 0.1:
        print(f"⚠️  WARNING: Low average confidence ({prob_stats['avg_confidence']:.3f}) - many predictions near 0.5 threshold")
    else:
        print(f"✓ Average confidence seems reasonable ({prob_stats['avg_confidence']:.3f})")
        
else:
    print("Evaluation failed - no valid predictions found.")
