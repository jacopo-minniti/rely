# rely/evaluate/score_self_consistency.py

import os
import json
import argparse
from collections import Counter
import math
import re

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from rely.utils import normalize_answer, MATH_SYSTEM_PROMPT, prompt_pattern
from rely.train.soft_prm.model import SoftClassificationPRMModel

# --- PUM Model Loading and Uncertainty Calculation ---

def load_pum_model(model_path):
    """Loads the PUM model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = SoftClassificationPRMModel.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = model.to(dtype=torch.bfloat16)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    
    print("PUM uncertainty model loaded.")
    return model, tokenizer, device

@torch.no_grad()
def get_trace_uncertainties(model, tokenizer, device, question, solutions):
    """
    Calculates the mean uncertainty for a batch of solution traces.
    """
    conversation_strs = []
    step_sep_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]

    for solution in solutions:
        assistant_response = solution['solution_path']
        steps = [s.strip() for s in assistant_response.split('''

''') if s.strip()]
        # Ensure there's at least one step for the <extra_0> token
        if not steps:
            steps = [""]
        
        formatted_content = "<extra_0>".join(steps) + "<extra_0>"
        
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": formatted_content}
        ]
        conv_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        conversation_strs.append(conv_str)

    if not conversation_strs:
        return []

    all_uncertainties = []
    batch_size = 8 

    for i in range(0, len(conversation_strs), batch_size):
        batch_conversations = conversation_strs[i:i + batch_size]
        try:
            inputs = tokenizer(batch_conversations, return_tensors="pt", padding=True, truncation=True, max_length=7000).to(device)
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            
            # Direct sigmoid probabilities from the custom model
            uncertainty_probs = outputs.logits
            token_masks = (inputs.input_ids == step_sep_id)
            
            has_separator = token_masks.any(dim=1)
            default_uncertainty = torch.tensor(0.5, device=device)
            
            # Calculate mean uncertainty ("average" method)
            masked_probs = uncertainty_probs * token_masks.float()
            sums = masked_probs.sum(dim=1)
            counts = token_masks.sum(dim=1).clamp(min=1)
            calculated_uncertainties = sums / counts
            
            final_uncertainties = torch.where(has_separator, calculated_uncertainties, default_uncertainty)
            all_uncertainties.extend(final_uncertainties.cpu().tolist())

            del inputs, outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warning: Error during batch uncertainty calculation: {e}")
            # Add default uncertainty for the failed batch
            all_uncertainties.extend([0.5] * len(batch_conversations))

    return all_uncertainties


# --- Evaluation Logic ---

def get_majority_vote(solutions):
    """Calculates majority vote from a list of solutions."""
    valid_answers = [s['final_answer'] for s in solutions if s.get('final_answer') and str(s['final_answer']).strip() and s['final_answer'] != "Not found"]
    if not valid_answers:
        return "N/A"
    
    normalized_answers = [normalize_answer(ans) for ans in valid_answers]
    normalized_answers = [ans for ans in normalized_answers if ans.strip()]
    
    if not normalized_answers:
        return "N/A"
        
    return Counter(normalized_answers).most_common(1)[0][0]

def process_json_file(path, pum_model, pum_tokenizer, pum_device, use_pum, top_k_percent):
    """
    Processes a single JSON file to check for correctness.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ground_truth = data.get('ground_truth')
        if ground_truth is None:
            return None, None # Skip if no ground truth

        normalized_gt = normalize_answer(str(ground_truth))
        if not normalized_gt:
            return None, None # Skip if ground truth is empty

        solutions = data.get('solutions', [])
        if not solutions:
            return False, None # No solutions, so incorrect

        # --- 1. Normal Majority Vote ---
        majority_vote = get_majority_vote(solutions)
        is_normal_correct = False
        if majority_vote != "N/A":
            normalized_mv = normalize_answer(str(majority_vote))
            if normalized_gt == normalized_mv:
                is_normal_correct = True

        # --- 2. PUM Weighted Majority Vote ---
        is_pum_correct = None
        if use_pum:
            if not pum_model or not pum_tokenizer:
                raise ValueError("PUM model and tokenizer must be loaded to use PUM-weighted scoring.")
            
            question = data.get('question')
            uncertainties = get_trace_uncertainties(pum_model, pum_tokenizer, pum_device, question, solutions)
            
            confidences = [1.0 - u for u in uncertainties]
            
            # Pair solutions with their confidence scores
            scored_solutions = sorted(zip(confidences, solutions), key=lambda x: x[0], reverse=True)
            
            # Keep only top-k percent
            num_to_keep = math.ceil(len(scored_solutions) * (top_k_percent / 100.0))
            top_k_solutions = [sol for conf, sol in scored_solutions[:num_to_keep]]
            
            pum_majority_vote = get_majority_vote(top_k_solutions)
            
            is_pum_correct = False
            if pum_majority_vote != "N/A":
                normalized_pum_mv = normalize_answer(str(pum_majority_vote))
                if normalized_gt == normalized_pum_mv:
                    is_pum_correct = True

        return is_normal_correct, is_pum_correct

    except Exception as e:
        print(f"Warning: Failed to process file {path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Evaluate self-consistency outputs with optional PUM-weighted majority voting.')
    parser.add_argument('input_dir', help='Directory containing result subfolders to process.')
    parser.add_argument('--use-pum', action='store_true', help='Enable PUM-weighted majority vote scoring.')
    parser.add_argument('--top-k-percent', type=int, default=50, help='Top k percent of traces to consider for PUM-weighted voting.')
    parser.add_argument('--uncertainty_model_path', type=str, help='Path to the PUM uncertainty model (required if --use-pum).')
    
    args = parser.parse_args()

    if args.use_pum and not args.uncertainty_model_path:
        parser.error("--uncertainty_model_path is required when --use-pum is specified.")

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"❌ Error: The directory does not exist: '{input_dir}'")
        exit(1)

    # --- Load PUM model if needed ---
    pum_model, pum_tokenizer, pum_device = None, None, None
    if args.use_pum:
        pum_model, pum_tokenizer, pum_device = load_pum_model(args.uncertainty_model_path)

    # --- Find and process summary files ---
    json_files = [os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files if f == 'summary.json']

    if not json_files:
        print(f"❌ No 'summary.json' files found in '{input_dir}' or its subfolders.")
        exit(1)

    # --- Initialize counters ---
    total_files = 0
    normal_correct_count = 0
    pum_correct_count = 0

    for path in tqdm(sorted(json_files), desc="Evaluating summaries"):
        is_normal_correct, is_pum_correct = process_json_file(
            path, pum_model, pum_tokenizer, pum_device, args.use_pum, args.top_k_percent
        )
        
        if is_normal_correct is None:
            continue # Skip file if it had errors or was invalid
            
        total_files += 1
        
        if is_normal_correct:
            normal_correct_count += 1
        
        if is_pum_correct is not None and is_pum_correct:
            pum_correct_count += 1

    # --- Calculate and Print Results ---
    normal_accuracy = (normal_correct_count / total_files) if total_files > 0 else 0.0
    
    pum_accuracy = None
    if args.use_pum:
        pum_accuracy = (pum_correct_count / total_files) if total_files > 0 else 0.0

    # Extract N from directory name (e.g., "n_32")
    dir_name = os.path.basename(input_dir.rstrip('/'))
    n_samples = 0
    match = re.search(r'n_(\d+)', dir_name)
    if match:
        n_samples = int(match.group(1))

    print(f"\n--- Evaluation Summary ---")
    print(f"Directory: {dir_name}")
    print(f"Number of samples evaluated: {total_files}")
    
    result = {
        "N": n_samples,
        "maj_accuracy": round(normal_accuracy, 4)
    }
    
    if pum_accuracy is not None:
        result["pum_maj_accuracy"] = round(pum_accuracy, 4)
        result["top_k_percent"] = args.top_k_percent

    print("\nResults:")
    print(json.dumps(result, indent=4))

if __name__ == '__main__':
    main()
