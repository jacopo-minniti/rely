import json
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
import numpy as np
import argparse
import os

# --- User's Existing Code ---
SYSTEM_PROMPT = """The following are multiple choice questions (with answers) about general knowledge. Think step by step and then finish your answer with 'The correct answer is (X)' where X is the correct letter choice.

EXAMPLE

Question: The quantum efficiency of a photon detector is 0.1. If 100 photons are sent into the detector, one after the other, the detector will detect photons
Options:
(A) an average of 10 times, with an rms deviation of about 4
(B) an average of 10 times, with an rms deviation of about 3
(C) an average of 10 times, with an rms deviation of about 1
(D) an average of 10 times, with an rms deviation of about 0.1

## Final Answer
[...Brief explanation of your answer...] The correct answer is (B).
"""

# Load the model and tokenizer
# It's recommended to use a GPU for this process
if torch.cuda.is_available():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        max_seq_length = 30000,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
else:
    print("Warning: No GPU found. This process will be very slow on a CPU.")
    # Exit or handle CPU-only case as needed
    exit()


def load_prompts(start_idx: int | None = None, end_idx: int | None = None):
    """Loads and formats prompts from the dataset.

    Parameters
    ----------
    start_idx : int | None
        Inclusive start index of the line in the dataset to process. If ``None`` starts from 0.
    end_idx : int | None
        Exclusive end index. If ``None`` reads until the end of file.
    """
    prompts: list[dict] = []

    # Stream the file line-by-line so that we can cheaply skip to the start_idx.
    with open("generations-mmlu-qwen3-8B.jsonl", "r") as f:
        for i, line in enumerate(f):
            if start_idx is not None and i < start_idx:
                continue  # Skip until we reach the first relevant line
            if end_idx is not None and i >= end_idx:
                break     # We have read all requested lines

            item = json.loads(line)

            # Build the prompt/response pair exactly as before
            user_prompt_part = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
            )
            completion_text = f"<think>\n{item['attempt']}<|im_end|>"
            prompts.append({
                "prompt": user_prompt_part,
                "completion": completion_text,
            })

    return prompts

def calculate_token_entropies(model, tokenizer, prompts):
    """
    Calculates the entropy for each token in the completion part of the prompts.
    This follows the paper's methodology.
    """
    all_entropies = []
    
    # The paper used a temperature of 1.0 for its analysis
    temperature = 1.0

    print(f"Calculating token entropies for {len(prompts)} prompts...")
    # Use tqdm for a progress bar
    for item in tqdm(prompts):
        full_text = item['prompt'] + item['completion']
        
        # Tokenize the full sequence
        inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
        
        # We need to find where the completion part starts to only calculate entropy for those tokens
        prompt_tokens = tokenizer(item['prompt'], return_tensors="pt")
        start_index = prompt_tokens.input_ids.shape[1]

        with torch.no_grad():
            # Get logits for the entire sequence in one forward pass
            outputs = model(**inputs)
            logits = outputs.logits

        # We are interested in the entropy of predicting token `i` given tokens `0..i-1`.
        # The logits at index `i-1` are used to predict token `i`.
        # So we iterate from the start of the completion to the end of the sequence.
        for i in range(start_index, inputs.input_ids.shape[1]):
            # Get the logits for the current token's prediction (logits at position i-1)
            token_logits = logits[0, i - 1, :]

            # Apply temperature (T=1.0 doesn't change logits but is included for correctness)
            scaled_logits = token_logits / temperature
            
            # Calculate probabilities using softmax
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Calculate the entropy for this token's distribution
            # Use torch.distributions.Categorical for a stable and efficient implementation
            entropy = torch.distributions.Categorical(probs=probabilities).entropy()
            
            all_entropies.append(entropy.item())
            
    return all_entropies


# --- Main Execution ---
# ------------------------- CLI helpers -------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute token-level entropies for a subset of prompts.")
    parser.add_argument("--start_idx", type=int, default=0, help="Inclusive start index of the prompts to process.")
    parser.add_argument("--end_idx", type=int, default=None, help="Exclusive end index of the prompts to process. If omitted, processes to the end of the file.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the numpy .npy file containing the entropies. If not provided, a name based on the index range is used.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Resolve the output path early so we can show it to the user immediately
    if args.output is None:
        end_display = "end" if args.end_idx is None else args.end_idx
        args.output = f"entropies_{args.start_idx}_{end_display}.npy"

    # Load only the needed slice of prompts
    prompts = load_prompts(args.start_idx, args.end_idx)

    if not prompts:
        raise ValueError("No prompts were loaded: check start/end indices or dataset path.")

    token_entropies = calculate_token_entropies(model, tokenizer, prompts)

    if token_entropies:
        entropies_arr = np.array(token_entropies, dtype=np.float32)

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        np.save(args.output, entropies_arr)

        print("\n--- Subset Results ---")
        print(f"Saved {len(entropies_arr)} entropies to {args.output}.")
        print(f"Average entropy (subset): {entropies_arr.mean():.4f}")
        print(f"Median entropy (subset): {np.median(entropies_arr):.4f}")
        print("Note: Aggregate the saved .npy files from all shards to compute the global 80th percentile.")
    else:
        print("No tokens were processed. Please check your data loading and prompt formatting.")