from typing import Optional, Union
import logging
import argparse
import os
import sys

# Add the current directory to Python path so we can import rely
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

from rely.inference import run_uats, UATSConfig, run_self_consistency, SelfConsistencyConfig
from rely.utils import MMLU_SYSTEM_PROMPT, load_dataset

def main():
    parser = argparse.ArgumentParser(description='Run UATS on a range of questions from a dataset')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting question index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None, help='Ending question index (exclusive)')
    parser.add_argument('--dataset_name', type=str, default="TIGER-Lab/MMLU-Pro", help='Dataset name')
    parser.add_argument('--split', type=str, default="validation", help='Dataset split')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help='Model name')
    parser.add_argument('--uncertainty_model_path', type=str, default="models/uncertainty_model", help='Path to uncertainty model')
    parser.add_argument('--uncertainty_scoring_method', type=str, default="last_step", choices=["product", "minimum", "average", "last_step"], help='Uncertainty scoring method')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.5, help='Uncertainty threshold for branching')
    parser.add_argument('--value_model_path', type=str, default="Qwen/Qwen2.5-Math-PRM-7B", help='HuggingFace path to value model')
    parser.add_argument('--value_scoring_method', type=str, default="product", choices=["product", "minimum", "average", "last_step"], help='Value scoring method')
    parser.add_argument('--budget', type=int, default=4000, help='Token budget')
    parser.add_argument('--max_step_tokens', type=int, default=512, help='Max tokens per step')
    parser.add_argument('--beam_width', type=int, default=4, help='Beam width')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature')
    parser.add_argument('--base_save_dir', type=str, default="uats_results", help='Base directory for saving results')
    
    # Device configuration arguments
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for policy/generation model')
    parser.add_argument('--uncertainty_device', type=str, default="cuda:1", help='Device for uncertainty model')
    parser.add_argument('--value_device', type=str, default="cuda:2", help='Device for value model')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.split)
    
    # Set end_idx if not provided
    if args.end_idx is None:
        args.end_idx = len(dataset)
    
    # Validate indices
    if args.start_idx < 0 or args.end_idx > len(dataset) or args.start_idx >= args.end_idx:
        raise ValueError(f"Invalid indices: start_idx={args.start_idx}, end_idx={args.end_idx}, dataset_size={len(dataset)}")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing questions {args.start_idx} to {args.end_idx-1} (total: {args.end_idx - args.start_idx})")
    
    # Process each question in the range
    for i in range(args.start_idx, args.end_idx):
        question = dataset[i]['question']
        options = dataset[i]['options']
        answer = dataset[i]['answer']

        options = '\n'.join(f"({chr(65 + idx)}) {opt}" for idx, opt in enumerate(options))
        prompt = f"Question: {question}\n\nOptions:\n{options}"
        
        # Create question-specific save directory
        question_save_dir = os.path.join(args.base_save_dir, f"question_{i:06d}")
        
        logger.info(f"Processing question {i}: {question[:100]}...")
        
        run_uats(
            user_question=prompt,
            system_prompt=MMLU_SYSTEM_PROMPT,
            config=UATSConfig(
                model_name=args.model_name,
                uncertainty_model_path=args.uncertainty_model_path,
                uncertainty_scoring_method=args.uncertainty_scoring_method,
                value_model_path=args.value_model_path,
                value_scoring_method=args.value_scoring_method,
                budget=args.budget,
                uncertainty_threshold=args.uncertainty_threshold,
                max_step_tokens=args.max_step_tokens,
                beam_width=args.beam_width,
                temperature=args.temperature,
                device=args.device,
                uncertainty_device=args.uncertainty_device,
                value_device=args.value_device,
            ),
            save_dir=question_save_dir,
            correct_answer=answer
        )
        
        logger.info(f"Completed question {i}")

if __name__ == "__main__":
    main()
