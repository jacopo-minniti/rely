import logging
import argparse
import os

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
    parser.add_argument('--model_name', type=str, default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit", help='Model name')
    parser.add_argument('--uncertainty_probe_path', type=str, default="models/uncertainty_probe.pth", help='Path to uncertainty probe')
    parser.add_argument('--value_probe_path', type=str, default="models/value_probe.pth", help='Path to value probe')
    parser.add_argument('--budget', type=int, default=1024, help='Token budget')
    parser.add_argument('--max_step_tokens', type=int, default=512, help='Max tokens per step')
    parser.add_argument('--beam_width', type=int, default=3, help='Beam width')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature')
    parser.add_argument('--base_save_dir', type=str, default="uats_results", help='Base directory for saving results')
    
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
                uncertainty_probe_path=args.uncertainty_probe_path,
                value_probe_path=args.value_probe_path,
                budget=args.budget,
                max_step_tokens=args.max_step_tokens,
                beam_width=args.beam_width,
                temperature=args.temperature,
            ),
            save_dir=question_save_dir,
            correct_answer=answer
        )
        
        logger.info(f"Completed question {i}")

if __name__ == "__main__":
    main()
