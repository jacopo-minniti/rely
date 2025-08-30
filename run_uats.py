import argparse
import logging
import re

from datasets import load_dataset

from rely.inference.uats.utils import run_uats, UATSConfig

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(
        description="Run UATS inference. Requires a running OpenAI-compatible server (e.g., vLLM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument("--output_dir", type=str, default="uats_results", help="Directory to save the output.")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name of the base generation model (served by vLLM).")
    parser.add_argument("--uncertainty_model_path", type=str, default="jacopo-minniti/Qwen2.5-Math-7B-PUM", help="Path to the uncertainty model.")
    parser.add_argument("--value_model_path", type=str, default="Qwen/Qwen2.5-Math-PRM-7B", help="Path to the value model.")

    # UATSConfig arguments
    parser.add_argument("--beam_width", type=int, default=4, help="Beam width for the search.")
    parser.add_argument("--max_branches", type=int, default=2, help="Maximum branches to explore based on uncertainty.")
    parser.add_argument("--budget", type=int, default=1024, help="Total token budget for generation.")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.5, help="Threshold to trigger branching.")
    parser.add_argument("--temperature", type=float, default=1, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Generation top_p.")
    parser.add_argument("--uncertainty_scoring_method", type=str, default="last_step", choices=["product", "average", "minimum", "last_step"], help="Scoring method for uncertainty.")
    parser.add_argument("--value_scoring_method", type=str, default="product", choices=["product", "average", "minimum", "last_step"], help="Scoring method for value.")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda:1", help="Main device for generation.")
    parser.add_argument("--uncertainty_device", type=str, default="cuda:0", help="Device for the uncertainty model.")
    parser.add_argument("--value_device", type=str, default="cuda:0", help="Device for the value model.")

    args = parser.parse_args()

    # Inform user about the server dependency
    logging.info("Please ensure an OpenAI-compatible API server (like vLLM) is running at http://localhost:8000/v1")
    logging.info(f"Serving the model: {args.model_name}")

    config = UATSConfig(
        model_name=args.model_name,
        uncertainty_model_path=args.uncertainty_model_path,
        value_model_path=args.value_model_path,
        uncertainty_scoring_method=args.uncertainty_scoring_method,
        value_scoring_method=args.value_scoring_method,
        beam_width=args.beam_width,
        max_branches=args.max_branches,
        budget=args.budget,
        uncertainty_threshold=args.uncertainty_threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        uncertainty_device=args.uncertainty_device,
        value_device=args.value_device,
    )

    NUM_SAMPLES = 1
    
    dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split='test')
    dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))

    questions = []
    correct_answers = []

    for example in dataset:
        question_text = example["problem"]
        # Remove asy blocks, which are not useful for the model
        cleaned_question = re.sub(r'\[asy\].*?\[/asy\]', '', question_text, flags=re.DOTALL).strip()
        questions.append(cleaned_question)
        correct_answers.append(example["answer"])

    run_uats(
        user_questions=questions, 
        correct_answers=correct_answers,
        config=config, 
        save_dir=args.output_dir
    )
