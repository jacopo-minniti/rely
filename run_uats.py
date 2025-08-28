import argparse
import logging
from rely.inference.uats.utils import run_uats, UATSConfig

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(
        description="Run UATS inference. Requires a running OpenAI-compatible server (e.g., vLLM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument("--question", type=str, required=True, help="The question to be answered.")
    parser.add_argument("--output_dir", type=str, default="uats_results", help="Directory to save the output.")
    parser.add_argument("--correct_answer", type=str, default=None, help="The correct answer to the question for evaluation.")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name of the base generation model (served by vLLM).")
    parser.add_argument("--uncertainty_model_path", type=str, required=True, help="Path to the uncertainty model.")
    parser.add_argument("--value_model_path", type=str, required=True, help="Path to the value model.")

    # UATSConfig arguments
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for the search.")
    parser.add_argument("--max_branches", type=int, default=2, help="Maximum branches to explore based on uncertainty.")
    parser.add_argument("--budget", type=int, default=1024, help="Total token budget for generation.")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.5, help="Threshold to trigger branching.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Generation top_p.")
    
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
    
    run_uats(
        user_question=args.question, 
        config=config, 
        save_dir=args.output_dir, 
        correct_answer=args.correct_answer
    )

if __name__ == "__main__":
    main()
