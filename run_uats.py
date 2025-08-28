import argparse
from rely.inference.uats.utils import run_uats, UATSConfig

def main():
    parser = argparse.ArgumentParser(description="Run UATS inference")
    parser.add_argument("--question", type=str, required=True, help="The question to be answered.")
    parser.add_argument("--output_dir", type=str, default="uats_results", help="Directory to save the output.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name of the base model.")
    parser.add_argument("--uncertainty_model_path", type=str, required=True, help="Path to the uncertainty model.")
    parser.add_argument("--value_model_path", type=str, required=True, help="Path to the value model.")
    args = parser.parse_args()

    config = UATSConfig(
        model_name=args.model_name,
        uncertainty_model_path=args.uncertainty_model_path,
        value_model_path=args.value_model_path,
    )
    
    run_uats(user_question=args.question, config=config, save_dir=args.output_dir)

if __name__ == "__main__":
    main()
