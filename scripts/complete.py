from rely.generate import Completer, CompleterConfig

def main():
    """
    Main function that runs the complete pipeline:
    """    

    input_dataset = "data/math_generations_qwen2.5.jsonl"  # Input dataset path
    completions_file = "math_completions_100.jsonl"  # Intermediate completions file
    
    completer_config = CompleterConfig(
        model="Qwen/Qwen3-1.7B",
        tp_size=1,
        dp_size=4,
        max_num_seqs=1024,
        forking_strategy="newline",
        completion_type="long",
        dataset=input_dataset,
        question_field="question"
    )

    completer = Completer(completer_config)
    completer.generate(
        output_file=completions_file,
        n_completions_per_item=8,
        max_new_tokens=4096,
        temperature=1.0,
        cot_percentage=1.0
    )

if __name__ == "__main__":
    exit(main())
