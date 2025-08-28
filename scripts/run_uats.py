import argparse
import json
import logging
import os
from multiprocessing import Process, Queue
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from rely.inference.uats.config import UATSConfig, Branch
from rely.inference.uats.guided_tree_search import GuidedTreeSearch

try:
    import pygraphviz as pgv
    PYGRAPHVIZ_AVAILABLE = True
except ImportError:
    PYGRAPHVIZ_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def model_server(model_path, device, task_queue, result_queue, model_type):
    """Generic server for running inference on a model."""
    if model_type == "value":
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    else: # uncertainty
        model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info(f"{model_type.capitalize()} model server started on {device}")

    while True:
        request_id, rank, text = task_queue.get()
        if text is None:  # Sentinel for stopping
            break
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            if model_type == "value":
                outputs = model(**inputs)
                # Take the logits of the last token
                score = outputs.logits[0, -1, 0].item()
            else: # uncertainty
                outputs = model(**inputs)
                # Apply softmax to get probabilities and take the score for the first class
                score = torch.softmax(outputs.logits, dim=-1)[0, 0].item()
        
        result_queue.put((request_id, score))

def worker(rank, config, question, uncertainty_task_queue, uncertainty_result_queue, value_task_queue, value_result_queue, output_dir):
    """Worker process to run the Guided Tree Search."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    gts = GuidedTreeSearch(
        tokenizer=tokenizer,
        config=config,
        uncertainty_task_queue=uncertainty_task_queue,
        uncertainty_result_queue=uncertainty_result_queue,
        value_task_queue=value_task_queue,
        value_result_queue=value_result_queue,
        worker_rank=rank,
    )

    final_branches, all_branches, tokens_used = gts.search(question)
    
    # --- Saving Mechanism ---
    # Create a directory for the question
    question_dir = os.path.join(output_dir, f"question_{rank}")
    os.makedirs(question_dir, exist_ok=True)

    # Save final branches as text files
    for i, branch in enumerate(final_branches):
        with open(os.path.join(question_dir, f"final_branch_{i}.txt"), "w") as f:
            f.write(branch.text)
            if branch.final_answer:
                f.write(branch.final_answer)

    # Save all branches and metadata as a JSON file
    output_data = {
        "question": question,
        "tokens_used": tokens_used,
        "final_branches": [b.to_dict() for b in final_branches],
        "all_branches": [b.to_dict() for b in all_branches],
    }
    with open(os.path.join(question_dir, "output.json"), "w") as f:
        json.dump(output_data, f, indent=4)

    # Create and save a graph of the branches
    if PYGRAPHVIZ_AVAILABLE:
        graph = pgv.Agraph(directed=True)
        for branch in all_branches:
            graph.add_node(branch.id, label=f"ID: {branch.id}\\nScore: {branch.score:.4f}\\nValue: {branch.value:.4f}")
            if branch.parent_id is not None:
                graph.add_edge(branch.parent_id, branch.id)
        graph.write(os.path.join(question_dir, "branches.dot"))
        graph.draw(os.path.join(question_dir, "branches.png"), prog="dot")
    else:
        logger.warning("pygraphviz not installed. Skipping graph visualization.")


def main():
    parser = argparse.ArgumentParser(description="Run UATS inference")
    parser.add_argument("--question", type=str, required=True, help="The question to be answered.")
    parser.add_argument("--output_dir", type=str, default="uats_output", help="Directory to save the output.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name of the base model.")
    parser.add_argument("--uncertainty_model_path", type=str, required=True, help="Path to the uncertainty model.")
    parser.add_argument("--value_model_path", type=str, required=True, help="Path to the value model.")
    args = parser.parse_args()

    config = UATSConfig(
        model_name=args.model_name,
        uncertainty_model_path=args.uncertainty_model_path,
        value_model_path=args.value_model_path,
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Multiprocessing Setup ---
    uncertainty_task_queue = Queue()
    uncertainty_result_queue = Queue()
    value_task_queue = Queue()
    value_result_queue = Queue()

    # Start model servers
    uncertainty_server = Process(target=model_server, args=(config.uncertainty_model_path, config.uncertainty_device, uncertainty_task_queue, uncertainty_result_queue, "uncertainty"))
    value_server = Process(target=model_server, args=(config.value_model_path, config.value_device, value_task_queue, value_result_queue, "value"))
    
    uncertainty_server.start()
    value_server.start()
    
    # Start worker
    worker_process = Process(target=worker, args=(0, config, args.question, uncertainty_task_queue, uncertainty_result_queue, value_task_queue, value_result_queue, args.output_dir))
.
    worker_process.start()
    worker_process.join()

    # Stop model servers
    uncertainty_task_queue.put((None, None, None))
    value_task_queue.put((None, None, None))
    uncertainty_server.join()
    value_server.join()

if __name__ == "__main__":
    main()
