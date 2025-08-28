import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from multiprocessing import Process, Queue

import torch
import pygraphviz as pgv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from .config import UATSConfig, Branch
from .guided_tree_search import GuidedTreeSearch
from rely.utils.text_utils import extract_final_answer

logger = logging.getLogger(__name__)


def _model_server(model_path, device, task_queue, result_queue, model_type):
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
                score = outputs.logits[0, -1, 0].item()
            else: # uncertainty
                outputs = model(**inputs)
                score = torch.softmax(outputs.logits, dim=-1)[0, 0].item()
        
        result_queue.put((request_id, score))

def _worker(rank, config, question, system_prompt, uncertainty_task_queue, uncertainty_result_queue, value_task_queue, value_result_queue, worker_result_queue):
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

    final_branches, all_branches, tokens_used = gts.search(question, system_prompt)
    
    final_branches_dicts = [b.to_dict() for b in final_branches]
    all_branches_dicts = [b.to_dict() for b in all_branches]
    
    worker_result_queue.put((final_branches_dicts, all_branches_dicts, tokens_used))


def run_uats(
    user_question: str,
    system_prompt: str,
    config: Optional[UATSConfig] = None,
    save_dir: Optional[Union[str, Path]] = "uats_results",
) -> List[Branch]:
    """High-level one-shot helper wrapping search & persistence."""

    if config is None:
        config = UATSConfig()

    os.makedirs(save_dir, exist_ok=True)
    
    uncertainty_task_queue = Queue()
    uncertainty_result_queue = Queue()
    value_task_queue = Queue()
    value_result_queue = Queue()
    worker_result_queue = Queue()

    uncertainty_server = Process(target=_model_server, args=(config.uncertainty_model_path, config.uncertainty_device, uncertainty_task_queue, uncertainty_result_queue, "uncertainty"))
    value_server = Process(target=_model_server, args=(config.value_model_path, config.value_device, value_task_queue, value_result_queue, "value"))
    
    uncertainty_server.start()
    value_server.start()
    
    worker_process = Process(target=_worker, args=(0, config, user_question, system_prompt, uncertainty_task_queue, uncertainty_result_queue, value_task_queue, value_result_queue, worker_result_queue))
    
    worker_process.start()
    final_branches_dicts, all_branches_dicts, tokens_used = worker_result_queue.get()
    worker_process.join()

    uncertainty_task_queue.put((None, None, None))
    value_task_queue.put((None, None, None))
    uncertainty_server.join()
    value_server.join()
    
    final_branches = [Branch.from_dict(b) for b in final_branches_dicts]
    all_branches = [Branch.from_dict(b) for b in all_branches_dicts]

    if save_dir is not None:
        logger.info(f"Saving branches to {save_dir}")
        save_branches(
            final_branches,
            all_branches,
            save_dir,
            tokens_used,
            user_question=user_question,
        )

    return final_branches


def save_branches(
    branches: List[Branch],
    all_branches: List[Branch],
    output_dir: Union[str, Path],
    tokens_used: int,
    user_question: Optional[str] = None,
) -> None:
    """Save the *active* branches at the end of the search to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_branches_data = []
    for branch in branches:
        branch_dict = branch.to_dict()
        full_text = branch.text
        if branch.final_answer:
            full_text += branch.final_answer
        
        extracted_answer = extract_final_answer(full_text)
        branch_dict['extracted_answer'] = extracted_answer
        final_branches_data.append(branch_dict)

    summary_data = {
        "question": user_question,
        "tokens_used": tokens_used,
        "final_branches": final_branches_data,
    }
    with open(os.path.join(output_path, "output.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    graph = pgv.Agraph(directed=True)
    for branch in all_branches:
        graph.add_node(branch.id, label=f"ID: {branch.id}\\nScore: {branch.score:.4f}\\nValue: {branch.value:.4f}")
        if branch.parent_id is not None:
            graph.add_edge(branch.parent_id, branch.id)
    graph.write(os.path.join(output_path, "branches.dot"))
    graph.draw(os.path.join(output_path, "branches.png"), prog="dot")
