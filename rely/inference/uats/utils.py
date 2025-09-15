import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from multiprocessing import Process, Queue
import multiprocessing as mp
from collections import Counter

import torch
import textwrap
import matplotlib.pyplot as plt
import networkx as nx
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification

from .config import UATSConfig, Branch
from .guided_tree_search import GuidedTreeSearch
from .scorer import Scorer
from rely.utils.text_utils import MATH_SYSTEM_PROMPT, extract_final_answer, normalize_answer

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _model_server(model_path, device, task_queue, result_queues, model_type, config):
    """
    Generic server for running inference on a model in batches.
    """
    if model_type == "value":
        model_class = AutoModel
        scoring_method = config.value_scoring_method
    elif model_type == "uncertainty":
        model_class = AutoModelForTokenClassification
        scoring_method = config.uncertainty_scoring_method
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model_class.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    scorer = Scorer(model, tokenizer, device, scoring_method, model_type)
    
    logger.info(f"{model_type.capitalize()} model server started on {device} with '{scoring_method}' scoring.")

    while True:
        request = task_queue.get()
        if request is None: # Sentinel to stop
            break
        
        request_id, rank, texts = request
        if texts is None:
            break
        
        scores = scorer.score_batch(texts)
        result_queues[rank].put((request_id, scores))

def run_uats(
    user_questions: List[str],
    system_prompt: str = MATH_SYSTEM_PROMPT,
    config: Optional[UATSConfig] = None,
    save_dir: Optional[Union[str, Path]] = "uats_results",
    correct_answers: Optional[List[Optional[str]]] = None,
    num_workers: int = 5,
) -> List[List[Branch]]:
    """High-level helper to run UATS for a list of questions with multiple workers."""

    if config is None:
        config = UATSConfig()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    uncertainty_task_queue = Queue()
    uncertainty_result_queues = {i: Queue() for i in range(num_workers)}
    value_task_queue = Queue()
    value_result_queues = {i: Queue() for i in range(num_workers)}
    
    uncertainty_server = Process(target=_model_server, args=(config.uncertainty_model_path, config.uncertainty_device, uncertainty_task_queue, uncertainty_result_queues, "uncertainty", config))
    value_server = Process(target=_model_server, args=(config.value_model_path, config.value_device, value_task_queue, value_result_queues, "value", config))
    
    uncertainty_server.start()
    value_server.start()

    all_results = []
    
    if correct_answers is None:
        answers_list: List[Optional[str]] = [None] * len(user_questions)
    else:
        answers_list = correct_answers

    question_task_queue = Queue()
    question_result_queue = Queue()
    
    for i, (question, correct_answer) in enumerate(zip(user_questions, answers_list)):
        question_task_queue.put((i, question, correct_answer))
    
    for _ in range(num_workers):
        question_task_queue.put(None)
    
    worker_processes = []
    for rank in range(num_workers):
        worker_process = Process(
            target=_question_worker, 
            args=(
                rank, 
                config, 
                system_prompt, 
                uncertainty_task_queue, 
                uncertainty_result_queues[rank],
                value_task_queue, 
                value_result_queues[rank],
                question_task_queue,
                question_result_queue,
                save_dir
            )
        )
        worker_process.start()
        worker_processes.append(worker_process)
    
    results_dict = {}
    for _ in range(len(user_questions)):
        question_idx, final_branches_dicts, all_branches_dicts, tokens_used = question_result_queue.get()
        final_branches = [Branch.from_dict(b) for b in final_branches_dicts]
        all_branches = [Branch.from_dict(b) for b in all_branches_dicts]
        results_dict[question_idx] = final_branches
        logger.info(f"Completed question {question_idx+1}/{len(user_questions)} using {tokens_used} tokens")
    
    for worker_process in worker_processes:
        worker_process.join()
    
    all_results = [results_dict[i] for i in range(len(user_questions))]

    uncertainty_task_queue.put(None)
    value_task_queue.put(None)
    uncertainty_server.join()
    value_server.join()

    return all_results


def _question_worker(
    rank: int, 
    config: UATSConfig, 
    system_prompt: str,
    uncertainty_task_queue: Queue,
    uncertainty_result_queue: Queue,
    value_task_queue: Queue,
    value_result_queue: Queue,
    question_task_queue: Queue,
    question_result_queue: Queue,
    save_dir: Optional[Union[str, Path]]
):
    """Worker process that processes questions from the queue."""
    logger.info(f"Worker {rank} started")
    
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
    
    while True:
        task = question_task_queue.get()
        if task is None:
            logger.info(f"Worker {rank} stopping")
            break
        
        question_idx, question, correct_answer = task
        start_time = time.time()
        logger.info(f"Worker {rank} processing question {question_idx+1}: {question[:100]}...")
        
        final_branches, all_branches, tokens_used = gts.search(question, system_prompt)
        
        final_branches_dicts = [b.to_dict() for b in final_branches]
        all_branches_dicts = [b.to_dict() for b in all_branches]

        if save_dir is not None:
            
            question_save_dir = Path(save_dir) / f"question_{question_idx}"
            logger.info(f"Worker {rank} saving results to {question_save_dir}")
            save_results(
                final_branches,
                all_branches,
                question_save_dir,
                user_question=question,
                correct_answer=correct_answer,
                total_tokens_generated=tokens_used,
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Worker {rank} completed question {question_idx+1} in {elapsed_time:.2f} seconds")
        
        question_result_queue.put((question_idx, final_branches_dicts, all_branches_dicts, tokens_used))


def save_results(
    final_beams: List[Branch],
    all_branches: List[Branch],
    output_dir: Union[str, Path],
    user_question: Optional[str] = None,
    correct_answer: Optional[str] = None,
    total_tokens_generated: Optional[int] = None,
) -> None:
    """Save the final results and search tree visualization to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    solutions = []
    for i, node in enumerate(final_beams):
        if not node.final_answer:
            node.final_answer = extract_final_answer(node.text) or ""
        
        termination_reason = "answer_found" if node.final_answer else "budget_reached"
        
        solution_data = {
            "beam_index": i + 1,
            "value": node.value,
            "final_answer": node.final_answer or "Not found",
            "depth": node.step_count,
            "termination_reason": termination_reason,
            "solution_path": node.text,
        }
        solutions.append(solution_data)

    ground_truth = normalize_answer(correct_answer)
    answers = [normalize_answer(s['final_answer']) for s in solutions if s['final_answer'] != "Not found"]
    majority_vote = Counter(answers).most_common(1)[0][0] if answers else "N/A"
    accuracy = "N/A"
    if ground_truth and answers:
        correct_answers_count = sum(1 for ans in answers if ans == ground_truth)
        accuracy = f"{correct_answers_count / len(answers):.2%}" if answers else "0.00%"

    summary = {
        "question": user_question,
        "ground_truth": ground_truth,
        "majority_vote": majority_vote,
        "accuracy": accuracy,
        "solutions": solutions,
        "total_tokens": total_tokens_generated
    }

    summary_filepath = output_path / "summary.json"
    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Saved summary to {summary_filepath}")

    try:
        _generate_tree_image(all_branches, output_path, correct_answer=correct_answer)
    except Exception as e:
        logger.warning(f"Could not generate tree image: {e}")


def _generate_tree_image(
    branches: List[Branch],
    output_dir: Path,
    filename: str = "search_tree.png",
    correct_answer: Optional[str] = None,
) -> None:
    """Render a simple tree visualisation of the explored search space using matplotlib and networkx."""
    
    if not branches:
        return
    
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        logger.warning("Matplotlib not available, skipping tree visualization")
        return
    
    G = nx.DiGraph()
    branch_map = {b.id: b for b in branches}
    
    for branch in branches:
        node_id = branch.id
        
        latest_step = ""
        if branch.parent_id is not None and branch.parent_id in branch_map:
            parent = branch_map[branch.parent_id]
            if branch.text.startswith(parent.text):
                latest_step = branch.text[len(parent.text):].strip()

        if not latest_step:
            assistant_marker = "assistant\n"
            marker_pos = branch.text.rfind(assistant_marker)
            if marker_pos != -1:
                latest_step = branch.text[marker_pos + len(assistant_marker):].strip()
            else:
                steps = branch.text.split('\n\n')
                latest_step = steps[-1].strip() if steps else branch.text.strip()
        
        latest_step = latest_step.replace('

, r'\

)
        latest_step = re.sub(r'boxed{(.*?)}', r'\1', latest_step)
        latest_step = (latest_step[:25] + '...') if len(latest_step) > 25 else latest_step
        
        wrapped_text = textwrap.fill(latest_step, width=30)
        
        score_lines = []
        if branch.value is not None:
            score_lines.append(f"v={branch.value:.2f}")
        if branch.uncertainty is not None:
            score_lines.append(f"u={branch.uncertainty:.2f}")
        label = "\n".join(score_lines) + f"\n{wrapped_text}" if score_lines else wrapped_text
        
        G.add_node(node_id, label=label, is_final=branch.is_final)
        
        if branch.parent_id is not None:
            G.add_edge(branch.parent_id, node_id)
        else:
            G.nodes[node_id]['is_root'] = True

    for branch in branches:
        if branch.is_final and branch.final_answer:
            final_answer_text = re.sub(r'boxed{(.*?)}', r'\1', branch.final_answer)
            final_answer_text = final_answer_text.replace('

, r'\

)
            final_node_id = f"final_{branch.id}"
            
            is_correct = False
            if correct_answer:
                normalized_answer = normalize_answer(final_answer_text)
                if normalized_answer != "Not found" and normalized_answer == normalize_answer(correct_answer):
                    is_correct = True
            
            G.add_node(final_node_id, label=final_answer_text, is_final_answer=True, is_correct=is_correct)
            G.add_edge(branch.id, final_node_id)
    
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    
    plt.figure(figsize=(20, 15))
    
    root_nodes = [node for node, data in G.nodes(data=True) if data.get('is_root')]
    correct_final_nodes = [node for node, data in G.nodes(data=True) if data.get('is_final_answer') and data.get('is_correct')]
    incorrect_final_nodes = [node for node, data in G.nodes(data=True) if data.get('is_final_answer') and not data.get('is_correct')]
    
    all_special_nodes = root_nodes + correct_final_nodes + incorrect_final_nodes
    intermediate_nodes = [node for node in G.nodes() if node not in all_special_nodes]

    nx.draw_networkx_nodes(G, pos, nodelist=intermediate_nodes, node_shape="s", node_color="lightblue", node_size=3500)
    nx.draw_networkx_nodes(G, pos, nodelist=root_nodes, node_shape="o", node_color="lightgreen", node_size=3500)
    nx.draw_networkx_nodes(G, pos, nodelist=correct_final_nodes, node_shape="o", node_color="lightgreen", node_size=3500)
    nx.draw_networkx_nodes(G, pos, nodelist=incorrect_final_nodes, node_shape="o", node_color="lightcoral", node_size=3500)
    
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=15, edge_color='gray', width=1.5)
    
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, font_size=8, font_weight="bold", horizontalalignment="center"
    )
    
    plt.margins(0.05)
    plt.tight_layout()
    png_path = output_dir / filename
    try:
        plt.savefig(png_path, bbox_inches='tight', dpi=200)
        logger.info(f"Tree image saved to {png_path}")
    except Exception as e:
        logger.warning(f"Could not save tree image: {e}")
    finally:
        plt.close()

