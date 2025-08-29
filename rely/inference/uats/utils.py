import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from multiprocessing import Process, Queue

import torch
import textwrap
import matplotlib.pyplot as plt
import networkx as nx
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification

from .config import UATSConfig, Branch
from .guided_tree_search import GuidedTreeSearch
from .scorer import Scorer
from rely.utils.text_utils import MATH_SYSTEM_PROMPT, extract_final_answer, normalize_answer

logger = logging.getLogger(__name__)


def _model_server(model_path, device, task_queue, result_queue, model_type, config):
    """
    Generic server for running inference on a model.
    Loads the correct model class based on model_type and uses the Scorer.
    """
    if model_type == "value":
        # The value model (PRM) is a custom model loaded with AutoModel.
        model_class = AutoModel
        scoring_method = config.value_scoring_method
    elif model_type == "uncertainty":
        # The uncertainty model (PUM) is a token classification model.
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
        request_id, rank, text = task_queue.get()
        if text is None:
            break
        
        score = scorer.score(text)
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
    system_prompt: str = MATH_SYSTEM_PROMPT,
    config: Optional[UATSConfig] = None,
    save_dir: Optional[Union[str, Path]] = "uats_results",
    correct_answer: Optional[str] = None,
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

    uncertainty_server = Process(target=_model_server, args=(config.uncertainty_model_path, config.uncertainty_device, uncertainty_task_queue, uncertainty_result_queue, "uncertainty", config))
    value_server = Process(target=_model_server, args=(config.value_model_path, config.value_device, value_task_queue, value_result_queue, "value", config))
    
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
            max_branch_tokens=config.budget,
            user_question=user_question,
            system_prompt=system_prompt,
            correct_answer=correct_answer,
        )

    return final_branches


def save_branches(
    branches: List[Branch],
    all_branches: List[Branch],
    output_dir: Union[str, Path],
    max_branch_tokens: int,
    user_question: Optional[str] = None,
    system_prompt: Optional[str] = None,
    correct_answer: Optional[str] = None,
) -> None:
    """Save the *active* branches at the end of the search to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not branches:
        return

    final_branches = branches
    
    logger.debug(f"Saving {len(final_branches)} final branches")
    
    if final_branches:
        step_counts = [b.step_count for b in final_branches]
        logger.info(f"Selected branches: max_step={max(step_counts)}, min_step={min(step_counts)}")

    summary_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "user_question": user_question,
        "system_prompt": system_prompt,
        "total_branches": len(final_branches),
        "branches": [],
    }
    
    all_answers = []
    correct_count = 0

    for i, branch in enumerate(final_branches):
        filepath = output_path / f"branch_{i}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Step count: {branch.step_count}\n")
            f.write(f"Score: {branch.score:.4f}\n")
            f.write(f"Uncertainty: {branch.uncertainty}\n")
            f.write(f"Value: {branch.value:.4f}\n")
            f.write(f"Total tokens: {branch.total_tokens}\n")
            f.write("\n--- Branch Text ---\n")
            f.write(branch.text)

        if not branch.final_answer:
            branch.final_answer = extract_final_answer(branch.text)

        extracted_answer = branch.final_answer or "Not found"
        normalized_answer = normalize_answer(extracted_answer)
        all_answers.append(normalized_answer)

        if correct_answer and normalized_answer != "Not found" and normalize_answer(correct_answer) == normalized_answer:
            correct_count += 1

        summary_data["branches"].append(
            {
                "branch_id": i,
                "step_count": branch.step_count,
                "score": branch.score,
                "uncertainty": branch.uncertainty,
                "value": branch.value,
                "total_tokens": branch.total_tokens,
                "extracted_answer": extracted_answer,
            }
        )

    if correct_answer is not None:
        hard_label = 1 if correct_count > 0 else 0
        soft_label = correct_count / len(all_answers) if all_answers else 0.0
        
        summary_data["correct_answer"] = correct_answer
        summary_data["all_answers"] = all_answers
        summary_data["hard_label"] = hard_label
        summary_data["soft_label"] = soft_label
        summary_data["correct_count"] = correct_count
        summary_data["total_answers"] = len(all_answers)

    summary_filepath = output_path / "results.json"
    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(final_branches)} branches and summary to {output_path}")

    _generate_tree_image(all_branches, output_path, correct_answer=correct_answer)


def _generate_tree_image(
    branches: List[Branch],
    output_dir: Path,
    filename: str = "search_tree.png",
    correct_answer: Optional[str] = None,
) -> None:
    """Render a simple tree visualisation of the explored search space using matplotlib and networkx."""
    
    if not branches:
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
        
        latest_step = re.sub(r'\\boxed{(.*?)}', r'\1', latest_step)
        latest_step = (latest_step[:25] + '...') if len(latest_step) > 25 else latest_step
        
        wrapped_text = textwrap.fill(latest_step, width=30)
        
        score_lines = []
        if branch.value is not None:
            score_lines.append(f"v={branch.value:.2f}")
        if branch.uncertainty is not None:
            score_lines.append(f"u={branch.uncertainty:.2f}")
        label = "\n".join(score_lines) + f"\n{wrapped_text}" if score_lines else wrapped_text
        
        G.add_node(node_id, label=label, is_final=branch.final_answer is not None)
        
        if branch.parent_id is not None:
            G.add_edge(branch.parent_id, node_id)
        else:
            G.nodes[node_id]['is_root'] = True

    for branch in branches:
        if branch.final_answer:
            final_answer_text = re.sub(r'\\boxed{(.*?)}', r'\1', branch.final_answer)
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
    
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>"), arrowsize=15, edge_color='gray', width=1.5)
    
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, font_size=8, font_weight="bold", horizontalalignment="center"
    )
    
    plt.margins(0.05)
    plt.tight_layout()
    png_path = output_dir / filename
    plt.savefig(png_path, bbox_inches='tight', dpi=200)
    plt.close()
    logger.info(f"Tree image saved to {png_path}")