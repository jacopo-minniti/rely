import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import UATSConfig, Branch
from .guided_tree_search import GuidedTreeSearch
from .uncertainty_model import UATSUncertaintyModel
from .value_model import UATSValueModel
from rely.utils.text_utils import MATH_SYSTEM_PROMPT, normalize_answer

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small helpers / utilities
# -----------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison by:
    - Converting to lowercase
    - Removing all whitespace (spaces, tabs, newlines)
    - Removing common punctuation that doesn't affect mathematical meaning
    - Handling special cases like fractions, decimals, etc.
    """
    if not answer or answer == "?":
        return answer
    
    # Convert to string and lowercase
    normalized = str(answer).lower()
    
    # Remove all whitespace
    normalized = re.sub(r'\s+', '', normalized)
    
    # Remove common punctuation that doesn't affect meaning
    # Keep mathematical operators and decimal points
    normalized = re.sub(r'[,;:()[\]{}"]', '', normalized)
    
    # Handle common mathematical expressions
    # Convert fractions like "1/2" to consistent format
    normalized = re.sub(r'(\d+)/(\d+)', r'\1/\2', normalized)
    
    # Remove trailing zeros after decimal point
    if '.' in normalized:
        normalized = normalized.rstrip('0').rstrip('.')
    
    return normalized


def extract_final_answer(text: str) -> Optional[str]:
    """
    Finds the last \\boxed{} in a string and extracts its content,
    correctly handling nested braces.
    """
    # 1. Find the starting position of the last \boxed{
    start_marker = r'\boxed{'
    last_box_start_pos = text.rfind(start_marker)

    # If \boxed{ is not found, return None
    if last_box_start_pos == -1:
        return None

    # 2. The actual content starts right after the marker
    content_start_pos = last_box_start_pos + len(start_marker)

    # 3. Use a counter (brace_level) to find the matching closing brace
    brace_level = 1
    for i in range(content_start_pos, len(text)):
        char = text[i]
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1

        # 4. When the brace level is 0, we've found the matching brace
        if brace_level == 0:
            # The content is the substring between the start and this point
            return text[content_start_pos:i]

    # If the loop finishes, it means a matching closing brace was not found
    return None


# -----------------------------------------------------------------------------
# Model loading helpers
# -----------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    max_seq_length: int = 16384,
    dtype: str = "bfloat16",
    load_in_4bit: bool = True,
):
    """Load a standard transformers model and its tokenizer ready for inference."""
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        device_map="auto" if device == "auto" else {"": device},
        trust_remote_code=True,
        max_position_embeddings=max_seq_length if hasattr(AutoModelForCausalLM, 'config') else None,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        max_length=max_seq_length,
    )
    
    # Set pad token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model to eval mode
    model.eval()
    
    return model, tokenizer


def create_uats_searcher(config: UATSConfig) -> GuidedTreeSearch:
    """Instantiate a fully-configured :class:`GuidedTreeSearch` ready to run."""

    logger.info(f"Loading model: {config.model_name}")
    model, tokenizer = load_model_and_tokenizer(config.model_name, device=config.device)
    logger.info("Model and tokenizer loaded successfully")

    # Load uncertainty model
    logger.info(f"Loading uncertainty model: {config.uncertainty_model_path}")
    try:
        uncertainty_model = UATSUncertaintyModel(
            model_path=config.uncertainty_model_path,
            device=config.uncertainty_device,
            scoring_method=config.uncertainty_scoring_method
        )
        logger.info("Uncertainty model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load uncertainty model: {e}")
        raise

    # Load value model
    logger.info(f"Loading value model: {config.value_model_path}")
    try:
        value_model = UATSValueModel(
            model_path=config.value_model_path,
            device=config.value_device,
            scoring_method=config.value_scoring_method
        )
        logger.info("Value model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load value model: {e}")
        raise

    return GuidedTreeSearch(
        model=model,
        tokenizer=tokenizer,
        uncertainty_model=uncertainty_model,
        value_model=value_model,
        config=config,
    )


# -----------------------------------------------------------------------------
# Search orchestration & persistence helpers
# -----------------------------------------------------------------------------

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

    # The branches passed here have already been filtered to beam_width in the search function
    # and have final answers generated, so we just save them as-is
    final_branches = branches
    
    logger.debug(f"Saving {len(final_branches)} final branches")
    
    # Log summary of selected branches
    if final_branches:
        step_counts = [b.step_count for b in final_branches]
        logger.info(f"Selected branches: max_step={max(step_counts)}, min_step={min(step_counts)}")

    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "user_question": user_question,
        "system_prompt": system_prompt,
        "total_branches": len(final_branches),
        "branches": [],
    }
    
    # Collect all answers for evaluation
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
            branch_text = branch.text
            if branch.final_answer:
                final_answer_text = branch.final_answer.strip()
                f.write(branch_text)
                f.write(final_answer_text)
            else:
                f.write(branch_text)

        extracted_answer = ""
        if branch.final_answer:
            full_text = branch.text + branch.final_answer
            extracted_answer = extract_final_answer(full_text)
            if extracted_answer:
                all_answers.append(extracted_answer)
                # Check if this answer is correct
                if correct_answer and normalize_answer(extracted_answer) == normalize_answer(correct_answer):
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

    # Add evaluation metrics if correct_answer is provided
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

    _generate_tree_image(all_branches, output_path)


def run_uats(
    user_question: str,
    system_prompt: str = MATH_SYSTEM_PROMPT,
    config: Optional[UATSConfig] = None,
    save_dir: Optional[Union[str, Path]] = "uats_results",
    correct_answer: Optional[str] = None,
    uncertainty_threshold: Optional[float] = None,
) -> List[Branch]:
    """High-level one-shot helper wrapping search & persistence."""

    if config is None:
        config = UATSConfig()
    
    # Override uncertainty threshold if provided
    if uncertainty_threshold is not None:
        config.uncertainty_threshold = uncertainty_threshold

    logger.info("Creating UATS searcher…")
    searcher = create_uats_searcher(config)
    logger.info("Starting UATS search…")
    final_branches, all_branches = searcher.search(user_question, system_prompt)
    logger.info(f"UATS search completed with {len(final_branches)} branches explored")

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


# -----------------------------------------------------------------------------
# Tree visualisation helpers
# -----------------------------------------------------------------------------

def _safe_text_for_matplotlib(text: str) -> str:
    """Escape or remove problematic characters for matplotlib text rendering."""
    # Replace dollar signs with escaped versions to prevent LaTeX parsing
    text = text.replace('$', r'\$')
    # Remove other problematic LaTeX characters
    text = text.replace('\\', '\\\\')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('_', '\\_')
    text = text.replace('^', '\\^')
    return text

def _generate_tree_image(
    branches: List[Branch],
    output_dir: Path,
    filename: str = "search_tree.png",
) -> None:
    """Render a simple tree visualisation of the explored search space using matplotlib and networkx."""
    import matplotlib.pyplot as plt
    import networkx as nx
    import textwrap
    from rely.inference.uats.utils import extract_final_answer
    
    if not branches:
        return
    
    G = nx.DiGraph()
    root_id = None
    
    # Add all nodes
    for branch in branches:
        node_id = branch.id
        
        # Extract only the latest step
        branch_text = branch.text
        if branch_text.endswith('<|im_end|>'):
            branch_text = branch_text[:-len('<|im_end|>')]
        if branch_text.endswith('\n\n'):
            branch_text = branch_text[:-2]
        
        steps = branch_text.split('\n\n')
        latest_step = steps[-1].strip() if steps else branch_text.strip()
        # get only first 25 chars and then ellipsis
        latest_step = latest_step[:25] + '...'
        
        # Safely escape the text for matplotlib
        safe_text = _safe_text_for_matplotlib(latest_step)
        
        # Wrap the text for better display in the node
        wrapped_text = textwrap.fill(safe_text, width=30)
        
        # Always show both scores if available, on separate lines
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
            root_id = node_id
            G.nodes[node_id]['is_root'] = True

    # Add final answer nodes for branches that have them
    for branch in branches:
        if branch.final_answer:
            full_text = branch.text + branch.final_answer
            final_answer = extract_final_answer(full_text)
            safe_answer = _safe_text_for_matplotlib(final_answer or "No answer")
            final_node_id = f"final_{branch.id}"
            # Determine correctness
            correct = False
            # Use normalized comparison
            if final_answer is not None and hasattr(branch, 'correct_answer') and branch.correct_answer is not None:
                correct = normalize_answer(final_answer) == normalize_answer(branch.correct_answer)
            elif final_answer is not None and hasattr(branch, 'expected_answer') and branch.expected_answer is not None:
                correct = normalize_answer(final_answer) == normalize_answer(branch.expected_answer)
            # Store correctness for coloring
            G.add_node(final_node_id, label=safe_answer, is_final_answer=True, is_correct=correct)
            G.add_edge(branch.id, final_node_id)
    
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    plt.figure(figsize=(20, 15))
    
    # Separate nodes by type for custom styling
    root_nodes = [node for node, data in G.nodes(data=True) if data.get('is_root')]
    final_answer_nodes = [node for node, data in G.nodes(data=True) if data.get('is_final_answer')]
    intermediate_nodes = [node for node in G.nodes() if node not in root_nodes and node not in final_answer_nodes]

    # Draw intermediate nodes as rectangles
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=intermediate_nodes,
        node_shape="s", # square/rectangle
        node_color="lightblue",
        node_size=3500
    )

    # Draw root nodes as green circles
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=root_nodes,
        node_shape="o",
        node_color="lightgreen",
        node_size=3500
    )

    # Draw final answer nodes: green if correct, red if incorrect
    correct_final_nodes = [node for node in final_answer_nodes if G.nodes[node].get('is_correct', False)]
    incorrect_final_nodes = [node for node in final_answer_nodes if not G.nodes[node].get('is_correct', False)]
    if correct_final_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=correct_final_nodes,
            node_shape="o",
            node_color="green",
            node_size=3500
        )
    if incorrect_final_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=incorrect_final_nodes,
            node_shape="o",
            node_color="red",
            node_size=3500
        )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        edge_color='gray',
        width=1.5
    )
    
    # Draw labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=8,
        font_weight="bold",
        horizontalalignment="center"
    )
    
    plt.margins(0.05) # Reduced margins
    plt.tight_layout()
    png_path = output_dir / filename
    plt.savefig(png_path, bbox_inches='tight', dpi=200)
    plt.close()
    logger.info(f"Tree image saved to {png_path}")