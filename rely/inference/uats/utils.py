import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import UATSConfig, Branch
from .guided_tree_search import GuidedTreeSearch
from .probes import load_probes
from .value_model import UATSValueModel
from rely.utils.text_utils import MMLU_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT
import random
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small helpers / utilities
# -----------------------------------------------------------------------------

# def extract_final_answer(branch_text: str, final_answer_text: str) -> str:
#     """Extract the final answer string (e.g. "(A)") from the concatenated texts."""

#     full_text = branch_text + final_answer_text

#     # Look for patterns like "(A)", "(B)", "(C)", "(D)" in the final answer section
#     if "## Final Answer" in full_text:
#         final_answer_section = full_text.split("## Final Answer")[-1].strip()
#     else:
#         final_answer_section = full_text

#     letter_pattern = r"\([A-Z]\)"
#     matches = re.findall(letter_pattern, final_answer_section)

#     if matches:
#         # Return the last match (most recent formatting)
#         return matches[-1].replace("(", "").replace(")", "")

#     return final_answer_section.strip()

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
# Model & probe loading helpers
# -----------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 16384,
    dtype: str = "bfloat16",
    load_in_4bit: bool = True,
):
    """Load a standard transformers model and its tokenizer ready for inference."""

    import torch
    from transformers import BitsAndBytesConfig
    
    # Set up quantization config if needed
    quantization_config = None
    # if load_in_4bit:
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
    #         bnb_4bit_use_double_quant=True,
    #     )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
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
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    logger.info("Model and tokenizer loaded successfully")

    hidden_size = model.config.hidden_size
    model_dtype = (
        model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype
    )

    # Load uncertainty probe
    logger.info(
        f"Loading uncertainty probe (hidden_size={hidden_size}, dtype={model_dtype}, device={config.probe_device})"
    )
    try:
        uncertainty_probe, _ = load_probes(
            hidden_size=hidden_size,
            model_dtype=model_dtype,
            uncertainty_probe_path=config.uncertainty_probe_path,
            value_probe_path=None,  # We don't need the value probe anymore
            device=config.probe_device,
        )
        logger.info("Uncertainty probe loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Probe file not found: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Failed to load probe: {e}")
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
        uncertainty_probe=uncertainty_probe,
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = f"_{random.randint(0, 100):03d}"
    output_path = Path(output_dir)
    run_dir = output_path / f"run_{timestamp}{random_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not branches:
        return

    # The branches passed here have already been filtered to beam_width in the search function
    # and have final answers generated, so we just save them as-is
    final_branches = branches
    
    logger.debug(f"Saving {len(final_branches)} final branches")
    
    # Log summary of selected branches
    if final_branches:
        step_counts = [b.step_count for b in final_branches]
        think_counts = sum(1 for b in final_branches if "</think>" in b.text)
        logger.info(f"Selected branches: max_step={max(step_counts)}, min_step={min(step_counts)}, with_</think>={think_counts}")

    summary_data = {
        "timestamp": timestamp,
        "user_question": user_question,
        "system_prompt": system_prompt,
        "total_branches": len(final_branches),
        "branches": [],
    }
    
    # Collect all answers for evaluation
    all_answers = []
    correct_count = 0

    for i, branch in enumerate(final_branches):
        filepath = run_dir / f"branch_{i}.txt"
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
            extracted_answer = extract_final_answer(branch.text, branch.final_answer)
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

    summary_filepath = run_dir / "results.json"
    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(final_branches)} branches and summary to {run_dir}")

    _generate_tree_image(all_branches, run_dir)


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

    logger.info("Creating UATS searcher…")
    searcher = create_uats_searcher(config)
    logger.info("Starting UATS search…")
    final_branches, all_branches = searcher.search(user_question, system_prompt)
    logger.info(f"UATS search completed with {len(final_branches)} branches explored")

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
        
        # Wrap the text for better display in the node
        wrapped_text = textwrap.fill(latest_step, width=30)
        
        label = ""
        if branch.uncertainty is not None and branch.value is not None:
            label = f"u={branch.uncertainty:.2f}, v={branch.value:.2f}\n\n{wrapped_text}"
        if branch.uncertainty is not None:
            label = f"u={branch.uncertainty:.2f}\n\n{wrapped_text}"
        if branch.value is not None:
            label = f"v={branch.value:.2f}\n\n{wrapped_text}"
        
        G.add_node(node_id, label=label, is_final=branch.final_answer is not None)
        
        if branch.parent_id is not None:
            G.add_edge(branch.parent_id, node_id)
        else:
            root_id = node_id
            G.nodes[node_id]['is_root'] = True

    # Add final answer nodes for branches that have them
    for branch in branches:
        if branch.final_answer:
            final_answer = extract_final_answer(branch.text, branch.final_answer)
            final_node_id = f"final_{branch.id}"
            G.add_node(final_node_id, label=final_answer, is_final_answer=True)
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
    
    # Draw root and final answer nodes as green circles
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=root_nodes + final_answer_nodes,
        node_shape="o", # circle
        node_color="lightgreen",
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