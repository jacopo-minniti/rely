import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from unsloth import FastLanguageModel
from ete3 import Tree, TreeStyle, faces, TextFace

from .config import UATSConfig, Branch
from .guided_tree_search import GuidedTreeSearch
from .probes import load_probes
from rely.utils.text_utils import MMLU_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small helpers / utilities
# -----------------------------------------------------------------------------

def extract_final_answer(branch_text: str, final_answer_text: str) -> str:
    """Extract the final answer string (e.g. "(A)") from the concatenated texts."""

    full_text = branch_text + final_answer_text

    # Look for patterns like "(A)", "(B)", "(C)", "(D)" in the final answer section
    if "## Final Answer" in full_text:
        final_answer_section = full_text.split("## Final Answer")[-1].strip()
    else:
        final_answer_section = full_text

    letter_pattern = r"\([A-D]\)"
    matches = re.findall(letter_pattern, final_answer_section)

    if matches:
        # Return the last match (most recent formatting)
        return matches[-1]

    return final_answer_section.strip()


# -----------------------------------------------------------------------------
# Model & probe loading helpers
# -----------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 4096,
    dtype: str = "bfloat16",
    load_in_4bit: bool = True,
):
    """Load an Unsloth model and its tokenizer ready for inference."""

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2× faster inference
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

    logger.info(
        f"Loading probes (hidden_size={hidden_size}, dtype={model_dtype}, device={config.probe_device})"
    )
    try:
        uncertainty_probe, value_probe = load_probes(
            hidden_size=hidden_size,
            model_dtype=model_dtype,
            uncertainty_probe_path=config.uncertainty_probe_path,
            value_probe_path=config.value_probe_path,
            device=config.probe_device,
        )
        logger.info("Probes loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Probe file not found: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Failed to load probe: {e}")
        raise

    return GuidedTreeSearch(
        model=model,
        tokenizer=tokenizer,
        uncertainty_probe=uncertainty_probe,
        value_probe=value_probe,
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
    beam_width: Optional[int] | None = None,
    user_question: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """Save the *active* branches at the end of the search to disk."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    run_dir = output_path / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not branches:
        return

    max_step_count = max(b.step_count for b in branches)
    final_branches = [b for b in branches if b.step_count == max_step_count]

    if beam_width is not None and len(final_branches) > beam_width:
        final_branches = sorted(final_branches, key=lambda b: b.score, reverse=True)[
            :beam_width
        ]

    logger.debug(f"Saving {len(final_branches)} final branches")

    summary_data = {
        "timestamp": timestamp,
        "user_question": user_question,
        "system_prompt": system_prompt,
        "total_branches": len(final_branches),
        "branches": [],
    }

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

    summary_filepath = run_dir / "results.json"
    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(final_branches)} branches and summary to {run_dir}")

    _generate_tree_image(all_branches, run_dir)


def run_uats(
    user_question: str,
    system_prompt: str = MMLU_SYSTEM_PROMPT,
    config: Optional[UATSConfig] = None,
    save_dir: Optional[Union[str, Path]] = "uats_results",
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
        beam_width=config.beam_width,
        user_question=user_question,
        system_prompt=system_prompt,
    )

    return final_branches


# -----------------------------------------------------------------------------
# Tree visualisation helpers (ETE3)
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
    final_answer_nodes = []
    
    # Add all nodes
    for branch in branches:
        node_id = branch.id
        
        # Extract only the latest step (split by '\n\n' and take the last part)
        steps = branch.text.split('\n\n')
        latest_step = steps[-1].strip() if steps else branch.text.strip()
        # Remove system prompt if present (look for <|im_start|>system)
        if '<|im_start|>system' in latest_step:
            # Find where the actual user content starts
            if '<|im_start|>user' in latest_step:
                user_start = latest_step.find('<|im_start|>user')
                latest_step = latest_step[user_start:]
            elif '<|im_start|>assistant' in latest_step:
                assistant_start = latest_step.find('<|im_start|>assistant')
                latest_step = latest_step[assistant_start:]
        
        # Clean up the step text
        if '<|im_start|>assistant' in latest_step:
            latest_step = latest_step.split('<|im_start|>assistant')[-1].strip()
        if '<|im_end|>' in latest_step:
            latest_step = latest_step.split('<|im_end|>')[0].strip()
        
        # Get the first sentence or first 40 chars
        first_sentence = latest_step.split('. ')[0][:40] + ("..." if len(latest_step.split('. ')[0]) > 40 else "")
        
        u_str = f"{branch.uncertainty:.2f}" if branch.uncertainty is not None else "?"
        v_str = f"{branch.value:.2f}" if branch.value is not None else "?"
        label = f"u={u_str}\nv={v_str}\n{first_sentence}"
        
        G.add_node(node_id, label=label, is_final=branch.final_answer is not None)
        
        if branch.parent_id is not None:
            G.add_edge(branch.parent_id, node_id)
        else:
            root_id = node_id
    
    # If no explicit root, add a dummy root
    if root_id is None:
        root_id = "Start"
        G.add_node(root_id, label="Start", is_root=True)
        for branch in branches:
            if branch.parent_id is None:
                G.add_edge(root_id, branch.id)
    
    # Add final answer nodes for branches that have them
    for branch in branches:
        if branch.final_answer:
            final_answer = extract_final_answer(branch.text, branch.final_answer)
            final_node_id = f"final_{branch.id}"
            final_answer_nodes.append(final_node_id)
            G.add_node(final_node_id, label=f"Final Answer:\n{final_answer}", is_final_answer=True)
            G.add_edge(branch.id, final_node_id)
    
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G)
    
    plt.figure(figsize=(15, 10))
    
    # Separate nodes by type
    root_nodes = []
    final_answer_nodes = []
    regular_nodes = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data.get('is_root', False):
            root_nodes.append(node)
        elif node_data.get('is_final_answer', False):
            final_answer_nodes.append(node)
        else:
            regular_nodes.append(node)
    
    # Draw regular nodes as squares
    if regular_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=regular_nodes,
            node_shape="s",
            node_color="lightblue",
            node_size=3000
        )
    
    # Draw root nodes as circles
    if root_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=root_nodes,
            node_shape="o",
            node_color="lightgreen",
            node_size=3500
        )
    
    # Draw final answer nodes as circles
    if final_answer_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=final_answer_nodes,
            node_shape="o",
            node_color="lightcoral",
            node_size=3000
        )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        edge_color='gray'
    )
    
    # Draw labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=8,
        font_weight="bold"
    )
    
    plt.margins(0.2)
    plt.tight_layout()
    png_path = output_dir / filename
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Tree image saved to {png_path}") 