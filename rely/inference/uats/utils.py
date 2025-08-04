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

    _generate_tree_image(final_branches, run_dir)


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
    branches = searcher.search(user_question, system_prompt)
    logger.info(f"UATS search completed with {len(branches)} branches explored")

    logger.info(f"Saving branches to {save_dir}")
    save_branches(
        branches,
        save_dir,
        max_branch_tokens=config.budget,
        beam_width=config.beam_width,
        user_question=user_question,
        system_prompt=system_prompt,
    )

    return branches


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

    if not branches:
        return

    # Helper to build a tree from branches
    def add_nodes_edges(branch, graph, parent=None, branch_id=0):
        # Each node is uniquely identified by (branch_id, step_count)
        node_id = f"B{branch_id}_S{branch.step_count}"
        # Shorten text for display
        text_snippet = branch.text.strip().split(". ")[0][:40] + ("..." if len(branch.text.strip()) > 40 else "")
        label = f"u={branch.uncertainty:.2f if branch.uncertainty is not None else '?'}\nv={branch.value:.2f}\n{text_snippet}"
        graph.add_node(node_id, label=label)
        if parent:
            graph.add_edge(parent, node_id)
        # No children in this context; branches are leaves
        return node_id

    # Build a tree: root -> each branch's steps
    G = nx.DiGraph()
    root_id = "Start"
    G.add_node(root_id, label="Start")
    for branch_id, branch in enumerate(branches):
        # Reconstruct the path for this branch
        steps = []
        # Try to split the text into steps by '\n\n' (node delimiter)
        step_texts = branch.text.split("\n\n")
        parent = root_id
        for step_idx, step_text in enumerate(step_texts):
            node_id = f"B{branch_id}_S{step_idx+1}"
            # Only display uncertainty/value for the leaf
            if step_idx == len(step_texts) - 1:
                u = branch.uncertainty if branch.uncertainty is not None else 0.0
                v = branch.value if branch.value is not None else 0.0
                snippet = step_text.strip().split(". ")[0][:40] + ("..." if len(step_text.strip()) > 40 else "")
                label = f"u={u:.2f}\nv={v:.2f}\n{snippet}"
            else:
                label = textwrap.shorten(step_text.strip(), width=30, placeholder="...")
            G.add_node(node_id, label=label)
            G.add_edge(parent, node_id)
            parent = node_id

    # Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G)

    # Draw
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos,
        with_labels=False,
        node_size=2500,
        node_color="lightblue",
        font_size=9,
        font_weight="bold",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        linewidths=1.5,
    )
    # Draw labels
    node_labels = nx.get_node_attributes(G, 'label')
    for node, (x, y) in pos.items():
        plt.text(
            x, y,
            node_labels.get(node, node),
            ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5),
            wrap=True
        )
    plt.margins(0.2)
    plt.tight_layout()
    png_path = output_dir / filename
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Tree image saved to {png_path}") 