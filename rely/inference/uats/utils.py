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
    """Render a circular tree visualisation of the explored search space."""

    if Tree is None or TreeStyle is None:
        logger.warning("ETE3 library is not installed; skipping tree image generation.")
        return

    if not branches:
        return

    try:
        # ------------------------------------------------------------------
        # Build a *hierarchical* tree so that individual branches visibly
        # diverge from the root and from one another.  Each reasoning step
        # becomes a level in the tree (depth = step_count).  A unique path
        # is created for every explored branch, meaning that nodes from
        # different branches are *not* merged – this guarantees a clear
        # separation of lines in the rendered image while keeping the logic
        # simple and deterministic.
        # ------------------------------------------------------------------
        root = Tree(name="Start")

        # Helper: create – or fetch if already created – a child with a
        # specific name under a parent node.  Returns the child node.
        def _get_or_create_child(parent, child_name):
            for ch in parent.children:
                if ch.name == child_name:
                    return ch
            return parent.add_child(name=child_name)

        for branch_id, branch in enumerate(branches):
            current = root
            # Build the path Step1 -> Step2 -> … -> StepN for this branch.
            for depth in range(1, branch.step_count + 1):
                step_node_name = f"B{branch_id}_step{depth}"
                current = _get_or_create_child(current, step_node_name)

            # Attach the *leaf* information (score, tokens, etc.).
            current.add_features(
                step_count=branch.step_count,
                score=branch.score,
                uncertainty=branch.uncertainty,
                value=branch.value,
                total_tokens=branch.total_tokens,
            )

        # ------------------------------------------------------------------
        # Styling – keep it minimal so that the branch structure is the star
        # ------------------------------------------------------------------
        ts = TreeStyle()
        ts.show_leaf_name = False  # We manually draw labels via layout_fn
        ts.show_branch_length = False
        ts.show_branch_support = False

        def layout(node):
            from ete3 import faces, TextFace  # Local import to keep global deps minimal
            # Root – give it a friendly label
            if node.is_root():
                faces.add_face_to_node(TextFace("Start", fsize=10, fgcolor="black"), node, 0)
                node.img_style["size"] = 12
                node.img_style["shape"] = "sphere"
                node.img_style["fgcolor"] = "black"
                return

            # Leaves – show collected metrics
            if not node.children:
                info_lines = [
                    f"Step: {getattr(node, 'step_count', '?')}",
                    f"Score: {getattr(node, 'score', 0):.3f}",
                    f"Tokens: {getattr(node, 'total_tokens', 0)}",
                ]
                if getattr(node, 'uncertainty', None) is not None:
                    info_lines.insert(2, f"Unc.: {getattr(node, 'uncertainty'): .3f}")
                label = "\n".join(info_lines)
                faces.add_face_to_node(TextFace(label, fsize=8), node, 0, position="branch-right")
                node.img_style["size"] = 8
                node.img_style["shape"] = "circle"
                node.img_style["fgcolor"] = "darkgreen"
            else:
                # Internal nodes – draw small grey circles.
                node.img_style["size"] = 4
                node.img_style["shape"] = "circle"
                node.img_style["fgcolor"] = "grey"

        ts.layout_fn = layout
        ts.mode = "r"  # Rectangular – better for binary-like layouts
        ts.branch_vertical_margin = 10

        png_path = output_dir / filename
        root.render(str(png_path), w=1000, units="px", tree_style=ts)
        logger.info(f"Tree image saved to {png_path}")

    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to generate tree image: {e}")
        import traceback

        logger.warning(f"Traceback: {traceback.format_exc()}") 