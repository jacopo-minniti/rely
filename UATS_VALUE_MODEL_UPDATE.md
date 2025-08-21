# UATS Value Model Update

This update replaces the value probe in UATS with an autoregressive LLM that has a classification head, similar to the approach used in SBS (Step-level Beam Search).

## Key Changes

### 1. Configuration Updates (`config.py`)
- **Removed**: `value_probe_path` parameter
- **Added**: `value_model_path` - HuggingFace model path (default: "Qwen/Qwen2.5-Math-PRM-7B")
- **Added**: `value_scoring_method` - How to combine step rewards ("product", "minimum", "average", "last_step")
- **Added**: `value_device` - Separate device for the value model (default: "cuda:1")

### 2. New Value Model (`value_model.py`)
- **UATSValueModel**: Autoregressive LLM with classification head for step-wise value estimation
- Uses `<extra_0>` tokens to separate reasoning steps
- Supports multiple scoring methods:
  - `"product"`: Product of all step rewards (penalizes weak steps)
  - `"minimum"`: Minimum step reward (focuses on weakest link)  
  - `"average"`: Average of all step rewards (balanced approach)
  - `"last_step"`: Only the last step reward (for new generations)

### 3. GuidedTreeSearch Updates (`guided_tree_search.py`)
- **Replaced**: `_value_probe()` method with `_value_model_score()`
- **Added**: Question storage for value model context
- **Added**: Reasoning text extraction from full prompts
- **Updated**: Constructor to accept `UATSValueModel` instead of value probe

### 4. Utility Updates (`utils.py`)
- **Updated**: `create_uats_searcher()` to load value model instead of value probe
- **Updated**: Probe loading to make value probe optional

### 5. Script Updates (`scripts/uats.py`)
- **Replaced**: `--value_probe_path` with `--value_model_path`
- **Added**: `--value_scoring_method` parameter with choices

### 6. Unsloth Removal
- **Removed**: All unsloth dependencies and imports
- **Replaced**: `FastLanguageModel.from_pretrained()` with standard `AutoModelForCausalLM.from_pretrained()`
- **Added**: Support for BitsAndBytesConfig for 4-bit quantization
- **Updated**: Default model from unsloth-specific to standard HuggingFace models

## Usage

### Basic Usage
```python
from rely.inference.uats import UATSConfig, run_uats

config = UATSConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    uncertainty_probe_path="models/c_uncertainty_probe.pth",
    value_model_path="Qwen/Qwen2.5-Math-PRM-7B",
    value_scoring_method="average",  # or "product", "minimum", "last_step"
    beam_width=3,
    budget=4000
)

branches = run_uats(
    user_question="What is 2 + 2?",
    config=config,
    save_dir="results"
)
```

### Command Line Usage
```bash
python scripts/uats.py \
    --start_idx 0 \
    --end_idx 10 \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --uncertainty_probe_path "models/c_uncertainty_probe.pth" \
    --value_model_path "Qwen/Qwen2.5-Math-PRM-7B" \
    --value_scoring_method "average" \
    --beam_width 4 \
    --budget 4000
```

## Value Scoring Methods

- **product**: Multiplies all step rewards. Good for penalizing paths with any weak steps.
- **minimum**: Uses the minimum step reward. Focuses on the single weakest link in reasoning.
- **average**: Averages all step rewards. Balanced approach considering all steps equally.
- **last_step**: Uses only the last step reward. Useful when comparing new generation quality.

## Technical Details

The value model processes reasoning text by:

1. Splitting reasoning into steps (by double newlines)
2. Inserting `<extra_0>` tokens between steps  
3. Creating chat template with system/user/assistant messages
4. Running through the autoregressive model with classification head
5. Extracting step-wise rewards using token masking
6. Combining rewards using the specified scoring method

This approach provides more nuanced step-level value estimation compared to simple probes, allowing the model to understand the semantic content and mathematical correctness of each reasoning step.
