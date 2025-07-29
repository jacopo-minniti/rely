# Rely Package

This package provides tools for uncertainty quantification in language models through fork-based analysis.

## Structure

The package is organized into three main modules:

### Complete Module (`rely.complete`)
Functions for generating model completions from various data sources using vLLM.

- `generate_from_dataset()`: Generate initial completions from Hugging Face datasets
- `complete_from_forks()`: Generate multiple completions from fork data
- `complete_from_jsonl()`: Generate completions from JSONL data

### Extract Module (`rely.extract`)
Functions for extracting model activations and creating fork points.

- `create_forks_from_dataset()`: Create fork points based on entropy thresholds
- `extract_activations()`: Extract model activations from data
- Parallel execution scripts: `fork_parallel.sh`, `activations_parallel.sh`

### Score Module (`rely.score`)
Functions for calculating various metrics from model outputs.

- `calculate_entropy_scores()`: Calculate entropy scores from completions
- `calculate_semantic_isotropy()`: Calculate semantic isotropy from embeddings

## Usage

### Basic API Usage

```python
from rely.api import *

# Generate initial completions from a dataset
generate_from_dataset(
    dataset_name="TIGER-Lab/MMLU-Pro",
    output_file="generations.jsonl",
    model="unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
)

# Create fork points from the generated data
create_forks_from_dataset(
    input_file="generations.jsonl",
    output_file="forks.pt",
    entropy_threshold=0.5195
)

# Generate multiple completions from fork data
complete_from_forks(
    input_file="forks.pt",
    output_file="completions.pt",
    n_completions_per_item=100
)

# Calculate entropy scores
calculate_entropy_scores(
    input_file="completions.pt",
    output_file="scores.pt"
)
```

### Command Line Usage

Each module can also be used from the command line:

```bash
# Generate completions
python -m rely.complete.generate --dataset TIGER-Lab/MMLU-Pro --output-file generations.jsonl

# Create forks
python -m rely.extract.fork --start_idx 0 --end_idx 1000 --output_path forks.pt

# Extract activations
python -m rely.extract.activations --input-file data.jsonl --output-file activations.pt --start-index 0 --end-index 100

# Calculate scores
python -m rely.score.entropy --input-file completions.pt --output-file scores.pt
```

### Parallel Execution

For large datasets, use the parallel execution scripts:

```bash
# Parallel fork creation
bash rely/extract/fork_parallel.sh

# Parallel activation extraction
bash rely/extract/activations_parallel.sh
```

## Dependencies

- torch
- vllm
- unsloth
- datasets
- tqdm
- numpy

## Installation

```bash
pip install -r requirements.txt
```

## Examples

See the original scripts in `scripts-8A100/` for examples of the original functionality that has been reorganized into this package structure. 