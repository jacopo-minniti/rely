# Unified Training Interface

This directory contains a unified training interface that supports both value head and uncertainty head training with a clean, composable API.

## Overview

The unified interface combines the best features from both `train-value.py` and `train-uncertainty.py` into a single, flexible training system that supports:

- **Value Head Training**: Classification tasks using variance, soft labels, or hard labels
- **Uncertainty Head Training**: Regression tasks using entropy or other continuous metrics
- **Flexible Data Formats**: Support for both `activations` and `cut_cot_activations` fields
- **Advanced Features**: Data balancing, early stopping, learning rate scheduling, input normalization

## Files

- `train.py` - The unified training interface
- `config.yaml` - Main configuration template
- `config-value-example.yaml` - Example config for value head training
- `config-uncertainty-example.yaml` - Example config for uncertainty head training

## Quick Start

### 1. Basic Usage

```bash
python train.py config.yaml
```

### 2. Value Head Training

```bash
python train.py config-value-example.yaml
```

### 3. Uncertainty Head Training

```bash
python train.py config-uncertainty-example.yaml
```

## Configuration Guide

### General Configuration

```yaml
# Basic settings
run_name: "my_experiment"
task: "classification"  # or "regression"
model_type: "mlp"       # or "linear"
output_file: "probe.pth"

# Data files
files:
  - "path/to/your/data.pt"

# Activation field (for compatibility)
activation_field: "activations"  # or "cut_cot_activations"
```

### Data Preprocessing

```yaml
# Use subset of dataset
percentage_dataset: 0.5  # Use 50% of data

# Normalize inputs
preprocessing:
  normalize_inputs: true  # Zero mean, unit variance
```

### Training Hyperparameters

```yaml
training:
  epochs: 128
  batch_size: 256
  learning_rate: 5.0e-4
  weight_decay: 1.0e-2
  val_split: 0.15
  patience: 15  # Early stopping
  
  # Learning rate scheduling
  scheduler:
    enabled: true
    factor: 0.5
    patience: 5
```

### Classification Task Configuration

```yaml
task_specific:
  classification:
    # New unified label system (recommended)
    label_type: "variance"  # Options: "hard", "soft", "variance"
    soft_label_threshold: 0.1
    
    # Legacy system (for backward compatibility)
    # label_field: "entropy"
    # data_split_threshold: 0.01
    
    # Evaluation
    val_threshold: 0.5
    
    # Data balancing
    balance_data: true
    balance_strategy: 'smote'  # Options: 'undersample', 'oversample', 'smote'
    smote_k_neighbors: 5
```

### Regression Task Configuration

```yaml
task_specific:
  regression:
    label_field: "entropy"  # Field name in your data
    use_transformation: false  # Apply logit transformation
```

### Model Configuration

```yaml
model_specific:
  mlp:
    hidden_dims: [512, 128]  # Hidden layer dimensions
    dropout_p: 0.6          # Dropout probability
```

## Key Features

### 1. Unified Label System

The interface supports both old and new label systems:

**New System (Recommended):**
```yaml
label_type: "variance"  # or "hard", "soft"
soft_label_threshold: 0.1
```

**Legacy System:**
```yaml
label_field: "entropy"
data_split_threshold: 0.01
```

### 2. Flexible Data Formats

Support for different activation field names:
- `activations` - Standard format
- `cut_cot_activations` - Uncertainty training format

### 3. Data Balancing Strategies

- **Undersampling**: Reduce majority class
- **Oversampling**: Duplicate minority class
- **SMOTE**: Synthetic Minority Over-sampling Technique

### 4. Advanced Training Features

- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rates
- **Input Normalization**: Optional feature standardization
- **Model Checkpointing**: Saves best model during training

## Output Structure

Each training run creates a timestamped directory with:

```
runs/
└── 2024-01-15_14-30-25_my_experiment/
    ├── config.yaml          # Saved configuration
    ├── results.json         # Training metrics
    ├── checkpoint.pt        # Best model weights
    └── probe.pth           # Final model
```

## Example Use Cases

### Value Head Training

For training a value head that predicts whether a model's output is correct:

```yaml
task: "classification"
activation_field: "activations"
task_specific:
  classification:
    label_type: "variance"
    soft_label_threshold: 0.1
    balance_data: true
    balance_strategy: 'smote'
```

### Uncertainty Head Training

For training an uncertainty head that predicts confidence scores:

```yaml
task: "regression"
activation_field: "cut_cot_activations"
task_specific:
  regression:
    label_field: "entropy"
    use_transformation: false
```

## Migration Guide

### From train-value.py

1. Use `label_type` instead of the old label system
2. Set `activation_field: "activations"`
3. Keep your existing balancing and evaluation settings

### From train-uncertainty.py

1. Use `label_field` in the regression section
2. Set `activation_field: "cut_cot_activations"`
3. The preprocessing and training settings remain the same

## Troubleshooting

### Common Issues

1. **"Dataset is empty"**: Check your file paths and data format
2. **"Not enough data for splits"**: Increase your dataset size or reduce `val_split`
3. **SMOTE errors**: Ensure minority class has enough samples for k-neighbors

### Data Format Requirements

Your data files should contain lists of dictionaries with:
- `activations` or `cut_cot_activations`: Model activation tensors
- `soft_label`, `hard_label`, `variance`, or `entropy`: Target labels

## API Reference

### TrainerConfig

```python
from train import TrainerConfig

# Load from YAML
config = TrainerConfig.from_yaml("config.yaml")

# Access configuration
config_dict = config.to_dict()
```

### Trainer

```python
from train import Trainer

# Create trainer
trainer = Trainer(config)

# Run training
metrics = trainer.train()
```

## Contributing

When adding new features:

1. Maintain backward compatibility
2. Add comprehensive configuration options
3. Update this README with examples
4. Test with both value and uncertainty head scenarios 