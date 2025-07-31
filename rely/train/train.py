import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import yaml
import json
from datetime import datetime
import numpy as np
from pathlib import Path
import random
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix,
    precision_recall_curve
)
from collections import Counter
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Union, Tuple
from pprint import pprint


@dataclass
class TrainerConfig:
    """Lightweight wrapper around the original YAML/dict configuration."""

    config: Dict[str, Any]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainerConfig":
        """Create a TrainerConfig from a YAML file path."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(config=data)

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying configuration dictionary (mutable)."""
        return self.config


class Trainer:
    """Unified training interface for both value and uncertainty heads."""

    def __init__(self, cfg: TrainerConfig):
        self.config = cfg.to_dict()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.run_dir: Optional[Path] = None

    # ---------- Public API ----------
    def train(self) -> dict:
        """Run training and return evaluation metrics as a dictionary."""
        self._create_run_dir()
        results = self._run_training()
        return results

    # ---------- Internals ----------
    def _create_run_dir(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = self.config.get("run_name", "unnamed_run")
        self.run_dir = Path("rely") / "train" / "runs" / f"{timestamp}_{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

    def _run_training(self) -> dict:
        """Core training logic with unified data processing."""
        config = self.config
        run_dir = self.run_dir
        device = self.device
        
        if run_dir is None:
            raise ValueError("Run directory not initialized. Call _create_run_dir() first.")

        # ---- Logging & Setup ----
        print(f"💡 Using device: {device}")
        print(f"🚀 Running task: {config['task']}")
        print(f"🧠 Using model type: {config['model_type']}")

        # ---- 1. Load activations ----
        activation_files = config['files']
        print(f"  - Found {len(activation_files)} file(s):")
        for f in sorted(activation_files):
            print(f"    - {f}")

        all_entries = []
        for file_path in sorted(activation_files):
            print(f"  - Loading {file_path}...")
            try:
                data_from_file = torch.load(file_path, weights_only=False)
                if isinstance(data_from_file, list):
                    # Tag each entry with its originating file so we can exclude certain files from validation
                    for entry in data_from_file:
                        if isinstance(entry, dict):
                            entry = dict(entry)  # Shallow copy to avoid mutating the original object
                            entry["__source_file__"] = file_path
                        all_entries.append(entry)
                else:
                    print(f"⚠️ Warning: File {Path(file_path).name} did not contain a list. Skipping.")
            except Exception as e:
                print(f"⚠️ Warning: Could not load or process {Path(file_path).name}. Error: {e}. Skipping.")

        if not all_entries:
            print("❌ Error: Dataset is empty after loading all files.")
            return {}

        # Shuffle and optional subsampling
        print(f"🔀 Shuffling {len(all_entries)} total entries...")
        random.shuffle(all_entries)
        
        # Get preprocessing configuration
        preprocess_cfg = config.get('preprocessing', {})
        percentage_dataset = float(preprocess_cfg.get('percentage_dataset', 1.0))
        if not 0 < percentage_dataset <= 1:
            print(f"⚠️ Warning: Invalid 'percentage_dataset' value {percentage_dataset}. Using 1.0 (100%).")
            percentage_dataset = 1.0
        if percentage_dataset < 1.0:
            keep_n = max(1, int(len(all_entries) * percentage_dataset))
            print(f"📉 Using only {keep_n}/{len(all_entries)} entries (~{percentage_dataset*100:.1f}%) as requested.")
            all_entries = all_entries[:keep_n]

        dataset_list = all_entries

        # ---- 2. Tensor preparation ----
        # Unified activation field handling
        activation_field = config.get('activation_field', 'activations')
        if activation_field == 'cut_cot_activations':
            X = torch.stack([e['cut_cot_activations'] for e in dataset_list]).float()
        else:
            X = torch.stack([e['activations'] for e in dataset_list]).float()
        
        # Optional normalization of input features
        if preprocess_cfg.get('normalize_inputs', False):
            mean = X.mean(dim=0, keepdim=True)
            std = X.std(dim=0, keepdim=True) + 1e-8
            X = (X - mean) / std
            print("🌀 Input features normalized (zero mean, unit variance).")
        input_dim = X.shape[1]

        # ---- 3. Label preparation (unified) ----
        if config['task'] == 'classification':
            clf_cfg = config['task_specific']['classification']
            y = self._prepare_classification_labels(dataset_list, clf_cfg)
            full_dataset = TensorDataset(X, y.unsqueeze(1))
        else:
            reg_cfg = config['task_specific']['regression']
            y = self._prepare_regression_labels(dataset_list, reg_cfg)
            full_dataset = TensorDataset(X, y.unsqueeze(1))

        # ---- 4. Train/Val split with optional file exclusion ----
        val_ratio = config['training']['val_split']
        raw_exclude = config.get('exclude_val', [])
        if raw_exclude is None:
            raw_exclude = []
        exclude_files = set(raw_exclude)

        # Separate indices based on whether their source file should be excluded from validation
        excluded_indices = [
            idx for idx, entry in enumerate(dataset_list)
            if entry.get("__source_file__") and (
                entry["__source_file__"] in exclude_files
                or Path(entry["__source_file__"]).name in exclude_files
            )
        ]
        eligible_indices = [idx for idx in range(len(dataset_list)) if idx not in excluded_indices]

        val_size = int(val_ratio * len(dataset_list))
        if val_size > len(eligible_indices):
            print(
                f"⚠️ Warning: Requested validation size ({val_size}) exceeds eligible samples "
                f"({len(eligible_indices)}). Reducing val_size to {len(eligible_indices)}."
            )
            val_size = len(eligible_indices)

        if val_size == 0:
            print("❌ Error: Validation set size is zero. Check 'val_split' or 'exclude_val' settings.")
            return {}

        val_indices = random.sample(eligible_indices, val_size)
        train_indices = [idx for idx in range(len(dataset_list)) if idx not in val_indices]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        print(f"Split dataset into {len(train_dataset)} training and {len(val_dataset)} validation samples.")
    
        # ---- 4b. Optional removal of near-zero labels from TRAINING set ----
        if preprocess_cfg.get('remove_zeroes', False):
            epsilon = float(preprocess_cfg.get('zero_tolerance', 0.01))  # tolerance window around 0
            train_dataset = self._remove_zeroes(full_dataset, train_dataset, epsilon)
            print(f"🧹 Removed samples with |label| <= {epsilon} from training set. New train size: {len(train_dataset)}")

        # ---- 5. Balance training set (optional) ----
        if config['task'] == 'classification' and config['task_specific']['classification'].get('balance_data', False):
            train_dataset = self._balance_training_set(full_dataset, train_dataset, config['task_specific']['classification'])

        # ---- 6. Dataloaders ----
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])

        # ---- 7. Model & Optimizer ----
        if config['model_type'] == 'linear':
            model = LinearProbe(input_dim).to(device)
        else:
            mlp_cfg = config['model_specific']['mlp']
            model = MLPProbe(input_dim, hidden_dims=mlp_cfg['hidden_dims'], dropout_p=mlp_cfg['dropout_p']).to(device)

        criterion = nn.BCEWithLogitsLoss() if config['task'] == 'classification' else nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training'].get('weight_decay', 0))

        scheduler = None
        sched_cfg = config['training'].get('scheduler', {})
        if sched_cfg.get('enabled', False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=sched_cfg.get('factor', 0.1), 
                patience=sched_cfg.get('patience', 5)
            )

        # ---- 8. Early stopping ----
        checkpoint_path = run_dir / 'checkpoint.pt'
        patience = config['training'].get('patience', 10)
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

        # ---- 9. Training loop ----
        print("\n--- Starting Training ---")
        for epoch in range(config['training']['epochs']):
            model.train()
            total_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)

            print(f"Epoch [{epoch+1}/{config['training']['epochs']}] -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            if scheduler:
                scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("🛑 Early stopping triggered!")
                break

        # ---- 10. Load best weights ----
        model.load_state_dict(torch.load(checkpoint_path))

        # ---- 11. Evaluation ----
        results = self._evaluate_model(model, val_loader, config, device)

        # ---- 12. Persist artifacts ----
        self._save_results(run_dir, results, model)

        return results

    def _prepare_classification_labels(self, dataset_list: List[Dict], clf_cfg: Dict) -> torch.Tensor:
        """Prepare labels for classification task with unified label types."""
        label_type = clf_cfg['label_type']
        threshold = clf_cfg.get('soft_label_threshold', 0.5)
        
        if label_type == 'hard':
            y = torch.tensor([entry['hard_label'] for entry in dataset_list]).float()
            print("  - Using 'hard_label' for training.")
        elif label_type == 'soft':
            y = torch.tensor([1.0 if entry['soft_label'] >= threshold else 0.0 for entry in dataset_list]).float()
            print(f"  - Using 'soft_label' binarized at threshold {threshold}.")
        elif label_type == 'entropy':
            y = torch.tensor([1.0 if entry['entropy'] >= threshold else 0.0 for entry in dataset_list]).float()
            print(f"  - Using 'entropy' binarized at threshold {threshold}.")
        elif label_type == 'variance':
            y = torch.tensor([1.0 if entry['variance'] >= threshold else 0.0 for entry in dataset_list]).float()
            print(f"  - Using 'variance' binarized at threshold {threshold}.")
        else:
            raise ValueError(f"Unknown label_type: {label_type}. Must be one of: 'hard', 'soft', 'entropy', 'variance'")
        
        # Print distribution
        positive_indices = (y == 1).nonzero(as_tuple=True)[0]
        negative_indices = (y == 0).nonzero(as_tuple=True)[0]
        n_positive = len(positive_indices)
        n_negative = len(negative_indices)
        print(f"  - Data distribution: {n_positive} positive (1) samples, {n_negative} negative (0) samples.")
        
        return y

    def _prepare_regression_labels(self, dataset_list: List[Dict], reg_cfg: Dict) -> torch.Tensor:
        """Prepare labels for regression task."""
        label_field = reg_cfg.get('label_field', 'soft_label')
        y = torch.tensor([e[label_field] for e in dataset_list]).float()
        print(f"  - Using '{label_field}' for regression training.")
        
        if reg_cfg.get('use_transformation', False):
            # y = logit_transform(y)
            y = torch.log(1 + y)
            print("  - Applied logit transformation to labels.")
        
        return y

    def _balance_training_set(self, full_dataset, train_dataset, clf_cfg):
        """Balance training set using various strategies."""
        strategy = clf_cfg.get('balance_strategy', 'undersample')
        print(f"⚖️ Balancing TRAINING dataset using '{strategy}' strategy...")

        train_indices = train_dataset.indices
        y_all = full_dataset.tensors[1].squeeze()
        train_y = y_all[train_indices]

        pos_idx = (train_y == 1).nonzero(as_tuple=True)[0]
        neg_idx = (train_y == 0).nonzero(as_tuple=True)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)

        if strategy == 'undersample':
            if n_pos > n_neg and n_neg > 0:
                perm = torch.randperm(n_pos)
                sampled_pos = pos_idx[perm[:n_neg]]
                balanced_rel = torch.cat([sampled_pos, neg_idx])
            elif n_neg > n_pos and n_pos > 0:
                perm = torch.randperm(n_neg)
                sampled_neg = neg_idx[perm[:n_pos]]
                balanced_rel = torch.cat([pos_idx, sampled_neg])
            else:
                balanced_rel = torch.arange(len(train_y))
            final_idx = [train_indices[int(i)] for i in balanced_rel]
            return Subset(full_dataset, final_idx)

        elif strategy == 'oversample':
            if n_pos > n_neg and n_neg > 0:
                majority, minority = pos_idx, neg_idx
            elif n_neg > n_pos and n_pos > 0:
                majority, minority = neg_idx, pos_idx
            else:
                minority = torch.tensor([])
                majority = torch.arange(len(train_y))
            if len(minority) > 0:
                n_to_add = len(majority) - len(minority)
                resampled = minority[torch.randint(0, len(minority), (n_to_add,))]
                balanced_rel = torch.cat([torch.arange(len(train_y)), resampled])
            else:
                balanced_rel = majority
            final_idx = [train_indices[int(i)] for i in balanced_rel]
            return Subset(full_dataset, final_idx)

        elif strategy == 'smote':
            smote_k = clf_cfg.get('smote_k_neighbors', 5)
            if n_neg <= smote_k or n_pos <= smote_k:
                print(f"⚠️ Warning: Minority class has {min(n_pos, n_neg)} samples, too few for SMOTE with k={smote_k}. Using original train data.")
                return train_dataset
            train_X_original = full_dataset.tensors[0][train_indices]
            X_np, y_np = train_X_original.numpy(), train_y.numpy()
            smote = SMOTE(random_state=42, k_neighbors=smote_k)
            resampled_data = smote.fit_resample(X_np, y_np)
            X_resampled, y_resampled = resampled_data[0], resampled_data[1]
            print(f"  - Original train distribution: {Counter(y_np)}")
            print(f"  - Resampled train distribution: {Counter(y_resampled)}")
            X_res_t = torch.from_numpy(X_resampled).float()
            y_res_t = torch.from_numpy(y_resampled).float()
            return TensorDataset(X_res_t, y_res_t.unsqueeze(1))

        else:
            print(f"⚠️ Warning: Unknown balance_strategy '{strategy}'. Using original training data.")
            return train_dataset

    def _remove_zeroes(self, full_dataset, train_dataset, epsilon: float = 0.01):
        """Remove training samples whose labels are approximately zero.

        Args:
            full_dataset (TensorDataset): The complete dataset (needed to reference original indices).
            train_dataset (Subset): Current training subset.
            epsilon (float): Samples with |label| <= epsilon will be discarded.

        Returns:
            Subset: A new training subset without near-zero-label samples.
        """
        if hasattr(train_dataset, 'indices'):
            # Standard case: `train_dataset` is a Subset produced by `random_split`
            train_indices = train_dataset.indices
            y_all = full_dataset.tensors[1].squeeze()
            train_y = y_all[train_indices]

            keep_mask = torch.abs(train_y) > epsilon
            if keep_mask.sum() == 0:
                print(f"⚠️ Warning: All training samples had |label| <= {epsilon}. Keeping original training set.")
                return train_dataset

            kept_indices = [train_indices[i] for i in keep_mask.nonzero(as_tuple=True)[0]]
            return Subset(full_dataset, kept_indices)

        # Fallback: if `train_dataset` is already a TensorDataset (e.g., after earlier processing)
        try:
            X_train, y_train = train_dataset.tensors  # type: ignore
            keep_mask = torch.abs(y_train.squeeze()) > epsilon
            if keep_mask.sum() == 0:
                print(f"⚠️ Warning: All training samples had |label| <= {epsilon}. Keeping original training set.")
                return train_dataset
            return TensorDataset(X_train[keep_mask], y_train[keep_mask])
        except AttributeError:
            # As a safety net, return the dataset unchanged if structure is unexpected
            print("⚠️ Warning: Could not apply zero-removal due to unexpected dataset structure.")
            return train_dataset

    def _find_optimal_threshold(self, labels, probs):
        """Find the optimal threshold that maximizes F1 score using precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        
        # Calculate F1 score for each threshold, avoiding division by zero
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        # Find the threshold that gives the best F1 score
        # Note: thresholds array is 1 shorter than precision/recall arrays
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude the last element of f1_scores
        optimal_threshold = thresholds[optimal_idx]
        best_f1 = f1_scores[optimal_idx]
        
        print(f"🔍 Auto-threshold optimization:")
        print(f"  - Optimal threshold: {optimal_threshold:.4f}")
        print(f"  - Best F1 score: {best_f1:.4f}")
        
        return float(optimal_threshold)

    def _evaluate_model(self, model, val_loader, config, device):
        """Evaluate model performance with unified metrics."""
        if config['task'] == 'classification':
            clf_cfg = config['task_specific']['classification']
            val_threshold = clf_cfg.get('val_threshold', 0.5)
            
            probs, labels = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X.to(device)).cpu()
                    probs.extend(torch.sigmoid(outputs).numpy())
                    labels.extend(batch_y.numpy())
            probs = np.array(probs).flatten()
            labels = np.array(labels).flatten()
            
            # Handle automatic threshold selection
            if val_threshold == "auto":
                threshold = self._find_optimal_threshold(labels, probs)
            else:
                threshold = float(val_threshold)
                print(f"🔍 Using fixed threshold: {threshold:.4f}")
            
            # Ensure threshold is a regular Python float for JSON serialization
            threshold = float(threshold)
            
            preds = (probs > threshold).astype(float)
            results = {
                'auroc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
                'f1_score': f1_score(labels, preds, zero_division=0),
                'accuracy': accuracy_score(labels, preds),
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'confusion_matrix': confusion_matrix(labels, preds).tolist(),
                'threshold': threshold,
            }
        else:
            use_transform = config['task_specific']['regression']['use_transformation']
            preds, labels = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X.to(device)).cpu()
                    preds_original = (torch.exp(outputs) - 1) if use_transform else outputs
                    labels_original = (torch.exp(batch_y) - 1) if use_transform else batch_y
                    preds.extend(preds_original.numpy())
                    labels.extend(labels_original.numpy())
            preds = np.array(preds).flatten()
            labels = np.array(labels).flatten()
            results = {
                'mean_squared_error': mean_squared_error(labels, preds),
                'mean_absolute_error': mean_absolute_error(labels, preds),
                'r_squared': r2_score(labels, preds),
            }
        return results

    def _save_results(self, run_dir: Path, results: dict, model):
        """Save training results and model."""
        output = {'config': self.config, 'results': results}
        with open(run_dir / 'results.json', 'w') as f:
            json.dump(output, f, indent=4)
        model_path = run_dir / Path(self.config['output_file']).name
        torch.save(model.state_dict(), model_path)
        print(f"\n💾 Artifacts saved to: {run_dir}")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'-- EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'-- Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LinearProbe(nn.Module):
    """A simple linear probe for regression or classification."""
    def __init__(self, input_dim):
        super(LinearProbe, self).__init__()
        self.layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.layer(x)


class MLPProbe(nn.Module):
    """A more advanced MLP probe with configurable depth, width, dropout, and batch norm."""
    def __init__(self, input_dim, hidden_dims=[512, 128], dropout_p=0.3):
        super(MLPProbe, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))  # Batch norm before activation
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            current_dim = h_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, 1)  # Final layer to produce logits

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)


def logit_transform(p, epsilon=1e-6):
    """Applies the logit transformation to a tensor of probabilities."""
    p = torch.clamp(p, min=epsilon, max=1-epsilon)
    return torch.log(p / (1 - p))


def inverse_logit_transform(x):
    """Applies the inverse of the logit transformation (sigmoid function)."""
    return torch.sigmoid(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a linear or MLP probe on model activations from files in a directory."
    )
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file.")
    config_path = parser.parse_args().config_file

    # Build objects using the new API
    try:
        trainer_cfg = TrainerConfig.from_yaml(config_path)
        print(f"✅ Configuration loaded from '{config_path}'")
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at '{config_path}'")
        exit()
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file: {e}")
        exit()

    trainer = Trainer(trainer_cfg)
    metrics = trainer.train()
    print("\n✅ Training finished. Metrics:")
    pprint(metrics) 