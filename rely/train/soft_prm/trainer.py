import os
import textwrap
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Optional, Union
import numpy as np

import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset, features
from sklearn.metrics import mean_squared_error, r2_score
from transformers import (
    BaseImageProcessor,
    DataCollator,
    DataCollatorForTokenClassification,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from trl import PRMConfig
from trl.trainer.utils import disable_dropout_in_model, generate_model_card

if is_wandb_available():
    import wandb


def compute_regression_metrics(eval_pred: EvalPrediction, mask_zeros: bool = False):
    """
    Computes R2 and MSE scores for regression tasks.
    Filters out predictions where the label is -100.
    """
    predictions, labels = eval_pred
    # Filter out ignored indices
    active_predictions = predictions[labels != -100]
    active_labels = labels[labels != -100]

    if mask_zeros:
        mask = active_labels >= 0.001
        active_predictions = active_predictions[mask]
        active_labels = active_labels[mask]

    active_labels_count = len(active_labels)
    if active_labels_count == 0:
        return {"r2": 0.0, "mse": 0.0, "active_labels_count": 0}

    r2 = r2_score(active_labels, active_predictions)
    mse = mean_squared_error(active_labels, active_predictions)

    return {"r2": r2, "mse": mse, "active_labels_count": active_labels_count}


class RegressionPRMTrainer(Trainer):
    """
    Initialize RegressionPRMTrainer.

    This trainer is adapted for PRM-style (Process Reward Model) training for regression tasks.
    It expects continuous, real-valued labels.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably a `RegressionPRMModel` or similar model.
        args (`PRMConfig`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default
            (`DataCollatorForTokenClassification`) will be used.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer ([`~transformers.PreTrainedTokenizerBase`], ...):
            Tokenizer used to process the data.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training.
        compute_metrics (`Callable[[transformers.EvalPrediction], dict]`, *optional*, defaults to `compute_regression_metrics`):
            The metrics to use for evaluation. Defaults to MSE and R2 for regression.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        mask_zeros (`bool`, *optional*, defaults to `False`):
            Whether to mask labels with values close to zero (< 0.001) during loss calculation.
    """

    _tag_names = ["trl", "prm", "regression"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[PRMConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        tokenizer: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        mask_zeros: bool = False,
    ):

        if args.disable_dropout:
            disable_dropout_in_model(model)
        
        if compute_metrics is None:
            compute_metrics = partial(compute_regression_metrics, mask_zeros=mask_zeros)

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "A tokenizer must be specified when using the default DataCollatorForTokenClassification"
                )
            data_collator = DataCollatorForTokenClassification(
                tokenizer, 
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt"
            )

        if "input_ids" not in train_dataset.column_names:
            with PartialState().main_process_first():
                fn_kwargs = {
                    "tokenizer": tokenizer,
                    "step_separator": args.step_separator,
                    "max_length": args.max_length,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    "train_on_last_step_only": args.train_on_last_step_only,
                }
                train_fn_kwargs = {**fn_kwargs, "is_eval": False}
                train_dataset = train_dataset.map(
                    self.tokenize_row,
                    fn_kwargs=train_fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    remove_columns=train_dataset.features,
                    desc="Tokenizing train dataset",
                    features=features.Features(
                        {
                            "labels": features.Sequence(features.Value("float32")),
                            "input_ids": features.Sequence(features.Value("int64")),
                        }
                    ),
                )

                eval_fn_kwargs = {**fn_kwargs, "is_eval": True}
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.map(
                        self.tokenize_row,
                        fn_kwargs=eval_fn_kwargs,
                        num_proc=args.dataset_num_proc,
                        remove_columns=eval_dataset.features,
                        desc="Tokenizing eval dataset",
                        features=features.Features(
                            {
                                "labels": features.Sequence(features.Value("float32")),
                                "input_ids": features.Sequence(features.Value("int64")),
                            }
                        ),
                    )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        print("AFTER PREPROCESSING")
        print(f"{eval_dataset[0]['labels']}\n{eval_dataset[1]['labels']}")

        # Set the mask_zeros on the model if it supports it
        if hasattr(self.model, "set_mask_zeros"):
            self.model.set_mask_zeros(mask_zeros)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def training_step(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: int) -> torch.Tensor:
        # Default training step behavior to get the loss
        loss = super().training_step(model, inputs)

        # Log every `logging_steps`
        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            # Perform a forward pass in eval mode to get logits
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]

                # Filter out ignored indices
                active_logits = logits[labels != -100]
                active_labels = labels[labels != -100]

                # Apply mask_zeros if enabled on the model
                if hasattr(model, "mask_zeros") and model.mask_zeros:
                    mask = active_labels >= 0.001
                    active_logits = active_logits[mask]
                    active_labels = active_labels[mask]

                if active_labels.numel() > 0:
                    # Move to CPU and convert to numpy for printing
                    active_labels_np = active_labels.detach().cpu().numpy()
                    active_logits_np = active_logits.detach().cpu().numpy()

                    print("\n--- TRAINING STEP DEBUG ---")
                    print(f"Step: {self.state.global_step}")
                    print(f"Sample labels:      {active_labels_np[:15]}")
                    print(f"Sample predictions: {active_logits_np[:15]}")
                    print(f"Label stats:      mean={np.mean(active_labels_np):.4f}, std={np.std(active_labels_np):.4f}")
                    print(f"Prediction stats: mean={np.mean(active_logits_np):.4f}, std={np.std(active_logits_np):.4f}")
                    print("---------------------------\n")
            # Switch back to train mode
            model.train()

        return loss

    @staticmethod
    def tokenize_row(
        features,
        tokenizer,
        step_separator,
        max_length,
        max_prompt_length,
        max_completion_length,
        train_on_last_step_only,
        is_eval,
    ):
        """
        Tokenize a row of the dataset for regression.

        This simplified version applies the chat template to the full conversation
        and then aligns labels with separator tokens.

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain "prompt", "completions", and "labels".
            tokenizer (`PreTrainedTokenizerBase`):
                Tokenizer used to process the data.
            # ... (other args)

        Returns:
            `dict[str, list]`:
                Tokenized sequences with "input_ids" and "labels".
        """
        if not hasattr(RegressionPRMTrainer.tokenize_row, "has_printed"):
            print("\n--- TOKENIZER DEBUG ---")
            print(f"Features for one sample: {features}")
            RegressionPRMTrainer.tokenize_row.has_printed = True

        # Construct assistant response, adding separator after each step
        assistant_response = "".join([comp + step_separator for comp in features["completions"]])
        
        # Format conversation using chat template
        messages = [
            {"role": "user", "content": features["prompt"]},
            {"role": "assistant", "content": assistant_response},
        ]

        # Tokenize the whole conversation
        input_ids = tokenizer.apply_chat_template(
            messages,
            max_length=max_length,
            truncation=True,
            add_generation_prompt=False,
        )

        # Create labels, aligning them with the separator token
        labels = [-100.0] * len(input_ids)
        
        separator_token_id = tokenizer.convert_tokens_to_ids(step_separator)

        separator_indices = [i for i, token_id in enumerate(input_ids) if token_id == separator_token_id]

        original_labels = [float(label) for label in features["labels"]]

        # The number of separators should match the number of labels.
        # Truncation might cut some off.
        num_steps = min(len(separator_indices), len(original_labels))

        if train_on_last_step_only and not is_eval:
            if num_steps > 0:
                last_label_idx = separator_indices[num_steps - 1]
                labels[last_label_idx] = original_labels[num_steps - 1]
        else:
            for i in range(num_steps):
                label_idx = separator_indices[i]
                labels[label_idx] = original_labels[i]

        return {"input_ids": input_ids, "labels": labels}

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        if "JOB_ID" in os.environ:
            tags.add("hf_jobs")

        tags.update(self._tag_names)

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.args.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            trainer_name="PRM",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))