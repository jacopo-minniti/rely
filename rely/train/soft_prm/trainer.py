import os
import textwrap
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Callable, Optional, Union
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
        print("\n--- PRE-MASK DEBUGGING ---")
        print(f"Unique labels before masking: {np.unique(active_labels)}")
        print("--------------------------\n")
        mask = active_labels >= 0.001
        active_predictions = active_predictions[mask]
        active_labels = active_labels[mask]

    active_labels_count = len(active_labels)
    if active_labels_count == 0:
        return {"r2": 0.0, "mse": 0.0, "active_labels_count": 0}

    # --- Debugging Prints ---
    print("\n--- METRICS DEBUGGING ---")
    print(f"Sample labels: {active_labels[:10]}")
    print(f"Sample predictions: {active_predictions[:10]}")
    print(f"Label mean: {np.mean(active_labels):.4f}, std: {np.std(active_labels):.4f}")
    print(f"Prediction mean: {np.mean(active_predictions):.4f}, std: {np.std(active_predictions):.4f}")
    print("-------------------------\\n")
    # --- End Debugging Prints ---

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

        # Set the mask_zeros on the model if it supports it
        if hasattr(self.model, "set_mask_zeros"):
            self.model.set_mask_zeros(mask_zeros)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

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

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain "prompt", "completions", and "labels".
                Labels should be continuous float values.
            tokenizer (`PreTrainedTokenizerBase`):
                Tokenizer used to process the data.
            # ... (other args are the same)

        Returns:
            `dict[str, list]`:
                Tokenized sequences with "input_ids" and "labels".
        """
        if not hasattr(RegressionPRMTrainer.tokenize_row, "has_printed"):
            print("\n--- TOKENIZER DEBUG ---")
            print(f"Features for one sample: {features}")
            RegressionPRMTrainer.tokenize_row.has_printed = True

        # Apply chat template to the prompt if tokenizer has one
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            # Format prompt as a user message for instruct models
            messages = [{"role": "user", "content": features["prompt"]}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompt_ids = tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"]
        else:
            # Fallback to original behavior if no chat template
            prompt_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        
        # Tokenize the completions
        completions_ids = [
            tokenizer(completion, add_special_tokens=False)["input_ids"] for completion in features["completions"]
        ]
        
        # MODIFIED: Cast labels to float instead of int
        if train_on_last_step_only and not is_eval:
            labels = [-100.0] * (len(features["labels"]) - 1) + [float(features["labels"][-1])]
        else:
            labels = [float(label) for label in features["labels"]]

        # Get the ID of the separator token and add it to the completions
        separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
        completions_ids = [completion + separator_ids for completion in completions_ids]

        # Create the label
        # Use -100.0 for float compatibility
        labels = [[-100.0] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]

        # Join the completions and labels steps
        completion_ids = list(chain(*completions_ids))
        labels = list(chain(*labels))

        if tokenizer.bos_token_id is not None:
            prompt_ids = [tokenizer.bos_token_id] + prompt_ids

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_ids = prompt_ids[-max_prompt_length:]
        if max_completion_length is not None:
            completion_ids = completion_ids[:max_completion_length]
            labels = labels[:max_completion_length]

        input_ids = prompt_ids + completion_ids
        # Use -100.0 for float compatibility
        labels = [-100.0] * len(prompt_ids) + labels

        if max_length is not None:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

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

        # docstyle-ignore
        citation = textwrap.dedent('''
        @article{uesato2022solving,
            title        = {{Solving Math Word Problems With Process- and Outcome-Based Feedback}},
            author       = {Uesato, Jonathan and Kushman, Nate and Kumar, Ramana and Song, Francis and Siegel, Noah and Wang, Lisa and Creswell, Antonia and Irving, Geoffrey and Higgins, Irina},
            year         = 2022,
            journal      = {arXiv preprint arXiv:2211.14275}
        }''')

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.args.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            trainer_name="PRM",
            trainer_citation=citation,
            paper_title="Solving math word problems with process-and outcome-based feedback",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))