# completer.py
import os
import random
import logging
from multiprocessing import Process
from time import sleep
from typing import Optional, Literal
from collections import namedtuple

import openai
from pydantic import BaseModel
from tqdm import tqdm

from rely.utils import format_prompt, load_dataset, save_dataset, merge, MATH_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)

# --- Mock objects to maintain compatibility with the output processing logic ---
# The original script expected a specific structure from vLLM's output.
# We create these simple mock objects to wrap the OpenAI API response
# so that the rest of the script can remain unchanged.
CompletionOutput = namedtuple("CompletionOutput", ["text"])
RequestOutput = namedtuple("RequestOutput", ["outputs"])


class CompleterConfig(BaseModel):
    # --- Online Inference Configuration ---
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "NULL"  # Dummy key for local vLLM server

    # --- Data and Prompt Configuration ---
    dp_size: int = 8  # Number of parallel workers to query the API
    forking_strategy: Literal["entropy", "newline"] = "newline"
    completion_type: Literal["short", "long"] = "long"
    system_prompt: str = MATH_SYSTEM_PROMPT
    dataset: str
    subset: Optional[str] = None
    split: str = "train"
    question_field: str = "question"


class Completer:

    def __init__(self, config: CompleterConfig):
        self.config = config

    def _format_prompt_with_completion_type(self, question: str, cut_cot: str) -> str:
        """
        Formats the prompt based on completion_type. For "short" completions, it adds
        a suffix to encourage the model to provide only the final answer.
        """
        if self.config.completion_type == "short":
            forcing_text = "\n\nI reasoned enough, the user wants a final answer.\n## Final Answer\n\\boxed{"
            modified_cot = cut_cot + forcing_text
            return format_prompt(question, system_prompt=self.config.system_prompt, cot=modified_cot)
        
        return format_prompt(question, system_prompt=self.config.system_prompt, cot=cut_cot)

    def generate(
        self,
        output_file: str,
        n_completions_per_item: int = 100,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        cot_percentage: float = 1.0,
    ):
        """
        Generates completions by querying the vLLM server with multiple parallel workers.

        Args:
            output_file: Path to save the generated completions.
            n_completions_per_item: Number of completions to generate for each CoT step.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature for generation.
            cot_percentage: Maximum percentage of CoT steps to sample from (0.0 to 1.0).
        """
        dp_size = self.config.dp_size

        if dp_size > 1:
            processes = []
            for rank in range(dp_size):
                proc = Process(
                    target=self._worker,
                    args=(
                        output_file,
                        n_completions_per_item,
                        max_new_tokens,
                        temperature,
                        cot_percentage,
                        rank,
                        dp_size,
                    ),
                )
                proc.start()
                processes.append(proc)

            exit_code = 0
            for proc in processes:
                proc.join()  # Wait for all processes to complete
                if proc.exitcode != 0:
                    logging.error(f"Process {proc.pid} exited with error code {proc.exitcode}.")
                    exit_code = proc.exitcode

            if exit_code == 0:
                self._merge_output_files(output_file, dp_size)
            else:
                logging.error("One or more processes failed. Skipping file merge.")
                raise RuntimeError(f"Completion failed with exit code {exit_code}")
        else:
            logging.info("Running in single-process mode (dp_size=1).")
            self._worker(
                output_file,
                n_completions_per_item,
                max_new_tokens,
                temperature,
                cot_percentage,
                0,  # rank
                1,  # dp_size
            )
            base, ext = os.path.splitext(output_file)
            # Rename the single output file to the final name
            os.rename(f"{base}.rank0{ext}", output_file)

        logging.info(f"Processing complete! Output written to {output_file}")

    def _worker(
        self,
        output_file,
        n_completions_per_item,
        max_new_tokens,
        temperature,
        cot_percentage,
        rank,
        dp_size,
    ):
        """
        A worker process that queries the vLLM server for a subset of the data.
        """
        logging.info(f"Starting worker: rank={rank}, pid={os.getpid()}")

        # --- Initialize OpenAI client to connect to the local vLLM server ---
        try:
            client = openai.OpenAI(
                base_url=self.config.api_base,
                api_key=self.config.api_key,
            )
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client on rank {rank}: {e}")
            return

        # --- Load and partition the dataset for this worker ---
        all_data = load_dataset(
            self.config.dataset,
            subset=self.config.subset,
            split=self.config.split,
            question_field=self.config.question_field,
        )
        if not all_data:
            logging.warning(f"No data loaded on rank {rank}. Exiting.")
            return

        chunk_size = (len(all_data) + dp_size - 1) // dp_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_data))
        data_chunk = all_data[start_idx:end_idx]

        if not data_chunk:
            logging.warning(f"Rank {rank} has no items to process. Exiting.")
            return
        logging.info(f"Rank {rank} preparing to process {len(data_chunk)} items.")

        # --- Prepare all prompts before making API calls ---
        prompts_to_process = []
        source_indices = []  # To map results back to original items

        for item_idx, item in enumerate(data_chunk):
            question = item.get("question", "")
            if not question:
                continue

            if self.config.forking_strategy == "entropy":
                cut_cot = item.get("cut_cot", "")
                prompt = self._format_prompt_with_completion_type(question, cut_cot)
                prompts_to_process.append(prompt)
                source_indices.append({"item_idx": item_idx, "cut_cot": cut_cot, "sample_idx": 0})
            else: # "newline" strategy
                attempt = item.get("attempt", "")
                if not attempt:
                    continue
                steps = attempt.split("\n\n")
                max_step_to_sample = int((len(steps) - 1) * cot_percentage)
                for step_index in range(max_step_to_sample + 1):
                    cut_cot = "\n\n".join(steps[:step_index + 1])
                    prompt = self._format_prompt_with_completion_type(question, cut_cot)
                    prompts_to_process.append(prompt)
                    source_indices.append({"item_idx": item_idx, "cut_cot": cut_cot, "sample_idx": step_index})

        if not prompts_to_process:
            logging.warning(f"Rank {rank} has no valid prompts. Exiting.")
            return

        # --- Make API calls and collect outputs ---
        logging.info(f"Rank {rank} starting generation for {len(prompts_to_process)} prompts...")
        all_outputs = []
        adjusted_max_tokens = min(50, max_new_tokens) if self.config.completion_type == "short" else max_new_tokens

        # Create progress bar for this worker
        progress_bar = tqdm(
            total=len(prompts_to_process),
            desc=f"Rank {rank} Processing",
            position=rank,
            leave=True
        )

        for i, prompt in enumerate(prompts_to_process):
            try:
                response = client.completions.create(
                    model=self.config.model,
                    prompt=prompt,
                    max_tokens=adjusted_max_tokens,
                    temperature=temperature,
                    n=n_completions_per_item,
                    stop=["<|eot_id|>"],  # Recommended stop token for Qwen models
                )
                # Wrap the response in our mock object for compatibility
                outputs = [CompletionOutput(c.text) for c in response.choices]
                all_outputs.append(RequestOutput(outputs))
            except Exception as e:
                logging.error(f"API call failed on rank {rank} for prompt {i}: {e}")
                # Add a dummy failure entry to keep lists aligned
                all_outputs.append(RequestOutput([CompletionOutput(f"API_ERROR: {e}")] * n_completions_per_item))
            
            # Update progress bar
            progress_bar.update(1)

        progress_bar.close()

        # --- Process and save the results for this worker's chunk ---
        processed_outputs = {}
        for source_info, output_group in zip(source_indices, all_outputs):
            item_idx = source_info["item_idx"]

            if item_idx not in processed_outputs:
                processed_outputs[item_idx] = {
                    "original_item": data_chunk[item_idx],
                    "samples": []
                }
            
            completions = [out.text for out in output_group.outputs]
            processed_outputs[item_idx]["samples"].append({
                "sample_idx": source_info["sample_idx"],
                "cut_cot": source_info["cut_cot"],
                "completions": completions
            })

        output_data = list(processed_outputs.values())
        base, ext = os.path.splitext(output_file)
        output_file_path = f"{base}.rank{rank}{ext}"

        save_dataset(output_data, output_file_path)
        logging.info(f"Rank {rank} finished. Output written to {output_file_path}")

    def _merge_output_files(self, output_file: str, dp_size: int) -> None:
        """Merges the temporary output files from each worker into a single file."""
        base, ext = os.path.splitext(output_file)
        input_files = [f"{base}.rank{rank}{ext}" for rank in range(dp_size)]
        logging.info(f"Merging output files: {input_files}")
        merge(input_files, output_file)
        logging.info(f"Final merged output written to {output_file}")