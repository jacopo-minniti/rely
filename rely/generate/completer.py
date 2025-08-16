import os
import random
from multiprocessing import Process
from time import sleep
import logging
from typing import Optional, Literal

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
from pydantic import BaseModel

from rely.utils import load_dataset, save_dataset, merge, MMLU_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)


class CompleterConfig(BaseModel):
    model: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
    tp_size: int = 1
    dp_size: int = 8
    max_num_seqs: int = 512
    forking_strategy: Literal["entropy", "newline"] = "newline"
    completion_type: Literal["short", "long"] = "long"
    system_prompt: str = MMLU_SYSTEM_PROMPT
    dataset: str  # Can be local file path or HF dataset name
    subset: Optional[str] = None  # For HF datasets
    split: str = "train"  # For HF datasets
    question_field: str = "question"


class Completer:

    def __init__(self, config: CompleterConfig):
        self.config = config


    def split_attempt(self, item: dict, cot_percentage: float = 1.0, used_steps: Optional[set] = None) -> str:
        if self.config.forking_strategy == "entropy":
            return item.get("cut_cot", "")
        else:
            attempt = item.get("attempt", "")
            if not attempt:
                return ""
            steps = attempt.split("\n\n")
            if len(steps) <= 1:
                return attempt
            
            max_step = len(steps) - 1
            # Calculate the maximum step based on cot_percentage
            max_sampled_step = int(max_step * cot_percentage)
            
            # Ensure different samples when used_steps is provided
            if used_steps is not None:
                available_steps = [i for i in range(max_sampled_step + 1) if i not in used_steps]
                if not available_steps:
                    # If all steps have been used, fall back to random selection
                    sampled_step = random.randint(0, max_sampled_step) if max_sampled_step > 0 else 0
                else:
                    sampled_step = random.choice(available_steps)
                used_steps.add(sampled_step)
            else:
                sampled_step = random.randint(0, max_sampled_step) if max_sampled_step > 0 else 0
                
            cut_steps = steps[:sampled_step + 1]

            return "\n\n".join(cut_steps)


    def build_prompt(self, question: str, cut_cot: str) -> str:
        if self.config.completion_type == "short":
            return f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{cut_cot}\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.</think>\n## Final Answer\n"
        else:
            return f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{cut_cot}\n\n"


    def generate(
        self,
        output_file: str,
        n_completions_per_item: int = 100,
        n_items_per_cot: int = 1,  # Number of random samples per CoT
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        cot_percentage: float = 1.0,  # Max percentage of CoT to sample from (0.0 to 1.0)
    ):
        """
        Generate completions from the dataset with support for multiple random samples per CoT.
        
        Args:
            output_file: Path to save the generated completions
            n_completions_per_item: Number of completions to generate per prompt
            n_items_per_cot: Number of random samples to generate from each CoT. 
                           Each sample will have n_completions_per_item completions.
                           For example, if n_items_per_cot=3 and n_completions_per_item=100,
                           each CoT will generate 3 different random cuts, each with 100 completions.
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature for generation
            cot_percentage: Maximum percentage of CoT steps to sample from (0.0 to 1.0)
        """
        node_size = 1
        node_rank = 0
        master_addr = "127.0.0.1"
        master_port = 13345
        gpu_memory_utilization = 0.9
        tp_size = self.config.tp_size
        dp_size = self.config.dp_size
        max_num_seqs = self.config.max_num_seqs
        model = self.config.model
        trust_remote_code = True

        if dp_size > 1:
            if node_size > 1:
                _master_addr = master_addr
                _master_port = master_port
            else:
                _master_addr = "127.0.0.1"
                _master_port = get_open_port()
                logging.info(f"Single node run. Using master address {_master_addr}:{_master_port}")
            if dp_size % node_size != 0:
                raise ValueError("dp_size must be divisible by node_size.")
            dp_per_node = dp_size // node_size
            processes = []
            for local_dp_rank in range(dp_per_node):
                global_dp_rank = node_rank * dp_per_node + local_dp_rank
                proc = Process(
                    target=self._worker,
                    args=(
                        output_file,
                        n_completions_per_item,
                        n_items_per_cot,
                        max_new_tokens,
                        temperature,
                        cot_percentage,
                        model,
                        trust_remote_code,
                        dp_size,
                        tp_size,
                        gpu_memory_utilization,
                        max_num_seqs,
                        global_dp_rank,
                        local_dp_rank,
                        _master_addr,
                        _master_port,
                    ),
                )
                proc.start()
                processes.append(proc)
            exit_code = 0
            for proc in processes:
                proc.join(timeout=35000)
                if proc.exitcode is None:
                    logging.warning(f"Process {proc.pid} timed out. Killing...")
                    proc.kill()
                    exit_code = 1
                elif proc.exitcode != 0:
                    logging.error(f"Process {proc.pid} exited with error code {proc.exitcode}.")
                    exit_code = proc.exitcode
            if exit_code == 0:
                self._merge_output_files(output_file, dp_size)
            else:
                logging.warning("One or more processes failed. Skipping file merge.")
            if exit_code != 0:
                raise RuntimeError(f"Completion failed with exit code {exit_code}")
        else:
            logging.info("Running in single-process mode (dp_size=1).")
            self._worker(
                output_file,
                n_completions_per_item,
                n_items_per_cot,
                max_new_tokens,
                temperature,
                cot_percentage,
                model,
                trust_remote_code,
                dp_size,
                tp_size,
                gpu_memory_utilization,
                max_num_seqs,
                0,  # global_dp_rank
                0,  # local_dp_rank
                '127.0.0.1',
                get_open_port()
            )
            base, ext = os.path.splitext(output_file)
            os.rename(f"{base}.rank0{ext}", output_file)
            logging.info(f"Processing complete! Output written to {output_file}")


    def _worker(self, output_file, n_completions_per_item, n_items_per_cot, max_new_tokens, temperature, cot_percentage, model, trust_remote_code, dp_size, tp_size, gpu_memory_utilization, max_num_seqs, global_dp_rank, local_dp_rank, master_addr, master_port):
        logging.info(f"Starting worker: global_rank={global_dp_rank}, local_rank={local_dp_rank}")
        os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = master_addr
        os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

        all_data = load_dataset(
            self.config.dataset,
            subset=self.config.subset,
            split=self.config.split,
            question_field=self.config.question_field,
        )
        if all_data is None:
            return
        
        total_items = len(all_data)
        chunk_size = total_items // dp_size
        remainder = total_items % dp_size

        def get_start_index(rank):
            return rank * chunk_size + min(rank, remainder)
            
        start_idx = get_start_index(global_dp_rank)
        end_idx = get_start_index(global_dp_rank + 1)
        data_chunk = all_data[start_idx:end_idx]

        if not data_chunk:
            logging.warning(f"DP rank {global_dp_rank} has no items to process. Exiting worker.")
            return
        logging.info(f"DP rank {global_dp_rank} preparing {len(data_chunk)} items.")

        prompts_to_process = []
        # CHANGE 1: Track the original item's index instead of copying the whole item.
        source_indices = []

        # Use enumerate to get a unique, hashable index for each item.
        for item_idx, item in enumerate(data_chunk):
            question = item.get("question", "")
            if not question:
                logging.warning(f"Skipping item on rank {global_dp_rank}: missing question")
                continue
            
            # Track sampled steps to ensure different samples
            used_steps = set()
            
            for sample_idx in range(n_items_per_cot):
                cut_cot = self.split_attempt(item, cot_percentage, used_steps)
                # You might want to check for empty cut_cot here as well.
                
                prompt = self.build_prompt(question, cut_cot)
                prompts_to_process.append(prompt)
                # Store the index and the cut_cot for later reconstruction.
                source_indices.append({
                    "item_idx": item_idx, 
                    "cut_cot": cut_cot,
                    "sample_idx": sample_idx
                })

        if not prompts_to_process:
            logging.warning(f"DP rank {global_dp_rank} has no valid prompts to process after filtering. Exiting.")
            return
        
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            trust_remote_code=trust_remote_code,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True
        )

        logging.info(f"DP rank {global_dp_rank} starting generation for {len(prompts_to_process)} prompts...")
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n_completions_per_item,
        )

        all_outputs = llm.generate(prompts_to_process, sampling_params)
        base, ext = os.path.splitext(output_file)
        output_file_path = f"{base}.rank{global_dp_rank}{ext}"

        # CHANGE 2: Use a much more efficient aggregation strategy.
        # This dictionary will hold the final structured data, indexed by the item's original index.
        processed_outputs = {}

        for source_info, output_group in zip(source_indices, all_outputs):
            item_idx = source_info["item_idx"]
            
            # If we haven't seen this item before, initialize its entry.
            # This only runs ONCE per original item, avoiding data duplication.
            if item_idx not in processed_outputs:
                processed_outputs[item_idx] = {
                    "original_item": data_chunk[item_idx],
                    "samples": []
                }
            
            # Append the new sample's results to the correct item.
            completions = [out.text for out in output_group.outputs]
            processed_outputs[item_idx]["samples"].append({
                "sample_idx": source_info["sample_idx"],
                "cut_cot": source_info["cut_cot"],
                "completions": completions
            })

        # CHANGE 3: Convert the dictionary values to a list for saving.
        output_data = list(processed_outputs.values())
        
        save_dataset(output_data, output_file_path)
        
        logging.info(f"DP rank {global_dp_rank} finished. Processed {len(output_data)} original items.")
        logging.info(f"Output for rank {global_dp_rank} written to {output_file_path}")
        
        sleep(1)


    def _merge_output_files(self, output_file: str, dp_size: int) -> None:
        base, ext = os.path.splitext(output_file)
        input_files = [f"{base}.rank{rank}{ext}" for rank in range(dp_size)]
        logging.info(f"Merging output files: {input_files}")
        merge(input_files, output_file)
        logging.info(f"Final merged output written to {output_file}")