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
    cot_percentage: float = 1.0  # Max percentage of CoT to sample from (0.0 to 1.0)


class Completer:

    def __init__(self, config: CompleterConfig):
        self.config = config


    def split_attempt(self, item: dict) -> str:
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
            max_sampled_step = int(max_step * self.config.cot_percentage)
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
        max_new_tokens: int = 512,
        temperature: float = 1.0,
    ):
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
                        max_new_tokens,
                        temperature,
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
                proc.join(timeout=15000)
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
                max_new_tokens,
                temperature,
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


    def _worker(self, output_file, n_completions_per_item, max_new_tokens, temperature, model, trust_remote_code, dp_size, tp_size, gpu_memory_utilization, max_num_seqs, global_dp_rank, local_dp_rank, master_addr, master_port):

        logging.info(f"Starting worker: global_rank={global_dp_rank}, local_rank={local_dp_rank}")
        os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = master_addr
        os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

        # Always use load_dataset, which supports both local and HF datasets
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
        logging.info(f"DP rank {global_dp_rank} preparing {len(data_chunk)} prompts (from index {start_idx} to {end_idx-1}).")

        prompts_to_process = []
        source_metadata = []

        for item in data_chunk:
            cut_cot = self.split_attempt(item)
            question = item.get("question", "")
            if not question or not cut_cot:
                logging.warning(f"Skipping item on rank {global_dp_rank}: missing question or cut_cot")
                continue
            prompt = self.build_prompt(question, cut_cot)
            prompts_to_process.append(prompt)
            source_metadata.append({
                "original_item": item,
                "cut_cot": cut_cot,
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
        processed_count = 0
        output_data = []

        for source_info, output_group in zip(source_metadata, all_outputs):
            completions = [out.text for out in output_group.outputs]
            out_data = source_info["original_item"].copy()
            out_data["cut_cot"] = source_info["cut_cot"]
            out_data["completions"] = completions
            output_data.append(out_data)
            processed_count += 1

        save_dataset(output_data, output_file_path)
        
        logging.info(f"DP rank {global_dp_rank} finished. Processed {processed_count} examples.")
        logging.info(f"Output for rank {global_dp_rank} written to {output_file_path}")
        
        sleep(1)


    def _merge_output_files(self, output_file: str, dp_size: int) -> None:
        base, ext = os.path.splitext(output_file)
        input_files = [f"{base}.rank{rank}{ext}" for rank in range(dp_size)]
        logging.info(f"Merging output files: {input_files}")
        merge(input_files, output_file)
        logging.info(f"Final merged output written to {output_file}")