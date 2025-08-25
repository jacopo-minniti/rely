import argparse
import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import transformers
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from rely.utils import MATH_SYSTEM_PROMPT


def make_step_rewards(logits, token_masks):
    """Extract step rewards using the exact same logic as pum-eval.py"""
    # Get probabilities from logits
    probabilities = F.softmax(logits, dim=-1)  # shape: [batch, seq_len, 2] for binary classification
    
    # Apply token masks to only consider step separator positions
    masked_probs = probabilities * token_masks.unsqueeze(-1)  # [batch, seq_len, 2]
    
    all_scores_res = []
    for i in range(probabilities.size(0)):  # for each batch
        sample_probs = masked_probs[i]  # [seq_len, 2]
        
        # Find positions where we have step separators
        step_positions = torch.where(token_masks[i])[0]
        
        # For binary classification, extract the positive class probability (index 1)
        step_rewards = []
        for pos in step_positions:
            pos_probs = sample_probs[pos]  # [2]
            positive_prob = pos_probs[1].item()  # probability of positive class
            step_rewards.append(positive_prob)
        
        all_scores_res.append(step_rewards)
    
    return all_scores_res


def collate_fn(batch, tokenizer, separator = '\n\n'):
    input_ids = []
    token_masks = []
    labels = []
    
    for i in batch:
        # Use the same chat template approach as pum-eval.py
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": i['prompt']},
            {"role": "assistant", "content": "\n\n".join(i['completions']) + "\n\n"},
        ]
        conversation_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

        prompt_ids = tokenizer.encode(
            conversation_str, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        
        # Create token mask for step separators (\n\n)
        token_mask = torch.zeros_like(prompt_ids, dtype=torch.bool)
        for token_idx, token_id in enumerate(prompt_ids[0]):
            decoded = tokenizer.decode([token_id])
            if '\n\n' in decoded:
                token_mask[0, token_idx] = True
        
        token_masks.append(token_mask)
        
        # Find the first False label index, or -1 if all are True
        i['label'] = i['labels'].index(False) if False in i['labels'] else -1
        labels.append(i['label'])
        input_ids.append(prompt_ids)
    
    # right pad input_ids and token_masks
    pad_token_id = tokenizer.pad_token_id
    max_len = max([i.size(-1) for i in input_ids])
    for i, input_idx in enumerate(input_ids):
        pad_len = max_len - input_idx.size(-1)
        input_ids[i] = torch.cat([
            input_idx.squeeze(), 
            torch.LongTensor(
                [pad_token_id] * pad_len
            )
        ])
        # Pad token masks with False
        token_masks[i] = torch.cat([
            token_masks[i].squeeze(),
            torch.BoolTensor([False] * pad_len)
        ])
    
    input_ids = torch.stack(input_ids)
    token_masks = torch.stack(token_masks)

    return dict(
        input_ids=input_ids,
        labels=labels,
        token_masks=token_masks
    )

def gather_objects(data, accelerator):
    world_size = accelerator.num_processes
    if world_size == 1:
        return data
        
    all_data = [None] * world_size
    torch.distributed.all_gather_object(all_data, data)
    
    if accelerator.is_main_process:
        result = []
        for process_data in all_data:
            result.extend(process_data)
        return result
    return None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    bs = args.batch_size
    num_of_workers = args.num_of_workers
    separator = args.separator
    model_path = args.model

    model_name = model_path.split('/')[-1]

    all_f1_scores = []
    save_dir = f'outputs/{model_name}'
    os.makedirs(save_dir, exist_ok=True)

    accelerator = Accelerator()
    print(f'Loading model from {model_path}')
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = accelerator.prepare(model)
    model.eval()

    configs = {
        'math': [207, 193], # error / correct num
    }

    for config, num in configs.items():
        dataset = load_dataset("jacopo-minniti/MATH-PUM-qwen2.5-1.5B", "pp-1", split="test")
        dataset = dataset.shuffle(seed=42).select(range(1000))

        sampler = None
        if accelerator.distributed_type == "MULTI_GPU":
            sampler = DistributedSampler(
                dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False,
            )
        dataloader = DataLoader(
            dataset, 
            batch_size=bs, 
            collate_fn=lambda x: x, 
            num_workers=num_of_workers,
            sampler=sampler,
            drop_last=False,
        )

        res_data = []
        for batch_ in tqdm(dataloader, disable=not accelerator.is_main_process):
            new_batch = deepcopy(batch_)

            batch = collate_fn(batch_, tokenizer, separator)
            input_ids = batch['input_ids'].to(accelerator.device)
            token_masks = batch['token_masks'].to(accelerator.device)
            labels = batch['labels']

            with accelerator.autocast(), torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            # Use the same step reward extraction as pum-eval.py
            step_rewards_batch = make_step_rewards(logits, token_masks)
            
            for i, step_rewards in enumerate(step_rewards_batch):
                label = labels[i]
                
                # Convert probabilities to binary predictions (>0.5 threshold)
                binary_preds = [1 if prob > 0.5 else 0 for prob in step_rewards]
                
                # Find the first 0 (False) prediction, or -1 if all are 1 (True)
                prediction_step = binary_preds.index(0) if 0 in binary_preds else -1
                
                new_batch[i]['label'] = label
                new_batch[i]['prediction'] = prediction_step
                new_batch[i]['match'] = prediction_step == label
                new_batch[i]['step_probabilities'] = step_rewards  # Store for debugging
            
            res_data.extend(new_batch)
        
        accelerator.wait_for_everyone()
        gathered_data = gather_objects(res_data, accelerator)

        if accelerator.is_main_process:
            data1 = [e for e in gathered_data if e['label'] != -1]
            data2 = [e for e in gathered_data if e['label'] == -1]
            
            with open(f'{save_dir}/{config}_error.jsonl', 'w') as f:
                for e in data1:
                    f.write(json.dumps(e) + '\n')
            with open(f'{save_dir}/{config}_correct.jsonl', 'w') as f:
                for e in data2:
                    f.write(json.dumps(e) + '\n')
            
            acc1 = np.mean([e['match'] for e in data1]) * 100
            acc2 = np.mean([e['match'] for e in data2]) * 100
            f1 = 2 * acc1 * acc2 / (acc1 + acc2)
            print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')

            all_f1_scores.append(f1)

    if accelerator.is_main_process:
        print(f'ProcessBench. Average F1: {np.mean(all_f1_scores):.1f}')

    if accelerator.distributed_type == "MULTI_GPU":
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=24)
    parser.add_argument("-w", "--num_of_workers", type=int, default=4)
    parser.add_argument("-s", "--separator", type=str, default="\n\n", help="It's important to use the same separator as the one used during TRL training")
    args = parser.parse_args()

    set_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)