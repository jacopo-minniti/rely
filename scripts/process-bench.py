# credit to the GitHub user @CJReinforce
# https://github.com/QwenLM/ProcessBench/pull/12
import argparse
import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


def compute_label_from_labels(labels_list):
    """
    Compute label as the index of the first False in the labels list,
    or -1 if no False is present.
    """
    try:
        return labels_list.index(False)
    except ValueError:
        return -1


def collate_fn(batch, tokenizer, separator = '\n'):
    input_ids = []
    score_ids = []
    labels = []
    separator_ids = tokenizer.encode(separator, add_special_tokens=False, return_tensors='pt')
    for i in batch:
        # Compute the label from labels list if it doesn't exist
        if 'label' not in i and 'labels' in i:
            i['label'] = compute_label_from_labels(i['labels'])
        
        prompt_ids = tokenizer(i['prompt'], add_special_tokens=False, return_tensors='pt')['input_ids']
        score_ids.append([])
        for completion in i['completions']:
            completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors='pt')['input_ids']
            prompt_ids = torch.cat([prompt_ids, completion_ids, separator_ids], dim=-1)
            score_ids[-1].append(prompt_ids.size(-1) - 1)
        labels.append(i['label'])
        input_ids.append(prompt_ids)
    
    # right pad input_ids
    pad_token_id = tokenizer.pad_token_id
    max_len = max([i.size(-1) for i in input_ids])
    for i, input_idx in enumerate(input_ids):
        input_ids[i] = torch.cat([
            input_idx.squeeze(), 
            torch.LongTensor(
                [pad_token_id] * (max_len - input_idx.size(-1))
            )
        ])
    input_ids = torch.stack(input_ids)

    return dict(
        input_ids=input_ids,
        labels=labels,
        score_ids=score_ids
    )
    
def find_first_zero(tensor):
    zeros = (tensor == 0).nonzero()
    return zeros[0].item() if zeros.numel() > 0 else -1

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

    configs = {
        'math': [75, 15], 
    }
    all_f1_scores = []
    save_dir = f'outputs/{model_name}'
    os.makedirs(save_dir, exist_ok=True)

    accelerator = Accelerator()
    print(f'Loading model from {model_path}')
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = accelerator.prepare(model)
    model.eval()

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
            
            # Ensure each item has a 'label' field computed from 'labels' if needed
            for item in new_batch:
                if 'label' not in item and 'labels' in item:
                    item['label'] = compute_label_from_labels(item['labels'])

            batch = collate_fn(batch_, tokenizer, separator)
            input_ids = batch['input_ids'].to(accelerator.device)
            labels = batch['labels']
            score_ids = batch['score_ids']

            with accelerator.autocast(), torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            for i, score_id in enumerate(score_ids):
                label = labels[i]
                pred = torch.argmax(logits[i, score_id], dim=-1)
                prediction_step = find_first_zero(pred)
                new_batch[i]['prediction'] = prediction_step
                new_batch[i]['match'] = prediction_step == label
            
            res_data.extend(new_batch)
        
        accelerator.wait_for_everyone()
        gathered_data = gather_objects(res_data, accelerator)

        if accelerator.is_main_process:
            data1 = [e for e in gathered_data if e['label'] != -1]
            data2 = [e for e in gathered_data if e['label'] == -1]
            # dataset length check
            # if len(data1) != num[0]:
            #     print(f'{config} error num mismatch: {len(data1)} != {num[0]}')
            # if len(data2) != num[1]:
            #     print(f'{config} correct num mismatch: {len(data2)} != {num[1]}')
            
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
    parser.add_argument("-m", "--model", default="jacopo-minniti/Qwen2.5-Math-7B-PUM-half_entropy", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=24)
    parser.add_argument("-w", "--num_of_workers", type=int, default=4)
    parser.add_argument("-s", "--separator", type=str, default="<extra_0>", help="It's important to use the same separator as the one used during TRL training")
    args = parser.parse_args()

    set_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)