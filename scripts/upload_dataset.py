#!/usr/bin/env python3
"""
Script to upload v2 dataset files to Hugging Face Hub without overwriting existing data.
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo
import argparse

def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def upload_as_separate_files(repo_id, files_to_upload, token=None):
    """Upload v2 files directly with their current names."""
    api = HfApi(token=token)
    
    # Try to create the repository if it doesn't exist
    try:
        print(f"Checking if repository {repo_id} exists...")
        create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
        print(f"✅ Repository {repo_id} is ready")
    except Exception as e:
        print(f"⚠️  Repository creation info: {e}")
    
    for file_name in files_to_upload:
        if Path(file_name).exists():
            print(f"Uploading {file_name}...")
            api.upload_file(
                path_or_fileobj=file_name,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"✅ Uploaded {file_name}")
        else:
            print(f"⚠️  File {file_name} not found")

def upload_as_configuration(repo_id, config_name="v2", token=None):
    """Upload as a new dataset configuration."""
    try:
        # Try to create the repository if it doesn't exist
        print(f"Checking if repository {repo_id} exists...")
        create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
        print(f"✅ Repository {repo_id} is ready")
        
        # Load the data
        train_data = load_jsonl("v1_mmlu_train.jsonl")
        test_data = load_jsonl("v1_mmlu_test.jsonl")
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        
        # Create dataset dict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        
        print(f"Uploading as configuration '{config_name}'...")
        dataset_dict.push_to_hub(
            repo_id, 
            config_name=config_name,
            token=token
        )
        print(f"✅ Uploaded configuration '{config_name}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload v2 dataset to Hugging Face Hub")
    parser.add_argument("repo_id", help="Repository ID (username/dataset-name)")
    parser.add_argument("--method", choices=["files", "config"], default="files",
                       help="Upload method: 'files' for separate files, 'config' for configuration")
    parser.add_argument("--config-name", default="v2", help="Configuration name (for config method)")
    parser.add_argument("--files", nargs="+", 
                       default=["math_dataset_train.jsonl", "math_dataset_test.jsonl"],
                       help="Files to upload (for files method)")
    parser.add_argument("--token", help="Hugging Face token (optional if logged in)")
    
    args = parser.parse_args()
    
    print(f"Repository: {args.repo_id}")
    print(f"Method: {args.method}")
    
    if args.method == "files":
        print(f"Files to upload: {args.files}")
        upload_as_separate_files(args.repo_id, args.files, args.token)
    else:
        upload_as_configuration(args.repo_id, args.config_name, args.token)

if __name__ == "__main__":
    # python upload_dataset.py jacopo-minniti/MATH-PUM-qwen2.5-1.5B --method config --config-name default
    main()
