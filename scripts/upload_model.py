#!/usr/bin/env python3
"""
Script to upload the value model to Hugging Face Hub
"""

from huggingface_hub import HfApi, login
import os

def upload_model():
    # Login to Hugging Face Hub (you'll need to provide your token)
    # You can get your token from https://huggingface.co/settings/tokens
    login()  # This will prompt for your token
    
    # Initialize the API
    api = HfApi()
    
    # Define paths and repo info
    model_path = "/Users/jacopominniti/Desktop/CodingProjects/UQ/rely/value_model_v2"
    repo_id = "jacopo-minniti/uats-value-model"
    
    # Create the repository (if it doesn't exist)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=False)
        print(f"Created repository: {repo_id}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Upload all files in the value_model_v2 directory
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload value model v2"
    )
    
    print(f"Model uploaded successfully to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    upload_model()
