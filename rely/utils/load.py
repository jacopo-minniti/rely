import os
import json
import torch
from typing import List, Dict, Any, Union, Optional
from pathlib import Path


def load_dataset(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load a dataset from either .pt or .jsonl file.
    
    Args:
        file_path: Path to the dataset file
    
    Returns:
        List[Dict[str, Any]]: The loaded dataset as a list of dictionaries
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is unsupported or data is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pt':
        # Load PyTorch file
        try:
            data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            raise ValueError(f"Error loading PyTorch file {file_path}: {e}")
    elif file_ext == '.jsonl':
        # Load JSONL file
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        data.append(json.loads(line))
        except Exception as e:
            raise ValueError(f"Error loading JSONL file {file_path}: {e}")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Only .pt and .jsonl are supported")
    
    # Validate that data is a list of dictionaries
    if not isinstance(data, list):
        raise ValueError(f"File {file_path} does not contain a list. Found: {type(data)}")
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"File {file_path} contains non-dictionary item at index {i}: {type(item)}")
    
    return data


def save_dataset(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save a dataset to either .pt or .jsonl file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where to save the dataset
    
    Raises:
        ValueError: If the file format is unsupported
    """
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pt':
        # Save as PyTorch file
        torch.save(data, file_path)
    elif file_ext == '.jsonl':
        # Save as JSONL file
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Only .pt and .jsonl are supported")


def validate_file_format(file_path: Union[str, Path]) -> bool:
    """
    Validate that a file has a supported format (.pt or .jsonl).
    
    Args:
        file_path: Path to the file to validate
    
    Returns:
        bool: True if format is supported, False otherwise
    """
    file_ext = Path(file_path).suffix.lower()
    return file_ext in ['.pt', '.jsonl']
