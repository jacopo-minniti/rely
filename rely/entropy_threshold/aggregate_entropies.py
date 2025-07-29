#!/usr/bin/env python
"""Aggregate entropies saved by shard runs of entropy-threshold.py and compute stats.

Usage:
    python aggregate_entropies.py path/to/outputs/*.npy
    # or simply run with no arguments to auto-discover files in ./entropy_outputs
"""
import sys
import glob
import numpy as np
from pathlib import Path


def main(paths: list[str]):
    if not paths:
        paths = glob.glob("entropy_outputs/entropies_*.npy")
        if not paths:
            print("No .npy files found. Provide paths explicitly or check output directory.")
            sys.exit(1)

    arrays = []
    for p in paths:
        arr = np.load(p)
        arrays.append(arr)
        print(f"Loaded {len(arr)} entropies from {p}")

    all_entropies = np.concatenate(arrays)
    print("\n--- Aggregated Statistics ---")
    print(f"Total tokens: {len(all_entropies)}")
    print(f"Average entropy: {all_entropies.mean():.4f}")
    print(f"Median entropy: {np.median(all_entropies):.4f}")
    print(f"80th percentile entropy threshold: {np.percentile(all_entropies, 80):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:]) 