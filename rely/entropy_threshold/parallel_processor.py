"""
Parallel Processor for entropy calculations.

This module provides the ParallelProcessor class for running entropy calculations
in parallel across multiple GPUs using tmux sessions.
"""

import os
import subprocess
import math
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class ParallelProcessor:
    """
    A class for running entropy calculations in parallel across multiple GPUs.
    
    This class handles the creation and management of tmux sessions for
    parallel processing of large datasets.
    """
    
    def __init__(
        self,
        script_path: str = "entropy-threshold.py",
        session_name: str = "entropy_run",
        output_dir: str = "entropy_outputs"
    ):
        """
        Initialize the ParallelProcessor.
        
        Parameters
        ----------
        script_path : str, default="entropy-threshold.py"
            Path to the entropy calculation script.
        session_name : str, default="entropy_run"
            Name for the tmux session.
        output_dir : str, default="entropy_outputs"
            Directory to save output files.
        """
        self.script_path = script_path
        self.session_name = session_name
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_chunk_sizes(
        self, 
        dataset_path: str, 
        num_shards: int = 8
    ) -> List[Tuple[int, int]]:
        """
        Calculate chunk sizes for parallel processing.
        
        Parameters
        ----------
        dataset_path : str
            Path to the dataset file.
        num_shards : int, default=8
            Number of shards to split the dataset into.
            
        Returns
        -------
        List[Tuple[int, int]]
            List of (start_idx, end_idx) tuples for each shard.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Count total lines in the dataset
        with open(dataset_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        # Calculate chunk size (round up to ensure all lines are covered)
        chunk_size = math.ceil(total_lines / num_shards)
        
        # Generate chunk boundaries
        chunks = []
        for shard in range(num_shards):
            start_idx = shard * chunk_size
            end_idx = min((shard + 1) * chunk_size, total_lines)
            
            if start_idx < total_lines:  # Only add chunks that have data
                chunks.append((start_idx, end_idx))
        
        return chunks
    
    def create_tmux_session(self, chunks: List[Tuple[int, int]], dataset_path: str):
        """
        Create a tmux session with parallel entropy calculations.
        
        Parameters
        ----------
        chunks : List[Tuple[int, int]]
            List of (start_idx, end_idx) tuples for each shard.
        dataset_path : str
            Path to the dataset file.
        """
        # Kill existing session if it exists
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", self.session_name],
                check=False,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            pass  # Session didn't exist, which is fine
        
        # Create new session
        subprocess.run([
            "tmux", "new-session", "-d", "-s", self.session_name, "-n", "entropy"
        ], check=True)
        
        # Create panes and run commands
        for shard, (start_idx, end_idx) in enumerate(chunks):
            output_file = os.path.join(self.output_dir, f"entropies_{shard}.npy")
            
            # Build the command
            cmd = [
                "CUDA_VISIBLE_DEVICES=" + str(shard),
                "python", self.script_path,
                "--start_idx", str(start_idx),
                "--end_idx", str(end_idx),
                "--output", output_file
            ]
            
            if shard == 0:
                # First pane - send command to existing pane
                subprocess.run([
                    "tmux", "send-keys", "-t", self.session_name, " ".join(cmd), "C-m"
                ], check=True)
            else:
                # Create new pane and send command
                subprocess.run([
                    "tmux", "split-window", "-t", self.session_name, "-h"
                ], check=True)
                subprocess.run([
                    "tmux", "select-layout", "-t", self.session_name, "tiled"
                ], check=True)
                subprocess.run([
                    "tmux", "send-keys", "-t", self.session_name, " ".join(cmd), "C-m"
                ], check=True)
        
        # Final layout adjustment
        subprocess.run([
            "tmux", "select-layout", "-t", self.session_name, "tiled"
        ], check=True)
    
    def run_parallel_processing(
        self, 
        dataset_path: str, 
        num_shards: int = 8,
        auto_attach: bool = False
    ) -> List[str]:
        """
        Run parallel entropy processing.
        
        Parameters
        ----------
        dataset_path : str
            Path to the dataset file.
        num_shards : int, default=8
            Number of shards to split the dataset into.
        auto_attach : bool, default=False
            Whether to automatically attach to the tmux session.
            
        Returns
        -------
        List[str]
            List of output file paths that will be created.
        """
        # Calculate chunks
        chunks = self.calculate_chunk_sizes(dataset_path, num_shards)
        
        # Create tmux session
        self.create_tmux_session(chunks, dataset_path)
        
        # Generate output file paths
        output_files = [
            os.path.join(self.output_dir, f"entropies_{shard}.npy")
            for shard in range(len(chunks))
        ]
        
        print(f"Started {len(chunks)} parallel entropy calculations")
        print(f"Session name: {self.session_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Expected output files: {output_files}")
        
        if auto_attach:
            print("Attaching to tmux session...")
            subprocess.run(["tmux", "attach", "-t", self.session_name])
        else:
            print(f"Attach manually with: tmux attach -t {self.session_name}")
        
        return output_files
    
    def check_session_status(self) -> Dict[str, str]:
        """
        Check the status of the tmux session.
        
        Returns
        -------
        Dict[str, str]
            Dictionary containing session status information.
        """
        try:
            # Check if session exists
            result = subprocess.run(
                ["tmux", "has-session", "-t", self.session_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {"status": "not_found", "message": "Session does not exist"}
            
            # Get session info
            info_result = subprocess.run(
                ["tmux", "list-sessions", "-F", "#{session_name}:#{session_windows}"],
                capture_output=True,
                text=True
            )
            
            # Check if any panes are still running
            running_result = subprocess.run(
                ["tmux", "list-panes", "-t", self.session_name, "-F", "#{pane_pid}"],
                capture_output=True,
                text=True
            )
            
            return {
                "status": "running",
                "session_name": self.session_name,
                "windows": info_result.stdout.strip(),
                "active_panes": len(running_result.stdout.strip().split('\n')) if running_result.stdout.strip() else 0
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def kill_session(self):
        """Kill the tmux session if it exists."""
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", self.session_name],
                check=True
            )
            print(f"Killed tmux session: {self.session_name}")
        except subprocess.CalledProcessError:
            print(f"Session {self.session_name} was not running or already killed")
    
    def wait_for_completion(self, timeout: Optional[int] = None) -> bool:
        """
        Wait for all parallel processes to complete.
        
        Parameters
        ----------
        timeout : int | None, default=None
            Timeout in seconds. If None, wait indefinitely.
            
        Returns
        -------
        bool
            True if all processes completed successfully, False otherwise.
        """
        import time
        
        start_time = time.time()
        
        while True:
            status = self.check_session_status()
            
            if status["status"] == "not_found":
                print("Session completed or was killed")
                return True
            
            if status["status"] == "error":
                print(f"Error checking session status: {status['message']}")
                return False
            
            # Check if timeout exceeded
            if timeout and (time.time() - start_time) > timeout:
                print(f"Timeout exceeded ({timeout}s)")
                return False
            
            print(f"Session still running... ({status.get('active_panes', 0)} active panes)")
            time.sleep(30)  # Check every 30 seconds 