
import os
import wandb
import yaml

def load_config_and_set_vars():
    # Try a few relative paths
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = "oat_evaluation/config.yaml"
    if not os.path.exists(config_path):
        config_path = "../config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml file not found in oat_evaluation... See config_example.yaml for an example, then create your own config.yaml file.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["HF_HOME"] = config["HF_HOME"]
    print(f"Setting HF_HOME: {os.environ['HF_HOME']}")
    os.environ["HF_HUB_CACHE"] = f"{config['HF_HOME']}/hub"
    os.environ["HF_TOKEN"] = config["HF_TOKEN"]

    wandb.login(key=config["WANDB_API_KEY"])
    return config

load_config_and_set_vars()

from typing import List, Tuple, Union
import numpy as np
import torch
from dataclasses import dataclass

from datetime import datetime
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
import subprocess
import pathlib
import time # Added for polling delay
from collections import deque # Added for managing command queue

def get_available_gpus():
    """Get the number of available GPUs"""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()

def run_command_on_gpu(gpu_id, command, log_dir):
    """Run training on a specific GPU with given command"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create timestamp for unique log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Kept original log file naming convention
    log_file = log_dir / f"{timestamp}_gpu{gpu_id}.log"
    # if log file already exists, append a number to the filename
    i = 1
    while os.path.exists(log_file):
        log_file = log_dir / f"{timestamp}_gpu{gpu_id}_{i}.log"
        i += 1
    
    # Modified print statement to include the command for better clarity
    print(f"Running training on GPU {gpu_id}. Command: {' '.join(command)}. Log file: {log_file}")
    
    # Open log file for writing
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=f,
            stderr=f,
            text=True
        )
    return process, log_file

def get_gpu_memory_usage(gpu_id):
    result = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=used_memory',
            '--format=csv,nounits,noheader', f'--id={gpu_id}'],
        stderr=subprocess.DEVNULL
    ).decode().strip()

    if not result:
        return 0  # No memory used by any process on this GPU

    return sum(int(x) for x in result.split('\n'))

def run_many_commands_on_gpus(commands, USE_ALL_GPUS=True, log_dir="oat_training/oat_training_logs"):
    """
    Run multiple training commands across available GPUs.
    If USE_ALL_GPUS is True, tasks are distributed dynamically to available GPUs.
    If USE_ALL_GPUS is False, tasks are run sequentially on GPU 0.
    
    Args:
        commands: List of command lists to execute (e.g., [["python", "script.py", "--arg1", "val1"], ...])
        USE_ALL_GPUS: Boolean indicating whether to use all GPUs or run sequentially on GPU 0.
    """
    num_gpus = get_available_gpus()
    
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True) # Ensure parent directories exist
    
    if num_gpus == 0:
        print("No GPUs available.")
        if USE_ALL_GPUS:
            print("Cannot dynamically schedule on multiple GPUs as none are available.")
            print("Falling back to sequential execution (attempts to use GPU 0 or CPU if script supports).")
        # Force sequential mode if no GPUs. This mode attempts GPU 0, matching original behavior.
        USE_ALL_GPUS = False 
    
    if not USE_ALL_GPUS:
        # Sequential execution mode (either explicitly requested or due to no GPUs)
        target_gpu_for_sequential = 0 
        print(f"Running {len(commands)} tasks sequentially. Target device: GPU {target_gpu_for_sequential} (if available).")
        
        if not commands:
            print("No commands to run.")
            return

        for i, cmd in enumerate(commands):
            print(f"\nStarting sequential task {i+1}/{len(commands)} on GPU {target_gpu_for_sequential}.")
            process, log_file = run_command_on_gpu(target_gpu_for_sequential, cmd, log_dir)
            
            try:
                process.wait() # Wait for this process to complete before starting the next one
                if process.returncode == 0:
                    print(f"Task {i+1} completed successfully. Log file: {log_file}")
                else:
                    print(f"Task {i+1} FAILED with return code {process.returncode}. Log file: {log_file}")
            except Exception as e: # Catch potential errors during wait (e.g., KeyboardInterrupt)
                print(f"Error waiting for task {i+1} (command: {' '.join(cmd)}): {e}. Log file: {log_file}")
        
        print("\nAll sequential tasks processed.")
        return

    # Dynamic GPU scheduling mode (USE_ALL_GPUS is True and num_gpus > 0)
    total_tasks = len(commands)
    if total_tasks == 0:
        print("No commands to run.")
        return
        
    print(f"Dynamically distributing {total_tasks} tasks across {num_gpus} available GPUs.")
    
    # Store commands as (original_index, command_list) for better tracking
    commands_queue = deque([(cmd_idx, cmd) for cmd_idx, cmd in enumerate(commands)])
    # active_processes_on_gpus maps gpu_id to (process, log_file, original_cmd_idx, command_list)
    active_processes_on_gpus = {}  
    
    completed_task_count = 0

    while completed_task_count < total_tasks:
        # Check for completed processes
        gpus_that_finished_task = []
        for gpu_id, (process, log_file, cmd_idx, cmd_list) in active_processes_on_gpus.items():
            if process.poll() is not None:  # Process has finished
                if process.returncode == 0:
                    print(f"\nTask {cmd_idx+1}/{total_tasks} (GPU {gpu_id}, {' '.join(cmd_list)}) completed successfully.")
                else:
                    print(f"\nTask {cmd_idx+1}/{total_tasks} (GPU {gpu_id}, {' '.join(cmd_list)}) FAILED with code {process.returncode}.")
                print(f"Log file: {log_file}")
                
                gpus_that_finished_task.append(gpu_id)
                completed_task_count += 1
        
        # Remove completed processes and free up their GPUs
        for gpu_id in gpus_that_finished_task:
            del active_processes_on_gpus[gpu_id]

        # Assign new tasks to available GPUs
        # Loop while there are commands in the queue AND there are free GPU slots
        while commands_queue and len(active_processes_on_gpus) < num_gpus:
            # Find the first available GPU with low memory usage
            assigned_gpu_id = -1
            for i in range(num_gpus):
                if i not in active_processes_on_gpus:
                    memory_usage = get_gpu_memory_usage(i)
                    print(f"GPU {i} memory usage: {memory_usage} MB")
                    if memory_usage < 100:  # Only use GPUs with less than 100MB memory usage
                        assigned_gpu_id = i
                        break
            
            if assigned_gpu_id != -1: # If a free GPU with low memory was found
                cmd_idx, cmd_to_run = commands_queue.popleft()
                print(f"\nAssigning task {cmd_idx+1}/{total_tasks} to GPU {assigned_gpu_id}.")
                process, log_file = run_command_on_gpu(assigned_gpu_id, cmd_to_run, log_dir)
                active_processes_on_gpus[assigned_gpu_id] = (process, log_file, cmd_idx, cmd_to_run)
            else:
                # No free GPU slot with low memory found
                break # Break from assigning more tasks in this cycle
        
        if completed_task_count == total_tasks:
            break # All tasks are done

        time.sleep(10) # Polling interval to check process statuses
    
    print("\nAll training commands have been processed.")


def wandb_log(name, value, step=None):
    if wandb.run:                       # safe if wandb disabled
        wandb.log({name: value}, step=step)

def print_timey(message: str):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"[{current_time}] {message}")
    wandb_log("event", message)

def print_mem_usage():
    print(f"[Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB] ", end="")
    print(f"[Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB]")

def dataset_to_list(dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> List:
    return [{"prompt": x["prompt"], "completion": x["completion"]} for x in dataset]

def get_quantile_with_bootstrapping(data, quantile, n_bootstrap=1000, seed=42):
    # Convert data to CPU tensor if it's a CUDA tensor
    if isinstance(data, torch.Tensor) and data.device.type == 'cuda':
        data = data.cpu()
    
    # Convert to tensor if it's not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    
    torch.manual_seed(seed)
    indices = torch.randint(0, len(data), (n_bootstrap, len(data)))
    bootstrap_samples = data[indices]
    bootstrap_quantiles = torch.quantile(bootstrap_samples, quantile, dim=1)
    return torch.mean(bootstrap_quantiles).item()


# TODO: Rename this function, once I'm sure it's not used elsewhere (e.g. by Max)
def print_mem_usage(print_mem: bool = True):
    if print_mem:
        print(f"    [Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB]")
        print(f"    [Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB]")
    return torch.cuda.memory_allocated() / 1024**2

@dataclass
class FlopCounter:
    num_flops: int = 0
    # num_activations: int = 0

def calculate_forward_flops(model_size, num_tokens):
    """
    Calculate approximate FLOPs for transformer forward pass operations.
    
    Args:
        model_size: Number of parameters in the model
        num_tokens: Number of tokens processed
    
    Returns:
        Estimated number of FLOPs for forward pass
    """
    # Basic estimate: k × d × 2, where k is tokens and d is model size
    forward_flops = num_tokens * model_size * 2
    
    return forward_flops

def calculate_backward_flops(model_size, num_tokens):
    """
    Calculate approximate FLOPs for transformer backward pass operations.
    
    Args:
        model_size: Number of parameters in the model
        num_tokens: Number of tokens processed
    
    Returns:
        Estimated number of FLOPs for backward pass
    """
    # Backward pass is approximately 2x the forward pass
    backward_flops = num_tokens * model_size * 2 * 2
    
    return backward_flops
