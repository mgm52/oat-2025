from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from dataclasses import dataclass
import yaml
import os
import wandb

from datetime import datetime
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
import torch


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
    
    for key in ["OPENAI_API_KEY", "HF_TOKEN", "STORAGE_HOME"]:
        if key in config:
            os.environ[key] = config[key]
    
    if "HF_HOME" in config:
        os.environ["HF_HOME"] = config["HF_HOME"]
        os.environ["HF_HUB_CACHE"] = f"{config['HF_HOME']}/hub"
    return config

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
    num_flops: Optional[int] = 0
    # num_activations: int = 0
    num_input_tokens: Optional[int] = 0
    num_output_tokens: Optional[int] = 0

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
