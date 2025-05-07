from typing import List, Tuple, Union
import numpy as np
import torch
import yaml
import os
import wandb

from datetime import datetime
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset

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
    os.environ["HF_HUB_CACHE"] = f"{config['HF_HOME']}/hub"
    os.environ["HF_TOKEN"] = config["HF_TOKEN"]
    
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