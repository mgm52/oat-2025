from dataclasses import dataclass
import yaml
import os
from datetime import datetime
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
    
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["HF_HOME"] = config["HF_HOME"]
    os.environ["HF_HUB_CACHE"] = f"{config['HF_HOME']}/hub"
    os.environ["HF_TOKEN"] = config["HF_TOKEN"]
    return config

def print_timey(message: str):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"[{current_time}] {message}")


def print_mem_usage(print_mem: bool = True):
    if print_mem:
        print(f"    [Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB]")
        print(f"    [Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB]")
    return torch.cuda.memory_allocated() / 1024**2


@dataclass
class FlopCounter:
    num_flops: int = 0
    # num_activations: int = 0
