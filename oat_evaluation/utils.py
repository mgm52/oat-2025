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

def calculate_flops(model_size, num_tokens, include_backward=False):
    """
    Calculate approximate FLOPs for transformer operations.
    
    Args:
        model_size: Number of parameters in the model
        num_tokens: Number of tokens processed
        include_backward: Whether to include backward pass (typically 2x forward)
    
    Returns:
        Estimated number of FLOPs
    """
    # Basic estimate: k × d × 2, where k is tokens and d is model size
    forward_flops = num_tokens * model_size * 2
    
    # Backward pass is approximately 2x the forward pass
    if include_backward:
        return forward_flops * 3
    else:
        return forward_flops