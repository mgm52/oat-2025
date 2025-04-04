import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc

torch.cuda.empty_cache()        # Clear any cached intermediates
gc.collect()

def get_optimizer_memory(optimizer):
    """Calculate GPU memory usage of optimizer states in MB."""
    if optimizer is None or not hasattr(optimizer, 'state'):
        return 0.0
    total_memory_bytes = 0
    for param, state in optimizer.state.items():
        for state_name, tensor in state.items():
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                total_memory_bytes += tensor.element_size() * tensor.nelement()
    return total_memory_bytes / (1024 ** 2)

def get_gpu_memory(obj, name=None):
    """Calculate GPU memory usage in MB for various objects."""
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        return (obj.element_size() * obj.nelement()) / (1024 ** 2)
    elif isinstance(obj, nn.Module):
        total_params = sum(p.numel() for p in obj.parameters() if p.is_cuda)
        param_size = next(p.element_size() for p in obj.parameters() if p.is_cuda) if total_params > 0 else 0
        return (total_params * param_size) / (1024 ** 2)
    elif isinstance(obj, torch.optim.Optimizer):
        opt_memory_mb = get_optimizer_memory(obj)
        print(f"Found optimizer '{name}', State memory: {opt_memory_mb:.2f} MB")
        return opt_memory_mb
    elif isinstance(obj, (Dataset, DataLoader)):
        # Datasets/DataLoaders don't live on GPU, but check their current batch if stored
        if hasattr(obj, 'batch') and isinstance(obj.batch, torch.Tensor) and obj.batch.is_cuda:
            return (obj.batch.element_size() * obj.batch.nelement()) / (1024 ** 2)
        return 0
    if name is not None:
        if isinstance(obj, (list, tuple)):
            return sum(get_gpu_memory(item, f"{name}[{i}]") for i, item in enumerate(obj) if get_gpu_memory(item) > 0)
        elif isinstance(obj, dict):
            return sum(get_gpu_memory(value, f"{name}[{key}]") for key, value in obj.items() if get_gpu_memory(value) > 0)
    return 0

# Take a snapshot of globals to avoid runtime errors
global_snapshot = list(globals().items())
gpu_objects = {}
for name, obj in global_snapshot:
    memory_mb = get_gpu_memory(obj, name)
    if memory_mb > 0:  # Only include objects using GPU memory
        gpu_objects[name] = memory_mb

# Print results
if gpu_objects:
    print("GPU memory usage by global variables:")
    for name, memory_mb in gpu_objects.items():
        print(f"Variable: {name}, Type: {type(globals()[name]).__name__}, Memory: {memory_mb:.2f} MB")
    print(f"Total GPU memory from listed objects: {sum(gpu_objects.values()):.2f} MB")
else:
    print("No GPU-using nn.Module or tensors found in globals.")

print(f"Current total GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Reserved GPU memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")