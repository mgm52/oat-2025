import torch


def print_mem_usage():
    print(f"    [Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB]")
    print(f"    [Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB]")
    return torch.cuda.memory_allocated() / 1024**2
