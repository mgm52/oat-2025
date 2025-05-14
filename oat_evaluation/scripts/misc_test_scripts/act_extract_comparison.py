#!/usr/bin/env python
# compare_hidden_states.py
#
# Compare hidden states captured by forward hooks with
# Hugging Face's built-in `output_hidden_states=True` output.
#
# Works with any pre-norm LLaMA-style model (e.g. Llama-2, Llama-3).

import argparse
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------- util helpers --------------------------- #

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, atol=1e-5, rtol=1e-4):
    equal = torch.allclose(t1, t2, atol=atol, rtol=rtol)
    max_diff = (t1 - t2).abs().max().item()
    return equal, max_diff

# Modified hook to place tensor at the correct index
def activation_extraction_hook_simple(destination: List[Optional[torch.Tensor]], index: int, debug_mode: bool = False):
    def hook_fn(module, input):
        # Ensure list is mutable if needed (though pre-sized should be ok)
        print(f"Hook for layer index {index} triggered with input {input}; input[0] shape: {input[0].shape}; index {index}")
        # Store the *single* full activation tensor for this layer/pass
        destination[index] = input[0]
    return hook_fn

@contextmanager
def add_fwd_hooks(module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]]):
    handles = []
    try:
        for module, hook_fn in module_forward_pre_hooks:
            handle = module.register_forward_pre_hook(hook_fn)
            handles.append(handle)
        yield
    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()





    def _get_block_modules(self) -> List[torch.nn.Module]:
        """Get the transformer block modules for hooking."""
        blocks = []
        for name, module in self._model.named_modules():
            # For Gemma models, the transformer blocks are in model.layers
            if isinstance(module, torch.nn.Module) and hasattr(module, 'self_attn'):
                blocks.append(module)
        return blocks






# --------------------------- main routine --------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Compare forward-hook activations with Hugging Face hidden_states."
    )
    parser.add_argument("--model", required=True, help="Model name or path (HF hub or local).")
    parser.add_argument("--prompt", default="Hello world", help="Text prompt.")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="Optional: generate this many tokens so the sequence grows.")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"], help="Model dtype.")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = get_device()

    print(f"Loading {args.model} on {device} …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": device}  # put everything on the chosen device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA often has no pad token

    # ---------------------------------------------------------------- #
    # Prepare prompt
    # ---------------------------------------------------------------- #
    enc = tokenizer(args.prompt, return_tensors="pt").to(device)

    # ---------------------------------------------------------------- #
    # Attach hooks
    # ---------------------------------------------------------------- #
    # We store: embedding output + each block input + final norm output
    num_layers = len(model.model.layers)          # transformer block count
    #hooked_states: List[Optional[torch.Tensor]] = [None] * (num_layers + 1)

    # 1. forward-pre hooks on every block (captures *input* to the block)
    #def make_pre_hook(idx):
    #    def hook(module, inputs):
    #        hooked_states[idx] = inputs[0]
    #    return hook

    target_layers = range(num_layers)
    print(f"target_layers: {target_layers}")

    _model_block_modules = []
    for name, module in model.named_modules():
        # For Gemma models, the transformer blocks are in model.layers
        if isinstance(module, torch.nn.Module) and hasattr(module, 'self_attn'):
            _model_block_modules.append(module)
    
    hooked_states = [None] * (num_layers+1)
    fwd_extraction_hooks = [(
        _model_block_modules[layer],
        activation_extraction_hook_simple(destination=hooked_states, index=i)
    ) for i, layer in enumerate(target_layers)
    ]

    with add_fwd_hooks(fwd_extraction_hooks):
        outputs = model.forward(**enc)

    # # 2. forward hook on final RMSNorm (`model.model.norm`)
    # def norm_forward_hook(module, inputs, output):
    #     hooked_states[-1] = output
    # handles.append(model.model.norm.register_forward_hook(norm_forward_hook))

    # ---------------------------------------------------------------- #
    # Forward pass (with hidden states)
    # ---------------------------------------------------------------- #
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
        if args.max_tokens > 0:
            # optional generation step to verify equality on longer seqs
            _ = model.forward(**enc,
                               output_hidden_states=True)

    hf_states = outputs.hidden_states          # tuple, len = num_layers + 1

    # ---------------------------------------------------------------- #
    # Compare
    # ---------------------------------------------------------------- #
    print("\nLayer-by-layer comparison:")
    all_equal = True
    print(f"Len(hooked_states): len{hooked_states}; len(hf_states): len{hf_states}")
    for i, (hooked, official) in enumerate(zip(hooked_states, hf_states)):
        same_shape = hooked is not None and hooked.shape == official.shape
        equal, max_diff = (False, float("inf")) if not same_shape \
            else compare_tensors(hooked, official)
        all_equal &= equal
        name = ("embeddings" if i == 0
                else ("final_norm" if i == num_layers else f"layer_{i}"))
        status = "✅" if equal else "❌"
        print(f"{name:<12} {status}  max |Δ| = {max_diff:.3e}  hooked shape={hooked.shape if hooked is not None else 'None'}  official shape={official.shape if official is not None else 'None'}")

    print("\nSummary:")
    if all_equal:
        print("✔️  Everything matches!  Hooked activations == hidden_states")
    else:
        print("⚠️  At least one mismatch detected (see ❌ rows above).")

if __name__ == "__main__":
    main()
