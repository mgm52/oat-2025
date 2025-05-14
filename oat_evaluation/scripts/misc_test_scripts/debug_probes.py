import sys
import torch
import traceback
from collections import OrderedDict

# --- Your existing remapping logic ---
from oat_evaluation.probes.abhay_checkpoints import CustomPickleModule, load_probes_with_remapping, AbhayCheckpointProbe


def inspect_checkpoint_structure(checkpoint_path: str):
    """
    Load the checkpoint and print out the full structure of each entry.
    """
    print(f"\nInspecting '{checkpoint_path}'...\n")
    try:
        data = torch.load(
            checkpoint_path,
            map_location="cpu",
            pickle_module=CustomPickleModule
        )
    except Exception as e:
        print("❌  Failed to load checkpoint:")
        traceback.print_exc()
        return

    print(f"Top-level type: {type(data)}\n")
    if not isinstance(data, dict):
        print("Expected a dict mapping layer→probe, but got something else.")
        return

    for layer_key, val in data.items():
        print(f"Key: {layer_key!r} (type: {type(layer_key)})")
        print(f"  Value Type: {type(val)}")
        if isinstance(val, OrderedDict):
            print(f"  ↳ OrderedDict with {len(val)} params:")
            for pname, param in val.items():
                if isinstance(param, torch.Tensor):
                    print(f"    • {pname:30} shape={param.shape} dtype={param.dtype}")
                else:
                    print(f"    • {pname:30} <non-tensor: {type(param)}>" )
        elif isinstance(val, torch.nn.Module):
            print(f"  ↳ A Module: {val.__class__.__name__}")
        else:
            short = repr(val)
            print("  ↳ repr:", short[:200] + ("…" if len(short)>200 else ""))
        print()


if __name__ == "__main__":
    # Replace with your desired probe path based on PROBE_TYPE / OAT_OR_BASE
    PROBE_TYPE = sys.argv[1] if len(sys.argv)>1 else "abhayllama"
    paths = {
        "linear": "/workspace/GIT_SHENANIGANS/oat-2025/checkpoints/probes/linear_probes_step_2048.pt",
        "mlp":    "/workspace/GIT_SHENANIGANS/oat-2025/checkpoints/probes/nonlinear_probes_step_2048.pt",
        "abhayllama": "/workspace/GIT_SHENANIGANS/oat-2025/checkpoints/probes/abhayllama_probes.pt",
    }

    chosen = paths.get(PROBE_TYPE)
    if not chosen:
        print(f"Unknown PROBE_TYPE '{PROBE_TYPE}'. Choose one of: {list(paths)}")
        sys.exit(1)

    inspect_checkpoint_structure(chosen)

    # Optionally: now try to instantiate the probe to see the warning/errors
    try:
        print("\nAttempting to build AbhayCheckpointProbe...\n")
        probe = AbhayCheckpointProbe(checkpoint_path=chosen)
        print("✅ Probe built successfully. Target layers:", probe.target_layers)
    except Exception as e:
        print("❌ Failed to initialize probe:")
        traceback.print_exc()
