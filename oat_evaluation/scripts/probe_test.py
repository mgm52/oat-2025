import torch
import pickle
import sys
from typing import Any, Dict

# Ensure the NEW path to LinearProbe is importable
try:
    from oat_training.src.probe_archs import LinearProbe
    print("Successfully imported LinearProbe from oat_training.src.probe_archs")
except ImportError as e:
    print(f"Error importing LinearProbe: {e}")
    print("Please ensure 'oat_training' is installed or '/workspace/GIT_SHENANIGANS/oat-2025' is in sys.path")
    sys.exit(1)

# --- Custom Unpickler to Remap Paths ---
class RemapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Define the mapping from old path to new class object
        if module == 'src.probe_archs' and name == 'LinearProbe':
            # Return the *actual* class object from its *new* location
            return LinearProbe
        # Fallback to default behavior for other classes
        return super().find_class(module, name)

# --- Custom Pickle Module Object ---
# This object mimics the standard 'pickle' module for torch.load
class CustomPickleModule:
    # Add the __name__ attribute that torch.load checks
    __name__ = "CustomPickleModuleForRemapping"

    # Provide the custom Unpickler class
    Unpickler = RemapUnpickler

    # Provide the load function, mimicking pickle.load
    # This should be a static method or a function attribute
    @staticmethod
    def load(f, **kwargs):
        # Instantiate *our* custom Unpickler and call its load method
        # Pass encoding if provided by torch.load
        encoding = kwargs.get('encoding', 'ASCII') # Default pickle encoding
        return CustomPickleModule.Unpickler(f, encoding=encoding).load()

    # Provide dumps/dump if necessary, though likely not needed for torch.load
    # Add dummy versions if torch complains they are missing
    # @staticmethod
    # def dumps(obj, protocol=None, *, fix_imports=True, buffer_callback=None):
    #     # This would need a proper implementation if torch.save used it via pickle_module
    #     raise NotImplementedError("Custom dumps not implemented")
    #
    # @staticmethod
    # def dump(obj, file, protocol=None, *, fix_imports=True, buffer_callback=None):
    #     # This would need a proper implementation if torch.save used it via pickle_module
    #     raise NotImplementedError("Custom dump not implemented")


# --- Loading Function ---
def load_probes_with_remapping(file_path: str) -> Dict[str, Any]:
    """Load probes using torch.load with a custom pickle module for remapping."""
    print(f"\nAttempting to load '{file_path}' with custom remapping (v2)...")
    try:
        # Use torch.load:
        # - weights_only=False: Essential for loading full Python objects.
        # - pickle_module: Provide our custom module-like object.
        probes = torch.load(
            file_path,
            map_location=torch.device('cpu'), # Good practice
            # Pass the custom module object itself
            pickle_module=CustomPickleModule,
            weights_only=False # MUST be False
        )
        print("Success loading with custom remapping (v2)!")
        return probes
    except Exception as e:
        print(f"Error loading with custom remapping (v2): {e}")
        # Print traceback for more details
        import traceback
        traceback.print_exc()
        return None

# --- Main Execution ---
if __name__ == "__main__":
    probe_file = "/workspace/GIT_SHENANIGANS/oat-2025/checkpoints/probes/probes_step_2048.pt"
    loaded_probes = load_probes_with_remapping(probe_file)

    if loaded_probes:
        print("\nLoaded probes structure:")
        # Example: Print keys or inspect the structure
        print("Keys:", loaded_probes.keys())
        # Check if the loaded object is indeed the correct type
        for k, v in loaded_probes.items():
             # Probes might be nested, check recursively or specifically
             if isinstance(v, LinearProbe):
                 print(f"Probe '{k}' loaded as type: {type(v)}")
             elif isinstance(v, dict): # Example check for nested dict
                 for nk, nv in v.items():
                     if isinstance(nv, LinearProbe):
                         print(f"Nested probe '{k}/{nk}' loaded as type: {type(nv)}")