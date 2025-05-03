import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from oat_evaluation.llms.autollm import AutoLLM

class AutoPEFT(AutoLLM):

    def __init__(self, base_model_path, adapter_path, dtype=torch.bfloat16, debug_mode=False):
        print(f"Loading base model from: {base_model_path}")
        super().__init__(base_model_path, dtype, debug_mode)
        print("Base model loaded.")

        print(f"Attempt 1 - loading adapter '{adapter_path}' onto the base model...")
        # No need to load PeftConfig separately here unless you need info from it beforehand
        try:
            self._model = PeftModel.from_pretrained(
                self._model,                 # Pass the loaded base model object
                adapter_path,                 # The adapter ID on the Hub
                subfolder="",        # Specify the subfolder containing the adapter
                # device_map is usually inferred from the base model, but can be specified if needed
            )
        except Exception as e:
            print(f"Error loading adapter: {e}")
            # Extract subfolder from adapter_path: everything after the second '/'
            subfolder = adapter_path.split('/')[2:].join('/')
            shorter_adapter_path = adapter_path.split('/')[:2].join('/')
            print(f"Attempt 2 - trying to load adapter with shorter path: {shorter_adapter_path}; subfolder: {subfolder}")
            self._model = PeftModel.from_pretrained(
                self._model,                 # Pass the loaded base model object
                shorter_adapter_path,                 # The adapter ID on the Hub
                subfolder=subfolder,        # Specify the subfolder containing the adapter
                # device_map is usually inferred from the base model, but can be specified if needed
            )
        print("PEFT Adapter loaded and merged.")

        super().prepare_model()

if __name__ == "__main__":

    adapter_path = "oat-workers/oat-gemma-apr2-checks"
    subfolder = "gemma2_lora_oat_generation_linear/lora_model_step_2048"
    base_path = "/workspace/gemma_2_9b_instruct" # your local weights path

    llm = AutoPEFT(base_path, adapter_path + "/" + subfolder)

    print(llm.generate_responses(["Hello, how are you?"], max_new_tokens=10))