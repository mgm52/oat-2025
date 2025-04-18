import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from oat_evaluation.llms.autollm import AutoLLM

class AutoPEFT(AutoLLM):

    def __init__(self, base_model_path, adapter_id, subfolder, dtype=torch.bfloat16, debug_mode=False):
        print(f"Loading base model from: {base_model_path}")
        super().__init__(base_model_path, dtype, debug_mode)
        print("Base model loaded.")

        print(f"Loading adapter '{adapter_id}' subfolder '{subfolder}' onto the base model...")
        # No need to load PeftConfig separately here unless you need info from it beforehand
        self._model = PeftModel.from_pretrained(
            self._model,                 # Pass the loaded base model object
            adapter_id,                 # The adapter ID on the Hub
            subfolder=subfolder,        # Specify the subfolder containing the adapter
            # device_map is usually inferred from the base model, but can be specified if needed
        )
        print("PEFT Adapter loaded and merged.")

        super().prepare_model()

if __name__ == "__main__":

    adapter_id = "oat-workers/oat-gemma-apr2-checks"
    subfolder = "gemma2_lora_oat_generation_linear/lora_model_step_2048"
    base_path = "/workspace/gemma_2_9b_instruct" # your local weights path

    llm = AutoPEFT(base_path, adapter_id, subfolder)

    print(llm.generate_responses(["Hello, how are you?"], max_new_tokens=10))