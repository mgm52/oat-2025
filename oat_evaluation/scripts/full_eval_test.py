from datetime import datetime
import gc
from oat_evaluation.utils import dataset_to_list, load_config_and_set_vars, print_mem_usage, print_timey
from oat_evaluation.evals.universal import batched_generate_and_score
load_config_and_set_vars() # needs to be at top to take effect

import traceback
import torch

from oat_evaluation.attacks.soft_suffix import SoftSuffixAttack
from oat_evaluation.data.standard_data import load_harmful_harmful_test_abhay_1000, load_harmless_blend_test_abhay_1533
from oat_evaluation.llms.autollm import AutoLLM
from oat_evaluation.llms.autopeft import AutoPEFT
from oat_evaluation.probes.abhay_checkpoints import AbhayCheckpointProbe
from oat_evaluation.utils import load_config_and_set_vars

from oat_evaluation.evals.universal import run_universal_eval
import wandb

def run_main():
    """
    Initialize models, load datasets once, and run evaluations with different offsets.
    """
    # Load config for model and probe paths

    print(f"Loading datasets once for all evaluations...")
    harmful = load_harmful_harmful_test_abhay_1000()
    harmful = dataset_to_list(harmful)

    # Override to load 15% xstest-blended set!
    harmless = load_harmless_blend_test_abhay_1533()
    harmless = dataset_to_list(harmless)

    print(f"Loaded {len(harmless)} harmless examples. First one: {harmless[0]}")

    attacks_to_evaluate = [
        SoftSuffixAttack(suffix_length=1, num_epochs=1, learning_rate=4e-2, batch_size=4, chunk_size=4, max_steps=150),
        SoftSuffixAttack(suffix_length=4, num_epochs=1, learning_rate=4e-2, batch_size=4, chunk_size=4, max_steps=150),
        SoftSuffixAttack(suffix_length=16, num_epochs=1, learning_rate=4e-2, batch_size=4, chunk_size=4, max_steps=150),
    ]
    
    MODEL_DEBUG_MODE = False
    MODEL_DTYPE = torch.bfloat16
    DO_NON_ATTACKED_EVAL = False

    # Initialize probe and model
    OAT_OR_BASE = "llama_base"  # options: "abhayllama", "llama_base", "gemma_oat_mlp", "gemma_oat_linear", "gemma_base", "latllama"

    config = load_config_and_set_vars()
    probes = config["PROBE_PATHS"]
    llms = config["MODEL_PATHS"]

    WANDB_PROJECT = "oat_evaluation" 
    WANDB_ENTITY = None # wandb username or team, or None for default

    for OAT_OR_BASE in ["abhayllama", "llama_base", "latllama"]: #["abhayllama", "llama_base", "gemma_oat_mlp", "gemma_oat_linear", "gemma_base", "latllama"]:
        try:
            if OAT_OR_BASE == "abhayllama":
                probe = AbhayCheckpointProbe(checkpoint_path= probes["llama_3_8b_oat_linear"])
                llm = AutoPEFT(llms["llama"], llms["llama_3_8b_oat_linear"], dtype=MODEL_DTYPE, debug_mode=MODEL_DEBUG_MODE)
            elif OAT_OR_BASE == "llama_base":
                probe = AbhayCheckpointProbe(checkpoint_path=probes["llama_3_8b_linear"])
                llm = AutoLLM(llms["llama"], dtype=MODEL_DTYPE, debug_mode=MODEL_DEBUG_MODE)
            elif OAT_OR_BASE == "gemma_oat_mlp":
                probe = AbhayCheckpointProbe(checkpoint_path=probes["gemma_2_9b_oat_mlp"])
                llm = AutoPEFT(llms["gemma"], llms["gemma_2_9b_oat_mlp"], dtype=MODEL_DTYPE, debug_mode=MODEL_DEBUG_MODE)
            elif OAT_OR_BASE == "gemma_oat_linear":
                probe = AbhayCheckpointProbe(checkpoint_path=probes["gemma_2_9b_oat_linear"])
                llm = AutoPEFT(llms["gemma"], llms["gemma_2_9b_oat_linear"], dtype=MODEL_DTYPE, debug_mode=MODEL_DEBUG_MODE)
            elif OAT_OR_BASE == "gemma_base":
                #raise NotImplementedError("Gemma base model probe not trained yet")
                probe = AbhayCheckpointProbe(checkpoint_path=probes["gemma_2_9b_oat_linear"])
                llm = AutoLLM(llms["gemma"], dtype=MODEL_DTYPE, debug_mode=MODEL_DEBUG_MODE)
            elif OAT_OR_BASE == "latllama":
                probe = AbhayCheckpointProbe(checkpoint_path=probes["llama_3_8b_lat_linear"])
                llm = AutoLLM(llms["llama"], dtype=MODEL_DTYPE, debug_mode=MODEL_DEBUG_MODE)
        except Exception as e:
            print(f"Error loading model {OAT_OR_BASE}: {e}")
            print(traceback.format_exc())
            continue
        
        llm.set_offsets(target_layer_offset=0, target_token_start_offset=0, target_token_end_offset=0)

        _, harmless_probe_scores = batched_generate_and_score(
            llm, [s["prompt"] for s in harmless], [s["completion"] for s in harmless], probe, forced=True
        )
        harmless_probe_scores = [x.item() for x in harmless_probe_scores]

        # Run evaluation for different layer offsets from -1 to 1
        #for max_steps in [32, 64, 128, 256, 512]:
        for attack in attacks_to_evaluate:
            for seed_num in range(5):

                timenow = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{OAT_OR_BASE}_{attack.name}_seed{seed_num}_{timenow}"
                 
                wandb.init(
                     project=WANDB_PROJECT,
                     entity=WANDB_ENTITY,
                     name=run_name,
                     group=f"{OAT_OR_BASE}_{attack.name}", # Group by model and attack type
                     config={
                         # TODO: alter to acquire from attack automatically
                         "model_type": OAT_OR_BASE,
                         "attack_name": attack.name,
                         "attack_params": {
                             "suffix_len": attack.suffix_length,
                             "num_epochs": attack.num_epochs,
                             "learning_rate": attack.learning_rate,
                             "batch_size": attack.batch_size,
                             "chunk_size": attack.chunk_size,
                             "max_steps": attack.max_steps,
                         },
                         "seed": seed_num,
                         "model_dtype": str(MODEL_DTYPE),
                         "harmless_probe_scores": harmless_probe_scores,
                         "model_name": llm.name,
                         "probe_name": probe.name,
                         "do_non_attacked_eval": DO_NON_ATTACKED_EVAL,
                     },
                     reinit=True # Allow re-initialization in loops
                )

                print(f"\n\n{'=' * 50}")
                print(f"STARTING ATTACK with SEED = {seed_num}: {attack}")
                print(f"{'=' * 50}\n")

                try:
                    run_universal_eval(attack=attack, llm=llm, harmful=harmful, harmless_probe_scores=harmless_probe_scores, probe=probe, do_non_attacked_eval=DO_NON_ATTACKED_EVAL,
                                    callback_steps=range(0, 150, 10), seed=seed_num)
                except Exception as e:
                    print(f"Error running attack: {e}")
                    print(traceback.format_exc())
                finally:
                    wandb.finish()
        print_timey("Finished all attacks for model. Deleting probe and model...")
        print_mem_usage()
        del probe
        del llm
        gc.collect()
        print_mem_usage()
        print_timey("Deleted probe and model.")
        
if __name__ == "__main__":
    run_main()
