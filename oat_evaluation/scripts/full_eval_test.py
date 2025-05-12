import hashlib
import os
import random
from oat_evaluation.attacks.attack import Attack
from oat_evaluation.llms.api_llm import ApiLLM
from oat_evaluation.utils import dataset_to_list, load_config_and_set_vars, print_mem_usage, print_timey
load_config_and_set_vars() # needs to be at top to take effect

from datetime import datetime
import time as time_lib
import gc
from oat_evaluation.attacks.perturbation import PerturbationAttack
from oat_evaluation.evals.universal import batched_generate_and_score

import traceback
import torch
import argparse

from oat_evaluation.attacks.soft_suffix import SoftSuffixAttack
from oat_evaluation.attacks.pair import PAIRAttack
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run full evaluation test with specified probe names')
    parser.add_argument('--probe_names', type=str, default="llama_3_8b_cb_linear_1",
                      help='Comma-separated list of probe names to evaluate (default: "llama_3_8b_cb_linear_1")')
    parser.add_argument('--group-name', type=str, default="",
                      help='Name of the group to run the evaluation in (default: "")')
    parser.add_argument('--num_harmful_test_samples', type=int, default=100,
                      help='Number of harmful test samples to use (default: 100)')
    parser.add_argument('--dont_perform_obfuscation', type=bool, default=False,
                      help='Whether to ignore probe during all attacks (default: False)')
    args = parser.parse_args()

    MINI_MODE = False
    print(f"MINI_MODE: {MINI_MODE}")

    print(f"Loading datasets once for all evaluations...")
    harmful = load_harmful_harmful_test_abhay_1000()
    harmful = dataset_to_list(harmful)

    harmless = load_harmless_blend_test_abhay_1533()
    harmless = dataset_to_list(harmless)
    harmless = harmless[:8] if MINI_MODE else harmless

    print(f"Loaded {len(harmless)} harmless examples. First one: {harmless[0]}")

    num_steps_soft_attack_standard = 75 if not MINI_MODE else 8
    remote_attacker_llm = ApiLLM(model_name = "gpt-4.1-mini", 
                           base_url = "https://api.openai.com/v1",
                           api_key_env_var="OPENAI_API_KEY",
                           )

    attacks_to_evaluate = [
        #lambda: PerturbationAttack(num_epochs=1, learning_rate=4e-2, batch_size=4, chunk_size=4, max_steps=num_steps_soft_attack_standard),
        #lambda: PerturbationAttack(num_epochs=1, learning_rate=2e-2, batch_size=4, chunk_size=4, max_steps=num_steps_soft_attack_standard),
        #lambda: PerturbationAttack(num_epochs=1, learning_rate=8e-2, batch_size=4, chunk_size=4, max_steps=num_steps_soft_attack_standard),

        # Trying to get CIs down
        # lambda: SoftSuffixAttack(suffix_length=1, num_epochs=1, learning_rate=3e-2, batch_size=4, chunk_size=4, max_steps=75),
        # lambda: SoftSuffixAttack(suffix_length=1, num_epochs=1, learning_rate=3e-2, batch_size=8, chunk_size=8, max_steps=75),
        # lambda: SoftSuffixAttack(suffix_length=1, num_epochs=1, learning_rate=4e-2, batch_size=4, chunk_size=4, max_steps=75),
        # lambda: SoftSuffixAttack(suffix_length=1, num_epochs=1, learning_rate=4e-2, batch_size=8, chunk_size=8, max_steps=75),

        # PAIR attacks
        lambda: PAIRAttack(
            n_concurrent_jailbreaks=1,
            max_num_iterations=2,
            keep_last_n_in_convs=2,
            max_json_attempts=2,
            max_new_attack_tokens=256,
            max_new_victim_tokens=256,
            attack_llm=remote_attacker_llm,
            judge_llm=remote_attacker_llm,
        ),
        lambda: PAIRAttack(
            n_concurrent_jailbreaks=1,
            max_num_iterations=8,
            keep_last_n_in_convs=4,
            max_json_attempts=2,
            max_new_attack_tokens=256,
            max_new_victim_tokens=256,
            attack_llm=remote_attacker_llm,
            judge_llm=remote_attacker_llm,
        ),

        # lambda: SoftSuffixAttack(suffix_length=1, num_epochs=1, learning_rate=3e-2, batch_size=8, chunk_size=8, max_steps=num_steps_soft_attack_standard),
        # lambda: SoftSuffixAttack(suffix_length=4, num_epochs=1, learning_rate=3e-2, batch_size=8, chunk_size=8, max_steps=num_steps_soft_attack_standard),
        # lambda: SoftSuffixAttack(suffix_length=16, num_epochs=1, learning_rate=3e-2, batch_size=8, chunk_size=8, max_steps=num_steps_soft_attack_standard),
        
        # lambda: PerturbationAttack(num_epochs=1, learning_rate=3e-2, batch_size=8, chunk_size=8, max_steps=num_steps_soft_attack_standard),
    ]
    
    MODEL_DEBUG_MODE = False
    MODEL_DTYPE = torch.bfloat16
    DO_NON_ATTACKED_EVAL = False

    # Initialize probe and model
    # OAT_OR_BASE = "llama_base"  # options: "abhayllama", "llama_base", "gemma_oat_mlp", "gemma_oat_linear", "gemma_base", "latllama"

    config = load_config_and_set_vars()
    probes = config["PROBE_PATHS"]
    llms = config["MODEL_PATHS"]

    WANDB_PROJECT = "oat_evaluation" 
    WANDB_ENTITY = None # wandb username or team, or None for default

    callback_steps=list(range(0, 15, 3)) + list(range(15, 75, 5)) + list(range(75, 150, 10))
    seeds = range(5) if not MINI_MODE else range(1)
    num_harmful_test_samples = args.num_harmful_test_samples if not MINI_MODE else 8
    num_harmful_train_samples = 600 if not MINI_MODE else 8

    process_id = os.getpid()
    process_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{process_id}"

    print(f"Probes to choose from: {probes.keys()}")
    print(f"Models to choose from: {llms.keys()}")

    probe_names_to_evaluate = [name.strip() for name in args.probe_names.split(',')]

    for pn in probe_names_to_evaluate:
        try:
            probe_details = probes[pn]
            model_details = llms[probe_details["model_name"]]

            probe = AbhayCheckpointProbe(checkpoint_path=probe_details["path"])
            if model_details["peft"]:
                llm = AutoPEFT(llms[model_details["base_model_name"]]["path"], model_details["path"], dtype=MODEL_DTYPE, debug_mode=MODEL_DEBUG_MODE)
            else:
                llm = AutoLLM(model_details["path"], dtype=MODEL_DTYPE, debug_mode=MODEL_DEBUG_MODE)
        except Exception as e:
            print(f"Error loading model/probe {probe_details['model_name']}: {e}")
            print(traceback.format_exc())
            continue
        
        llm.set_offsets(target_layer_offset=0, target_token_start_offset=0, target_token_end_offset=0)

        with torch.no_grad():
            _, harmless_probe_scores = batched_generate_and_score(
                llm, [s["prompt"] for s in harmless], [s["completion"] for s in harmless], probe, forced=True
            )
        harmless_probe_scores = [x.item() for x in harmless_probe_scores]

        # Run evaluation for different layer offsets from -1 to 1
        #for max_steps in [32, 64, 128, 256, 512]:
        for ai in range(len(attacks_to_evaluate)):
            attack_id = hashlib.sha256(f"{ai}{time_lib.time()}{random.randint(0, 1000000)}{process_id}".encode()).hexdigest()
            is_slow = False
            for seed_num in seeds:
                if seed_num > 0 and is_slow:
                    print(f"Skipping slow attack (attack idx {ai}) for seed {seed_num} because it's already been run for seed 0")
                    continue

                # Instantiate attack
                attack: Attack = attacks_to_evaluate[ai]()

                is_slow = is_slow or attack.is_slow
                if is_slow:
                    num_test_samples_override = num_harmful_test_samples // 2
                else:
                    num_test_samples_override = num_harmful_test_samples

                # Initialize wandb
                timenow = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{pn}_{ai}_{attack.name}_seed{seed_num}_{timenow}"
                 
                wandb.init(
                     project=WANDB_PROJECT,
                     entity=WANDB_ENTITY,
                     name=run_name,
                     group=args.group_name if args.group_name else process_id, # Group evals within a single process
                     config={
                         # TODO: alter to acquire from attack automatically
                         "attack_name": attack.name,
                         "attack_str": str(attack),
                         # attack_params are important: they're used to identify identical attacks across runs
                         # TODO: change to make Attack generate this itself
                         "attack_params": {
                             # SoftSuffix parameters
                             "suffix_len": attack.suffix_length if isinstance(attack, SoftSuffixAttack) else None,
                             "num_epochs": attack.num_epochs if hasattr(attack, 'num_epochs') else None,
                             "learning_rate": attack.learning_rate if hasattr(attack, 'learning_rate') else None,
                             "batch_size": attack.batch_size if hasattr(attack, 'batch_size') else None,
                             "chunk_size": attack.chunk_size if hasattr(attack, 'chunk_size') else None,
                             "max_steps": attack.max_steps if hasattr(attack, 'max_steps') else None,
                             # PAIR parameters
                             "n_concurrent_jailbreaks": attack.n_concurrent_jailbreaks if isinstance(attack, PAIRAttack) else None,
                             "max_num_iterations": attack.max_num_iterations if isinstance(attack, PAIRAttack) else None,
                             "keep_last_n_in_convs": attack.keep_last_n_in_convs if isinstance(attack, PAIRAttack) else None,
                             "max_json_attempts": attack.max_json_attempts if isinstance(attack, PAIRAttack) else None,
                             "max_new_attack_tokens": attack.max_new_attack_tokens if isinstance(attack, PAIRAttack) else None,
                             "max_new_victim_tokens": attack.max_new_victim_tokens if isinstance(attack, PAIRAttack) else None,
                             # Common parameters
                             "num_harmful_train_samples": num_harmful_train_samples,
                             "num_harmful_test_samples": num_test_samples_override,
                             "performing_obfuscation": (not args.dont_perform_obfuscation) and attack.can_perform_obfuscation,
                         },
                         "seed": seed_num,
                         "model_dtype": str(MODEL_DTYPE),
                         "harmless_probe_scores": harmless_probe_scores,
                         "model_type": llm.name,
                         "model_name": probe_details["model_name"],
                         "model_path": model_details["path"],
                         "probe_type": probe.name,
                         "probe_name": pn,
                         "probe_path": probe_details["path"],
                         "do_non_attacked_eval": DO_NON_ATTACKED_EVAL,
                         "defender_flop_cost": model_details["flops"] + probe_details["flops"],
                         "attack_id": attack_id,
                     },
                     reinit=True # Allow re-initialization in loops
                )

                print(f"\n\n{'=' * 50}")
                print_timey(f"STARTING ATTACK with SEED = {seed_num}: {attack}")
                print(f"{'=' * 50}\n")

                try:
                    run_universal_eval(attack=attack, llm=llm, harmful=harmful, harmless_probe_scores=harmless_probe_scores, probe=probe, do_non_attacked_eval=DO_NON_ATTACKED_EVAL,
                                    callback_steps=callback_steps, seed=seed_num, num_harmful_test_samples=num_test_samples_override, num_harmful_train_samples=num_harmful_train_samples,
                                    dont_perform_obfuscation=args.dont_perform_obfuscation)
                except Exception as e:
                    print(f"Error running attack: {e}")
                    print(traceback.format_exc())
                finally:
                    wandb.finish()
                    del attack
        print_timey("Finished all attacks for model. Deleting probe and model...")
        print_mem_usage()
        del probe
        del llm
        gc.collect()
        print_mem_usage()
        print_timey("Deleted probe and model.")
        
if __name__ == "__main__":
    run_main()
