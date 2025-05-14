from datetime import datetime
import os
from oat_evaluation.utils import load_config_and_set_vars, run_many_commands_on_gpus
load_config_and_set_vars() # needs to be at top to take effect


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # If group name is not provided, each process will output to a different group in W&B
    wandb_group_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_lowstep_softs"
    commands = [
          # 0.1x test samples
          #["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_linear", "--group-name", wandb_group_name, "--num_harmful_test_samples", "10"],
          # 0.5x test samples
          #["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_linear", "--group-name", wandb_group_name, "--num_harmful_test_samples", "50"],
          # Standard
          #["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_linear", "--group-name", wandb_group_name, "--num_harmful_test_samples", "100"],
          # 2x test samples
          #["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_linear", "--group-name", wandb_group_name, "--num_harmful_test_samples", "200"],
          # 4x test samples
          #["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_linear", "--group-name", wandb_group_name, "--num_harmful_test_samples", "400"],

     # Regular evals

        # PAIR batch
         
        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_highpgd", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256"],
        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_highpgd", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--dont_perform_obfuscation", "True"],
         
        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_linear", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256"],
        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_linear", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--dont_perform_obfuscation", "True"],

        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_cb_linear_midlr", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256"],
        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_cb_linear_midlr", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--dont_perform_obfuscation", "True"],

        # Soft suffix / pert batch

        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_lat_linear", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256"],
         
        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_linear", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256"],

        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_cb_linear_lowlr", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256"],
        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_cb_linear_midlr", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256"],
        #  ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_cb_linear_highlr", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256"],

         # Step 2048 models
         ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_lowpgd", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],
         ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_recreation", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],
         ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_highpgd", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],

         # Step 1792 models
     #     ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_highpgd_1792", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],
     #     ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_recreation_1792", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],
     #     ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_lowpgd_1792", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],

         # Step 1536 models
         ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_highpgd_1536", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],
         ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_recreation_1536", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],
         ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_lowpgd_1536", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],

         # Step 1280 models
     #     ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_highpgd_1280", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],
     #     ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_recreation_1280", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],
     #     ["python", "oat_evaluation/scripts/eval_scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear_lowpgd_1280", "--group-name", wandb_group_name, "--num_harmful_test_samples", "256", "--num_samples_when_slow", "20"],

    ]

    run_many_commands_on_gpus(commands, USE_ALL_GPUS=True, log_dir="oat_evaluation/outputs/eval_logs")