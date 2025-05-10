from oat_evaluation.utils import load_config_and_set_vars, run_many_commands_on_gpus
load_config_and_set_vars() # needs to be at top to take effect


if __name__ == "__main__":
    commands = [
         ["python", "oat_evaluation/scripts/full_eval_test.py", "--probe_names", "llama_3_8b_cb_linear_1"],
         ["python", "oat_evaluation/scripts/full_eval_test.py", "--probe_names", "llama_3_8b_oat_linear"],
         ["python", "oat_evaluation/scripts/full_eval_test.py", "--probe_names", "llama_3_8b_lat_linear"],
         ["python", "oat_evaluation/scripts/full_eval_test.py", "--probe_names", "llama_3_8b_linear"],
    ]
    run_many_commands_on_gpus(commands, USE_ALL_GPUS=True, log_dir="oat_evaluation/eval_logs")