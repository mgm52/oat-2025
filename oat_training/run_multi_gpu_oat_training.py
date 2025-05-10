from oat_evaluation.utils import load_config_and_set_vars, run_many_commands_on_gpus
load_config_and_set_vars() # needs to be at top to take effect


if __name__ == "__main__":
    commands = [
        # TRAINING PROBES ONLY
        # ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "8", "--start-adv-training-at-step", "9999999", "--freeze-model-during-warmup", "True",
        #  "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct"],

        # ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "1024", "--start-adv-training-at-step", "9999999", "--freeze-model-during-warmup", "True",
        #  "--model-path", "GraySwanAI/Llama-3-8B-Instruct-RR"],

        # TRAINING LLAMA 8B
        #  ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "2048", "--start-adv-training-at-step", "1024",
        #  "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--nickname", "regular"],

        #  ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "2048", "--start-adv-training-at-step", "1024",
        #  "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--adapter-lr", str(2e-5), "--nickname", "0.5modelLR"],

        #  ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "2048", "--start-adv-training-at-step", "1024",
        #  "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--adapter-lr", str(1e-4), "--nickname", "2modelLR"],
        
        #  ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "2048", "--start-adv-training-at-step", "1024",
        #  "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--pgd-iterations", "16", "--nickname", "16pgd"],

        #  ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "2048", "--start-adv-training-at-step", "1024",
        #  "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--pgd-iterations", "64", "--nickname", "64pgd"],

        #  ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "nonlinear", 
        #  "--n-steps", "2048", "--start-adv-training-at-step", "1024",
        #  "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--nickname", "nonlinear"],

        # ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "2048", "--start-adv-training-at-step", "9999999",
        #  "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--nickname", "no_adv"],

        # TRAINING LLAMA 1B and 3B
         ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
         "--n-steps", "2048", "--start-adv-training-at-step", "1024",
         "--model-path", "meta-llama/Llama-3.2-1B-Instruct", "--nickname", "31_1b_4layers", "--layers", "3,9,11,15"],

         ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
         "--n-steps", "2048", "--start-adv-training-at-step", "1024",
         "--model-path", "meta-llama/Llama-3.2-1B-Instruct", "--nickname", "31_1b_6layers", "--layers", "2,4,6,9,12,14"],

         ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
         "--n-steps", "2048", "--start-adv-training-at-step", "1024",
         "--model-path", "meta-llama/Llama-3.2-3B-Instruct", "--nickname", "31_3b"],

    ]
    run_many_commands_on_gpus(commands, USE_ALL_GPUS=True)