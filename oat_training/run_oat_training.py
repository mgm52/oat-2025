from oat_evaluation.llms.autollm import AutoLLM
from oat_evaluation.utils import load_config_and_set_vars, print_mem_usage, print_timey
load_config_and_set_vars() # needs to be at top to take effect

from datetime import datetime
import argparse
import json
import os
import random
import warnings
from typing import Literal

warnings.filterwarnings("ignore")  # Suppress all other warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress transformer warnings

import numpy as np

# Third-party library imports
import torch
from datasets import load_dataset

# Local imports
from src.probe_training import (
    train_oat_probe_and_model,
    save_probes,
    Probe
)
from src.probe_archs import (
    LinearProbe,
    NonlinearProbe
)
from src.encoders import (
    EleutherSparseAutoencoder,
    DeepmindSparseAutoencoder
)

MODEL_TOKEN_IDS = {
    "llama3": {
        "ASSISTANT_TOKEN": 78191,
        "USER_TOKEN": 882,
        "NEWLINE_TOKEN": 271,
        "TURN_START_TOKEN_ID": 128000,  # '<|begin_of_text|>' or '<|start_header_id|>'? - Using 128000 based on example. Adjust if needed.
        "TURN_END_TOKEN_ID": 128009,  # '<|eot_id|>'
        "HEADER_END_TOKEN_ID": 128007, # '<|end_header_id|>'
    },
    "gemma2": {
        "ASSISTANT_TOKEN": 2516,  # 'model' token ID
        "USER_TOKEN": 1645,  # 'user' token ID
        "NEWLINE_TOKEN": 108, # '\n' token ID
        "TURN_START_TOKEN_ID": 106, # '<start_of_turn>'
        "TURN_END_TOKEN_ID": 107, # '<end_of_turn>'
        # Gemma doesn't seem to have a direct equivalent to HEADER_END_TOKEN_ID in the same way Llama3 does for delimiting user/assistant roles immediately.
        # We'll rely on the combination of TURN_START and USER/ASSISTANT tokens.
    },
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train probes with configurable masking and probe types"
    )
    parser.add_argument(
        "--masking-type",
        type=str,
        choices=["instruction", "generation"],
        default="generation",
        help="Type of masking to use (instruction or generation)",
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["linear", "nonlinear"],
        default="linear",
        help="Type of probe to use (linear or nonlinear)",
    )
    parser.add_argument(
        "--no-lora-probes",
        action="store_true",
        help="Disable LoRA probes (enabled by default)",
    )
    parser.add_argument(
        "--start-adv-training-at-step",
        type=int,
        default=1024,
        help="Step at which to start adversarial training",
    )
    parser.add_argument(
        "--freeze-probes-during-adv-training",
        type=bool,
        default=True,
        help="Whether to freeze probes during adversarial training",
    )
    parser.add_argument(
        "--freeze-model-during-warmup",
        type=bool,
        default=False,
        help="Whether to freeze model during warmup",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of training steps",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Path to the model to use",
    )
    parser.add_argument(
        "--adversary-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the adversary",
    )
    parser.add_argument(
        "--adapter-lr",
        type=float,
        default=5e-5,
        help="Learning rate for the adapter",
    )
    parser.add_argument(
        "--probe-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the probes",
    )
    parser.add_argument(
        "--pgd-iterations",
        type=int,
        default=32,
        help="Number of PGD iterations for adversarial training",
    )
    parser.add_argument(
        "--kl-penalty",
        type=float,
        default=10.0,
        help="KL divergence penalty weight",
    )
    parser.add_argument(
        "--nickname",
        type=str,
        default="oat",
        help="Nickname for the run",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="4,8,12,16,20,24",
        help="Layers to train on",
    )
    return parser.parse_args()


def create_linear_probe():
    return LinearProbe(llm._model.config.hidden_size)


def create_nonlinear_probe():
    return NonlinearProbe(llm._model.config.hidden_size, 64)


def get_probe_creator(probe_type: Literal["linear", "nonlinear"]):
    if probe_type == "linear":
        return lambda: LinearProbe(llm._model.config.hidden_size)
    else:
        return lambda: NonlinearProbe(llm._model.config.hidden_size, 64)


def sample_examples_from_datasets(
    datasets, proportions, total_examples=1000, only_prompts=False
):
    # This function samples examples from multiple datasets, ensuring that the final list has the desired proportions
    # of examples from each dataset. The final list is shuffled.

    # Ensure the proportions sum to 1
    if len(datasets) != len(proportions):
        raise ValueError("Number of datasets must match number of proportions")

    if abs(sum(proportions) - 1) > 1e-6:
        raise ValueError("Proportions must sum to 1")

    examples = []
    np.random.seed(42)
    for dataset, proportion in zip(datasets, proportions):
        n_samples = int(total_examples * proportion)

        # Ensure we don't try to sample more examples than available
        sampled_indices = np.random.choice(len(dataset), size=n_samples, replace=True)
        sampled = dataset.select(sampled_indices)

        if only_prompts:
            examples.extend([item["prompt"] for item in sampled])
        else:
            examples.extend(
                [f"{item['prompt']} {item['completion']}" for item in sampled]
            )

    # Shuffle the final list to mix examples from different datasets
    random.Random(42).shuffle(examples)

    return examples


def split_dataset(dataset, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0):
    # Function to split dataset into train, validation, and test sets
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1"

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_set = dataset[:train_size]
    val_set = dataset[train_size : train_size + val_size]
    test_set = dataset[train_size + val_size :]

    return train_set, val_set, test_set


def is_token_after_specific_token(seq_idx, token, tokens, target_token_id):
    """Checks if the current token immediately follows the target_token_id."""
    return seq_idx >= 1 and tokens[seq_idx - 1] == target_token_id

def is_at_newline_after_specific_token(seq_idx, token, tokens, target_token_id, newline_token_id):
    """Checks if the current token is a newline and follows the sequence <turn_start><specific_token><newline> or <header_end><newline>."""
    # Llama3: <start_header_id> <role> <end_header_id> \n -> Check for newline after <end_header_id>
    # Gemma2: <start_of_turn> <role> \n -> Check for newline after <role>
    # Check Llama3 pattern: pos-2 is end_header_id and pos-3 is role
    if 'HEADER_END_TOKEN_ID' in MODEL_TOKEN_IDS["llama3"] and \
       target_token_id == MODEL_TOKEN_IDS["llama3"]["USER_TOKEN"] or \
       target_token_id == MODEL_TOKEN_IDS["llama3"]["ASSISTANT_TOKEN"]:
        if seq_idx >= 2 and tokens[seq_idx-1] == MODEL_TOKEN_IDS["llama3"]["HEADER_END_TOKEN_ID"] and token == newline_token_id:
             # check if pos-2 is the target role
             # Need to look further back for the actual role token before the header end
             if seq_idx >= 3 and tokens[seq_idx-2] == target_token_id:
                 return True # It's newline after header end, after target role

    # Check Gemma2 / General pattern: pos-2 is target_token_id
    if seq_idx >= 2 and tokens[seq_idx - 2] == target_token_id and token == newline_token_id:
        return True

    return False


def get_token_ranges(masking_type: Literal["instruction", "generation"], model_name: str):
    if model_name not in MODEL_TOKEN_IDS:
        raise ValueError(f"Unknown model_name: {model_name}. Supported models: {list(MODEL_TOKEN_IDS.keys())}")

    ids = MODEL_TOKEN_IDS[model_name]
    TURN_END_TOKEN_ID = ids["TURN_END_TOKEN_ID"]
    ASSISTANT_TOKEN_ID = ids["ASSISTANT_TOKEN"]
    USER_TOKEN_ID = ids["USER_TOKEN"]
    NEWLINE_TOKEN_ID = ids["NEWLINE_TOKEN"]


    # Define closures that capture the correct token IDs
    def is_at_newline_after_assistant_closure(seq_idx, token, tokens):
        return is_at_newline_after_specific_token(seq_idx, token, tokens, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID)

    def is_at_newline_after_user_closure(seq_idx, token, tokens):
        return is_at_newline_after_specific_token(seq_idx, token, tokens, USER_TOKEN_ID, NEWLINE_TOKEN_ID)

    def is_token_after_assistant_closure(seq_idx, token, tokens):
        # For Llama3, generation starts after <end_header_id>\n
        if model_name == 'llama3':
             # Check if current token is after the newline following the assistant header end
             if seq_idx >= 2 and \
                tokens[seq_idx-1] == NEWLINE_TOKEN_ID and \
                tokens[seq_idx-2] == ids["HEADER_END_TOKEN_ID"]:
                 # Verify that the assistant token preceded the header end
                 if seq_idx >= 3 and tokens[seq_idx-3] == ASSISTANT_TOKEN_ID:
                     return True
        # For Gemma2, generation starts after <start_of_turn>model\n
        elif model_name == 'gemma2':
             if seq_idx >= 2 and \
                tokens[seq_idx-1] == NEWLINE_TOKEN_ID and \
                tokens[seq_idx-2] == ASSISTANT_TOKEN_ID:
                 return True
        return False # Default case

    # Note: The logic for 'only_return_on_tokens_between' and 'only_choose_prompt_tokens_between'
    # seems to target the first token *after* the newline following the role identifier.
    # The 'only_probe_tokens_between' logic targets the entire generation/instruction section.

    token_ranges = {
        "only_return_on_tokens_between": [ # Region in which to enforce low KL divergence of logits
            is_at_newline_after_assistant_closure, # Start checking *after* the newline following assistant marker
            TURN_END_TOKEN_ID, # End of generation (if not present, range goes as far as possible)
        ],
        "only_choose_prompt_tokens_between": [ # Region in which to modify / generate soft prompts
            is_at_newline_after_user_closure, # Start checking *after* the newline following user marker
            TURN_END_TOKEN_ID,  # End of instruction
        ]
    }

    if masking_type == "generation":
        # Probe tokens *within* the assistant's response
        token_ranges["only_probe_tokens_between"] = [ # Region in which to probe
            is_token_after_assistant_closure, # Start probing from the first token of the assistant's response, including the initial newline
            TURN_END_TOKEN_ID  # End of generation (if not present, range goes as far as possible)
        ]
    elif masking_type == "instruction":
         # Probe tokens *within* the user's instruction
        token_ranges["only_probe_tokens_between"] = [ # Region in which to probe
            is_at_newline_after_user_closure, # Start probing from the first token of the user's instruction (after newline)
            TURN_END_TOKEN_ID
        ]

    return token_ranges


def main():
    global llm

    args = parse_args()
    # Allow specifying model type via argument later if needed
    # For now, let's keep it hardcoded for testing

    print(f"Going to attempt to load model from {args.model_path}...")


    model_path = args.model_path
    model_path_name = model_path.split("/")[-1]
    probes_folder = f"./oat_training/oat_training_results/{model_path_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.n_steps}_steps_{args.nickname}_{random.randint(0, 999999999999)}/"
    
    # While the folder already exists, we'll append a random number to the folder name
    while os.path.exists(probes_folder):
        probes_folder = f"{probes_folder}_{random.randint(0, 999999999999)}/"

    masking_type = args.masking_type
    use_lora_probes = not args.no_lora_probes
    name = f"{model_path_name}_lora_oat_{masking_type}_{'nonlinear' if args.probe_type == 'nonlinear' else 'linear'}"

    model_type = "llama3" if "llama" in model_path_name.lower() else "gemma2" if "gemma" in model_path_name.lower() else None
    if model_type is None:
        raise ValueError(f"Unknown model type: {model_path_name}")

    # Load model and dataset
    llm = AutoLLM(model_path)
    if model_type == "llama3":
        #encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
        #encoder = EleutherSparseAutoencoder.load_custom_model_without_sae(model_path)
        jailbreaks_dataset = load_dataset(
            "Mechanistic-Anomaly-Detection/llama3-jailbreaks"
        )
    elif model_type == "gemma2":
        #encoder = DeepmindSparseAutoencoder.load_gemma2_sae(None, 11)
        #encoder = DeepmindSparseAutoencoder.load_custom_model_without_sae(model_path)
        jailbreaks_dataset = load_dataset(
            "Mechanistic-Anomaly-Detection/gemma2-jailbreaks"
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    forget_examples_train = sample_examples_from_datasets(
        [jailbreaks_dataset["circuit_breakers_train"]], [1.0]
    )

    retain_examples_train = sample_examples_from_datasets(
        [jailbreaks_dataset["xstest"], jailbreaks_dataset["benign_instructions_train"]],
        [0.15, 0.85],
    )

    # Also get examples with just the prompts
    forget_examples_train_prompts = sample_examples_from_datasets(
        [jailbreaks_dataset["circuit_breakers_train"]], [1.0], only_prompts=True
    )

    retain_examples_train_prompts = sample_examples_from_datasets(
        [jailbreaks_dataset["xstest"], jailbreaks_dataset["benign_instructions_train"]],
        [0.15, 0.85],
        only_prompts=True,
    )

    # Train/test split
    forget_examples_train, forget_examples_val, _ = split_dataset(forget_examples_train)
    retain_examples_train, retain_examples_val, _ = split_dataset(retain_examples_train)
    _, forget_examples_val_prompts, _ = split_dataset(forget_examples_train_prompts)
    # Make sure retain validation prompts are also split if needed for eval
    _, retain_examples_val_prompts, _ = split_dataset(retain_examples_train_prompts)

    ranges = get_token_ranges(masking_type, model_type) # Pass model_type here
    only_return_on_tokens_between = ranges["only_return_on_tokens_between"]
    only_choose_prompt_tokens_between = ranges["only_choose_prompt_tokens_between"]
    only_probe_tokens_between = ranges["only_probe_tokens_between"]

    time_at_start_of_process = datetime.now().strftime("%Y%m%d_%H%M%S")

    probes, lora_model, info = train_oat_probe_and_model(
        llm=llm,
        positive_examples=forget_examples_train,
        negative_examples=retain_examples_train,
        create_probe_fn=get_probe_creator(args.probe_type),
        layers=[int(layer) for layer in args.layers.split(",")],
        max_length=512,
        n_steps_per_logging=8,
        batch_size=2,
        n_grad_accum=8,
        adversary_lr=args.adversary_lr,
        adapter_lr=args.adapter_lr,
        probe_lr=args.probe_lr,
        n_steps=args.n_steps,
        run_softprompt_eval_every=256,
        pgd_iterations=args.pgd_iterations,
        kl_penalty=args.kl_penalty,
        device="cuda",
        only_return_on_tokens_between=only_return_on_tokens_between,
        only_choose_prompt_tokens_between=only_choose_prompt_tokens_between,
        only_probe_tokens_between=only_probe_tokens_between,
        adversarial_training=True,
        use_lora_adapter=use_lora_probes,
        softprompt_evals_data={
            "test_negative_examples": retain_examples_val,
            "test_positive_examples": forget_examples_val,
            "test_positive_prompts": forget_examples_val_prompts,
            # Pass model-specific ranges to evaluation too
            "only_return_on_tokens_between": only_return_on_tokens_between,
            "only_choose_prompt_tokens_between": only_choose_prompt_tokens_between,
            "only_probe_tokens_between": only_probe_tokens_between,
            # It might be useful to pass the model_type or specific token IDs here as well
            # depending on how the evaluation function uses these ranges.
        },
        checkpoint_dir=probes_folder,
        checkpoint_every=256,
        start_adv_training_at_step=args.start_adv_training_at_step,
        freeze_probes_during_adversarial_training=args.freeze_probes_during_adv_training,
        freeze_model_during_warmup=args.freeze_model_during_warmup,
    )

    # Save results
    save_probes(
        probes=probes,
        save_path=os.path.join(probes_folder, f"{name}_probes.pt"),
    )

    # Check if the model is a lora adapter, if so save it
    if use_lora_probes and hasattr(lora_model, 'save_pretrained') and (args.start_adv_training_at_step < args.n_steps or not args.freeze_model_during_warmup):
        lora_model.save_pretrained(os.path.join(probes_folder, f"{name}_model"))
    elif not hasattr(lora_model, 'save_pretrained'):
         print("Warning: Skipping saving model because it doesn't have 'save_pretrained'")

    # Save the info
    with open(os.path.join(probes_folder, f"{name}_info.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main()
