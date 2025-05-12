from oat_evaluation.utils import load_config_and_set_vars, print_mem_usage
load_config_and_set_vars() # needs to be at top to take effect

import random
from typing import List, Tuple, Dict, Union

import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset, IterableDatasetDict, IterableDataset


def sample_examples(
    dataset_list: List, proportions: List[float], total: int = 1000, only_prompts: bool = False
) -> List:
    """
    Sample from multiple datasets according to proportions.
    """
    if len(dataset_list) != len(proportions) or not np.isclose(sum(proportions), 1.0):
        raise ValueError("Datasets and proportions must match and sum to 1.")

    examples = []
    for ds, prop in zip(dataset_list, proportions):
        count = int(total * prop)
        idx = np.random.choice(len(ds), size=count, replace=True)
        sampled = ds.select(idx)
        if only_prompts:
            examples.extend(item["prompt"] for item in sampled)
        else:
            examples.extend(
                {"prompt": item["prompt"], "completion": item.get("completion", item.get("response"))}
                for item in sampled
            )
    random.Random(42).shuffle(examples)
    return examples


def load_harmful_harmful_cb_5000():
    harm = load_dataset("justinphan3110/circuit_breakers_train", split="train")
    harm = harm.map(lambda x: {
        "prompt": x["prompt"],
        "completion": x["response"]
    })
    return harm

def load_harmless_ultra_208k():
    ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    def format_ultrachat(example):
        messages = example["messages"]
        prompt = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        completion = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
        return {"prompt": prompt, "completion": completion}
    ultra = ultra.map(format_ultrachat)
    return ultra

# Note that this includes both harmful and harmless examples
def load_xstest_450():
    xstest = load_dataset("natolambert/xstest-v2-copy", split="gpt4")
    xstest = xstest.map(lambda x: {
        "prompt": x["prompt"],
        "completion": x["completion"]
    })
    return xstest


def strip_llama_prompt_tokens(prompt: str) -> str:
    return prompt.replace("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n", "").replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "")

def load_harmful_harmful_training_abhay_3000():
    # Folder is "llama3 jailbreaks" but we strip llama-specific tokens
    harm = load_dataset("Mechanistic-Anomaly-Detection/llama3-jailbreaks", split="circuit_breakers_train")
    harm = harm.map(lambda x: {
        "prompt": strip_llama_prompt_tokens(x["prompt"]),
        "completion": x["completion"]
    })
    return harm

def load_harmful_harmful_test_abhay_1000():
    # Folder is "llama3 jailbreaks" but we strip llama-specific tokens
    harm = load_dataset("Mechanistic-Anomaly-Detection/llama3-jailbreaks", split="circuit_breakers_test")
    harm = harm.map(lambda x: {
        "prompt": strip_llama_prompt_tokens(x["prompt"]),
        "completion": x["completion"]
    })
    return harm

def load_harmless_training_abhayultra_11961():
    # Folder is "llama3 jailbreaks" but we strip llama-specific tokens
    ultra = load_dataset("Mechanistic-Anomaly-Detection/llama3-jailbreaks", split="benign_instructions_train")
    ultra = ultra.map(lambda x: {
        "prompt": strip_llama_prompt_tokens(x["prompt"]),
        "completion": x["completion"]
    })
    return ultra

def load_harmless_training_abhayxstest_230():
    # Folder is "llama3 jailbreaks" but we strip llama-specific tokens
    xstest = load_dataset("Mechanistic-Anomaly-Detection/llama3-jailbreaks", split="xstest")
    xstest = xstest.map(lambda x: {
        "prompt": strip_llama_prompt_tokens(x["prompt"]),
        "completion": x["completion"]
    })
    return xstest

def load_harmless_test_abhayutra_3979():
    # Folder is "llama3 jailbreaks" but we strip llama-specific tokens
    ultra = load_dataset("Mechanistic-Anomaly-Detection/llama3-jailbreaks", split="benign_instructions_test")
    ultra = ultra.map(lambda x: {
        "prompt": strip_llama_prompt_tokens(x["prompt"]),
        "completion": x["completion"]
    })
    return ultra

# Restriction: abhay didn't specify a test set for xstest, so there may be some overlap with his training set...
# For our purposes, this just means the probe 1%FPR thresholds we learn may be veeeeery slightly lower than they should be
def load_harmless_blend_test_abhay_1533():
    # Folder is "llama3 jailbreaks" but we strip llama-specific tokens
    ultra = load_harmless_test_abhayutra_3979()
    xstest = load_harmless_training_abhayxstest_230()

    # 15% xstest (230 samples); 85% ultra (1303 samples) -> total 1533
    num_ultra_samples = int(len(xstest) * 0.85 / 0.15) # Calculate needed ultra samples
    # Ensure we don't request more samples than available
    num_ultra_samples = min(num_ultra_samples, len(ultra))

    ultra_subset = ultra.select(range(num_ultra_samples)) # Select the first num_ultra_samples

    return concatenate_datasets([ultra_subset, xstest])


