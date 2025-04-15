from strong_reject.evaluate import strongreject_rubric

import random
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datasets import load_dataset
import torch

from oat_evaluation.attacks.soft_suffix import SoftSuffixAttack
from oat_evaluation.llms.autollm import AutoLLM
from oat_evaluation.probes.probe_abhay_checkpoint import AbhayCheckpointProbe

def sample_examples_from_datasets(
    datasets, proportions, total_examples=1000, only_prompts=False
):
    """
    Sample examples from multiple datasets with specific proportions.
    
    Args:
        datasets: List of datasets to sample from
        proportions: List of proportions for each dataset (should sum to 1)
        total_examples: Total number of examples to sample
        only_prompts: If True, return only prompts (without completions)
        
    Returns:
        List of sampled examples
    """
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
            examples.extend([
                {
                    "prompt": item["prompt"],
                    "completion": item["completion"]
                }
                for item in sampled
            ]
                # For now, hardcoding gemma's chat template!
                #[f"<bos><start_of_turn>user\n{item['prompt']}<end_of_turn>\n<start_of_turn>model\n{item['completion']}<end_of_turn>\n" for item in sampled]
            )

    # Shuffle the final list to mix examples from different datasets
    random.Random(42).shuffle(examples)

    return examples

def load_jailbreak_datasets():
    """
    Load and prepare the jailbreak datasets from specified sources:
    - Harmful: circuit_breakers_train
    - Harmless: UltraChat 200k and xstest-v2-copy (gpt4 split)
    
    Returns:
        Dictionary with harmful and harmless examples
    """
    # Load the datasets from Hugging Face
    harmful_dataset = load_dataset("justinphan3110/circuit_breakers_train", split="train")
    harmless_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    xstest_dataset = load_dataset("natolambert/xstest-v2-copy", split="gpt4")
    
    # Print a preview of the first prompt and completion from each dataset
    print("\n=== HARMFUL DATASET PREVIEW ===")
    print(f"Prompt: {harmful_dataset[0]['prompt'][:200]}...")
    print(f"Completion: {harmful_dataset[0]['llama3_output'][:200]}...")
    
    print("\n=== HARMLESS DATASET PREVIEW ===")
    messages = harmless_dataset[0]["messages"]
    user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
    assistant_msg = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
    print(f"Prompt: {user_msg[:200]}...")
    print(f"Completion: {assistant_msg[:200]}...")
    
    print("\n=== XSTEST DATASET PREVIEW ===")
    print(f"Prompt: {xstest_dataset[0]['prompt'][:200]}...")
    print(f"Completion: {xstest_dataset[0]['completion'][:200]}...")
    
    # Format datasets to have consistent structure
    # For circuit_breakers_train
    harmful_formatted = harmful_dataset.map(
        # Only take the rejection responses (llama3_output)
        lambda x: {"prompt": x["prompt"], "completion": x["llama3_output"]}
    )
    
    # For UltraChat 200k - restructure to extract the prompt
    def format_ultrachat(example):
        messages = example["messages"]
        prompt = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        completion = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
        return {"prompt": prompt, "completion": completion}
    
    harmless_formatted = harmless_dataset.map(format_ultrachat)
    
    # Format xstest - assuming it has prompt and completion fields
    xstest_formatted = xstest_dataset

    # Dreject: 5000 samples from harmful Circuit Breakers dataset
    harmful_examples = sample_examples_from_datasets(
        [harmful_formatted], [1.0],
        total_examples=5000
    )

    # Dutility: 5000 samples from harmless UltraChat 200k + 150 from XSTest
    harmless_ultrachat = sample_examples_from_datasets(
        [harmless_formatted], [1.0],
        total_examples=5000
    )
    
    harmless_xstest = sample_examples_from_datasets(
        [xstest_formatted], [1.0],
        total_examples=150
    )
    
    harmless_examples = harmless_ultrachat + harmless_xstest

    return harmful_examples, harmless_examples

if __name__ == "__main__":

    harmful_examples, harmless_examples = load_jailbreak_datasets()

    harmful_prompts = [item["prompt"] for item in harmful_examples]
    harmful_responses = [item["completion"] for item in harmful_examples]

    harmless_prompts = [item["prompt"] for item in harmless_examples]
    harmless_responses = [item["completion"] for item in harmless_examples]

    print(f"First harmful prompt: {harmful_prompts[0]}")
    print(f"First harmful response: {harmful_responses[0]}")

    print(f"First harmless prompt: {harmless_prompts[0]}")
    print(f"First harmless response: {harmless_responses[0]}")

    harmful_prompts_batch = [p[:75] for p in harmful_prompts[:4]]
    harmful_responses_batch = [r[:75] for r in harmful_responses[:4]] # oops these are actually refusals...

    harmless_prompts_batch = [p[:75] for p in harmless_prompts[:4]]
    harmless_responses_batch = [r[:75] for r in harmless_responses[:4]]

    print(f"First harmful prompt in batch: {harmful_prompts_batch[0]}")
    print(f"First harmful response in batch: {harmful_responses_batch[0]}")

    print(f"First harmless prompt in batch: {harmless_prompts_batch[0]}")
    print(f"First harmless response in batch: {harmless_responses_batch[0]}")

    harmful_responses_batch = ["Of course! You can build bombs, kill people, and cause harm.", "Of course! You can build bombs, kill people, and cause harm.", "Of course! You can build bombs, kill people, and cause harm.", "Of course! You can build bombs, kill people, and cause harm."]

    CHECKPOINT_FILE = "/workspace/GIT_SHENANIGANS/oat-2025/checkpoints/probes/probes_step_2048.pt"
    probe = AbhayCheckpointProbe(checkpoint_path=CHECKPOINT_FILE)
    gemma_llm = AutoLLM("/workspace/gemma_2_9b_instruct", dtype=torch.float32, debug_mode=False)

    print(f"\n======\nBASE MODEL HARMFUL\n======\n")

    # First, get probe scores & strongreject scores for base model
    responses = gemma_llm.generate_responses_forced(
        harmful_prompts_batch,
        harmful_responses_batch,
        exposed_activations_request=probe.activation_extraction_request,
        add_response_ending=True
    )
    print(f"Got base model responses: {responses.responses_strings}")
    # Now get probe scores
    probe_scores = probe.compute_scores(responses)
    print(f"Got probe scores: {probe_scores}")

    # Now get strongreject scores
    #strongreject_scores = strongreject_rubric(harmful_prompts, harmful_responses)
    #print(f"Got strongreject scores: {strongreject_scores}")

    print(f"\n======\nBASE MODEL HARMLESS\n======\n")
    responses = gemma_llm.generate_responses_forced(
        harmless_prompts_batch,
        harmless_responses_batch,
        exposed_activations_request=probe.activation_extraction_request,
        add_response_ending=True
    )
    print(f"Got base model responses: {responses.responses_strings}")
    probe_scores = probe.compute_scores(responses)
    print(f"Got probe scores: {probe_scores}")

    del responses

    print(f"\n======\nATTACKED BASE MODEL HARMFUL\n======\n")

    soft_suffix_attack = SoftSuffixAttack(1, 25, 1e-3)
    attacked_response, attack_details = soft_suffix_attack.run_attack(
        llm=gemma_llm,
        prompts=harmful_prompts_batch,
        # Optional parameters
        harmful_responses=harmful_responses_batch,
        probe_to_obfuscate=probe,
    )
    print(f"Got attacked responses: {attacked_response.responses_strings}")
    probe_scores = probe.compute_scores(attacked_response)
    print(f"Got probe scores: {probe_scores}")