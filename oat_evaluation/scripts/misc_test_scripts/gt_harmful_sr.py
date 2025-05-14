from typing import List, Dict, Tuple

import numpy as np
from datasets import load_dataset
from strong_reject.evaluate import strongreject_rubric

# Configuration
DATASET_NAME = "justinphan3110/circuit_breakers_train"
DATASET_SPLIT = "train"
TEST_SET_SIZE = 400  # Number of examples to use for testing


def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000, ci=0.95):
    """
    Calculate bootstrap confidence intervals for a statistic.

    Args:
        data: Input data (list or array)
        statistic_func: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level (default: 0.95 for 95% CI)

    Returns:
        tuple: (lower bound, upper bound) of the confidence interval
    """
    if len(data) == 0:
        return (np.nan, np.nan)
    bootstrap_stats = []
    data_array = np.array(data) # Convert to numpy array for efficient sampling
    n_data = len(data_array)
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample_indices = np.random.choice(n_data, size=n_data, replace=True)
        sample = data_array[sample_indices]
        bootstrap_stats.append(statistic_func(sample))

    # Calculate confidence interval
    lower_bound = np.percentile(bootstrap_stats, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrap_stats, (1 + ci) / 2 * 100)

    return lower_bound, upper_bound


def calculate_refusal_rate(sr_scores: List[float]) -> float:
    """
    Calculate the refusal rate (proportion of sr_scores equal to 0).
    """
    if not sr_scores:
        return 0.0
    return (np.array(sr_scores) == 0).mean()


def load_and_prepare_data(
    dataset_name: str, dataset_split: str, num_examples: int
) -> List[Dict]:
    """
    Load dataset, format, and select a subset.
    """
    print(f"Loading dataset: {dataset_name} [{dataset_split}]")
    try:
        dataset = load_dataset(dataset_name, split=dataset_split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    # Ensure we don't request more examples than available
    num_available = len(dataset)
    if num_examples > num_available:
        print(f"Warning: Requested {num_examples} examples, but only {num_available} available. Using {num_available}.")
        num_examples = num_available
    
    # Select the first 'num_examples'
    selected_data = dataset.select(range(num_examples))

    # Format into {"prompt": ..., "completion": ...}
    formatted_data = []
    for item in selected_data:
        # Adjust keys based on the dataset structure if needed
        prompt = item.get("prompt")
        completion = item.get("completion", item.get("response")) # Handle both common keys

        if prompt is None or completion is None:
            print(f"Warning: Skipping item due to missing 'prompt' or 'completion'/'response': {item}")
            continue
        
        # Truncate prompt and completion to 150 characters
        truncated_prompt = str(prompt)[:150]
        truncated_completion = str(completion)[:150]
        
        formatted_data.append({"prompt": truncated_prompt, "completion": truncated_completion})

    print(f"Loaded and formatted {len(formatted_data)} examples (truncated to 150 chars).")
    return formatted_data


def evaluate_strongreject_on_data(
    data: List[Dict],
) -> Tuple[List[Dict], List[float]]:
    """
    Compute strongreject results on the provided data.
    """
    print(f"Evaluating StrongReject on {len(data)} examples...")
    results = []
    sr_scores = []
    for i, item in enumerate(data):
        prompt = item["prompt"]
        completion = item["completion"]
        try:
            res = strongreject_rubric(prompt, completion)
            results.append(res)
            sr_scores.append(res["score"])
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            # Assign a default score or skip? Let's skip and note it.
            # Optionally add placeholder result: results.append({"error": str(e), "score": np.nan})
            # sr_scores.append(np.nan) # Or skip adding to sr_scores

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} examples...")

    print(f"Finished StrongReject evaluation.")
    # Filter out potential NaNs if error handling added them
    sr_scores_valid = [s for s in sr_scores if not np.isnan(s)]
    return results, sr_scores_valid


def main():
    np.random.seed(42) # For reproducibility of bootstrap

    # 1. Load and prepare the harmful test data
    harmful_test_data = load_and_prepare_data(
        DATASET_NAME, DATASET_SPLIT, TEST_SET_SIZE
    )

    if not harmful_test_data:
        print("No data loaded. Exiting.")
        return

    # 2. Evaluate StrongReject on the loaded data
    sr_results, sr_scores = evaluate_strongreject_on_data(harmful_test_data)

    if not sr_scores:
        print("No valid StrongReject scores obtained. Exiting.")
        return

    # 3. Calculate statistics
    avg_sr_score = np.mean(sr_scores)
    refusal_rate = calculate_refusal_rate(sr_scores)

    # 4. Calculate confidence intervals
    sr_score_ci = bootstrap_confidence_interval(sr_scores, np.mean)
    refusal_rate_ci = bootstrap_confidence_interval(np.array(sr_scores) == 0, np.mean)

    # 5. Print summary
    print("\n=== SUMMARY ===")
    print(f"Dataset: {DATASET_NAME} [{DATASET_SPLIT}]")
    print(f"Evaluated examples: {len(sr_scores)}")
    print(f"Average StrongReject score: {avg_sr_score:.4f} [95% CI: {sr_score_ci[0]:.4f}, {sr_score_ci[1]:.4f}]")
    print(f"Refusal rate (SR Score == 0): {refusal_rate:.4f} [95% CI: {refusal_rate_ci[0]:.4f}, {refusal_rate_ci[1]:.4f}]")
    print("\nNote: Scores are based on the *original completions* provided in the dataset, not on model generations.")


if __name__ == "__main__":
    main()