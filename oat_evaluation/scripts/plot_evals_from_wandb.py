import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os
import re # For filename sanitization

# --- Configuration ---
# Replace with your W&B entity (username or team) and project name
DEFAULT_WANDB_ENTITY = "mgm52"  # e.g., "my_username" or "my_team"
DEFAULT_WANDB_PROJECT = "oat_evaluation"
SAVE_DIR = "oat_evaluation/eval_plots"

# Metrics to fetch and plot
METRIC_MAP = {
    "Jailbreak Rate (Attacked, No Probe)": "eval/jailbreak_rate_attacked",
    "Avg SR Score (Attacked, No Probe)": "eval/avg_sr_score_attacked",
    "Jailbreak Rate (Attacked, With Probe)": "eval/jailbreak_rate_probe_attacked",
    "Avg SR Score (Attacked, With Probe)": "eval/avg_sr_score_probe_attacked",
}

# --- Helper Functions ---

def get_attack_config_key(config):
    """Generates a unique, hashable key for an attack configuration."""
    attack_name = config.get("attack_name", "UnknownAttack")
    attack_params = config.get("attack_params", {})
    params_tuple = tuple(sorted(attack_params.items(), key=lambda item: str(item[0])))
    return (attack_name, params_tuple)

def get_model_probe_key(config):
    """Generates a unique, hashable key for a model and probe combination."""
    model_name = config.get("model_name", "UnknownModel")
    probe_name = config.get("probe_name", "UnknownProbe")
    return (model_name, probe_name)

def format_attack_params_for_title(params_tuple):
    """Creates a readable string representation of attack parameters for plot titles."""
    parts = []
    for k, v in params_tuple:
        if isinstance(v, float):
            v_str = f"{v:.2e}"
        elif isinstance(v, (list, tuple)):
            if len(v) > 3:
                v_str = f"[{str(v[0])}, ..., {str(v[-1])}]({len(v)})"
            else:
                v_str = str(v)
        elif v is None:
            v_str = "None"
        else:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)

def sanitize_filename(name_part, max_length=50):
    """Sanitizes a string to be filename-safe and truncates if necessary."""
    if not isinstance(name_part, str):
        name_part = str(name_part)
    # Remove invalid characters (allow alphanumeric, underscore, hyphen, dot)
    name_part = re.sub(r'[^a-zA-Z0-9_.-]', '', name_part)
    # Replace multiple dots or hyphens with a single one
    name_part = re.sub(r'\.+', '.', name_part)
    name_part = re.sub(r'-+', '-', name_part)
    # Replace whitespace (should be caught by first sub, but just in case)
    name_part = re.sub(r'\s+', '_', name_part).strip('_.')
    # Truncate
    return name_part[:max_length]


# --- Data Fetching and Processing --- (Identical to previous version)

def fetch_and_process_wandb_data(entity, project, group_names):
    """
    Fetches run data from W&B for specified groups and processes it for plotting.
    """
    api = wandb.Api()
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    runs_to_process = []
    for group_name in group_names:
        try:
            print(f"Fetching runs for group: {group_name}...")
            runs = api.runs(f"{entity}/{project}", filters={"group": group_name})
            runs_to_process.extend(list(runs))
        except wandb.errors.CommError as e:
            print(f"Error fetching runs for group '{group_name}': {e}")
            continue
    
    if not runs_to_process:
        print("No runs found for the specified group(s).")
        return {}
        
    print(f"Found {len(runs_to_process)} runs across {len(group_names)} group(s) to process.")
    history_keys_to_fetch = ["_step"] + list(METRIC_MAP.values())

    for i, run in enumerate(runs_to_process):
        print(f"Processing run {i+1}/{len(runs_to_process)}: {run.name} (ID: {run.id})")
        config = run.config
        attack_config_key = get_attack_config_key(config)
        model_probe_key = get_model_probe_key(config)

        try:
            history_df = run.history(keys=history_keys_to_fetch, pandas=True)
            if history_df.empty:
                print(f"  Warning: Run {run.id} has empty history for requested keys. Skipping.")
                continue
        except Exception as e:
            print(f"  Warning: Could not fetch/process history for run {run.id}. Error: {e}. Skipping.")
            continue

        for _, row in history_df.iterrows():
            step = row.get("_step")
            if pd.isna(step):
                continue
            
            for wandb_metric_key in METRIC_MAP.values():
                metric_value = row.get(wandb_metric_key)
                if pd.notna(metric_value):
                    raw_data[attack_config_key][model_probe_key][int(step)][wandb_metric_key].append(metric_value)
    
    processed_data = defaultdict(lambda: defaultdict(dict))
    for attack_config_key, model_level_data in raw_data.items():
        for model_probe_key, step_level_data in model_level_data.items():
            for wandb_metric_key in METRIC_MAP.values():
                aggregated_steps = []
                aggregated_means = []
                aggregated_ci_lowers = []
                aggregated_ci_uppers = []
                sorted_recorded_steps = sorted(step_level_data.keys())

                for step in sorted_recorded_steps:
                    seed_values = step_level_data[step].get(wandb_metric_key, [])
                    if seed_values:
                        aggregated_steps.append(step)
                        mean_val = np.mean(seed_values)
                        aggregated_means.append(mean_val)
                        if len(seed_values) > 1:
                            lower, upper = np.percentile(seed_values, [2.5, 97.5])
                        else:
                            lower, upper = mean_val, mean_val
                        aggregated_ci_lowers.append(lower)
                        aggregated_ci_uppers.append(upper)
                
                if aggregated_steps:
                    processed_data[attack_config_key][model_probe_key][wandb_metric_key] = \
                        (aggregated_steps, aggregated_means, aggregated_ci_lowers, aggregated_ci_uppers)
    return processed_data


# --- Plotting Function ---

def plot_results(processed_data, entity, project, group_names_list):
    """
    Generates and saves plots to SAVE_DIR based on the processed W&B data.
    """
    if not processed_data:
        print("No data available for plotting.")
        return

    # Create the save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(SAVE_DIR)}")

    num_metrics_to_plot = len(METRIC_MAP)

    for attack_config_key, model_level_data in processed_data.items():
        attack_name_str = attack_config_key[0]
        attack_params_tuple = attack_config_key[1]
        
        formatted_params_str_title = format_attack_params_for_title(attack_params_tuple)
        
        fig_title = (f"Attack: {attack_name_str}\n"
                     f"Params: {formatted_params_str_title}\n"
                     f"W&B Groups: {', '.join(group_names_list)}")
        
        # Determine subplot layout
        ncols = 2
        nrows = (num_metrics_to_plot + ncols - 1) // ncols 
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows), sharex=True, squeeze=False)
        axes = axes.flatten()

        plot_idx = 0
        for display_name, wandb_metric_key in METRIC_MAP.items():
            ax = axes[plot_idx]
            has_data_for_this_subplot = False
            
            for model_probe_key, metric_specific_results in model_level_data.items():
                model_name_str, probe_name_str = model_probe_key
                line_label = f"Model: {model_name_str}, Probe: {probe_name_str}"
                
                if wandb_metric_key in metric_specific_results:
                    steps, means, ci_lowers, ci_uppers = metric_specific_results[wandb_metric_key]
                    if steps:
                        ax.plot(steps, means, label=line_label, marker='o', linestyle='-')
                        ax.fill_between(steps, ci_lowers, ci_uppers, alpha=0.2)
                        has_data_for_this_subplot = True
            
            ax.set_xlabel("Attack Steps")
            ax.set_ylabel(display_name.split('(')[0].strip())
            ax.set_title(display_name)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            if has_data_for_this_subplot:
                 ax.legend(fontsize='small', loc='best')
            else:
                ax.text(0.5, 0.5, "No data for this metric", 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax.transAxes, color='grey')
            plot_idx += 1

        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(fig_title, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # --- Create filename and save ---
        sane_attack_name = sanitize_filename(attack_name_str, 30)
        sane_params_str = sanitize_filename(format_attack_params_for_title(attack_params_tuple), 70)
        sane_groups_str = sanitize_filename("_".join(sorted(group_names_list)), 40)

        filename_parts = [
            "plot",
            sane_attack_name,
            f"params_{sane_params_str}" if sane_params_str else "no_params",
            f"groups_{sane_groups_str}" if sane_groups_str else "no_groups"
        ]
        filename = "_".join(filter(None, filename_parts)) + ".png"
        # Further ensure filename is not excessively long
        filename = filename[:200] + ".png" if len(filename) > 200 else filename


        save_path = os.path.join(SAVE_DIR, filename)
        try:
            plt.savefig(save_path)
            print(f"  Saved plot: {save_path}")
        except Exception as e:
            print(f"  Error saving plot {save_path}: {e}")
        
        plt.close(fig) # Close the figure to free memory

# --- Main Execution --- (Identical to previous version)

if __name__ == "__main__":
    wandb_entity = input(f"Enter W&B entity (or press Enter for default '{DEFAULT_WANDB_ENTITY}'): ").strip()
    if not wandb_entity:
        wandb_entity = DEFAULT_WANDB_ENTITY

    wandb_project = input(f"Enter W&B project (or press Enter for default '{DEFAULT_WANDB_PROJECT}'): ").strip()
    if not wandb_project:
        wandb_project = DEFAULT_WANDB_PROJECT

    group_names_str = input("Enter W&B group names (comma-separated): ")
    group_names_list = [name.strip() for name in group_names_str.split(',') if name.strip()]

    if not wandb_project:
        print("W&B project is required.")
    elif not group_names_list:
        print("No group names entered. Exiting.")
    else:
        print(f"\nAttempting to fetch data for entity='{wandb_entity}', project='{wandb_project}', groups={group_names_list}")
        
        # wandb.login() # Uncomment if needed

        processed_wandb_data = fetch_and_process_wandb_data(wandb_entity, wandb_project, group_names_list)
        
        if not processed_wandb_data:
            print("\nNo data was processed. Ensure group names and project/entity are correct and runs have completed logging.")
        else:
            print("\nData processing complete. Generating and saving plots...")
            plot_results(processed_wandb_data, wandb_entity, wandb_project, group_names_list)
            print("\nPlotting finished.")