import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os
import re # For filename sanitization
import textwrap # For title wrapping
import seaborn as sns # For styling
from scipy.stats import bootstrap # For BCa confidence intervals

# --- Configuration ---
# Replace with your W&B entity (username or team) and project name
DEFAULT_WANDB_ENTITY = "mgm52"  # e.g., "my_username" or "my_team"
DEFAULT_WANDB_PROJECT = "oat_evaluation"
BASE_SAVE_DIR = "oat_evaluation/eval_plots"

# Metrics to fetch and plot
METRIC_MAP = {
    "Jailbreak Rate (Attacked, No Probe)": "eval/jailbreak_rate_attacked",
    "Avg SR Score (Attacked, No Probe)": "eval/avg_sr_score_attacked",
    "Jailbreak Rate (Attacked, With Probe)": "eval/jailbreak_rate_probe_attacked",
    "Avg SR Score (Attacked, With Probe)": "eval/avg_sr_score_probe_attacked",
}

# --- Figure Size Configuration (inches) ---
# (Based on ICML style: \textwidth=6.75in, \columnwidth=3.25in)
ONE_COL_WIDTH_INCHES = 3.25 * 1.5
TWO_COL_WIDTH_INCHES = 6.75 * 1.5
DEFAULT_SUBPLOT_ASPECT_RATIO = 0.6 # height / width_per_subplot
N_BOOTSTRAP_RESAMPLES = 2000 # Fewer for faster dev (e.g., 999), more for publication (e.g., 9999)

# --- Helper Functions ---

def get_attack_config_key(config):
    """Generates a unique, hashable key for an attack configuration."""
    attack_name = config.get("attack_name", "UnknownAttack")
    attack_params = config.get("attack_params", {})
    # Ensure lists/tuples within params are converted to tuples for hashability
    frozen_params = {}
    for k, v in attack_params.items():
        if isinstance(v, list):
            frozen_params[k] = tuple(v)
        else:
            frozen_params[k] = v
    params_tuple = tuple(sorted(frozen_params.items(), key=lambda item: str(item[0])))
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
        elif isinstance(v, (list, tuple)): # Should be tuple now from get_attack_config_key
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
    name_part = re.sub(r'[^\w\s-]', '', name_part).strip() # Allow alphanumeric, whitespace, hyphen
    name_part = re.sub(r'[-\s]+', '-', name_part) # Replace whitespace/multiple hyphens with single hyphen
    name_part = re.sub(r'[^a-zA-Z0-9_.-]', '', name_part) # Final stricter pass
    return name_part[:max_length].strip('_.')


# --- Data Fetching and Processing ---

def fetch_and_process_wandb_data(entity, project, group_names):
    """
    Fetches run data from W&B for specified groups and processes it for plotting,
    using BCa for confidence intervals.
    """
    api = wandb.Api()
    # raw_data: [attack_config_key][model_probe_key][step][wandb_metric_key] -> list_of_values_from_seeds
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    seed_counts = defaultdict(lambda: defaultdict(int))  # Track seed counts per model-probe
    
    runs_to_process = []
    for group_name in group_names:
        try:
            print(f"Fetching runs for group: {group_name}...")
            # Construct filter: {"group": group_name, "state": "finished"} # Optionally filter by state
            runs = api.runs(f"{entity}/{project}", filters={"group": group_name})
            runs_to_process.extend(list(runs))
        except wandb.errors.CommError as e:
            print(f"Error fetching runs for group '{group_name}': {e}")
            continue
    
    if not runs_to_process:
        print("No runs found for the specified group(s).")
        return {}, {}
        
    print(f"Found {len(runs_to_process)} runs across {len(group_names)} group(s) to process.")
    history_keys_to_fetch = ["_step"] + list(METRIC_MAP.values())

    # Temporary dict to count unique seeds per (attack_config, model_probe)
    # to avoid double counting if a run appears in multiple fetched groups (though unlikely with group filter)
    # or if a seed is accidentally run multiple times with identical config.
    # For simplicity, we'll assume seed implies unique run data for a given config.
    # We will just count distinct runs contributing to a point.

    processed_run_ids = set() # To avoid processing the same run multiple times if it matches multiple group queries
    
    for i, run in enumerate(runs_to_process):
        if run.id in processed_run_ids:
            continue
        processed_run_ids.add(run.id)

        print(f"Processing run {i+1}/{len(runs_to_process)}: {run.name} (ID: {run.id})")
        config = run.config
        attack_config_key = get_attack_config_key(config)
        model_probe_key = get_model_probe_key(config)
        
        # Increment seed count for this specific attack_config and model_probe combination
        # This logic is slightly simplified; if you need *true* unique seed counts per step point,
        # it would need to be more granular. This counts runs contributing to the model_probe_key.
        seed_counts[attack_config_key][model_probe_key] +=1

        try:
            # Fetch all history for keys at once
            history_df = run.history(keys=history_keys_to_fetch, pandas=True, samples=run.summary.get("_step", 500) + 50) # Fetch more samples if many steps
            if history_df.empty:
                print(f"  Warning: Run {run.id} has empty history for requested keys. Skipping.")
                continue
        except Exception as e:
            print(f"  Warning: Could not fetch/process history for run {run.id}. Error: {e}. Skipping.")
            continue

        for _, row in history_df.iterrows():
            step = row.get("_step")
            if pd.isna(step): # Skip rows where _step is NaN
                continue
            step = int(step) # Ensure step is an integer
            
            for wandb_metric_key in METRIC_MAP.values():
                metric_value = row.get(wandb_metric_key)
                if pd.notna(metric_value):
                    raw_data[attack_config_key][model_probe_key][step][wandb_metric_key].append(metric_value)
    
    # processed_data: [attack_config_key][model_probe_key][wandb_metric_key] -> (steps, means, ci_lowers, ci_uppers)
    processed_data = defaultdict(lambda: defaultdict(dict))
    
    for attack_config_key, model_level_data in raw_data.items():
        for model_probe_key, step_level_data in model_level_data.items():
            for wandb_metric_key in METRIC_MAP.values():
                aggregated_steps = []
                aggregated_means = []
                aggregated_ci_lowers = []
                aggregated_ci_uppers = []
                
                # Sort steps numerically
                sorted_recorded_steps = sorted(step_level_data.keys())

                for step in sorted_recorded_steps:
                    seed_values = step_level_data[step].get(wandb_metric_key, [])
                    
                    if seed_values:
                        current_mean = np.mean(seed_values)
                        aggregated_steps.append(step)
                        aggregated_means.append(current_mean)
                        
                        if len(seed_values) >= 2: # BCa needs at least 2 data points
                            # Scipy bootstrap expects data as a tuple of samples, here (sample_array,)
                            data_array = np.array(seed_values)
                            try:
                                res = bootstrap((data_array,), np.mean, confidence_level=0.95,
                                                method='BCa', n_resamples=N_BOOTSTRAP_RESAMPLES,
                                                random_state=0) # For reproducibility of CIs
                                lower_ci, upper_ci = res.confidence_interval.low, res.confidence_interval.high
                            except Exception as e:
                                print(f"  Warning: BCa bootstrap failed for {attack_config_key}, {model_probe_key}, metric {wandb_metric_key}, step {step}. Error: {e}. Falling back to percentile.")
                                lower_ci, upper_ci = np.percentile(seed_values, [2.5, 97.5])
                        elif len(seed_values) == 1:
                            lower_ci, upper_ci = current_mean, current_mean # No CI for single point
                        else: # Should not happen if seed_values is not empty
                            lower_ci, upper_ci = np.nan, np.nan

                        aggregated_ci_lowers.append(lower_ci)
                        aggregated_ci_uppers.append(upper_ci)
                
                if aggregated_steps: # If any data was actually processed for this metric
                    processed_data[attack_config_key][model_probe_key][wandb_metric_key] = \
                        (aggregated_steps, aggregated_means, aggregated_ci_lowers, aggregated_ci_uppers)
                        
    return processed_data, seed_counts


# --- Plotting Function ---

# --- Plotting Function ---

def plot_results(processed_data, seed_counts, group_names_list):
    """
    Generates and saves plots to SAVE_DIR based on the processed W&B data.
    Uses Seaborn styling, BCa CIs (calculated in fetch_and_process), and saves as PDF.
    """
    if not processed_data:
        print("No data available for plotting.")
        return

    # Apply Seaborn styling and Matplotlib best practices
    sns.set_context("paper") # Options: paper, notebook, talk, poster
    sns.set_style("darkgrid") # Options: darkgrid, whitegrid, dark, white, ticks
    
    # Use constrained_layout. This helps in automatically adjusting subplot params
    # to prevent labels, titles, etc., from overlapping.
    # It works well with fig.suptitle().
    plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams['text.usetex'] = True # Uncomment if LaTeX is installed and math rendering is needed

    # Create the save directory if it doesn't exist
    sane_group_dir_name = sanitize_filename("__".join(sorted(group_names_list)), 100)
    save_dir = os.path.join(BASE_SAVE_DIR, sane_group_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(save_dir)}")

    num_metrics_to_plot = len(METRIC_MAP)
    fig_idx = 0

    for attack_config_key, model_level_data in processed_data.items():
        attack_name_str = attack_config_key[0]
        attack_params_tuple = attack_config_key[1]
        
        formatted_params_str_title = format_attack_params_for_title(attack_params_tuple)
        
        model_info_parts = []
        num_models_in_plot = len(model_level_data.keys())
        for model_probe_key in model_level_data.keys():
            model_name_str, probe_name_str = model_probe_key
            run_count = seed_counts.get(attack_config_key, {}).get(model_probe_key, 0)
            model_info_parts.append(f"{model_name_str} ({run_count} runs)")
        
        model_info_str = " | ".join(model_info_parts)
        
        # Prepare title components with text wrapping
        # The width for textwrap.fill should ideally be responsive to fig_width,
        # but a fixed moderate width is often acceptable.
        title_wrap_width = 70 # Adjusted for typical figure widths
        wrapped_attack = textwrap.fill(f"Attack: {attack_name_str}", width=title_wrap_width)
        wrapped_params = textwrap.fill(f"Params: {formatted_params_str_title}", width=title_wrap_width)
        wrapped_models = textwrap.fill(f"Models: {model_info_str}", width=title_wrap_width)
        wrapped_groups = textwrap.fill(f"W&B Groups: {', '.join(group_names_list)}", width=title_wrap_width)
        
        fig_title_str = f"{wrapped_attack}\n{wrapped_params}\n{wrapped_models}\n{wrapped_groups}"
        
        ncols = min(2, num_metrics_to_plot) 
        if num_metrics_to_plot == 1:
            ncols = 1
            fig_width = ONE_COL_WIDTH_INCHES
        else:
            fig_width = TWO_COL_WIDTH_INCHES

        nrows = (num_metrics_to_plot + ncols - 1) // ncols 
        subplot_width = fig_width / ncols
        subplot_height = subplot_width * DEFAULT_SUBPLOT_ASPECT_RATIO
        
        # Adjust figure height: add space for title.
        # The 1.5 inch includes general top/bottom margins and space for the suptitle.
        # constrained_layout will use this space.
        # Approximate height needed for title: 4 lines * ~0.2-0.25 inch/line (for 11pt font) ~ 1 inch.
        # Add some padding.
        title_height_allowance = 1.5 # inches, if title exists
        bottom_margin_allowance = 0.5 # inches
        fig_height = nrows * subplot_height + bottom_margin_allowance
        if fig_title_str:
            fig_height += title_height_allowance
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), 
                                 sharex=True, squeeze=False)
        axes = axes.flatten() 

        plot_idx = 0
        for display_name, wandb_metric_key in METRIC_MAP.items():
            if plot_idx >= len(axes): 
                break 
            ax = axes[plot_idx]
            has_data_for_this_subplot = False
            max_step_val = 0

            for model_probe_key, metric_specific_results in model_level_data.items():
                model_name_str, probe_name_str = model_probe_key
                line_label = f"{model_name_str}"
                if probe_name_str and probe_name_str.lower() != "unknownprobe" and probe_name_str.lower() != "none":
                     line_label += f", Probe: {probe_name_str}"
                
                if wandb_metric_key in metric_specific_results:
                    steps, means, ci_lowers, ci_uppers = metric_specific_results[wandb_metric_key]
                    if steps: 
                        ax.plot(steps, means, label=line_label, marker='o', linestyle='-', markersize=4)
                        ax.fill_between(steps, ci_lowers, ci_uppers, alpha=0.2)
                        has_data_for_this_subplot = True
                        if steps and max(steps) > max_step_val:
                            max_step_val = max(steps)
            
            ax.set_xlabel("Attack Steps")
            ax.set_ylabel(display_name.split('(')[0].strip()) 
            ax.set_title(display_name, fontsize=10)

            if max_step_val > 0:
                ax.set_xlim(left=-0.05 * max_step_val, right=max_step_val * 1.05) 
            
            if has_data_for_this_subplot:
                 ax.legend(fontsize='small', loc='best')
            else:
                ax.text(0.5, 0.5, "No data for this metric", 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax.transAxes, color='grey')
            plot_idx += 1

        for i in range(plot_idx, nrows * ncols):
            fig.delaxes(axes[i])

        # Use fig.suptitle() for the main title.
        # constrained_layout is designed to make space for suptitle.
        # The previous use of fig.text() for the title is not automatically handled
        # by constrained_layout for spacing, which likely caused the overlap.
        if fig_title_str:
            fig.suptitle(fig_title_str, fontsize=11) # y parameter can be adjusted if needed, default is 0.98

        sane_attack_name = sanitize_filename(attack_name_str, 30)
        sane_params_str = sanitize_filename(format_attack_params_for_title(attack_params_tuple), 70)
        sane_groups_str = sanitize_filename("_".join(sorted(group_names_list)), 40)

        filename_parts = [
            "plot", sane_attack_name,
            f"params_{sane_params_str}" if sane_params_str else "no_params",
            f"groups_{sane_groups_str}" if sane_groups_str else "no_groups",
            f"{fig_idx}"
        ]
        base_filename = "_".join(filter(None, filename_parts))
        base_filename = base_filename[:200] 
        
        save_path_pdf = os.path.join(save_dir, base_filename + ".pdf")
        save_path_png = os.path.join(save_dir, base_filename + ".png")

        try:
            # bbox_inches='tight' is used to ensure everything fits when saving.
            # This works well with constrained_layout.
            plt.savefig(save_path_pdf, bbox_inches='tight')
            print(f"  Saved PDF plot: {save_path_pdf}")
            plt.savefig(save_path_png, bbox_inches='tight', dpi=150)
            print(f"  Saved PNG plot: {save_path_png}")
        except Exception as e:
            print(f"  Error saving plot {base_filename}: {e}")
        
        plt.close(fig)
        fig_idx += 1

# --- Main Execution ---

if __name__ == "__main__":
    wandb_entity = input(f"Enter W&B entity (or press Enter for default '{DEFAULT_WANDB_ENTITY}'): ").strip()
    if not wandb_entity:
        wandb_entity = DEFAULT_WANDB_ENTITY

    wandb_project = input(f"Enter W&B project (or press Enter for default '{DEFAULT_WANDB_PROJECT}'): ").strip()
    if not wandb_project:
        wandb_project = DEFAULT_WANDB_PROJECT
    
    # Example group names: "20240101_my_experiment_gcm,20240102_another_run_gcm"
    # For testing with dummy data, ensure your W&B project has runs in these groups
    DEFAULT_GROUP_NAMES = "20250510_203803_1392405_ci_test" #"20250510_161748,20250510_161752,20250510_161754"
    group_names_str = input(f"Enter W&B group names (comma-separated) (default: '{DEFAULT_GROUP_NAMES}'): ").strip()
    if not group_names_str:
        group_names_str = DEFAULT_GROUP_NAMES
    group_names_list = [name.strip() for name in group_names_str.split(',') if name.strip()]

    if not wandb_project:
        print("W&B project is required.")
    elif not group_names_list:
        print("No group names entered. Exiting.")
    else:
        print(f"\nAttempting to fetch data for entity='{wandb_entity}', project='{wandb_project}', groups={group_names_list}")
        
        # wandb.login() # Make sure you are logged in. Can be done via CLI `wandb login` as well.

        processed_wandb_data, seed_counts = fetch_and_process_wandb_data(wandb_entity, wandb_project, group_names_list)
        
        if not processed_wandb_data:
            print("\nNo data was processed. Ensure group names and project/entity are correct and runs have completed logging.")
        else:
            print(f"\nData processing complete. Found data for {len(processed_wandb_data)} attack configurations.")
            print("Generating and saving plots...")
            # Removed entity and project from plot_results as they are not used there
            plot_results(processed_wandb_data, seed_counts, group_names_list)
            print("\nPlotting finished.")