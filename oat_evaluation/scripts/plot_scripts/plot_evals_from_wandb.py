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
import argparse # For command line arguments

# --- Configuration ---
# Replace with your W&B entity (username or team) and project name
DEFAULT_WANDB_ENTITY = "mgm52"  # e.g., "my_username" or "my_team"
DEFAULT_WANDB_PROJECT = "oat_evaluation"
DEFAULT_GROUP_NAMES = "20250511_043048_3142425_big_eval_repaired" #"20250510_161748,20250510_161752,20250510_161754"
BASE_SAVE_DIR = "oat_evaluation/outputs/eval_plots"

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
        elif isinstance(v, dict):
            # HACK for now, due to oversharing in api_llm strings in some evals
            frozen_params[k] = v["model_name"]
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
        print(f" Attempting to parse param: key='{k}', value='{v}', type={type(v)}")
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
    history_keys_to_fetch = ["eval/current_attack_step"] + list(METRIC_MAP.values())

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

        # TODO: REPLACE HACK - used to fix PAIRAttack runs...
        if "20250512_015933_2360201_pair_eval" in group_names:
            print(f"  HACK: Extracting run_attack_index from run.name: {run.name}")
            run_attack_index = int(run.name.split("_PAIRAttack")[0].split("_")[-1])
            attack_config_key = (attack_config_key[0], attack_config_key[1], run_attack_index)
            print(f"  HACK: New attack_config_key: {attack_config_key}")
        else:
            print(f"  HACK: 20250512_015933_2360201_pair_eval not in group_names. Using attack_config_key: {attack_config_key}")

        model_probe_key = get_model_probe_key(config)
        
        # Increment seed count for this specific attack_config and model_probe combination
        # This logic is slightly simplified; if you need *true* unique seed counts per step point,
        # it would need to be more granular. This counts runs contributing to the model_probe_key.
        seed_counts[attack_config_key][model_probe_key] +=1

        try:
            # Fetch all history for keys at once
            history_df = run.history(keys=history_keys_to_fetch, pandas=True, samples=run.summary.get("eval/current_attack_step", 500) + 50) # Fetch more samples if many steps
            if history_df.empty:
                print(f"  Warning: Run {run.id} has empty history for requested keys. Skipping.")
                continue
        except Exception as e:
            print(f"  Warning: Could not fetch/process history for run {run.id}. Error: {e}. Skipping.")
            continue

        for _, row in history_df.iterrows():
            step = row.get("eval/current_attack_step")
            if pd.isna(step): # Skip rows where step is NaN
                continue
            step = int(step) # Ensure step is an integer
            
            for wandb_metric_key in METRIC_MAP.values():
                metric_value = row.get(wandb_metric_key)

                print(f"  DEBUG RAW: Run {run.id}, Step {step}, Metric '{wandb_metric_key}': "
                      f"Raw value='{metric_value}', Type={type(metric_value)}")

                if pd.notna(metric_value): # Handles None, pd.NA, and actual np.nan (if it somehow came as float)
                    val_to_append = None
                    original_val_for_debug = metric_value # Store for debug print in except
                    original_type_for_debug = type(metric_value)

                    try:
                        if isinstance(metric_value, str):
                            # Explicitly handle string 'nan' (case-insensitive, stripped)
                            if metric_value.strip().lower() == 'nan':
                                val_to_append = np.nan # Convert to float np.nan
                            else:
                                val_to_append = float(metric_value) # Try to convert other strings
                        elif isinstance(metric_value, (int, float)): # Already numeric
                            val_to_append = float(metric_value) # Ensure it's a standard float
                        else:
                            # Attempt conversion for other types that might be convertible (e.g., np.float32)
                            val_to_append = float(metric_value)
                        
                        # CRUCIAL DEBUG PRINT: See what happened after conversion logic
                        print(f"  DEBUG CONVERSION: Original='{original_val_for_debug}' (type: {original_type_for_debug}), "
                              f"Attempted to append='{val_to_append}' (type: {type(val_to_append)})")

                        if val_to_append is not None: # Should always be true if no exception
                            raw_data[attack_config_key][model_probe_key][step][wandb_metric_key].append(val_to_append)
                        # No else needed here, as val_to_append should be set or an exception raised

                    except (ValueError, TypeError) as e:
                        print(f"  Warning (Conversion Error): Run {run.name} (ID: {run.id}), metric '{wandb_metric_key}', step {step}: "
                              f"Could not convert value '{original_val_for_debug}' (type: {original_type_for_debug}) to float. Skipping. Error: {e}")
                # else:
                #     print(f"  DEBUG SKIP pd.notna: Run {run.id}, Step {step}, Metric '{wandb_metric_key}', value '{metric_value}' was pd.NA or None.")    
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

                    print(f"  DEBUG: For Mean Calc - Attack: {attack_config_key}, Model/Probe: {model_probe_key}, "
                          f"Metric: {wandb_metric_key}, Step: {step}, seed_values: {seed_values}, "
                          f"Types in seed_values: {[type(v) for v in seed_values]}")

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

def plot_results(processed_data, seed_counts, group_names_list, model_to_skip=None):
    """
    Generates and saves plots to SAVE_DIR based on the processed W&B data.
    • If every series in a subplot has only one x-value, the subplot is rendered
      as a bar chart whose x-axis is the model / probe label.
    • Otherwise the subplot is rendered as the usual line chart with shaded CIs.
    
    Args:
        processed_data: The processed W&B data
        seed_counts: Counts of seeds per model-probe combination
        group_names_list: List of W&B group names
        model_to_skip: Optional model name to skip in plotting
    """
    if not processed_data:
        print("No data available for plotting.")
        return

    sns.set_context("paper")
    sns.set_style("darkgrid")
    plt.rcParams['figure.constrained_layout.use'] = True

    sane_group_dir_name = sanitize_filename("__".join(sorted(group_names_list)), 100)
    save_dir = os.path.join(BASE_SAVE_DIR, sane_group_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(save_dir)}")

    num_metrics_to_plot = len(METRIC_MAP)
    fig_idx = 0

    for attack_config_key, model_level_data in processed_data.items():
        if len(attack_config_key) == 2: attack_name_str, attack_params_tuple = attack_config_key
        elif len(attack_config_key) == 3: attack_name_str, attack_params_tuple, run_attack_index = attack_config_key
        formatted_params_str_title = format_attack_params_for_title(attack_params_tuple)

        # ---------- build nice long figure title ----------
        model_info_parts = []
        for model_probe_key in model_level_data.keys():
            model_name_str, probe_name_str = model_probe_key
            # Skip the specified model if provided
            if model_to_skip and model_name_str == model_to_skip:
                continue
            run_count = seed_counts.get(attack_config_key, {}).get(model_probe_key, 0)
            model_info_parts.append(f"{model_name_str} ({run_count} runs)")
        title_wrap_width = 70
        fig_title_str = "\n".join(
            textwrap.fill(t, width=title_wrap_width)
            for t in (
                f"Attack: {attack_name_str}",
                f"Params: {formatted_params_str_title}",
                f"Models: {' | '.join(model_info_parts)}",
                f"W&B Groups: {', '.join(group_names_list)}",
            )
        )

        # ---------- figure & axes grid ----------
        ncols = 1 if num_metrics_to_plot == 1 else 2
        fig_width = ONE_COL_WIDTH_INCHES if ncols == 1 else TWO_COL_WIDTH_INCHES
        nrows = (num_metrics_to_plot + ncols - 1) // ncols
        subplot_height = (fig_width / ncols) * DEFAULT_SUBPLOT_ASPECT_RATIO
        fig_height = nrows * subplot_height + 2.0  # room for title
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), sharex=False
        )
        axes = axes.flatten()

        # ---------- one subplot per metric ----------
        for plot_idx, (display_name, wandb_metric_key) in enumerate(METRIC_MAP.items()):
            ax = axes[plot_idx]
            series = []

            # gather data for this metric across all (model, probe)
            for model_probe_key, metric_specific_results in model_level_data.items():
                model_name_str, probe_name_str = model_probe_key
                # Skip the specified model if provided
                if model_to_skip and model_name_str == model_to_skip:
                    continue
                    
                if wandb_metric_key not in metric_specific_results:
                    continue
                steps, means, ci_lowers, ci_uppers = metric_specific_results[wandb_metric_key]
                label = probe_name_str if probe_name_str.lower() not in {"unknownprobe", "none"} else model_name_str
                series.append(
                    {
                        "label": label,
                        "steps": steps,
                        "means": means,
                        "lower": ci_lowers,
                        "upper": ci_uppers,
                    }
                )

            # decide chart type: bar if every series has a single unique x
            unique_steps = {step for s in series for step in s["steps"]}
            draw_bar = len(unique_steps) == 1

            if draw_bar:
                # ----- BAR CHART -----
                xpos = np.arange(len(series))
                heights = [s["means"][0] for s in series]
                err_lower = [s["means"][0] - s["lower"][0] for s in series]
                err_upper = [s["upper"][0] - s["means"][0] for s in series]
                ax.bar(xpos, heights, yerr=[err_lower, err_upper], capsize=3)
                ax.set_xticks(xpos, [s["label"] for s in series], rotation=15, ha="right")
                ax.set_xlabel("Model / Probe")
            else:
                # ----- LINE CHART -----
                max_step_val = 0
                for s in series:
                    ax.plot(s["steps"], s["means"], label=s["label"], marker="o", linestyle="-", markersize=4)
                    ax.fill_between(s["steps"], s["lower"], s["upper"], alpha=0.2)
                    max_step_val = max(max_step_val, max(s["steps"]))
                ax.set_xlabel("Attack Steps")
                if series:
                    ax.set_xlim(left=-0.05 * max_step_val, right=max_step_val * 1.05)
                    ax.legend(fontsize="small", loc="best")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data for this metric",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        color="grey",
                    )

            # common y-axis label & title
            ax.set_ylabel(display_name.split("(")[0].strip())
            ax.set_title(display_name, fontsize=10)

        # remove any unused axes (e.g. odd number of plots)
        for idx in range(len(METRIC_MAP), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(fig_title_str, fontsize=11)

        # ---------- save ----------
        sane_attack_name = sanitize_filename(attack_name_str, 30)
        sane_params_str = sanitize_filename(format_attack_params_for_title(attack_params_tuple), 70)
        sane_groups_str = sanitize_filename("_".join(sorted(group_names_list)), 40)
        filename_base = "_".join(
            filter(
                None,
                [
                    "plot",
                    sane_attack_name,
                    f"params_{sane_params_str}" if sane_params_str else "no_params",
                    f"groups_{sane_groups_str}" if sane_groups_str else "no_groups",
                    f"{fig_idx}",
                ],
            )
        )[:200]

        for ext in ("pdf", "png"):
            out_path = os.path.join(save_dir, f"{filename_base}.{ext}")
            try:
                plt.savefig(out_path, bbox_inches="tight", dpi=150 if ext == "png" else None)
                print(f"  Saved {ext.upper()} plot: {out_path}")
            except Exception as e:
                print(f"  Error saving plot {out_path}: {e}")

        plt.close(fig)
        fig_idx += 1


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation results from W&B runs.")
    parser.add_argument("--entity", default=DEFAULT_WANDB_ENTITY, help="W&B entity (user or team)")
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--groups", default=DEFAULT_GROUP_NAMES, help="Comma-separated list of W&B group names")
    parser.add_argument("--model-to-skip", help="Model name to skip in plotting")
    args = parser.parse_args()

    group_names_list = [name.strip() for name in args.groups.split(',') if name.strip()]

    if not args.project:
        print("W&B project is required.")
    elif not group_names_list:
        print("No group names entered. Exiting.")
    else:
        print(f"\nAttempting to fetch data for entity='{args.entity}', project='{args.project}', groups={group_names_list}")
        if args.model_to_skip:
            print(f"Will skip model: {args.model_to_skip}")
        
        processed_wandb_data, seed_counts = fetch_and_process_wandb_data(args.entity, args.project, group_names_list)
        
        if not processed_wandb_data:
            print("\nNo data was processed. Ensure group names and project/entity are correct and runs have completed logging.")
        else:
            print(f"\nData processing complete. Found data for {len(processed_wandb_data)} attack configurations.")
            print("Generating and saving plots...")
            plot_results(processed_wandb_data, seed_counts, group_names_list, args.model_to_skip)
            print("\nPlotting finished.")