# plot_evals_flops_from_wandb.py

import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os
import re # For filename sanitization
import textwrap # For title wrapping
import seaborn as sns # For styling
import argparse # For command line arguments

# --- Configuration ---
DEFAULT_WANDB_ENTITY = "mgm52"
DEFAULT_WANDB_PROJECT = "oat_evaluation"
DEFAULT_GROUP_NAMES = "20250511_043048_3142425_big_eval_repaired"
BASE_SAVE_DIR_FLOPS = "oat_evaluation/outputs/eval_plots_flops"

COLOR_METRICS = {
    "Avg SR Score (Attacked)": "eval/avg_sr_score_attacked",
    "Jailbreak Rate (Attacked)": "eval/jailbreak_rate_attacked",
    "Avg SR Score Probe (Attacked)": "eval/avg_sr_score_probe_attacked",
    "Jailbreak Rate Probe (Attacked)": "eval/jailbreak_rate_probe_attacked",
}
X_AXIS_METRIC_HISTORY = "eval/flop_cost"
Y_AXIS_CONFIG_KEY = "defender_flop_cost"
ATTACK_NAME_CONFIG_KEY = "attack_name"
ATTACK_STEP_HISTORY_KEY = "eval/current_attack_step"

SCATTER_FIG_WIDTH_INCHES = 10  # Increased width
SCATTER_FIG_HEIGHT_INCHES = 7 # Slightly decreased height to make it less square

# --- Helper Functions ---

def sanitize_filename(name_part, max_length=50):
    if not isinstance(name_part, str):
        name_part = str(name_part)
    name_part = re.sub(r'[^\w\s-]', '', name_part).strip()
    name_part = re.sub(r'[-\s]+', '-', name_part)
    name_part = re.sub(r'[^a-zA-Z0-9_.-]', '', name_part)
    return name_part[:max_length].strip('_.')

def get_defender_id_legend_label(config): # Renamed for clarity
    """Generates a label for the legend, preferring probe name."""
    probe_name = config.get("probe_name", "UnknownProbe")
    model_name = config.get("model_name", "UnknownModel") # Keep for full ID if needed elsewhere

    sane_probe_name = sanitize_filename(probe_name, 30)
    # Use probe name if available and not generic, otherwise fallback
    if probe_name and sane_probe_name.lower() not in {"unknownprobe", "none", ""}:
        return sane_probe_name
    else: # Fallback to model name if probe name is not informative
        return sanitize_filename(model_name, 20)

def get_full_defender_id_for_grouping(config): # New function for unique grouping key
    """Generates a unique ID for grouping data by defender, including model and probe."""
    model_name = config.get("model_name", "UnknownModel")
    probe_name = config.get("probe_name", "UnknownProbe")
    sane_model_name = sanitize_filename(model_name, 20)
    sane_probe_name = sanitize_filename(probe_name, 30)
    return f"{sane_model_name}-{sane_probe_name}"

# --- NEW: Heat-map plotting -----------------------------------------------

def plot_heatmaps_flops(plot_df, group_names_list,
                        attacker_exp_step=0.1,   # one tick every ~0.25 dex
                        defender_exp_step=0.02,
                        aggfunc="mean"):          # "mean", "median", "max", …
    """
    Build a 2-D heat-map where the x-axis is attacker-FLOPs (log-binned),
    the y-axis is defender-FLOPs (log-binned) and colour = a metric.

    Parameters
    ----------
    attacker_exp_step / defender_exp_step
        Size (in log-10 space) of each bin.  Smaller = finer grid.
    aggfunc : str or callable
        How to aggregate when more than one point falls in a cell.
    """
    if plot_df.empty:
        print("No data available for heat-maps.")
        return

    sns.set_context("talk")
    sns.set_style("whitegrid")

    sane_group_dir = sanitize_filename("__".join(sorted(group_names_list)), 100)
    save_dir = os.path.join(BASE_SAVE_DIR_FLOPS, sane_group_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Work out global bin edges (log-space) once so every panel lines up.
    att_min, att_max = plot_df[X_AXIS_METRIC_HISTORY].min(), plot_df[X_AXIS_METRIC_HISTORY].max()
    def_min, def_max = plot_df["defender_flops"].min(), plot_df["defender_flops"].max()

    att_edges = np.arange(np.floor(np.log10(att_min)),
                          np.ceil(np.log10(att_max))+attacker_exp_step,
                          attacker_exp_step)
    def_edges = np.arange(np.floor(np.log10(def_min)),
                          np.ceil(np.log10(def_max))+defender_exp_step,
                          defender_exp_step)

    plot_idx_global = 0
    for attack_name, attack_df in plot_df.groupby("attack_name"):
        # Get all defenders for this attack
        all_defenders = sorted(attack_df['defender_legend_label'].unique())
        
        # Count points per defender (only counting points with valid data)
        defender_counts = {}
        for defender in all_defenders:
            # Count how many points this defender contributes to this attack
            # Only counting points with non-zero defender flops
            defender_df = attack_df[
                (attack_df['defender_legend_label'] == defender) & 
                (attack_df['defender_flops'] > 0)
            ]
            defender_counts[defender] = len(defender_df)
        
        # Create a string with defender names and their point counts
        defenders_with_counts = [f"{defender} ({defender_counts[defender]} pts)" for defender in all_defenders]
        defenders_str = ", ".join(defenders_with_counts)
        
        for metric_label, metric_key in COLOR_METRICS.items():
            df_use = attack_df.dropna(subset=[metric_key])
            if df_use.empty:
                continue

            # Update defender counts for this specific metric
            defender_counts_for_metric = {}
            for defender in all_defenders:
                # Count how many points this defender contributes to this metric
                # Only counting points with non-zero defender flops and valid metric value
                defender_df = df_use[
                    (df_use['defender_legend_label'] == defender) & 
                    (df_use['defender_flops'] > 0) &
                    (pd.notna(df_use[metric_key]))
                ]
                defender_counts_for_metric[defender] = len(defender_df)
            
            # Create a string with defender names and their point counts for this metric
            defenders_with_counts_for_metric = [f"{defender} ({defender_counts_for_metric[defender]} pts)" 
                                              for defender in all_defenders 
                                              if defender_counts_for_metric[defender] > 0]  # Only include defenders with points
            defenders_str_for_metric = ", ".join(defenders_with_counts_for_metric)

            # Bin in log-space
            df_use["att_bin"] = pd.cut(np.log10(df_use[X_AXIS_METRIC_HISTORY]),
                                       bins=att_edges, include_lowest=True)
            df_use["def_bin"] = pd.cut(np.log10(df_use["defender_flops"]),
                                       bins=def_edges, include_lowest=True)

            # Pivot to 2-D grid
            pivot = (df_use
                     .pivot_table(values=metric_key,
                                  index="def_bin", columns="att_bin",
                                  aggfunc=aggfunc))

            # Keep bins in ascending (small→big) order for nicer axes
            pivot = pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)

            # Build the plot
            fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

            # Colour-bar bounds identical to your scatter version
            if metric_key == "eval/avg_sr_score_attacked" or metric_key == "eval/avg_sr_score_probe_attacked":
                vmin, vmax, cmap = 0, 1, "coolwarm_r"
            elif metric_key == "eval/jailbreak_rate_attacked" or metric_key == "eval/jailbreak_rate_probe_attacked":
                vmin, vmax, cmap = 0, 1, "viridis"
            else:
                vmin, vmax, cmap = pivot.min().min(), pivot.max().max(), "viridis"

            sns.heatmap(pivot,
                        vmin=vmin, vmax=vmax, cmap=cmap,
                        cbar_kws={"label": metric_label},
                        ax=ax)

            # Axis tick labels: show bin centres, formatted as FLOPs
            att_ticklabels = [f"{10**bin.mid:.2g}" for bin in pivot.columns]
            def_ticklabels = [f"{10**bin.mid:.2g}" for bin in pivot.index]
            ax.set_xticklabels(att_ticklabels, rotation=45, ha="right", fontsize="small")
            ax.set_yticklabels(def_ticklabels, rotation=0, fontsize="small")

            ax.set_xlabel(f"Attacker FLOP Cost ({X_AXIS_METRIC_HISTORY})")
            ax.set_ylabel(f"Defender FLOPs ({Y_AXIS_CONFIG_KEY})")
            
            # Create and wrap title with defenders
            title = f"Heat-map | Attack: {attack_name} | Defenders: {defenders_str_for_metric} | Colour: {metric_label}"
            wrapped_title = "\n".join(textwrap.wrap(title, width=80))
            ax.set_title(wrapped_title, fontsize=11, pad=10)

            # Save – same scheme as scatter plots
            fname = "_".join([
                "heatmap",
                sanitize_filename(attack_name, 30),
                sanitize_filename(metric_label.replace(' ', '_'), 40),
                f"groups_{sane_group_dir}",
                str(plot_idx_global)
            ])[:200]

            for ext in ("pdf", "png"):
                out = os.path.join(save_dir, f"{fname}.{ext}")
                plt.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
                print(f"  Saved {ext.upper()} heat-map: {out}")
            plt.close(fig)
            plot_idx_global += 1

# --- Data Fetching and Processing (Modified to use new defender ID functions) ---

def fetch_and_process_wandb_data_flops(entity, project, group_names, probe_name_filter=""):
    api = wandb.Api()
    runs_to_process = []
    for group_name in group_names:
        try:
            print(f"Fetching runs for group: {group_name}...")
            runs = api.runs(f"{entity}/{project}", filters={"group": group_name, "state": "finished"})
            runs_to_process.extend(list(runs))
        except wandb.errors.CommError as e:
            print(f"Error fetching runs for group '{group_name}': {e}")
            continue
    
    if not runs_to_process:
        print("No runs found for the specified group(s).")
        return pd.DataFrame()
        
    print(f"Found {len(runs_to_process)} runs across {len(group_names)} group(s) to process.")
    history_keys_to_fetch = [X_AXIS_METRIC_HISTORY, ATTACK_STEP_HISTORY_KEY] + list(COLOR_METRICS.values())
    
    all_selected_points = []
    processed_run_ids = set()

    for i, run in enumerate(runs_to_process):
        if run.id in processed_run_ids:
            continue
        processed_run_ids.add(run.id)

        # print(f"Processing run {i+1}/{len(runs_to_process)}: {run.name} (ID: {run.id})") # Less verbose
        config = run.config
        
        defender_flops = config.get(Y_AXIS_CONFIG_KEY)
        attack_name = config.get(ATTACK_NAME_CONFIG_KEY, "UnknownAttack")
        
        # Filter by probe name if specified
        if probe_name_filter:
            probe_name = config.get("probe_name", "")
            if not probe_name or probe_name_filter.lower() not in probe_name.lower():
                continue
        
        # Use the new functions for defender IDs
        defender_grouping_id = get_full_defender_id_for_grouping(config)
        defender_legend_label = get_defender_id_legend_label(config)


        if not defender_flops or defender_flops <= 0:
            # print(f"  Skipping run {run.name} due to zero/missing defender_flops ({defender_flops}).")
            continue

        try:
            history_df = run.history(keys=history_keys_to_fetch, pandas=True, samples=1000) # ensure enough samples
            if history_df.empty:
                # print(f"  Warning: Run {run.id} has empty history. Skipping.")
                continue
        except Exception as e:
            # print(f"  Warning: Could not fetch history for run {run.id}. Error: {e}. Skipping.")
            continue

        valid_history_points = []
        for _, row in history_df.iterrows():
            def get_val(r, key_orig):
                val = r.get(key_orig)
                if val is None: val = r.get(key_orig.lower())
                return val

            attacker_flops_val = get_val(row, X_AXIS_METRIC_HISTORY)
            if attacker_flops_val is None or pd.isna(attacker_flops_val) or attacker_flops_val <= 0:
                continue
            attacker_flops_val = pd.to_numeric(attacker_flops_val, errors='coerce')
            if pd.isna(attacker_flops_val) or attacker_flops_val <= 0:
                continue

            point_data = {X_AXIS_METRIC_HISTORY: attacker_flops_val}
            has_valid_color_metric = False
            for color_metric_key in COLOR_METRICS.values():
                val = get_val(row, color_metric_key)
                val_numeric = pd.to_numeric(val, errors='coerce')
                point_data[color_metric_key] = val_numeric
                if pd.notna(val_numeric): has_valid_color_metric = True
            
            if not has_valid_color_metric: continue

            attack_step_val = get_val(row, ATTACK_STEP_HISTORY_KEY)
            point_data[ATTACK_STEP_HISTORY_KEY] = pd.to_numeric(attack_step_val, errors='coerce')
            valid_history_points.append(point_data)

        if not valid_history_points:
            # print(f"  No valid history points for run {run.name}.")
            continue

        valid_history_points.sort(key=lambda x: (
            x[X_AXIS_METRIC_HISTORY], 
            x[ATTACK_STEP_HISTORY_KEY] if pd.notna(x[ATTACK_STEP_HISTORY_KEY]) else float('inf')
        ))
        
        num_vhp = len(valid_history_points)
        indices_to_select = set()
        if num_vhp > 0: indices_to_select.add(0)
        if num_vhp > 1: indices_to_select.add(num_vhp - 1)
        if num_vhp > 2:
            for p_val in range(10, 100, 10):
                idx = int(p_val / 100.0 * (num_vhp - 1))
                indices_to_select.add(idx)
                
        for idx in sorted(list(indices_to_select)):
            selected_point_hist_data = valid_history_points[idx]
            final_point_data = {
                'attack_name': attack_name,
                'defender_grouping_id': defender_grouping_id, # For grouping
                'defender_legend_label': defender_legend_label, # For legend
                'defender_flops': defender_flops,
                'run_id': run.id,
                'original_step_for_point': selected_point_hist_data.get(ATTACK_STEP_HISTORY_KEY)
            }
            final_point_data[X_AXIS_METRIC_HISTORY] = selected_point_hist_data[X_AXIS_METRIC_HISTORY]
            for cm_key in COLOR_METRICS.values():
                final_point_data[cm_key] = selected_point_hist_data[cm_key]
            all_selected_points.append(final_point_data)
            
    return pd.DataFrame(all_selected_points)


# --- Plotting Function (Modified) ---

def plot_results_flops(plot_df, group_names_list):
    if plot_df.empty:
        print("No data available for plotting.")
        return

    sns.set_context("talk")
    sns.set_style("whitegrid")
    # Setting constrained_layout globally or per figure is fine.
    # If set globally, ensure it's before any figure creation.
    # plt.rcParams['figure.constrained_layout.use'] = True 

    sane_group_dir_name = sanitize_filename("__".join(sorted(group_names_list)), 100)
    save_dir = os.path.join(BASE_SAVE_DIR_FLOPS, sane_group_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(save_dir)}")

    # Markers based on the *legend label* now, for consistency in the legend.
    unique_defender_legend_labels = sorted(plot_df['defender_legend_label'].unique())
    available_markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'H', 'd', '<', '>']
    defender_label_to_marker = {
        label: available_markers[i % len(available_markers)]
        for i, label in enumerate(unique_defender_legend_labels)
    }
    
    plot_idx_global = 0

    for attack_name, attack_group_df in plot_df.groupby('attack_name'):
        if attack_group_df.empty:
            continue

        for color_metric_display_name, color_metric_key in COLOR_METRICS.items():
            current_plot_df = attack_group_df.dropna(subset=[color_metric_key])
            if current_plot_df.empty:
                continue

            # Enable constrained_layout for this specific figure
            fig, ax = plt.subplots(figsize=(SCATTER_FIG_WIDTH_INCHES, SCATTER_FIG_HEIGHT_INCHES),
                                   constrained_layout=True) 
            
            metric_values = current_plot_df[color_metric_key]
            if color_metric_key == "eval/avg_sr_score_attacked" or color_metric_key == "eval/avg_sr_score_probe_attacked":
                vmin_actual, vmax_actual = 0, 10; cmap = "coolwarm_r"
            elif color_metric_key == "eval/jailbreak_rate_attacked" or color_metric_key == "eval/jailbreak_rate_probe_attacked":
                vmin_actual, vmax_actual = 0, 1; cmap = "viridis"
            else:
                vmin_actual, vmax_actual = metric_values.min(), metric_values.max(); cmap = "viridis"

            data_min, data_max = metric_values.min(), metric_values.max()
            vmin_plot = max(data_min, vmin_actual) if pd.notna(data_min) else vmin_actual
            vmax_plot = min(data_max, vmax_actual) if pd.notna(data_max) else vmax_actual
            if vmin_plot == vmax_plot:
                 vmax_plot = vmin_plot + 0.1 if vmax_plot < vmax_actual else vmax_actual
                 if vmin_plot == vmax_actual: vmin_plot = vmin_actual - 0.1 if vmin_plot > vmin_actual else vmin_actual
            if vmin_plot >= vmax_plot and not (vmin_plot == 0 and vmax_plot == 0):
                vmax_plot = vmin_plot + 0.1

            legend_handles = []
            sc = None
            
            # Group by the full defender ID for data separation, but use legend label for marker and legend entry
            for defender_group_id_val, defender_data_for_id in current_plot_df.groupby('defender_grouping_id'):
                if defender_data_for_id.empty:
                    continue
                
                # Get the consistent legend label for this group
                # All rows in defender_data_for_id should have the same 'defender_legend_label'
                legend_label_for_this_group = defender_data_for_id['defender_legend_label'].iloc[0]
                marker_style = defender_label_to_marker[legend_label_for_this_group]
                
                sc = ax.scatter(
                    defender_data_for_id[X_AXIS_METRIC_HISTORY],
                    defender_data_for_id['defender_flops'],
                    c=defender_data_for_id[color_metric_key],
                    marker=marker_style,
                    s=150, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                    alpha=0.8, edgecolors='k', linewidths=0.5
                )
            
            # Create legend handles based on unique legend labels
            for legend_label, marker in defender_label_to_marker.items():
                # Only add to legend if this label was actually plotted
                if legend_label in current_plot_df['defender_legend_label'].unique():
                     legend_handles.append(plt.Line2D([0], [0], marker=marker, color='w', label=legend_label,
                                           markerfacecolor='grey', markersize=10)) # Reduced marker size in legend
            
            if legend_handles:
                ax.legend(handles=legend_handles, title="Defender", 
                          bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                          fontsize='small') # Smaller font for legend

            if sc is not None:
                 cbar = fig.colorbar(sc, ax=ax, pad=0.02, aspect=30) # aspect can control colorbar thickness
                 cbar.set_label(color_metric_display_name, rotation=270, labelpad=15)
                 cbar.ax.tick_params(labelsize='small') # Smaller ticks on colorbar
            else:
                 ax.text(0.5, 0.5, "No data points", transform=ax.transAxes, ha='center', va='center', color='red')

            ax.set_xlabel(f"Attacker FLOP Cost ({X_AXIS_METRIC_HISTORY}) (log scale)")
            ax.set_ylabel(f"Defender FLOPs ({Y_AXIS_CONFIG_KEY}) (log scale)")
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize='small') # Smaller axis ticks


            title_parts = [
                f"Attack: {attack_name}",
                f"Color: {color_metric_display_name}",
                # f"Groups: {', '.join(group_names_list)}" # Potentially too long for suptitle
            ]
            # Shorter title, and let constrained_layout handle it.
            # fig.suptitle(" | ".join(title_parts), fontsize=14) # Removed y, constrained_layout should adjust
            # Or set title on the ax directly if suptitle is problematic with constrained_layout
            
            # Add defenders to title
            defenders_in_plot = sorted(current_plot_df['defender_legend_label'].unique())
            
            # Count points per defender (only counting points with valid data)
            defender_counts = {}
            for defender in defenders_in_plot:
                # Count how many points this defender contributes to this particular plot
                # Only counting points with non-zero defender flops and valid color metric
                defender_df = current_plot_df[
                    (current_plot_df['defender_legend_label'] == defender) & 
                    (current_plot_df['defender_flops'] > 0) &
                    (pd.notna(current_plot_df[color_metric_key]))
                ]
                defender_counts[defender] = len(defender_df)
            
            # Create a string with defender names and their point counts
            defenders_with_counts = [f"{defender} ({defender_counts[defender]} pts)" for defender in defenders_in_plot]
            defenders_str = ", ".join(defenders_with_counts)
            
            # Wrap long titles
            title = f"Attack: {attack_name} | Defenders: {defenders_str} | Color: {color_metric_display_name}"
            wrapped_title = "\n".join(textwrap.wrap(title, width=80))
            ax.set_title(wrapped_title, fontsize=11, pad=10)


            sane_attack_name = sanitize_filename(attack_name, 30)
            sane_metric_name = sanitize_filename(color_metric_display_name.replace(" ", "_"), 40)
            sane_groups_str = sanitize_filename("_".join(sorted(group_names_list)), 40)
            filename_base = "_".join(filter(None, ["flops_scatter", sane_attack_name, sane_metric_name,
                                                  f"groups_{sane_groups_str}" if sane_groups_str else "no_groups",
                                                  f"{plot_idx_global}"]))[:200]

            for ext in ("pdf", "png"):
                out_path = os.path.join(save_dir, f"{filename_base}.{ext}")
                try:
                    plt.savefig(out_path, bbox_inches="tight", dpi=150 if ext == "png" else None)
                    print(f"  Saved {ext.upper()} plot: {out_path}")
                except Exception as e:
                    print(f"  Error saving plot {out_path}: {e}")
            
            plt.close(fig)
            plot_idx_global += 1

# --- Main Execution (ensure it calls the modified fetch function) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FLOP costs from W&B runs")
    parser.add_argument("--entity", type=str, default=DEFAULT_WANDB_ENTITY,
                        help=f"W&B entity (default: {DEFAULT_WANDB_ENTITY})")
    parser.add_argument("--project", type=str, default=DEFAULT_WANDB_PROJECT,
                        help=f"W&B project (default: {DEFAULT_WANDB_PROJECT})")
    parser.add_argument("--groups", type=str, default=DEFAULT_GROUP_NAMES,
                        help=f"Comma-separated W&B group names (default: {DEFAULT_GROUP_NAMES})")
    parser.add_argument("--must-include-in-probe-name", type=str, default="",
                        help="Only include defenders with this string in their probe_name")
    args = parser.parse_args()
    
    wandb_entity = args.entity
    wandb_project = args.project
    group_names_list = [name.strip() for name in args.groups.split(',') if name.strip()]
    probe_name_filter = args.must_include_in_probe_name

    if not wandb_project:
        print("W&B project is required.")
    elif not group_names_list:
        print("No group names entered. Exiting.")
    else:
        print(f"\nFetching: entity='{wandb_entity}', project='{wandb_project}', groups={group_names_list}")
        if probe_name_filter:
            print(f"Filtering to defenders with probe names containing: '{probe_name_filter}'")
        
        processed_df = fetch_and_process_wandb_data_flops(wandb_entity, wandb_project, group_names_list, probe_name_filter)
        
        if processed_df.empty:
            print("\nNo data processed. Check criteria (positive FLOPs, etc.).")
        else:
            print(f"\nData processing complete. Found {len(processed_df)} plot points from {processed_df['run_id'].nunique()} runs.")
            print(f"Plotting data for {processed_df['attack_name'].nunique()} unique attack types.")
            # print("\nUnique Defender Legend Labels:", sorted(processed_df['defender_legend_label'].unique()))
            # print("\nSample of processed data for plotting:")
            # print(processed_df[['attack_name', 'defender_grouping_id', 'defender_legend_label', X_AXIS_METRIC_HISTORY, 'defender_flops', COLOR_METRICS["Avg SR Score (Attacked)"]]].head())

            plot_results_flops(processed_df, group_names_list)
            plot_heatmaps_flops(processed_df, group_names_list)
            print("\nPlotting finished.")