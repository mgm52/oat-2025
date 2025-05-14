import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.ticker as mticker

def parse_log(log_content):
    entries = []

    # Updated to capture SEED
    model_attack_start_pattern = re.compile(
        r"STARTING ATTACK WITH MAX STEPS = \d+, SUFFIX LENGTH = (\d+), MODEL = (\w+), SEED = (\d+)"
    )
    step_eval_pattern = re.compile(
        r"Executing callback <lambda> for step (\d+)..."
    )
    flop_cost_pattern = re.compile(r"Current attack FLOP cost: (\d+)")

    summary_probe_pattern = re.compile(
        r"=== SUMMARY ===.*?"
        r"SYSTEM: LLM \+ PROBE \(1%FPR refusal threshold\).*?"
        r"After suffix attack.*?"
        r"Avg SR score w/ probe: ([0-9.-]+) \[95% CI: ([0-9.-]+), ([0-9.-]+)\].*?" # Original CIs ignored for plotting
        r"Refusal rate w/ probe: ([0-9.-]+) \[95% CI: ([0-9.-]+), ([0-9.-]+)\]",    # Original CIs ignored for plotting
        re.S
    )
    summary_no_probe_pattern = re.compile(
        r"=== SUMMARY ===.*?"
        r"SYSTEM: LLM\n.*?"
        r"After suffix attack.*?"
        r"Avg SR score: ([0-9.-]+) \[95% CI: ([0-9.-]+), ([0-9.-]+)\].*?"          # Original CIs ignored for plotting
        r"Refusal rate: ([0-9.-]+) \[95% CI: ([0-9.-]+), ([0-9.-]+)\]",           # Original CIs ignored for plotting
        re.S
    )

    model_attack_markers = list(re.finditer(model_attack_start_pattern, log_content))

    for i, model_attack_match in enumerate(model_attack_markers):
        suffix_length = int(model_attack_match.group(1))
        model_name = model_attack_match.group(2)
        seed = int(model_attack_match.group(3)) # Capture seed

        current_block_start_idx = model_attack_match.end()
        if i + 1 < len(model_attack_markers):
            next_block_start_idx = model_attack_markers[i+1].start()
            current_model_suffix_block_content = log_content[current_block_start_idx:next_block_start_idx]
        else:
            current_model_suffix_block_content = log_content[current_block_start_idx:]

        step_eval_markers = list(re.finditer(step_eval_pattern, current_model_suffix_block_content))

        if not step_eval_markers:
            continue

        for j, step_match in enumerate(step_eval_markers):
            actual_max_steps = int(step_match.group(1))
            
            summary_search_region_start_offset = step_match.end()
            if j + 1 < len(step_eval_markers):
                summary_search_region_end_offset = step_eval_markers[j+1].start()
                summary_block_for_step = current_model_suffix_block_content[summary_search_region_start_offset:summary_search_region_end_offset]
            else:
                summary_block_for_step = current_model_suffix_block_content[summary_search_region_start_offset:]

            flop_cost = np.nan
            flop_match = flop_cost_pattern.search(summary_block_for_step)
            if flop_match:
                flop_cost = int(flop_match.group(1))

            entry = {
                "suffix_length": suffix_length,
                "max_steps": actual_max_steps,
                "flop_cost": flop_cost,
                "source": model_name,
                "seed": seed, # Add seed to entry
                # Metrics initialized to NaN
                "jailbreak_rate_probe": np.nan,
                "sr_score_probe": np.nan,
                "jailbreak_rate_no_probe": np.nan,
                "sr_score_no_probe": np.nan,
            }
            found_any_summary_for_this_step = False

            summary_probe_match = summary_probe_pattern.search(summary_block_for_step)
            if summary_probe_match:
                sr_score_p, _, _, refusal_p, _, _ = map(float, summary_probe_match.groups()) # Ignored CIs from log
                entry["sr_score_probe"] = sr_score_p
                entry["jailbreak_rate_probe"] = 1.0 - refusal_p
                found_any_summary_for_this_step = True

            summary_no_probe_match = summary_no_probe_pattern.search(summary_block_for_step)
            if summary_no_probe_match:
                sr_score_np, _, _, refusal_np, _, _ = map(float, summary_no_probe_match.groups()) # Ignored CIs from log
                entry["sr_score_no_probe"] = sr_score_np
                entry["jailbreak_rate_no_probe"] = 1.0 - refusal_np
                found_any_summary_for_this_step = True
            
            if found_any_summary_for_this_step:
                entries.append(entry)

    return pd.DataFrame(entries)

def aggregate_metrics_for_plotting(df, x_col, metric_col):
    """Aggregates metrics by source and x_col, calculating mean and percentiles."""
    if df.empty or metric_col not in df.columns or x_col not in df.columns:
        return pd.DataFrame()
    
    # Drop rows where the metric itself is NaN before aggregation
    df_filtered = df.dropna(subset=[metric_col, x_col, 'source'])
    if df_filtered.empty:
        return pd.DataFrame()

    # Define percentile functions robust to few data points
    def p025(x):
        return np.percentile(x, 2.5) if len(x.dropna()) > 0 else np.nan
    def p975(x):
        return np.percentile(x, 97.5) if len(x.dropna()) > 0 else np.nan
    def mean_val(x):
        return x.dropna().mean() if len(x.dropna()) > 0 else np.nan

    aggregated = df_filtered.groupby(['source', x_col]).agg(
        mean=(metric_col, mean_val),
        p025=(metric_col, p025),
        p975=(metric_col, p975),
        count=(metric_col, 'count') # Count of seeds for this point
    ).reset_index()
    return aggregated


if __name__ == "__main__":
    # --- Configuration ---
    log_file_paths = [
        "/workspace/GIT_SHENANIGANS/oat-2025/SEEDY_EVAL_TEST.log",
    ]
    output_dir = "oat_evaluation/scripts/plots_chonky_combined_logs_lines_flops_seeds_ci" 
    os.makedirs(output_dir, exist_ok=True)

    combined_log_content = ""
    files_read_count = 0
    for file_path in log_file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                combined_log_content += f.read() + "\n"
                files_read_count += 1
            print(f"Successfully read and appended: {file_path}")
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}. Skipping.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}. Skipping.")

    if files_read_count == 0:
        print("Error: No log files were successfully read. Exiting."); exit(1)
    if not combined_log_content.strip():
        print("Error: Combined log content is empty. Exiting."); exit(1)

    df = parse_log(combined_log_content)

    if df.empty:
        print("Error: No data parsed. Check logs and regex."); exit(1)

    if 'source' in df.columns:
        labels = sorted(df["source"].unique())
        if not labels: print("Error: No 'source' found."); exit(1)
    else: print("Error: 'source' column missing."); exit(1)

    numeric_cols = [
        'suffix_length', 'max_steps', 'flop_cost', 'seed',
        'jailbreak_rate_probe', 'sr_score_probe',
        'jailbreak_rate_no_probe', 'sr_score_no_probe'
    ]
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        else: print(f"Warning: Expected numeric column '{col}' not found.")

    if df[['max_steps', 'suffix_length', 'source', 'seed']].isnull().values.any():
         print("Critical Warning: NaN values in 'max_steps', 'suffix_length', 'source', or 'seed'.")
    if 'flop_cost' in df and df['flop_cost'].isnull().all():
        print("Warning: 'flop_cost' column all NaN. FLOP plots may be empty.")

    sns.set(style="whitegrid")
    hue_order = labels 

    metrics_to_plot = {
        "jailbreak_rate_probe": "Jailbreak Rate (w/ Probe)",
        "sr_score_probe": "Avg SR Score (w/ Probe)",
        "jailbreak_rate_no_probe": "Jailbreak Rate (No Probe)",
        "sr_score_no_probe": "Avg SR Score (No Probe)",
    }
    
    x_axis_configs = {
        "max_steps": {"label": "MAX STEPS (Attack Iterations)", "col_name": "max_steps", "ticks_from_data": True},
        "flop_cost": {"label": "FLOP Cost", "col_name": "flop_cost", "ticks_from_data": False} # FLOP cost is continuous
    }

    for suffix_length in sorted(df["suffix_length"].dropna().unique()):
        subset_sf = df[df["suffix_length"] == suffix_length].copy()
        if subset_sf.empty:
            print(f"Skipping suffix length {suffix_length}: No data."); continue

        for x_config_name, x_config in x_axis_configs.items():
            x_col = x_config["col_name"]
            x_label = x_config["label"]

            # Ensure the x_axis column has valid data for this suffix_length
            if x_col not in subset_sf.columns or subset_sf[x_col].isnull().all():
                print(f"Skipping {x_label} plots for suffix {suffix_length}: No valid '{x_col}' data.")
                continue
            
            # Determine x-ticks for MAX_STEPS plots specifically
            current_x_ticks = None
            if x_config["ticks_from_data"] and x_col == "max_steps":
                current_x_ticks = sorted(subset_sf[x_col].dropna().unique())
                if not current_x_ticks:
                    print(f"Skipping {x_label} plots for suffix {suffix_length}: No unique '{x_col}' values found for ticks.")
                    continue


            for metric_col, y_label in metrics_to_plot.items():
                print(f"Processing: Suffix={suffix_length}, Metric='{y_label}', X-axis='{x_label}'")
                
                # Aggregate data for the current x_col and metric_col
                aggregated_data = aggregate_metrics_for_plotting(subset_sf, x_col, metric_col)
                
                if aggregated_data.empty or aggregated_data['mean'].isnull().all():
                    print(f"  Skipping plot: No aggregated data or all means are NaN for {metric_col} vs {x_col}.")
                    continue

                plt.figure(figsize=(12, 7))
                
                # Palette for consistent colors across plots for the same models
                palette = sns.color_palette(n_colors=len(hue_order))
                color_map = {model: palette[i] for i, model in enumerate(hue_order)}

                for model_name in hue_order:
                    model_plot_data = aggregated_data[aggregated_data['source'] == model_name].sort_values(by=x_col)
                    
                    if not model_plot_data.empty and not model_plot_data['mean'].isnull().all():
                        color = color_map.get(model_name)
                        plt.plot(model_plot_data[x_col], model_plot_data['mean'], label=model_name, marker='o', color=color)
                        
                        # Only plot fill_between if p025 and p975 are not all NaN and different from mean
                        # And if there's more than one seed for at least some points
                        valid_ci_data = model_plot_data.dropna(subset=['p025', 'p975'])
                        if not valid_ci_data.empty and (valid_ci_data['count'] > 1).any():
                             plt.fill_between(valid_ci_data[x_col], valid_ci_data['p025'], valid_ci_data['p975'], alpha=0.2, color=color)
                
                plt.title(f"{suffix_length}-Token Attack: {y_label} vs {x_label}\n(Mean +/- 95% CI from Seeds)", fontsize=14)
                plt.xlabel(x_label, fontsize=12)
                plt.ylabel(y_label, fontsize=12)

                if "Jailbreak Rate" in y_label: plt.ylim(0, 1.05)
                else: # For SR Score, auto-adjust y-limits with some padding
                    min_y_val = aggregated_data.dropna(subset=['p025'])['p025'].min() if not aggregated_data.dropna(subset=['p025']).empty else aggregated_data['mean'].min()
                    max_y_val = aggregated_data.dropna(subset=['p975'])['p975'].max() if not aggregated_data.dropna(subset=['p975']).empty else aggregated_data['mean'].max()
                    if pd.notna(min_y_val) and pd.notna(max_y_val):
                         padding = (max_y_val - min_y_val) * 0.1 if (max_y_val - min_y_val) > 0 else 0.1
                         plt.ylim(min_y_val - padding, max_y_val + padding if max_y_val > min_y_val - padding else min_y_val + padding + 0.2)


                if x_config["ticks_from_data"] and current_x_ticks:
                    plt.xticks(ticks=current_x_ticks, labels=current_x_ticks)
                elif x_col == "flop_cost":
                    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                    # Consider log scale: plt.xscale('log') and appropriate tick formatting

                plt.legend(title="System", title_fontsize='13', fontsize='11', loc='best')
                plt.tight_layout()
                
                plot_filename_metric_short = metric_col.replace("_probe","P").replace("_no_probe","NP").replace("jailbreak_rate","JB").replace("sr_score","SR")
                plot_filename = os.path.join(output_dir, f"{x_config_name}_{plot_filename_metric_short}_ci_{suffix_length}.png")
                plt.savefig(plot_filename)
                print(f"  Saved plot: {plot_filename}")
                plt.close()

    print("Processing complete.")