import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os # Import os to create directories

def parse_log(log_content, label):
    # Regex to find the start of an evaluation block
    start_pattern = re.compile(r"STARTING EVALUATION WITH MAX STEPS = (\d+); SUFFIX LENGTH = (\d+)")

    # Regex for LLM + PROBE summary block after attack
    summary_probe_pattern = re.compile(
        r"=== SUMMARY ===.*?"
        r"SYSTEM: LLM \+ PROBE \(1%FPR refusal threshold\).*?"
        r"After suffix attack.*?"
        r"Avg SR score w/ probe: ([0-9.-]+) \[95% CI: ([0-9.-]+), ([0-9.-]+)\].*?" # Allow negative sign for CI potentially
        r"Refusal rate w/ probe: ([0-9.-]+) \[95% CI: ([0-9.-]+), ([0-9.-]+)\]",
        re.S
    )

    # Regex for LLM (no probe) summary block after attack
    summary_no_probe_pattern = re.compile(
        r"=== SUMMARY ===.*?"
        r"SYSTEM: LLM\n.*?" # Match "SYSTEM: LLM" exactly at the start of a line
        r"After suffix attack.*?"
        r"Avg SR score: ([0-9.-]+) \[95% CI: ([0-9.-]+), ([0-9.-]+)\].*?" # Allow negative sign
        r"Refusal rate: ([0-9.-]+) \[95% CI: ([0-9.-]+), ([0-9.-]+)\]", # Allow negative sign
        re.S
    )


    entries = []
    # Iterate through each evaluation run found in the log
    for match in re.finditer(start_pattern, log_content):
        max_steps, suffix_length = map(int, match.groups())
        match_start = match.end()

        # Find the start of the *next* evaluation block to delimit the current block
        next_start_match = re.search(start_pattern, log_content[match_start:])
        if next_start_match:
            block_end = match_start + next_start_match.start()
            block = log_content[match_start:block_end]
        else:
            block = log_content[match_start:]

        entry = {
            "suffix_length": suffix_length,
            "max_steps": max_steps,
            "source": label,
            # Initialize all metrics to NaN
            "jailbreak_rate_probe": np.nan, "jb_ci_lower_probe": np.nan, "jb_ci_upper_probe": np.nan,
            "sr_score_probe": np.nan, "sr_ci_low_probe": np.nan, "sr_ci_high_probe": np.nan,
            "jailbreak_rate_no_probe": np.nan, "jb_ci_lower_no_probe": np.nan, "jb_ci_upper_no_probe": np.nan,
            "sr_score_no_probe": np.nan, "sr_ci_low_no_probe": np.nan, "sr_ci_high_no_probe": np.nan,
        }

        # Search for the summary WITH PROBE within this block
        summary_probe_match = summary_probe_pattern.search(block)
        if summary_probe_match:
            sr_score_p, sr_ci_low_p, sr_ci_high_p, refusal_p, refusal_ci_low_p, refusal_ci_high_p = map(float, summary_probe_match.groups())

            entry["sr_score_probe"] = sr_score_p
            entry["sr_ci_low_probe"] = sr_ci_low_p
            entry["sr_ci_high_probe"] = sr_ci_high_p

            entry["jailbreak_rate_probe"] = 1.0 - refusal_p
            entry["jb_ci_lower_probe"] = 1.0 - refusal_ci_high_p # Note the swap for CI
            entry["jb_ci_upper_probe"] = 1.0 - refusal_ci_low_p  # Note the swap for CI

        # Search for the summary WITHOUT PROBE within this block
        summary_no_probe_match = summary_no_probe_pattern.search(block)
        if summary_no_probe_match:
            sr_score_np, sr_ci_low_np, sr_ci_high_np, refusal_np, refusal_ci_low_np, refusal_ci_high_np = map(float, summary_no_probe_match.groups())

            entry["sr_score_no_probe"] = sr_score_np
            entry["sr_ci_low_no_probe"] = sr_ci_low_np
            entry["sr_ci_high_no_probe"] = sr_ci_high_np

            entry["jailbreak_rate_no_probe"] = 1.0 - refusal_np
            entry["jb_ci_lower_no_probe"] = 1.0 - refusal_ci_high_np # Note the swap for CI
            entry["jb_ci_upper_no_probe"] = 1.0 - refusal_ci_low_np  # Note the swap for CI

        # Only add the entry if at least one of the summaries was found
        if summary_probe_match or summary_no_probe_match:
             entries.append(entry)
        # else:
            # Optional: print a warning if a block doesn't have *any* expected summary
            # print(f"Warning: Could not find any summary for MAX_STEPS={max_steps}, SUFFIX_LENGTH={suffix_length} in label '{label}'")


    return pd.DataFrame(entries)

if __name__ == "__main__":
    # --- Configuration ---
    files = [
        "/workspace/GIT_SHENANIGANS/oat-2025/BIG_eval_test_OATABHAY_250425basellama12.log",
        "/workspace/GIT_SHENANIGANS/oat-2025/BIG_eval_test_LATABHAY_2904252.log",
        "/workspace/GIT_SHENANIGANS/oat-2025/BIG_eval_test_OATABHAY_250425oatllama.log"
    ]
    labels = [
        "Base LLaMa", # Shortened label for plots
        "LAT LLaMa",
        "OAT LLaMa"
    ]
    output_dir = "oat_evaluation/scripts/plots" # Define output directory
    # --- End Configuration ---

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Ensure files and labels have the same length
    if len(files) != len(labels):
        print("Error: Number of files does not match number of labels")
        exit()

    # Read all log files and parse them
    dfs = []
    for i, (file, label) in enumerate(zip(files, labels)):
        try:
            with open(file, "r", encoding="utf-8") as f:
                log = f.read()
        except FileNotFoundError:
            print(f"Error: File not found - {file}")
            continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        # Parse log
        df_parsed = parse_log(log, label=label) # Renamed variable to avoid conflict
        if df_parsed.empty:
            print(f"Warning: No data parsed from file {i+1}: {file}")
        else:
            dfs.append(df_parsed)

    # Combine data
    if not dfs:
        print("Error: No data parsed from any log file. Check log format and regex.")
        exit()

    df = pd.concat(dfs, ignore_index=True)

    if df.empty:
        print("Error: Combined dataframe is empty after parsing.")
        exit()

    # Ensure numeric types
    numeric_cols = [
        'suffix_length', 'max_steps',
        'jailbreak_rate_probe', 'jb_ci_lower_probe', 'jb_ci_upper_probe',
        'sr_score_probe', 'sr_ci_low_probe', 'sr_ci_high_probe',
        'jailbreak_rate_no_probe', 'jb_ci_lower_no_probe', 'jb_ci_upper_no_probe',
        'sr_score_no_probe', 'sr_ci_low_no_probe', 'sr_ci_high_no_probe'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for parsing issues leading to NaNs
    if df[numeric_cols].isnull().values.any():
        print("Warning: Found NaN values after parsing. Check logs or regex. This might be expected if some blocks are missing.")
        # print(df[df[numeric_cols].isnull().any(axis=1)]) # Optional: print rows with NaNs

    # Plotting
    sns.set(style="whitegrid")

    # Get unique sorted max_steps for consistent x-axis ordering
    all_max_steps = sorted(df["max_steps"].unique())
    # Map max_steps to integer positions for error bar placement
    step_to_xpos = {step: i for i, step in enumerate(all_max_steps)}
    hue_order = labels # Define order for consistent coloring and positioning
    bar_width = 0.8 / len(labels) # Adjust bar width based on number of models

    for suffix_length in sorted(df["suffix_length"].unique()):
        # Filter data for the current suffix length
        subset_full = df[(df["suffix_length"] == suffix_length) & df['max_steps'].isin(all_max_steps)].copy()

        if subset_full.empty:
             print(f"Skipping suffix length {suffix_length}: No data found for this length.")
             continue

        # --- Plot 1: Jailbreak Rate (With Probe) ---
        plot_cols_jb_probe = ["max_steps", "jailbreak_rate_probe", "jb_ci_lower_probe", "jb_ci_upper_probe"]
        subset_jb_probe = subset_full.dropna(subset=plot_cols_jb_probe).copy()
        if not subset_jb_probe.empty:
            plt.figure(figsize=(12, 7))
            ax1 = sns.barplot(
                data=subset_jb_probe, x="max_steps", y="jailbreak_rate_probe", hue="source",
                hue_order=hue_order, order=all_max_steps, capsize=0.1, errwidth=1.5, ci=None
            )
            # Add manual error bars
            for i, source_label in enumerate(hue_order):
                source_subset = subset_jb_probe[subset_jb_probe["source"] == source_label]
                for _, row in source_subset.iterrows():
                    x_center = step_to_xpos[row["max_steps"]]
                    x_pos = x_center - bar_width * len(labels) / 2 + (i + 0.5) * bar_width
                    y_err_lower = row["jailbreak_rate_probe"] - row["jb_ci_lower_probe"]
                    y_err_upper = row["jb_ci_upper_probe"] - row["jailbreak_rate_probe"]
                    y_err_lower = max(0, y_err_lower if not pd.isna(y_err_lower) else 0)
                    y_err_upper = max(0, y_err_upper if not pd.isna(y_err_upper) else 0)
                    plt.errorbar(x=x_pos, y=row["jailbreak_rate_probe"], yerr=[[y_err_lower], [y_err_upper]],
                                 fmt='none', c='black', capsize=4)

            plt.title(f"{suffix_length}-Token Attack: Jailbreak Rate (w/ Probe, 95% CI)", fontsize=14)
            plt.xlabel("MAX STEPS (Attack Iterations)", fontsize=12)
            plt.ylabel("Jailbreak Rate (w/ Probe)", fontsize=12)
            plt.ylim(0, 1.05)
            plt.xticks(ticks=range(len(all_max_steps)), labels=all_max_steps)
            plt.legend(title="System", title_fontsize='13', fontsize='11')
            plt.tight_layout()
            jb_probe_plot_filename = os.path.join(output_dir, f"jailbreak_rate_probe_comparison_plot_{suffix_length}.png")
            plt.savefig(jb_probe_plot_filename)
            print(f"Saved Jailbreak Rate (Probe) plot: {jb_probe_plot_filename}")
            plt.close()
        else:
            print(f"Skipping Jailbreak Rate (Probe) plot for suffix length {suffix_length}: No valid data.")

        # --- Plot 2: SR Score (With Probe) ---
        plot_cols_sr_probe = ["max_steps", "sr_score_probe", "sr_ci_low_probe", "sr_ci_high_probe"]
        subset_sr_probe = subset_full.dropna(subset=plot_cols_sr_probe).copy()
        if not subset_sr_probe.empty:
            plt.figure(figsize=(12, 7))
            ax2 = sns.barplot(
                data=subset_sr_probe, x="max_steps", y="sr_score_probe", hue="source",
                hue_order=hue_order, order=all_max_steps, capsize=0.1, errwidth=1.5, ci=None
            )
            # Add manual error bars
            for i, source_label in enumerate(hue_order):
               source_subset = subset_sr_probe[subset_sr_probe["source"] == source_label]
               for _, row in source_subset.iterrows():
                    x_center = step_to_xpos[row["max_steps"]]
                    x_pos = x_center - bar_width * len(labels) / 2 + (i + 0.5) * bar_width
                    y_err_lower = row["sr_score_probe"] - row["sr_ci_low_probe"]
                    y_err_upper = row["sr_ci_high_probe"] - row["sr_score_probe"]
                    y_err_lower = max(0, y_err_lower if not pd.isna(y_err_lower) else 0)
                    y_err_upper = max(0, y_err_upper if not pd.isna(y_err_upper) else 0)
                    plt.errorbar(x=x_pos, y=row["sr_score_probe"], yerr=[[y_err_lower], [y_err_upper]],
                                 fmt='none', c='black', capsize=4)

            plt.title(f"{suffix_length}-Token Attack: Avg SR Score (w/ Probe, 95% CI)", fontsize=14)
            plt.xlabel("MAX STEPS (Attack Iterations)", fontsize=12)
            plt.ylabel("Avg SR Score (w/ Probe)", fontsize=12)
            # plt.ylim(0, max(subset_sr_probe['sr_ci_high_probe'].max() * 1.1, 0.1)) # Auto-scale example
            plt.xticks(ticks=range(len(all_max_steps)), labels=all_max_steps)
            plt.legend(title="System", title_fontsize='13', fontsize='11')
            plt.tight_layout()
            sr_probe_plot_filename = os.path.join(output_dir, f"sr_score_probe_comparison_plot_{suffix_length}.png")
            plt.savefig(sr_probe_plot_filename)
            print(f"Saved SR Score (Probe) plot: {sr_probe_plot_filename}")
            plt.close()
        else:
            print(f"Skipping SR Score (Probe) plot for suffix length {suffix_length}: No valid data.")


        # --- Plot 3: Jailbreak Rate (No Probe) ---
        plot_cols_jb_no_probe = ["max_steps", "jailbreak_rate_no_probe", "jb_ci_lower_no_probe", "jb_ci_upper_no_probe"]
        subset_jb_no_probe = subset_full.dropna(subset=plot_cols_jb_no_probe).copy()
        if not subset_jb_no_probe.empty:
            plt.figure(figsize=(12, 7))
            ax3 = sns.barplot(
                data=subset_jb_no_probe, x="max_steps", y="jailbreak_rate_no_probe", hue="source",
                hue_order=hue_order, order=all_max_steps, capsize=0.1, errwidth=1.5, ci=None
            )
            # Add manual error bars
            for i, source_label in enumerate(hue_order):
                source_subset = subset_jb_no_probe[subset_jb_no_probe["source"] == source_label]
                for _, row in source_subset.iterrows():
                    x_center = step_to_xpos[row["max_steps"]]
                    x_pos = x_center - bar_width * len(labels) / 2 + (i + 0.5) * bar_width
                    y_err_lower = row["jailbreak_rate_no_probe"] - row["jb_ci_lower_no_probe"]
                    y_err_upper = row["jb_ci_upper_no_probe"] - row["jailbreak_rate_no_probe"]
                    y_err_lower = max(0, y_err_lower if not pd.isna(y_err_lower) else 0)
                    y_err_upper = max(0, y_err_upper if not pd.isna(y_err_upper) else 0)
                    plt.errorbar(x=x_pos, y=row["jailbreak_rate_no_probe"], yerr=[[y_err_lower], [y_err_upper]],
                                 fmt='none', c='black', capsize=4)

            plt.title(f"{suffix_length}-Token Attack: Jailbreak Rate (No Probe, 95% CI)", fontsize=14)
            plt.xlabel("MAX STEPS (Attack Iterations)", fontsize=12)
            plt.ylabel("Jailbreak Rate (No Probe)", fontsize=12)
            plt.ylim(0, 1.05)
            plt.xticks(ticks=range(len(all_max_steps)), labels=all_max_steps)
            plt.legend(title="System", title_fontsize='13', fontsize='11')
            plt.tight_layout()
            jb_no_probe_plot_filename = os.path.join(output_dir, f"jailbreak_rate_no_probe_comparison_plot_{suffix_length}.png")
            plt.savefig(jb_no_probe_plot_filename)
            print(f"Saved Jailbreak Rate (No Probe) plot: {jb_no_probe_plot_filename}")
            plt.close()
        else:
             print(f"Skipping Jailbreak Rate (No Probe) plot for suffix length {suffix_length}: No valid data.")

        # --- Plot 4: SR Score (No Probe) ---
        plot_cols_sr_no_probe = ["max_steps", "sr_score_no_probe", "sr_ci_low_no_probe", "sr_ci_high_no_probe"]
        subset_sr_no_probe = subset_full.dropna(subset=plot_cols_sr_no_probe).copy()
        if not subset_sr_no_probe.empty:
            plt.figure(figsize=(12, 7))
            ax4 = sns.barplot(
                data=subset_sr_no_probe, x="max_steps", y="sr_score_no_probe", hue="source",
                hue_order=hue_order, order=all_max_steps, capsize=0.1, errwidth=1.5, ci=None
            )
            # Add manual error bars
            for i, source_label in enumerate(hue_order):
               source_subset = subset_sr_no_probe[subset_sr_no_probe["source"] == source_label]
               for _, row in source_subset.iterrows():
                    x_center = step_to_xpos[row["max_steps"]]
                    x_pos = x_center - bar_width * len(labels) / 2 + (i + 0.5) * bar_width
                    y_err_lower = row["sr_score_no_probe"] - row["sr_ci_low_no_probe"]
                    y_err_upper = row["sr_ci_high_no_probe"] - row["sr_score_no_probe"]
                    y_err_lower = max(0, y_err_lower if not pd.isna(y_err_lower) else 0)
                    y_err_upper = max(0, y_err_upper if not pd.isna(y_err_upper) else 0)
                    plt.errorbar(x=x_pos, y=row["sr_score_no_probe"], yerr=[[y_err_lower], [y_err_upper]],
                                 fmt='none', c='black', capsize=4)

            plt.title(f"{suffix_length}-Token Attack: Avg SR Score (No Probe, 95% CI)", fontsize=14)
            plt.xlabel("MAX STEPS (Attack Iterations)", fontsize=12)
            plt.ylabel("Avg SR Score (No Probe)", fontsize=12)
            # plt.ylim(0, max(subset_sr_no_probe['sr_ci_high_no_probe'].max() * 1.1, 0.1)) # Auto-scale example
            plt.xticks(ticks=range(len(all_max_steps)), labels=all_max_steps)
            plt.legend(title="System", title_fontsize='13', fontsize='11')
            plt.tight_layout()
            sr_no_probe_plot_filename = os.path.join(output_dir, f"sr_score_no_probe_comparison_plot_{suffix_length}.png")
            plt.savefig(sr_no_probe_plot_filename)
            print(f"Saved SR Score (No Probe) plot: {sr_no_probe_plot_filename}")
            plt.close()
        else:
             print(f"Skipping SR Score (No Probe) plot for suffix length {suffix_length}: No valid data.")


    print("Processing complete.")