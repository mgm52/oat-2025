import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for isnan check if needed, though pandas usually handles it

def parse_log(log_content, label):
    # Regex to find the start of an evaluation block
    start_pattern = re.compile(r"STARTING EVALUATION WITH MAX STEPS = (\d+); SUFFIX LENGTH = (\d+)")

    # Regex to find the specific summary block for LLM + PROBE after attack
    # Captures: SR Score, SR CI Low, SR CI High, Refusal Rate, Refusal CI Low, Refusal CI High
    summary_block_pattern = re.compile(
        r"=== SUMMARY ===.*?"  # Start of summary block
        r"SYSTEM: LLM \+ PROBE \(1%FPR refusal threshold\).*?" # Find the right system block
        r"After suffix attack.*?" # Ensure we are looking after the attack
        r"Avg SR score w/ probe: ([0-9.]+) \[95% CI: ([0-9.]+), ([0-9.]+)\].*?" # Capture SR score + CI
        r"Refusal rate w/ probe: ([0-9.]+) \[95% CI: ([0-9.]+), ([0-9.]+)\]", # Capture Refusal rate + CI
        re.S # Allow '.' to match newlines
    )

    entries = []
    # Iterate through each evaluation run found in the log
    for match in re.finditer(start_pattern, log_content):
        max_steps, suffix_length = map(int, match.groups())
        match_start = match.end() # Where the STARTING EVALUATION line ends

        # Find the start of the *next* evaluation block to delimit the current block
        next_start_match = re.search(start_pattern, log_content[match_start:])
        if next_start_match:
            block_end = match_start + next_start_match.start()
            block = log_content[match_start:block_end]
        else:
            # If no next block, take the rest of the log content
            block = log_content[match_start:]

        # Search for the summary within this specific block
        summary_match = summary_block_pattern.search(block)
        if summary_match:
            # Extract captured groups
            sr_score, sr_ci_low, sr_ci_high, refusal_after, refusal_ci_low, refusal_ci_high = map(float, summary_match.groups())

            # Calculate Jailbreak Rate and its CI
            jailbreak_rate = 1.0 - refusal_after
            # Correct CI calculation: JB_low = 1 - Refusal_high, JB_high = 1 - Refusal_low
            jb_ci_lower = 1.0 - refusal_ci_high
            jb_ci_upper = 1.0 - refusal_ci_low

            entries.append({
                "suffix_length": suffix_length,
                "max_steps": max_steps,
                "jailbreak_rate": jailbreak_rate,
                "jb_ci_lower": jb_ci_lower,
                "jb_ci_upper": jb_ci_upper,
                "sr_score": sr_score,
                "sr_ci_low": sr_ci_low,
                "sr_ci_high": sr_ci_high,
                "source": label
            })
        # else:
            # Optional: print a warning if a block doesn't have the expected summary
            # print(f"Warning: Could not find summary for MAX_STEPS={max_steps}, SUFFIX_LENGTH={suffix_length} in label '{label}'")


    return pd.DataFrame(entries)

if __name__ == "__main__":
    # --- Configuration ---
    file1 = "/workspace/GIT_SHENANIGANS/oat-2025/BIG_eval_test_OATABHAY_250425basellama12.log"
    file2 = "/workspace/GIT_SHENANIGANS/oat-2025/BIG_eval_test_OATABHAY_250425oatllama.log"
    label1 = "Base LLaMa + probe"
    label2 = "OAT LLaMa + probe"
    # --- End Configuration ---

    # Read both log files
    try:
        with open(file1, "r", encoding="utf-8") as f:
            log1 = f.read()
    except FileNotFoundError:
        print(f"Error: File not found - {file1}")
        exit()
    except Exception as e:
        print(f"Error reading {file1}: {e}")
        exit()

    try:
        with open(file2, "r", encoding="utf-8") as f:
            log2 = f.read()
    except FileNotFoundError:
        print(f"Error: File not found - {file2}")
        exit()
    except Exception as e:
        print(f"Error reading {file2}: {e}")
        exit()


    # Parse logs
    df1 = parse_log(log1, label=label1)
    df2 = parse_log(log2, label=label2)

    # Combine data
    if df1.empty and df2.empty:
        print("Error: No data parsed from either log file. Check log format and regex.")
        exit()
    elif df1.empty:
        print("Warning: No data parsed from file 1.")
    elif df2.empty:
        print("Warning: No data parsed from file 2.")

    df = pd.concat([df1, df2], ignore_index=True)

    if df.empty:
        print("Error: Combined dataframe is empty after parsing.")
        exit()

    # Ensure numeric types (pandas usually handles this from map(float), but good practice)
    numeric_cols = ['suffix_length', 'max_steps', 'jailbreak_rate', 'jb_ci_lower', 'jb_ci_upper',
                    'sr_score', 'sr_ci_low', 'sr_ci_high']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce forces non-numerics to NaN


    # Check for parsing issues leading to NaNs
    if df[numeric_cols].isnull().values.any():
        print("Warning: Found NaN values after parsing. Check logs or regex.")
        print(df[df[numeric_cols].isnull().any(axis=1)])


    # Plotting
    sns.set(style="whitegrid")

    # Get unique sorted max_steps for consistent x-axis ordering
    all_max_steps = sorted(df["max_steps"].unique())
    # Map max_steps to integer positions for error bar placement
    step_to_xpos = {step: i for i, step in enumerate(all_max_steps)}
    hue_order = [label1, label2] # Define order for consistent coloring and positioning
    bar_width = 0.4 # Adjust as needed for bar separation

    for suffix_length in sorted(df["suffix_length"].unique()):
        subset = df[(df["suffix_length"] == suffix_length) & df['max_steps'].isin(all_max_steps)].copy() # Filter relevant data
        subset = subset.dropna(subset=numeric_cols) # Drop rows with NaN in critical columns


        if subset.empty:
            print(f"Skipping suffix length {suffix_length}: No valid data after filtering/dropping NaNs.")
            continue

        # --- Plot 1: Jailbreak Rate ---
        plt.figure(figsize=(12, 7)) # Slightly wider figure
        ax1 = sns.barplot(
            data=subset,
            x="max_steps",
            y="jailbreak_rate",
            hue="source",
            hue_order=hue_order, # Ensure consistent order
            order=all_max_steps, # Ensure consistent x-axis order
            capsize=0.1, # Smaller capsize for potentially crowded bars
            errwidth=1.5,
            ci=None # We will draw manual error bars
        )

        # Add manual error bars for Jailbreak Rate
        for i, source_label in enumerate(hue_order):
           source_subset = subset[subset["source"] == source_label]
           for _, row in source_subset.iterrows():
                if pd.isna(row["max_steps"]) or pd.isna(row["jailbreak_rate"]) or pd.isna(row["jb_ci_lower"]) or pd.isna(row["jb_ci_upper"]):
                    continue # Skip if any necessary value is NaN
                # Calculate x position: center of the group + offset for hue
                x_center = step_to_xpos[row["max_steps"]]
                x_pos = x_center - bar_width / 2 + (i + 0.5) * bar_width / len(hue_order)

                y_err_lower = row["jailbreak_rate"] - row["jb_ci_lower"]
                y_err_upper = row["jb_ci_upper"] - row["jailbreak_rate"]

                # Ensure errors are non-negative
                y_err_lower = max(0, y_err_lower)
                y_err_upper = max(0, y_err_upper)

                plt.errorbar(
                    x=x_pos,
                    y=row["jailbreak_rate"],
                    yerr=[[y_err_lower], [y_err_upper]],
                    fmt='none',
                    c='black',
                    capsize=4 # Smaller capsize for error bars
                )

        plt.title(f"{suffix_length}-Token Universal Soft-Suffix Attack Results (95% CI)", fontsize=14)
        plt.xlabel("MAX STEPS (Attack Iterations)", fontsize=12)
        plt.ylabel("Jailbreak Rate", fontsize=12)
        plt.ylim(0, 1.05) # Extend ylim slightly
        plt.xticks(ticks=range(len(all_max_steps)), labels=all_max_steps) # Ensure all steps are labeled
        plt.legend(title="System", title_fontsize='13', fontsize='11')
        plt.tight_layout() # Adjust layout
        jb_plot_filename = f"oat_evaluation/scripts/plots/jailbreak_rate_comparison_plot_{suffix_length}.png"
        plt.savefig(jb_plot_filename)
        print(f"Saved Jailbreak Rate plot: {jb_plot_filename}")
        plt.close() # Close the figure


        # --- Plot 2: SR Score ---
        plt.figure(figsize=(12, 7)) # Slightly wider figure
        ax2 = sns.barplot(
            data=subset,
            x="max_steps",
            y="sr_score",
            hue="source",
            hue_order=hue_order, # Ensure consistent order
            order=all_max_steps, # Ensure consistent x-axis order
            capsize=0.1, # Smaller capsize
            errwidth=1.5,
            ci=None # We will draw manual error bars
        )

        # Add manual error bars for SR Score
        for i, source_label in enumerate(hue_order):
           source_subset = subset[subset["source"] == source_label]
           for _, row in source_subset.iterrows():
                if pd.isna(row["max_steps"]) or pd.isna(row["sr_score"]) or pd.isna(row["sr_ci_low"]) or pd.isna(row["sr_ci_high"]):
                    continue # Skip if any necessary value is NaN
                # Calculate x position: center of the group + offset for hue
                x_center = step_to_xpos[row["max_steps"]]
                x_pos = x_center - bar_width / 2 + (i + 0.5) * bar_width / len(hue_order)

                y_err_lower = row["sr_score"] - row["sr_ci_low"]
                y_err_upper = row["sr_ci_high"] - row["sr_score"]

                # Ensure errors are non-negative
                y_err_lower = max(0, y_err_lower)
                y_err_upper = max(0, y_err_upper)

                plt.errorbar(
                    x=x_pos,
                    y=row["sr_score"],
                    yerr=[[y_err_lower], [y_err_upper]],
                    fmt='none',
                    c='black',
                    capsize=4 # Smaller capsize for error bars
                )

        plt.title(f"{suffix_length}-Token Attack: Avg SR Score w/ Probe (95% CI)", fontsize=14)
        plt.xlabel("MAX STEPS (Attack Iterations)", fontsize=12)
        plt.ylabel("Avg SR Score", fontsize=12)
        # Let y-axis scale automatically for SR score unless you know the typical range
        # plt.ylim(0, max(subset['sr_ci_high'].max() * 1.1, 0.1)) # Example: auto-scale with some padding
        plt.xticks(ticks=range(len(all_max_steps)), labels=all_max_steps) # Ensure all steps are labeled
        plt.legend(title="System", title_fontsize='13', fontsize='11')
        plt.tight_layout() # Adjust layout
        sr_plot_filename = f"oat_evaluation/scripts/plots/sr_score_comparison_plot_{suffix_length}.png"
        plt.savefig(sr_plot_filename)
        print(f"Saved SR Score plot: {sr_plot_filename}")
        plt.close() # Close the figure

    print("Processing complete.")