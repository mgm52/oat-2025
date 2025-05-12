from oat_evaluation.utils import load_config_and_set_vars
load_config_and_set_vars()

import wandb
import re
import os
import numpy as np # For np.mean, though values are already means in logs
from datetime import datetime

# --- Configuration ---
WANDB_ENTITY = "mgm52"  # Replace with your W&B entity (username or team)
WANDB_PROJECT = "oat_evaluation"  # Replace with your W&B project name
GROUP_NAME = "20250511_043048_3142425_big_eval" # Original group to read from
NEW_GROUP_NAME_SUFFIX = "_repaired" # Suffix for the new group where parsed logs will be stored

# Set your W&B API key as an environment variable:
# export WANDB_API_KEY="your_api_key"
# Or login via `wandb login`

# *** Path to local W&B run logs ***
LOCAL_LOG_BASE_PATH = "/workspace/GIT_SHENANIGANS/oat-2025/wandb" # <--- SET THIS PATH

# --- Regular Expressions ---
timestamp_prefix = r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\]\s*"
optional_lineno_prefix = r"(?:\d+\s+[\d\-:\.]+\s+)?"

step_pattern = re.compile(timestamp_prefix + r"Executing callback <lambda> for step (\d+)...")
flop_pattern = re.compile(timestamp_prefix + r"About to run evaluation... Current attack FLOP cost: (\d+)")
summary_start_pattern = re.compile(optional_lineno_prefix + r"=== SUMMARY ===")
sr_pattern = re.compile(optional_lineno_prefix + r"Avg SR score: ([\d.]+) \[95% CI: ([\d.]+), ([\d.]+)\]")
refusal_pattern = re.compile(optional_lineno_prefix + r"Refusal rate: ([\d.]+) \[95% CI: ([\d.]+), ([\d.]+)\]")
sr_probe_pattern = re.compile(optional_lineno_prefix + r"Avg SR score w/ probe: ([\d.]+) \[95% CI: ([\d.]+), ([\d.]+)\]")
refusal_probe_pattern = re.compile(optional_lineno_prefix + r"Refusal rate w/ probe: ([\d.]+) \[95% CI: ([\d.]+), ([\d.]+)\]")

def parse_log_and_create_new_run(original_run, new_group_name):
    print(f"Processing original run: {original_run.name} (ID: {original_run.id})")
    log_file_path = ""
    try:
        # Construct the local log file path from the original run's metadata
        try:
            created_at_str = original_run.createdAt
            if created_at_str.endswith('Z'): # Handle UTC 'Z' for datetime.fromisoformat
                 created_at_str = created_at_str[:-1] + '+00:00'
            dt_obj = datetime.fromisoformat(created_at_str)
            formatted_ts = dt_obj.strftime("%Y%m%d_%H%M%S")
        except (ValueError, AttributeError) as e_dt:
            print(f"  Error parsing original_run.createdAt timestamp '{original_run.createdAt}' for run {original_run.id}: {e_dt}. Skipping.")
            return 0, None # Return blocks_logged_count and new_run_id

        run_dir_name = f"run-{formatted_ts}-{original_run.id}"
        log_file_path = os.path.join(LOCAL_LOG_BASE_PATH, run_dir_name, "files", "output.log")

        if not os.path.exists(log_file_path):
            print(f"  Local log file not found: {log_file_path}. Skipping original run {original_run.id}.")
            return 0, None
        
        print(f"  Reading local log file: {log_file_path}")
        with open(log_file_path, "r", encoding="utf-8") as f:
            log_lines = f.readlines()

    except Exception as e:
        print(f"  Error accessing or reading local log for original run {original_run.id} (path: {log_file_path}): {e}. Skipping.")
        return 0, None

    metrics_to_commit_for_run = []
    current_attack_step = None
    flop_cost = None
    temp_metrics = {}
    state = 0

    for line_num, line in enumerate(log_lines):
        line = line.strip()
        # State machine logic (identical to previous version)
        if state == 0:
            match = step_pattern.search(line)
            if match:
                current_attack_step = int(match.group(1))
                temp_metrics = {"eval/current_attack_step": current_attack_step}
                flop_cost = None
                state = 1
        elif state == 1:
            match = flop_pattern.search(line)
            if match:
                flop_cost = int(match.group(1))
                temp_metrics["eval/flop_cost"] = flop_cost
                state = 2
            else:
                step_match_reset = step_pattern.search(line)
                if step_match_reset:
                    print(f"  L{line_num+1}: Resetting: New step found before FLOP for step {current_attack_step}")
                    current_attack_step = int(step_match_reset.group(1))
                    temp_metrics = {"eval/current_attack_step": current_attack_step}
                    flop_cost = None
                    state = 1
        elif state == 2:
            if summary_start_pattern.search(line):
                state = 3
            else:
                step_match_reset = step_pattern.search(line)
                if step_match_reset:
                    print(f"  L{line_num+1}: Resetting: New step found before SUMMARY for step {current_attack_step}")
                    current_attack_step = int(step_match_reset.group(1))
                    temp_metrics = {"eval/current_attack_step": current_attack_step}
                    flop_cost = None
                    state = 1
        elif state == 3:
            match = sr_pattern.search(line)
            if match:
                temp_metrics["eval/avg_sr_score_attacked"] = float(match.group(1))
                # ... (rest of SR metrics)
                state = 4
            else:
                if step_pattern.search(line) or flop_pattern.search(line) or summary_start_pattern.search(line):
                    print(f"  L{line_num+1}: Resetting: Unexpected line while expecting SR score for step {current_attack_step}")
                    state = 0; temp_metrics = {}; current_attack_step = None; flop_cost = None
                    match_step = step_pattern.search(line)
                    if match_step:
                        current_attack_step = int(match_step.group(1))
                        temp_metrics = {"eval/current_attack_step": current_attack_step}
                        state = 1
        elif state == 4:
            match = refusal_pattern.search(line)
            if match:
                a_refusal_rate = float(match.group(1))
                temp_metrics["eval/refusal_rate_attacked"] = a_refusal_rate
                temp_metrics["eval/jailbreak_rate_attacked"] = 1.0 - a_refusal_rate
                # ... (rest of refusal metrics)
                state = 5
            else:
                if step_pattern.search(line) or flop_pattern.search(line) or summary_start_pattern.search(line):
                    print(f"  L{line_num+1}: Resetting: Unexpected line while expecting Refusal rate for step {current_attack_step}")
                    state = 0; temp_metrics = {}; current_attack_step = None; flop_cost = None
                    match_step = step_pattern.search(line)
                    if match_step:
                        current_attack_step = int(match_step.group(1))
                        temp_metrics = {"eval/current_attack_step": current_attack_step}
                        state = 1
        elif state == 5:
            match = sr_probe_pattern.search(line)
            if match:
                temp_metrics["eval/avg_sr_score_probe_attacked"] = float(match.group(1))
                # ... (rest of SR probe metrics)
                state = 6
            else:
                if step_pattern.search(line) or flop_pattern.search(line) or summary_start_pattern.search(line):
                    print(f"  L{line_num+1}: Resetting: Unexpected line while expecting SR w/ probe for step {current_attack_step}")
                    state = 0; temp_metrics = {}; current_attack_step = None; flop_cost = None
                    match_step = step_pattern.search(line)
                    if match_step:
                        current_attack_step = int(match_step.group(1))
                        temp_metrics = {"eval/current_attack_step": current_attack_step}
                        state = 1
        elif state == 6:
            match = refusal_probe_pattern.search(line)
            if match:
                a_adjusted_refusal_rate = float(match.group(1))
                temp_metrics["eval/refusal_rate_probe_attacked"] = a_adjusted_refusal_rate
                temp_metrics["eval/jailbreak_rate_probe_attacked"] = 1.0 - a_adjusted_refusal_rate
                # ... (rest of refusal probe metrics)
                if current_attack_step is not None and "eval/flop_cost" in temp_metrics:
                    metrics_to_commit_for_run.append((current_attack_step, temp_metrics.copy()))
                    print(f"  L{line_num+1}: Successfully parsed block for step {current_attack_step}. Metrics count: {len(temp_metrics)}")
                else:
                    print(f"  L{line_num+1}: Parsed final metric but step/FLOP was missing. Block invalid. Step: {current_attack_step}, FLOP present: {'eval/flop_cost' in temp_metrics}")
                state = 0
                current_attack_step = None
                flop_cost = None
                temp_metrics = {}
            else:
                if step_pattern.search(line) or flop_pattern.search(line) or summary_start_pattern.search(line):
                    print(f"  L{line_num+1}: Resetting: Unexpected line while expecting Refusal w/ probe for step {current_attack_step}")
                    state = 0; temp_metrics = {}; current_attack_step = None; flop_cost = None
                    match_step = step_pattern.search(line)
                    if match_step:
                        current_attack_step = int(match_step.group(1))
                        temp_metrics = {"eval/current_attack_step": current_attack_step}
                        state = 1

    if not metrics_to_commit_for_run:
        print(f"  No valid metric blocks found in logs for original run {original_run.id}. No new run will be created.")
        return 0, None

    # Initialize a NEW W&B run
    new_run_name = f"{original_run.name}{NEW_GROUP_NAME_SUFFIX}" if original_run.name else f"{original_run.id}{NEW_GROUP_NAME_SUFFIX}"
    
    # Prepare tags for the new run
    new_tags = original_run.tags.copy() if original_run.tags else []
    new_tags.append("repaired_log_data")
    if NEW_GROUP_NAME_SUFFIX.strip("_"): # Add suffix as a tag if it's meaningful
        new_tags.append(NEW_GROUP_NAME_SUFFIX.strip("_"))


    print(f"  Creating new W&B run '{new_run_name}' in group '{new_group_name}' to log {len(metrics_to_commit_for_run)} metric blocks.")
    
    new_run_logger = None
    try:
        if os.environ.get("WANDB_MODE") in ["offline", "disabled"]:
            print("  WANDB_MODE is offline or disabled. Cannot log metrics to a new run. Printing instead.")
            print(f"    Original Run ID: {original_run.id}")
            print(f"    New Run Name (would be): {new_run_name}")
            print(f"    New Group Name (would be): {new_group_name}")
            print(f"    Config (from original): {original_run.config}")
            for step, metrics_dict in metrics_to_commit_for_run:
                 print(f"      Would log for attack step {step}: {metrics_dict}")
            return len(metrics_to_commit_for_run), f"offline_run_for_{original_run.id}"


        new_run_logger = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            group=new_group_name,
            name=new_run_name,
            config=original_run.config, # Copy config from the original run
            tags=new_tags,
            notes=f"Repaired log data from original run: ID='{original_run.id}', Name='{original_run.name}'.\nOriginal notes: {original_run.notes}"
            # No 'id' or 'resume' - this creates a new run
        )
        for step, metrics_dict in metrics_to_commit_for_run:
            # Log with the 'current_attack_step' as the W&B step if you want plots vs attack_step
            new_run_logger.log(metrics_dict, step=step) 
            print(f"    Logged metrics for attack step {step} to new run {new_run_logger.id}")
        
        new_run_id = new_run_logger.id
        new_run_logger.finish()
        print(f"  Successfully created new run {new_run_id} and logged {len(metrics_to_commit_for_run)} blocks.")
        return len(metrics_to_commit_for_run), new_run_id
    
    except Exception as e:
        print(f"  Error initializing or logging to new W&B run for original run {original_run.id}: {e}")
        if new_run_logger: # If init succeeded but log/finish failed
            new_run_logger.finish(exit_code=1) # Mark as crashed
        return 0, None


def main():
    if not WANDB_ENTITY or WANDB_ENTITY == "YOUR_ENTITY":
        print("Please set WANDB_ENTITY in the script.")
        return
    if not WANDB_PROJECT or WANDB_PROJECT == "YOUR_PROJECT_NAME":
        print("Please set WANDB_PROJECT in the script.")
        return
    if not LOCAL_LOG_BASE_PATH or LOCAL_LOG_BASE_PATH == "/path/to/your/local/wandb/logs":
        print("Please set LOCAL_LOG_BASE_PATH to the root of your local wandb run directories.")
        return
    if "WANDB_API_KEY" not in os.environ and os.environ.get("WANDB_MODE") not in ["offline", "disabled"]:
        print("WANDB_API_KEY environment variable not set. Run `wandb login` or set WANDB_MODE=offline.")
        # Allow to continue if WANDB_MODE is offline, but print warning.

    api = wandb.Api()
    
    print(f"Fetching original runs from group: '{GROUP_NAME}' in project {WANDB_ENTITY}/{WANDB_PROJECT}")
    try:
        original_runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": GROUP_NAME})
    except Exception as e:
        print(f"Error fetching original runs: {e}")
        return

    total_original_runs = len(original_runs)
    print(f"Found {total_original_runs} original runs in group '{GROUP_NAME}'.")
    
    if total_original_runs == 0:
        print("No original runs found. Exiting.")
        return

    new_group_name = f"{GROUP_NAME}{NEW_GROUP_NAME_SUFFIX}"
    print(f"Parsed metrics will be logged to new runs in group: '{new_group_name}'")

    total_blocks_logged_all_new_runs = 0
    new_runs_created_count = 0
    processed_original_runs_count = 0

    for original_run in original_runs:
        processed_original_runs_count += 1
        print(f"\n--- Processing original run {processed_original_runs_count}/{total_original_runs} ---")
        
        # Pass the original run object and the target new group name
        blocks_logged_for_this_new_run, new_run_id = parse_log_and_create_new_run(original_run, new_group_name)
        
        if new_run_id: # If a new run was successfully created (or would have been in offline mode)
            total_blocks_logged_all_new_runs += blocks_logged_for_this_new_run
            new_runs_created_count += 1
        
    print(f"\n--- Summary ---")
    print(f"Processed {processed_original_runs_count} original runs from group '{GROUP_NAME}'.")
    print(f"Created {new_runs_created_count} new runs in group '{new_group_name}'.")
    print(f"Total metric blocks logged across all new runs: {total_blocks_logged_all_new_runs}")

if __name__ == "__main__":
    main()