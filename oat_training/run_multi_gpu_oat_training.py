from oat_evaluation.utils import load_config_and_set_vars, print_mem_usage, print_timey
load_config_and_set_vars() # needs to be at top to take effect

import subprocess
import os
import torch
import datetime
import pathlib
import time # Added for polling delay
from collections import deque # Added for managing command queue

def get_available_gpus():
    """Get the number of available GPUs"""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()

def run_training_on_gpu(gpu_id, command, log_dir):
    """Run training on a specific GPU with given command"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create timestamp for unique log files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Kept original log file naming convention
    log_file = log_dir / f"training_{timestamp}_gpu{gpu_id}.log"
    
    # Modified print statement to include the command for better clarity
    print(f"Running training on GPU {gpu_id}. Command: {' '.join(command)}. Log file: {log_file}")
    
    # Open log file for writing
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=f,
            stderr=f,
            text=True
        )
    return process, log_file

def main(commands, USE_ALL_GPUS=True):
    """
    Run multiple training commands across available GPUs.
    If USE_ALL_GPUS is True, tasks are distributed dynamically to available GPUs.
    If USE_ALL_GPUS is False, tasks are run sequentially on GPU 0.
    
    Args:
        commands: List of command lists to execute (e.g., [["python", "script.py", "--arg1", "val1"], ...])
        USE_ALL_GPUS: Boolean indicating whether to use all GPUs or run sequentially on GPU 0.
    """
    num_gpus = get_available_gpus()
    
    log_dir = pathlib.Path("oat_training/oat_training_logs")
    log_dir.mkdir(parents=True, exist_ok=True) # Ensure parent directories exist
    
    if num_gpus == 0:
        print("No GPUs available.")
        if USE_ALL_GPUS:
            print("Cannot dynamically schedule on multiple GPUs as none are available.")
            print("Falling back to sequential execution (attempts to use GPU 0 or CPU if script supports).")
        # Force sequential mode if no GPUs. This mode attempts GPU 0, matching original behavior.
        USE_ALL_GPUS = False 
    
    if not USE_ALL_GPUS:
        # Sequential execution mode (either explicitly requested or due to no GPUs)
        target_gpu_for_sequential = 0 
        print(f"Running {len(commands)} tasks sequentially. Target device: GPU {target_gpu_for_sequential} (if available).")
        
        if not commands:
            print("No commands to run.")
            return

        for i, cmd in enumerate(commands):
            print(f"\nStarting sequential task {i+1}/{len(commands)} on GPU {target_gpu_for_sequential}.")
            process, log_file = run_training_on_gpu(target_gpu_for_sequential, cmd, log_dir)
            
            try:
                process.wait() # Wait for this process to complete before starting the next one
                if process.returncode == 0:
                    print(f"Task {i+1} completed successfully. Log file: {log_file}")
                else:
                    print(f"Task {i+1} FAILED with return code {process.returncode}. Log file: {log_file}")
            except Exception as e: # Catch potential errors during wait (e.g., KeyboardInterrupt)
                print(f"Error waiting for task {i+1} (command: {' '.join(cmd)}): {e}. Log file: {log_file}")
        
        print("\nAll sequential tasks processed.")
        return

    # Dynamic GPU scheduling mode (USE_ALL_GPUS is True and num_gpus > 0)
    total_tasks = len(commands)
    if total_tasks == 0:
        print("No commands to run.")
        return
        
    print(f"Dynamically distributing {total_tasks} tasks across {num_gpus} available GPUs.")
    
    # Store commands as (original_index, command_list) for better tracking
    commands_queue = deque([(cmd_idx, cmd) for cmd_idx, cmd in enumerate(commands)])
    # active_processes_on_gpus maps gpu_id to (process, log_file, original_cmd_idx, command_list)
    active_processes_on_gpus = {}  
    
    completed_task_count = 0

    while completed_task_count < total_tasks:
        # Check for completed processes
        gpus_that_finished_task = []
        for gpu_id, (process, log_file, cmd_idx, cmd_list) in active_processes_on_gpus.items():
            if process.poll() is not None:  # Process has finished
                if process.returncode == 0:
                    print(f"\nTask {cmd_idx+1}/{total_tasks} (GPU {gpu_id}, {' '.join(cmd_list)}) completed successfully.")
                else:
                    print(f"\nTask {cmd_idx+1}/{total_tasks} (GPU {gpu_id}, {' '.join(cmd_list)}) FAILED with code {process.returncode}.")
                print(f"Log file: {log_file}")
                
                gpus_that_finished_task.append(gpu_id)
                completed_task_count += 1
        
        # Remove completed processes and free up their GPUs
        for gpu_id in gpus_that_finished_task:
            del active_processes_on_gpus[gpu_id]

        # Assign new tasks to available GPUs
        # Loop while there are commands in the queue AND there are free GPU slots
        while commands_queue and len(active_processes_on_gpus) < num_gpus:
            assigned_gpu_id = -1
            # Find the first available GPU ID
            for i in range(num_gpus):
                if i not in active_processes_on_gpus:
                    assigned_gpu_id = i
                    break
            
            if assigned_gpu_id != -1: # If a free GPU was found
                cmd_idx, cmd_to_run = commands_queue.popleft()
                print(f"\nAssigning task {cmd_idx+1}/{total_tasks} to GPU {assigned_gpu_id}.")
                process, log_file = run_training_on_gpu(assigned_gpu_id, cmd_to_run, log_dir)
                active_processes_on_gpus[assigned_gpu_id] = (process, log_file, cmd_idx, cmd_to_run)
            else:
                # No free GPU slot found (all num_gpus are busy)
                break # Break from assigning more tasks in this cycle
        
        if completed_task_count == total_tasks:
            break # All tasks are done

        time.sleep(5) # Polling interval to check process statuses
    
    print("\nAll training commands have been processed.")

if __name__ == "__main__":
    commands = [
        # TRAINING PROBES ONLY
        # ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "8", "--start-adv-training-at-step", "9999999", "--freeze-model-during-warmup", "True",
        #  "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct"],

        # ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "1024", "--start-adv-training-at-step", "9999999", "--freeze-model-during-warmup", "True",
        #  "--model-path", "GraySwanAI/Llama-3-8B-Instruct-RR"],

         ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
         "--n-steps", "2048", "--start-adv-training-at-step", "1024", "--freeze-model-during-warmup", "False",
         "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct"],

         ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
         "--n-steps", "2048", "--start-adv-training-at-step", "1024", "--freeze-model-during-warmup", "False",
         "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--adapter-lr", str(float("2e-5"))],

         ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
         "--n-steps", "2048", "--start-adv-training-at-step", "1024", "--freeze-model-during-warmup", "False",
         "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--adapter-lr", str(float("1e-4"))],
        
         ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
         "--n-steps", "2048", "--start-adv-training-at-step", "1024", "--freeze-model-during-warmup", "False",
         "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--pgd-iterations", "16"],

         ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
         "--n-steps", "2048", "--start-adv-training-at-step", "1024", "--freeze-model-during-warmup", "False",
         "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct", "--pgd-iterations", "64"],

         ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "nonlinear", 
         "--n-steps", "2048", "--start-adv-training-at-step", "1024", "--freeze-model-during-warmup", "False",
         "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct"],

        #  ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "2048", "--start-adv-training-at-step", "1024", "--freeze-model-during-warmup", "False",
        #  "--model-path", "meta-llama/Llama-3.2-1B-Instruct"],

        #  ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
        #  "--n-steps", "2048", "--start-adv-training-at-step", "1024", "--freeze-model-during-warmup", "False",
        #  "--model-path", "meta-llama/Llama-3.2-3B-Instruct"],

        ["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", 
         "--n-steps", "2048", "--start-adv-training-at-step", "9999999", "--freeze-model-during-warmup", "False",
         "--model-path", "meta-llama/Meta-Llama-3-8B-Instruct"],

        # TRAINING PROBES AND MODEL
        #["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "linear", "--n-steps", "2048"],
        #["python", "oat_training/run_oat_training.py", "--masking-type", "generation", "--probe-type", "nonlinear", "--n-steps", "2048"],
    ]
    main(commands, USE_ALL_GPUS=True)