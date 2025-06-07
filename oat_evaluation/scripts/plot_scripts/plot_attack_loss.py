import re
import matplotlib.pyplot as plt

log_path = "/workspace/GIT_SHENANIGANS/oat-2025/oat_evaluation/outputs/eval_logs/20250525_192850_gpu0.log"

def parse_log_runs(log_path):
    runs = []
    current_run = []
    last_step = -1
    current_step = 0

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Reset step when encountering attack training marker
            if "=== ATTACK TRAINING ===" in line:
                if current_run:
                    runs.append(current_run)
                    current_run = []
                current_step = 0
                last_step = -1
                continue
                
            loss = None
            if "Buffer min loss" in line:
                match_loss = re.search(r"Buffer min loss: ([\d.]+)", line)
                if match_loss:
                    loss = float(match_loss.group(1))
            if "GCG current_loss: " in line:
                match_loss = re.search(r"GCG current_loss: ([\d.]+)", line)
                if match_loss:
                    loss = float(match_loss.group(1))
            
            if loss is not None:
                step = None
                if "Step" in line:
                    match_step = re.search(r"Step (\d+):", line)
                    if match_step:
                        step = int(match_step.group(1))

                # If no step found, increment current_step
                if step is None:
                    current_step += 1
                    step = current_step
                else:
                    current_step = step

                # Check if step decreased — implies new run
                if step < last_step and current_run:
                    runs.append(current_run)
                    current_run = []

                current_run.append((step, loss))
                last_step = step

    if current_run:
        runs.append(current_run)

    return runs

def plot_runs(runs):
    # Filter runs with more than 1000 steps
    valid_runs = [run for run in runs if max(steps for steps, _ in run) > 99]
    n_runs = len(valid_runs)
    
    if n_runs == 0:
        print("No runs with more than 1000 steps found")
        return
        
    # Calculate grid dimensions
    n_cols = min(4, n_runs)  # Max 4 columns
    n_rows = (n_runs + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 5 * n_rows))
    
    for i, run in enumerate(valid_runs):
        plt.subplot(n_rows, n_cols, i + 1)
        steps, losses = zip(*run)
        plt.plot(steps, losses, marker='o')
        plt.title(f"Run {i+1}")
        plt.xlabel("Step")
        plt.ylabel("Buffer Min Loss")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("all_runs_loss_plot_sweep_gcg.png")
    plt.close()

if __name__ == "__main__":
    runs = parse_log_runs(log_path)
    plot_runs(runs)
