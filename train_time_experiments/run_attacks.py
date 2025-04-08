import os
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_attack(checkpoint_dir, step, max_num_eval_prompts):
    print(f"Running step {step} from {checkpoint_dir}")
    with subprocess.Popen([
        "python",
        "attack_evals.py",
        f"--checkpoint_dir={checkpoint_dir}",
        f"--adversary_checkpoint={step}",
        f"--max_num_eval_prompts={max_num_eval_prompts}"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            print(line, end='')  # Pipe stdout to the console
        stderr = process.stderr.read()
        returncode = process.wait()
    
    if returncode != 0:
        print(f"[ERROR] step {step} from {checkpoint_dir} failed:\n{stderr}")
    else:
        print(f"[DONE] step {step} from {checkpoint_dir} completed.")
    
    return returncode

# Prepare jobs
jobs = []
dirs = ["checkpoints/adversaries_obf", "checkpoints/adversaries_unobf"]

for dir_path in dirs:
    checkpoint_files = glob.glob(os.path.join(dir_path, "checkpoint_step_*.pt"))
    for filepath in sorted(checkpoint_files):
        filename = os.path.basename(filepath)
        step = filename[len("checkpoint_step_"):-len(".pt")]
        jobs.append((dir_path, filename))

max_num_eval_prompts = 128
# Run in parallel with up to 3 workers
with ProcessPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(run_attack, dir_path, filename, max_num_eval_prompts) for dir_path, filename in jobs]
    for future in as_completed(futures):
        future.result()  # to raise any exceptions
