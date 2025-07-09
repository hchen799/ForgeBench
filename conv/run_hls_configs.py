import os
import subprocess
import concurrent.futures
import time

def run_hls(run_path):
    tcl_script = os.path.join(run_path, "run_hls.tcl")
    if os.path.isfile(tcl_script):
        print(f"Running HLS flow in {run_path}...")
        result = subprocess.run(["vitis_hls", "-f", "run_hls.tcl"], cwd=run_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_file = os.path.join(run_path, "vitis_hls.log")
        if result.returncode == 0:
            print(f"HLS flow completed successfully in {run_path}.")
        else:
            print(f"HLS flow failed in {run_path}. Check {log_file} for details.")
    else:
        print(f"No 'run_hls.tcl' found in {run_path}, skipping...")

def run_hls_on_dirs(base_dir="hls_files_FPT"):
    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' does not exist. Nothing to run.")
        return
    
    run_dirs = [os.path.join(base_dir, d)
                for d in sorted(os.listdir(base_dir))
                if os.path.isdir(os.path.join(base_dir, d))]
    
    # cap the number of simultaneous workers at 64
    max_workers = 8
    num_workers = min(max_workers, len(run_dirs))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # this will schedule all of run_dirs,
        # but only run `num_workers` at a time
        executor.map(run_hls, run_dirs)
if __name__ == "__main__":
    start = time.time()
    run_hls_on_dirs()
    end = time.time()
    print(f"Total time taken: {end - start} seconds.")
