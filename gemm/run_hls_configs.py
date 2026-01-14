import os
import subprocess
import concurrent.futures
import time
from tqdm import tqdm

def run_hls(run_path):
    tcl_script = os.path.join(run_path, "run_hls.tcl")
    if not os.path.isfile(tcl_script):
        return f"Skipped: {run_path} (No TCL file)"
    
    try:
        # Using subprocess.run. 
        # Note: capture_output=True is a cleaner way to write stdout=PIPE, stderr=PIPE
        result = subprocess.run(
            ["vitis_hls", "-f", "run_hls.tcl"], 
            cwd=run_path, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            return f"Success: {run_path}"
        else:
            return f"Failed: {run_path}"
    except Exception as e:
        return f"Error in {run_path}: {str(e)}"

def run_hls_on_dirs(base_dir="hls_files"):
    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' does not exist.")
        return
    
    # Gather all valid directories
    run_dirs = [
        os.path.join(base_dir, d) 
        for d in sorted(os.listdir(base_dir)) 
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    # limit concurrency to 50 here
    MAX_CONCURRENT_JOBS = 50 
    
    # ThreadPoolExecutor handles the queueing for you
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        # We use list() to force the iterator to consume and tqdm to track progress
        list(tqdm(executor.map(run_hls, run_dirs), total=len(run_dirs), desc="Processing HLS configs"))

if __name__ == "__main__":
    start = time.time()
    run_hls_on_dirs("large_hls_files")
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds.")