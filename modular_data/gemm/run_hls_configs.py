import os
import subprocess
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
from joblib import Parallel, delayed

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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
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
        if os.path.isdir(os.path.join(base_dir, d)) and d == "diff_orders_module"
    ]

    # run_dirs = run_dirs[:100]  # Limit to first 1000 directories for testing

    # limit concurrency to 50 here
    MAX_CONCURRENT_JOBS = 50 
    all_results = []
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as ex:
        futures = []
        for run in run_dirs:
            futures.append(
                ex.submit(run_hls, run)
            )
        for fut in tqdm(as_completed(futures), total=len(futures)):
            all_results.append(fut.result())

    # ThreadPoolExecutor handles the queueing for you
    # with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
    #     # We use list() to force the iterator to consume and tqdm to track progress
    #     tqdm(executor.map(run_hls, run_dirs), total=len(run_dirs), desc="Processing HLS configs")

    # Parallel(n_jobs=MAX_CONCURRENT_JOBS, backend="multiprocessing")(
    #     delayed(run_hls)(run_path) for run_path in tqdm(run_dirs, desc="Processing HLS configs")
    # )

if __name__ == "__main__":
    start = time.time()
    run_hls_on_dirs("hls_files")
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds.")