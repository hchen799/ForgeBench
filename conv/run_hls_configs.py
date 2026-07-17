import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm


def run_hls(run_path):
    tcl_script = os.path.join(run_path, "run_hls.tcl")
    if not os.path.isfile(tcl_script):
        return f"Skipped: {run_path} (No TCL file)"

    try:
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


def run_hls_on_dirs(base_dir="large_hls_files"):
    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' does not exist.")
        return

    run_dirs = [
        os.path.join(base_dir, d)
        for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    MAX_CONCURRENT_JOBS = 64
    all_results = []
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as ex:
        futures = [ex.submit(run_hls, run) for run in run_dirs]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            all_results.append(fut.result())


if __name__ == "__main__":
    start = time.time()
    run_hls_on_dirs("large_hls_files")
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds.")
