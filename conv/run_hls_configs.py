import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op progress wrapper
    def tqdm(x, **kwargs):
        return x


def _prune_autopilot(run_path):
    """Delete the ~25 MB/design .autopilot build DB once synth is done.

    Prune-as-we-go: a full sweep leaves hundreds of GB of these intermediates
    and previously filled the scratch disk to 100%. The reports we extract live
    under solution1/{syn,impl}/report, not here, so dropping .autopilot as each
    design completes is lossless and bounds peak usage to the ~64 live workers.
    """
    ap = os.path.join(run_path, "project_1", "solution1", ".autopilot")
    if os.path.isdir(ap):
        shutil.rmtree(ap, ignore_errors=True)


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
        _prune_autopilot(run_path)
        if result.returncode == 0:
            return f"Success: {run_path}"
        else:
            return f"Failed: {run_path}"
    except Exception as e:
        _prune_autopilot(run_path)
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
