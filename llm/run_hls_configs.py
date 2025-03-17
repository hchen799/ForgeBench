import os
import subprocess
def run_hls_on_dirs(base_dir="hls_files"):
    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' does not exist. Nothing to run.")
        return
    
    all_results = []
    
    for run_dir in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, run_dir)
        tcl_script = os.path.join(run_path, "run_hls.tcl")
        report_path = os.path.join(run_path, "project_1/solution1/syn/report/top_csynth.rpt")

        if os.path.isfile(tcl_script):
            print(f"Running HLS flow in {run_path}...")
            result = subprocess.run(["vitis_hls", "-f", "run_hls.tcl"], cwd=run_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # log_file = os.path.join(run_path, "hls_run.log")
            # with open(log_file, "w") as f:
            #     f.write(result.stdout)
            #     f.write("\n--- STDERR ---\n")
            #     f.write(result.stderr)
            log_file = os.path.join(run_path, "vitis_hls.log")
            if result.returncode == 0:
                print(f"HLS flow completed successfully in {run_path}.")
            else:
                print(f"HLS flow failed in {run_path}. Check {log_file} for details.")
        else:
            print(f"No 'run_hls.tcl' found in {run_path}, skipping...")

if __name__ == "__main__":
    run_hls_on_dirs()
