import argparse
import subprocess

def run_csim():
    # Replace with your actual csim command and script
    command = ["vitis_hls", "-f", "csim_script.tcl"]
    subprocess.run(command)

def run_csynth():
    # Replace with your actual csynth command and script
    command = ["vitis_hls", "-f", "csynth_script.tcl"]
    subprocess.run(command)

def run_cosim():
    # Replace with your actual cosim command and script
    command = ["vitis_hls", "-f", "cosim_script.tcl"]
    subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(
        description="Launch HLS process: csim, csynth, or cosim based on user input."
    )
    parser.add_argument(
        "process", 
        choices=["csim", "csynth", "cosim"],
        help="Specify which HLS process to run: csim, csynth, or cosim."
    )
    args = parser.parse_args()

    if args.process == "csim":
        run_csim()
    elif args.process == "csynth":
        run_csynth()
    elif args.process == "cosim":
        run_cosim()

if __name__ == "__main__":
    main()
