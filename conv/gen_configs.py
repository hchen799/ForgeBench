import os
import json
import shutil
from generate_code import (
    generate_top_function,
    generate_top_h,
    generate_testbench_code,
    generate_dram_txt_files,
    generate_full_tcl_file
)

def load_json_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_run_directory(run_name, base_dir="runs"):
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def run_hls_flow(config_path, base_dir="runs", FPGA_name="xczu9eg-ffvb1156-2-e", clock_period=10, task=["csynth", "export_ip"]):
    config = load_json_config(config_path)
    run_name = os.path.splitext(os.path.basename(config_path))[0]
    run_dir = create_run_directory(run_name, base_dir)
    
    brams = config["brams"]
    drams = config["drams"]
    ops = config["ops"]
    output_dram_names = config["output_dram_names"]
    data_type = config.get("data_type", "ap_fixed<16, 5>")
    top_func_name = config.get("top_func_name", "top")
    
    # Generate and save files in run directory
    top_code = generate_top_function(brams, drams, ops, data_type, top_func_name)
    with open(os.path.join(run_dir, "top.cpp"), "w") as f:
        f.write(top_code)
    
    top_h_code = generate_top_h(drams, data_type, top_func_name)
    with open(os.path.join(run_dir, "top.h"), "w") as f:
        f.write(top_h_code)
    
    tb_code = generate_testbench_code(drams, output_dram_names, data_type, top_func_name)
    with open(os.path.join(run_dir, "tb_top.cpp"), "w") as f:
        f.write(tb_code)
    
    generate_dram_txt_files(drams, seed=42)
    for dram in drams:
        dram_txt = f"{dram['name']}.txt"
        if os.path.exists(dram_txt):
            shutil.move(dram_txt, os.path.join(run_dir, dram_txt))
    
    generate_full_tcl_file(drams, FPGA_name, clock_period, task, output_filename=os.path.join(run_dir, "run_hls.tcl"))
    print(f"Generated files for {run_name} in {run_dir}")

if __name__ == "__main__":
    test_case_dir = "test_case_configs"
    base_output_dir = "hls_files"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # for file in os.listdir(test_case_dir):
    for file in ["conv_A.json", "conv_B.json", "conv_C.json", "config_CIN16_HIN28_WIN28_COUT16_K1_IPF1_KPF11_KPF21_BPF1_OPF1_UFCIN1_UFCOU1_ap_fixed_16_5_.json"]:
        if file.endswith(".json"):
            config_path = os.path.join(test_case_dir, file)
            task = ["csynth"]
            run_hls_flow(config_path, base_output_dir, task=task)