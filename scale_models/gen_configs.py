import os
import json
import shutil
from production_types import REVISION, annotate_storage, arithmetic, c_type
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

def run_hls_flow(config_path, base_dir="runs", FPGA_name="xczu9eg-ffvb1156-2-e", clock_period=10, task=["csynth",]):
    config = load_json_config(config_path)
    run_name = os.path.splitext(os.path.basename(config_path))[0]
    run_dir = create_run_directory(run_name, base_dir)
    
    ops = config["ops"]
    brams, drams = annotate_storage(config["brams"], config["drams"], ops)
    output_dram_names = config["output_dram_names"]
    data_type = c_type(config.get("data_type", "ap_fixed<16, 5>"))     # generic spellings (fixed<16,5,rnd,sat>) -> Vitis type
    arith = arithmetic(config)
    top_func_name = config.get("top_func_name", "top")
    fpga_name = config.get("FPGA_name", FPGA_name)
    clock= config.get("clock_period", clock_period)
    tasks = config.get("task", task)
    config.update(brams=brams, drams=drams, production_revision=REVISION)
    if config.get("data_type") not in (None, data_type):         # resolved config records the C++ type; keep the JSON spelling too
        config.update(data_type_spec=config["data_type"], data_type=data_type)
    # Validate and construct the code before replacing any saved project file.
    top_code = generate_top_function(brams, drams, ops, data_type, top_func_name, arith)
    top_h_code = generate_top_h(drams, data_type, top_func_name)
    tb_code = generate_testbench_code(drams, output_dram_names, data_type, top_func_name)

    # Generate and save files in run directory
    with open(os.path.join(run_dir, "top.cpp"), "w") as f:
        f.write(top_code)
    
    with open(os.path.join(run_dir, "top.h"), "w") as f:
        f.write(top_h_code)
    
    with open(os.path.join(run_dir, "tb_top.cpp"), "w") as f:
        f.write(tb_code)
    
    # generate_dram_txt_files(drams, seed=42)
    # for dram in drams:
    #     dram_txt = f"{dram['name']}.txt"
    #     if os.path.exists(dram_txt):
    #         shutil.move(dram_txt, os.path.join(run_dir, dram_txt))
    
    generate_full_tcl_file(drams, fpga_name, clock, tasks, output_filename=os.path.join(run_dir, "run_hls.tcl"))
    with open(os.path.join(run_dir, "resolved_config.json"), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Generated files for {run_name} in {run_dir}")


if __name__ == "__main__":
    test_case_dir = "auto_generated_configs"
    base_output_dir = "hls_files"
    os.makedirs(base_output_dir, exist_ok=True)
    
    for file in os.listdir(test_case_dir):
        if file.endswith(".json"):
            config_path = os.path.join(test_case_dir, file)
            task = ["csynth", "export_ip"]
            run_hls_flow(config_path, base_output_dir, task=task)
