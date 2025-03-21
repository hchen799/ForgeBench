# /// script
# dependencies = [
#     "pandas",
# ]
# ///

import os
import xml.etree.ElementTree as ET
import pandas as pd



HLS_DIR = "hls_files/"
OUTPUT_CSV = "hls_utilization.csv"

def parse_utilization(xml_path, function_name=None, project_name=None):
    if not os.path.exists(xml_path):
        print(f"Warning: XML file '{xml_path}' not found.")
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract overall utilization
    area_report = root.find("AreaReport")
    if area_report is None:
        print(f"Warning: AreaReport not found in '{xml_path}'.")
        return None

    resources = area_report.find("Resources")
    available_resources = area_report.find("AvailableResources")

    if resources is None or available_resources is None:
        print(f"Warning: Resources or AvailableResources missing in '{xml_path}'.")
        return None

    overall_dsp = int(resources.find("DSP").text)
    overall_lut = int(resources.find("LUT").text)
    total_dsp = int(available_resources.find("DSP").text)
    total_lut = int(available_resources.find("LUT").text)

    utilization = {
        "Project": project_name,
        "Function": "Overall",
        "DSP_Utilization": overall_dsp / total_dsp if total_dsp else 0,
        "LUT_Utilization": overall_lut / total_lut if total_lut else 0,
    }

    results = [utilization]

    # Extract function-specific utilization if provided
    if function_name:
        rtl_modules = root.find("RtlModules")
        if rtl_modules is not None:
            for module in rtl_modules.findall("RtlModule"):
                if f"inst/grp_{function_name}_fu_" in module.get("CELL", ""):
                    func_resources = module.find("Resources")
                    if func_resources is not None:
                        func_dsp = int(func_resources.get("DSP", 0))
                        func_lut = int(func_resources.get("LUT", 0))
                        results.append({
                            "Project": project_name,
                            "Function": function_name,
                            "DSP_Utilization": func_dsp / total_dsp if total_dsp else 0,
                            "LUT_Utilization": func_lut / total_lut if total_lut else 0,
                        })
                    break

    return results

def process_all_hls_projects(base_dir, function_dict=None):
    """
    Traverses all subdirectories in the given base directory and extracts utilization stats.
    function_dict: Dictionary where keys are subdirectory names and values are function names to parse.
    """
    all_results = []

    for subdir in os.listdir(base_dir):
        project_path = os.path.join(base_dir, subdir, "project_1/solution1/impl/report/verilog/export_impl.xml")
        if os.path.exists(project_path):
            print(f"Processing {subdir}...")
            function_name = function_dict.get(subdir) if function_dict else None
            utilization = parse_utilization(project_path, function_name, subdir)
            if utilization:
                all_results.extend(utilization)
        else:
            print(f"Warning: XML file not found for '{subdir}'.")

    return all_results

# Define function names for specific subdirectories if needed
function_mappings = {
    "diff_dims_module_small": "tiled_gemm",
    "diff_dims_module_large": "tiled_gemm",
    "mult_op_module_dot": "dot_product",
    "mult_op_module_mmv": "mmv_ij",
    "mult_op_module_mm": "gemm_ijk",
    "diff_orders_module": "common_gemm_ikj",
    "vec_mtx_module": "mmv_ij",
    # Add more mappings as needed
}

# Run processing for all projects
data = process_all_hls_projects(HLS_DIR, function_mappings)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"Utilization data saved to {OUTPUT_CSV}")
