# /// script
# dependencies = [
#     "pandas",
# ]
# ///

import os
import xml.etree.ElementTree as ET
import pandas as pd



HLS_DIR = "hls_files/"
OUTPUT_CSV = "hls_synth_utilization.csv"

def parse_utilization(xml_path, project_name=None, function_name=None):
    if not os.path.exists(xml_path):
        print(f"Warning: XML file '{xml_path}' not found.")
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract overall utilization
    area_report = root.find("AreaEstimates")
    if area_report is None:
        print(f"Warning: AreaEstimates not found in '{xml_path}'.")
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
        "Function": function_name if function_name else "Overall",
        "LUT_Utilization": (overall_lut / total_lut) * 100 if total_lut else 0,
        "DSP_Utilization": (overall_dsp / total_dsp) * 100 if total_dsp else 0,
    }

    results = [utilization]

    # # Extract function-specific utilization if provided
    # if function_name:
    #     rtl_modules = root.find("RtlModules")
    #     if rtl_modules is not None:
    #         for module in rtl_modules.findall("RtlModule"):
    #             if f"inst/grp_{function_name}_fu_" in module.get("CELL", ""):
    #                 func_resources = module.find("Resources")
    #                 if func_resources is not None:
    #                     func_dsp = int(func_resources.get("DSP", 0))
    #                     func_lut = int(func_resources.get("LUT", 0))
    #                     results.append({
    #                         "Project": project_name,
    #                         "Function": function_name,
    #                         "DSP_Utilization": (func_dsp / total_dsp) * 100 if total_dsp else 0,
    #                         "LUT_Utilization": (func_lut / total_lut) * 100 if total_lut else 0,
    #                     })
    #                 break

    return results

def process_all_hls_projects(base_dir, function_dict=None):
    """
    Traverses all subdirectories in the given base directory and extracts utilization stats.
    function_dict: Dictionary where keys are subdirectory names and values are function names to parse.
    """
    all_results = []

    for subdir in os.listdir(base_dir):
        project_path = os.path.join(base_dir, subdir, "project_1/solution1/syn/report/csynth.xml")
        if os.path.exists(project_path):
            print(f"Processing {subdir}...")
            function_name = function_dict.get(subdir) if function_dict else None
            utilization = parse_utilization(project_path, subdir)
            if utilization:
                all_results.extend(utilization)
            if function_name:
                function_path = os.path.join(base_dir, subdir, f"project_1/solution1/syn/report/{function_name}_csynth.xml")
                function_utilization = parse_utilization(function_path, subdir, function_name)
                if function_utilization:
                    all_results.extend(function_utilization)
        else:
            print(f"Warning: XML file not found for '{subdir}'.")

    return all_results

# Define function names for specific subdirectories if needed
function_mappings = {
"tiled_attn_module": "grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_",

"Llama_GPT_module": "grouped_multihead_attention_8_32_8_4_rope_ap_fixed_16_5_",
# "Llama_GPT_module": "dropout_8_32_ap_fixed_16_5_",
# "Llama_GPT_module": "matrix_add_8_32_ap_fixed_16_5_",
# "Llama_GPT_module": "matmul_8_32_128_ap_fixed_16_5_",
# "Llama_GPT_module": "matmul_8_128_32_ap_fixed_16_5_",

"attn_breakdown_module": "grouped_multihead_attention_8_32_8_4_rope_ap_fixed_16_5_",

}

# Run processing for all projects
data = process_all_hls_projects(HLS_DIR, function_mappings)

# Convert to DataFrame
df = pd.DataFrame(data)

# Round numeric columns to 2 decimal places
df['DSP_Utilization'] = df['DSP_Utilization'].round(2)
df['LUT_Utilization'] = df['LUT_Utilization'].round(2)

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"Utilization data saved to {OUTPUT_CSV}")
