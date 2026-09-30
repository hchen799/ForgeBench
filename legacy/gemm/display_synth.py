# /// script
# dependencies = [
#     "pandas",
# ]
# ///

import pandas as pd
import sys

def process_utilization(csv_path):
    df = pd.read_csv(csv_path)
    
    # Extract base name more accurately
    df['Base'] = df['Project'].str.extract(r'(^[^_]+_p\d+)')
    df['Is_Module'] = df['Project'].str.contains('_module_')
    
    # Calculate total LUT and DSP utilization for each base name
    totals = df[~df['Is_Module']].groupby('Base')[['LUT_Utilization', 'DSP_Utilization']].sum()
    totals = totals.reset_index()
    totals['Project'] = totals['Base'] + '_total'
    
    # Process module-specific calculations
    module_df = df[df['Is_Module']].copy()
    module_df['Module_Type'] = module_df['Project'].str.extract(r'(_module_\w+)')
    
    # Compute shared utilization across same module types within each base
    shared = module_df.groupby(['Base', 'Module_Type'])[['LUT_Utilization', 'DSP_Utilization']].sum().reset_index()
    shared['Project'] = shared['Base'] + shared['Module_Type'] + '_shared'
    
    # Compute overall module utilization for each base
    overall_modules = module_df.groupby('Base')[['LUT_Utilization', 'DSP_Utilization']].sum().reset_index()
    overall_modules['Project'] = overall_modules['Base'] + '_module_overall'
    
    # Merge total values back to compute relative changes for module overall
    overall_modules = overall_modules.merge(totals, on='Base', suffixes=('', '_Total'))
    overall_modules['LUT_Change'] = (overall_modules['LUT_Utilization'] / overall_modules['LUT_Utilization_Total']).round(2)
    overall_modules['DSP_Change'] = (overall_modules['DSP_Utilization'] / overall_modules['DSP_Utilization_Total']).round(2)
    
    # Format values as (LUT, DSP)
    df['Formatted'] = '(' + df['LUT_Utilization'].astype(str) + ', ' + df['DSP_Utilization'].astype(str) + ')'
    totals['Formatted'] = '(' + totals['LUT_Utilization'].astype(str) + ', ' + totals['DSP_Utilization'].astype(str) + ')'
    shared['Formatted'] = '(' + shared['LUT_Utilization'].astype(str) + ', ' + shared['DSP_Utilization'].astype(str) + ')'
    overall_modules['Formatted'] = '(' + overall_modules['LUT_Change'].astype(str) + ', ' + overall_modules['DSP_Change'].astype(str) + ')'
    
    # Concatenate results
    result = pd.concat([df, totals, shared, overall_modules], ignore_index=True)[['Project', 'Formatted']]
    
    print(result.to_csv(index=False))

if __name__ == "__main__":
    process_utilization(sys.argv[1])
