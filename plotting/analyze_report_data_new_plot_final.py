import csv
import matplotlib.pyplot as plt
import os
import pandas as pd

def read_csv_file(filename):

    # prepare empty lists
    data_dict = {
        'projects': [],
        'functions': [],
        'cycles': [],
        'latencies': [],
        'brams': [],
        'dsps': [],
        'ffs': [],
        'luts': [],
        'bram_utils': [],
        'lut_utils': [],
        'ff_utils': [],
        'dsp_utils': []
    }

    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_dict['projects'].append(row['Project'])
            data_dict['functions'].append(row['Function'])
            data_dict['cycles'].append(int(row['CYCLE']))
            data_dict['latencies'].append(row['LATENCY'])  # keeps the “76.179 ms” / “0.150 sec” strings
            data_dict['brams'].append(int(row['BRAM']))
            data_dict['dsps'].append(int(row['DSP']))
            data_dict['ffs'].append(int(row['FF']))
            data_dict['luts'].append(int(row['LUT']))
            data_dict['bram_utils'].append(float(row['BRAM_Utilization']))
            data_dict['lut_utils'].append(float(row['LUT_Utilization']))
            data_dict['ff_utils'].append(float(row['FF_Utilization']))
            data_dict['dsp_utils'].append(float(row['DSP_Utilization']))
    return data_dict

def process_directory(dirname):
    """
    Read all CSV files from a relative subdirectory `dirname` and aggregate
    their data into a single dictionary with the same keys returned by
    `read_csv_file`.

    Args:
        dirname (str): Path to a relative subdirectory (relative to this
                       script file) containing CSV files.

    Returns:
        dict: Aggregated dictionary containing lists for each field.
    """

    # Build base path relative to this script so callers can pass a
    # subdirectory name (e.g. "data")
    base_dir = os.path.join(os.path.dirname(__file__), dirname)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    # Prepare an empty aggregated dict with the same keys as read_csv_file
    aggregated = {
        'projects': [],
        'functions': [],
        'cycles': [],
        'latencies': [],
        'brams': [],
        'dsps': [],
        'ffs': [],
        'luts': [],
        'bram_utils': [],
        'lut_utils': [],
        'ff_utils': [],
        'dsp_utils': []
    }

    # Iterate over CSV files in the directory and extend the aggregated lists
    for fname in sorted(os.listdir(base_dir)):
        if not fname.lower().endswith('.csv'):
            continue
        path = os.path.join(base_dir, fname)
        try:
            data = read_csv_file(path)
        except Exception as e:
            # Propagate or log as needed; here we raise to alert the caller
            raise RuntimeError(f"Failed to read CSV '{path}': {e}") from e

        # Extend aggregated lists with values from this file
        for key in aggregated.keys():
            # If the CSV reader returned slightly different keys, skip silently
            if key in data:
                aggregated[key].extend(data[key])

    return aggregated


def build_dataframe_from_dirs(dirnames):
    """
    Build a single pandas DataFrame from one or more relative subdirectories.

    Args:
        dirnames (str | list[str]): A single relative subdirectory name or a
            list of relative subdirectory names (relative to this script).

    Returns:
        pandas.DataFrame: DataFrame containing the concatenated rows from all
            CSV files in the provided directories. A new column `source_dir`
            identifies which directory each row came from.

    Example:
        df = build_dataframe_from_dirs(['dir_a', 'dir_b', 'dir_c'])
    """
    if isinstance(dirnames, str):
        dirnames = [dirnames]

    frames = []
    for d in dirnames:
        # Reuse existing aggregation helper
        data = process_directory(d)

        # Convert aggregated dict to DataFrame
        df = pd.DataFrame(data)

        # Add source directory column to allow different coloring later
        df['source_dir'] = d

        frames.append(df)

    if not frames:
        # Return empty DataFrame with expected columns
        cols = ['projects', 'functions', 'cycles', 'latencies', 'brams', 'dsps', 'ffs', 'luts',
                'bram_utils', 'lut_utils', 'ff_utils', 'dsp_utils', 'source_dir']
        return pd.DataFrame(columns=cols)

    # Concatenate and return
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result


def filter_dataframe(df, bram_min=18, bram_max=100, dsp_max=100, lut_max=100, cycles_max=1e7):
    """
    Filter a DataFrame of synthesis results by resource/utilization thresholds.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns at least
            'bram_utils', 'dsp_utils', 'lut_utils', 'cycles'. Other columns
            are preserved.
        bram_min (float): Minimum allowed BRAM utilization (percent).
        bram_max (float): Maximum allowed BRAM utilization (percent).
        dsp_max (float): Maximum allowed DSP utilization (percent).
        lut_max (float): Maximum allowed LUT utilization (percent).
        cycles_max (float): Maximum allowed cycle count.

    Returns:
        tuple: (filtered_df, removed_indices)
            `filtered_df` is a DataFrame containing only rows that pass all
            filters. `removed_indices` is a sorted list of integer indices
            (from the original DataFrame) that were removed.

    Example:
        filtered, removed = filter_dataframe(df)
    """
    if df is None or df.empty:
        return df.copy(), []

    df2 = df.copy()

    # Ensure numeric columns for comparisons
    for col in ('bram_utils', 'dsp_utils', 'lut_utils', 'cycles'):
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors='coerce')

    indices_bram_utils = []
    if 'bram_utils' in df2.columns:
        indices_bram_utils = df2.index[(df2['bram_utils'] > bram_max) | (df2['bram_utils'] < bram_min)].tolist()

    indices_dsp_utils = []
    if 'dsp_utils' in df2.columns:
        indices_dsp_utils = df2.index[df2['dsp_utils'] > dsp_max].tolist()

    indices_lut_utils = []
    if 'lut_utils' in df2.columns:
        indices_lut_utils = df2.index[df2['lut_utils'] > lut_max].tolist()

    indices_cycle = []
    if 'cycles' in df2.columns:
        indices_cycle = df2.index[df2['cycles'] > cycles_max].tolist()

    union_indices = sorted(set(indices_bram_utils) | set(indices_dsp_utils) | set(indices_lut_utils) | set(indices_cycle))

    # Remove rows with union indices and return filtered DataFrame
    filtered_df = df2.drop(index=union_indices).reset_index(drop=True)
    return filtered_df, union_indices


# Load data from three subdirectories
df = build_dataframe_from_dirs(['gemm_data', 'conv_data', 'llm_data'])

# Filter the combined DataFrame
filtered_df, removed_indices = filter_dataframe(df, bram_min=0, cycles_max=1e9)

print(f"Total points: {len(df)}")
print(f"Removed points: {len(removed_indices)}")
print(f"Valid points: {len(filtered_df)}")


# Define visualization parameters
FIGSIZE = (36, 10) # Increased width and height for a 1x4 layout
LABEL_FONT = 32
TICK_FONT = 28
TITLE_FONT = 32
MARKER_SIZE = 30 # Kept as requested

dir_config = {
    'gemm_data': {'color': 'lightseagreen', 'marker': 's', 'label': 'GEMM'},
    'conv_data': {'color': 'coral', 'marker': 's', 'label': 'CONV'},
    'llm_data': {'color': 'mediumslateblue', 'marker': 's', 'label': 'LLM'},
}

x_vars = [
'bram_utils', 
'dsp_utils', 
'lut_utils',
    'bram_utils', 
]
y_vars = [
'cycles',
 'cycles',
 'cycles',
    'dsp_utils', 
]
titles = [
'BRAM Util vs Cycles', 
'DSP Util vs Cycles', 
'LUT Util vs Cycles',
    'BRAM vs DSP Util', 
]
x_labels = [
'BRAM Utilization (%)', 
'DSP Utilization (%)', 
'LUT Utilization (%)',
    'BRAM Utilization (%)', 
]

fig, axes = plt.subplots(1, 4, figsize=FIGSIZE)

for i, ax in enumerate(axes):
    x_col = x_vars[i]
    y_col = y_vars[i]
    
    for source_dir in ['gemm_data', 'conv_data', 'llm_data']:
        subset = filtered_df[filtered_df['source_dir'] == source_dir]
        config = dir_config[source_dir]
        ax.scatter(subset[x_col], subset[y_col], 
                   marker=config['marker'], s=MARKER_SIZE, c=config['color'], 
                   label=config['label'], alpha=0.7)
    
    # Text and Labels
    ax.set_title(titles[i], fontsize=TITLE_FONT, fontweight='bold', pad=20)
    ax.set_xlabel(x_labels[i], fontsize=LABEL_FONT, fontweight='bold')
    ax.set_ylabel(y_col.replace('dsp_utils', 'DSP Utilization (%)').title(), fontsize=LABEL_FONT, fontweight='bold')
    # make the y axis log

    # Tick Formatting
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    
    # Offset Text Font Size (The 1e7)
    ax.yaxis.get_offset_text().set_fontsize(TICK_FONT-4)
    
    # Axis Limits
    ax.set_xlim(left=-5 if 'bram' not in x_col else 15, right=105)
    if i == 3:
        ax.set_ylim(-5, 105)
    else:
        ax.set_yscale('log')


# import fontmanager
from matplotlib import font_manager as fm


# Create a single legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()

#set fm to bold
prop = fm.FontProperties()
prop.set_weight('bold')
prop.set_size(LABEL_FONT)
# make marker opacity 1.0 in legend
# make marker sqaure in legend
for handle in handles:
    handle.set_alpha(1.0)


fig.legend(handles, labels, 
           loc='lower center', 
           bbox_to_anchor=(0.5, -0.02), # Positioned below the axes
           ncol=3, 
           fontsize=LABEL_FONT,
           markerscale=6.0, # This makes the legend markers 4x larger than the plot markers
           frameon=True,
           handletextpad=0.2,
           prop=prop)

# Adjust layout to make room for the bottom legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.2) # Manually ensure space for the legend

plt.savefig('combined_utilization_bottom_legend.png', dpi=300, bbox_inches='tight')
plt.show()