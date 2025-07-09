import csv
import matplotlib.pyplot as plt
# prepare empty lists
projects = []
functions = []
cycles = []
latencies = []
brams = []
dsps = []
ffs = []
luts = []
bram_utils = []
lut_utils = []
ff_utils = []
dsp_utils = []

with open('hls_synth_utilization.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        projects.append(row['Project'])
        functions.append(row['Function'])
        cycles.append(int(row['CYCLE']))
        latencies.append(row['LATENCY'])  # keeps the “76.179 ms” / “0.150 sec” strings
        brams.append(int(row['BRAM']))
        dsps.append(int(row['DSP']))
        ffs.append(int(row['FF']))
        luts.append(int(row['LUT']))
        bram_utils.append(float(row['BRAM_Utilization']))
        lut_utils.append(float(row['LUT_Utilization']))
        ff_utils.append(float(row['FF_Utilization']))
        dsp_utils.append(float(row['DSP_Utilization']))

# now you have 12 lists, one per column:
print(bram_utils)
print(dsp_utils)

plt.figure()
plt.scatter(bram_utils, dsp_utils)
plt.xlabel('BRAM Utilization (%)')
plt.ylabel('DSP Utilization (%)')
plt.title('BRAM vs DSP Utilization Scatter Plot')
plt.tight_layout()
# save to PNG in the current directory
plt.savefig('bram_vs_dsp_utilization.png', dpi=300)


# --- 2) BRAM Utilization vs Cycles ---
plt.figure()
plt.scatter(bram_utils, cycles)
plt.xlabel('BRAM Utilization (%)')
plt.ylabel('Cycles')
plt.title('BRAM Utilization vs Cycles')
plt.tight_layout()
plt.savefig('bram_util_vs_cycles.png', dpi=300)


# --- 3) DSP Utilization vs Cycles ---
plt.figure()
plt.scatter(dsp_utils, cycles)
plt.xlabel('DSP Utilization (%)')
plt.ylabel('Cycles')
plt.title('DSP Utilization vs Cycles')
plt.tight_layout()
plt.savefig('dsp_util_vs_cycles.png', dpi=300)


# --- 4) LUT Utilization vs Cycles ---
plt.figure()
plt.scatter(lut_utils, cycles)
plt.xlabel('LUT Utilization (%)')
plt.ylabel('Cycles')
plt.title('LUT Utilization vs Cycles')
plt.tight_layout()
plt.savefig('lut_util_vs_cycles.png', dpi=300)
