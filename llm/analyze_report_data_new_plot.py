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



# with open('hls_synth_utilization_1.csv', newline='') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         projects.append(row['Project'])
#         functions.append(row['Function'])
#         cycles.append(int(row['CYCLE']))
#         latencies.append(row['LATENCY'])  # keeps the “76.179 ms” / “0.150 sec” strings
#         brams.append(int(row['BRAM']))
#         dsps.append(int(row['DSP']))
#         ffs.append(int(row['FF']))
#         luts.append(int(row['LUT']))
#         bram_utils.append(float(row['BRAM_Utilization']))
#         lut_utils.append(float(row['LUT_Utilization']))
#         ff_utils.append(float(row['FF_Utilization']))
#         dsp_utils.append(float(row['DSP_Utilization']))
        
with open('hls_synth_utilization_2.csv', newline='') as f:
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

with open('hls_synth_utilization_3.csv', newline='') as f:
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
        
with open('hls_synth_utilization_4.csv', newline='') as f:
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
        
with open('hls_synth_utilization_5.csv', newline='') as f:
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
        
with open('hls_synth_utilization_6.csv', newline='') as f:
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
        
# with open('hls_synth_utilization_4.csv', newline='') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         projects.append(row['Project'])
#         functions.append(row['Function'])
#         cycles.append(int(row['CYCLE']))
#         latencies.append(row['LATENCY'])  # keeps the “76.179 ms” / “0.150 sec” strings
#         brams.append(int(row['BRAM']))
#         dsps.append(int(row['DSP']))
#         ffs.append(int(row['FF']))
#         luts.append(int(row['LUT']))
#         bram_utils.append(float(row['BRAM_Utilization']))
#         lut_utils.append(float(row['LUT_Utilization']))
#         ff_utils.append(float(row['FF_Utilization']))
#         dsp_utils.append(float(row['DSP_Utilization']))        

print(len(bram_utils))
#print(bram_utils)
indices_bram_utils = [i for i, v in enumerate(bram_utils) if (v > 100 or v < 18) ]
indices_dsp_utils = [i for i, v in enumerate(dsp_utils) if v > 100]
indices_lut_utils = [i for i, v in enumerate(lut_utils) if v > 100]
indices_cycle = [i for i, v in enumerate(cycles) if v > 1e7]


union_indices = set(indices_bram_utils) | set(indices_dsp_utils) | set(indices_lut_utils) | set(indices_cycle)
union_indices = sorted(union_indices)

brams_selected = []
dsps_selected = []
ffs_selected = []
luts_selected = []

bram_utils_selected = []
lut_utils_selected = []
ff_utils_selected = []
dsp_utils_selected = []

cycles_selected = []
latencies_selected = []

for i in range (len(projects)):
    if i not in union_indices:
        
        brams_selected.append(brams[i])
        dsps_selected.append(dsps[i])
        ffs_selected.append(ffs[i])
        luts_selected.append(luts[i])
        bram_utils_selected.append(bram_utils[i])
        lut_utils_selected.append(lut_utils[i])
        ff_utils_selected.append(ff_utils[i])
        dsp_utils_selected.append(dsp_utils[i])
        
        cycles_selected.append(cycles[i])
        latencies_selected.append(latencies[i])
        

print(len(union_indices))
FIGSIZE = (9, 6)

LABEL_FONT  = 18
TICK_FONT   = 18
TITLE_FONT  = 18

print(len(union_indices))
plt.figure(figsize=FIGSIZE)
plt.scatter(bram_utils_selected, dsp_utils_selected, marker='x', s = 10, c = 'lightseagreen')
plt.xlabel('BRAM Utilization (%)', fontsize=LABEL_FONT)
plt.ylabel('DSP Utilization (%)', fontsize=LABEL_FONT)
plt.title('BRAM vs DSP Utilization', fontsize=TITLE_FONT)
plt.xticks([0, 20, 40, 60, 80, 100], fontsize=TICK_FONT)
plt.xlim(15, 105)
plt.yticks([0, 20, 40, 60, 80, 100], fontsize=TICK_FONT)
plt.ylim(-5, 105)
plt.tight_layout()
# save to PNG in the current directory
plt.savefig('bram_vs_dsp_utilization.png', dpi=300)


# --- 2) BRAM Utilization vs Cycles ---
plt.figure(figsize=FIGSIZE)
plt.scatter(bram_utils_selected, cycles_selected, marker='x', s = 10, c = 'lightseagreen')
plt.xlabel('BRAM Utilization (%)', fontsize=LABEL_FONT)
plt.ylabel('Cycles', fontsize=LABEL_FONT)
plt.title('BRAM Utilization vs Cycles', fontsize=TITLE_FONT)
plt.xticks([0, 20, 40, 60, 80, 100], fontsize=TICK_FONT)
plt.yticks(fontsize=TICK_FONT)
plt.xlim(15, 105)
plt.tight_layout()
plt.savefig('bram_util_vs_cycles.png', dpi=300)


# --- 3) DSP Utilization vs Cycles ---
plt.figure(figsize=FIGSIZE)
plt.scatter(dsp_utils_selected, cycles_selected, marker='x', s = 10, c = 'lightseagreen')
plt.xlabel('DSP Utilization (%)', fontsize=LABEL_FONT)
plt.ylabel('Cycles', fontsize=LABEL_FONT)
plt.title('DSP Utilization vs Cycles', fontsize=TITLE_FONT)
plt.xticks([0, 20, 40, 60, 80, 100], fontsize=TICK_FONT)
plt.yticks(fontsize=TICK_FONT)
plt.xlim(-5, 105)
plt.tight_layout()
plt.savefig('dsp_util_vs_cycles.png', dpi=300)


# --- 4) LUT Utilization vs Cycles ---
plt.figure(figsize=FIGSIZE)
plt.scatter(lut_utils_selected, cycles_selected, marker='x', s = 10, c = 'lightseagreen')
plt.xlabel('LUT Utilization (%)', fontsize=LABEL_FONT)
plt.ylabel('Cycles', fontsize=LABEL_FONT)
plt.title('LUT Utilization vs Cycles', fontsize=TITLE_FONT)
plt.xticks([0, 20, 40, 60, 80, 100], fontsize=TICK_FONT)
plt.yticks(fontsize=TICK_FONT)
plt.xlim(-5, 105)
plt.tight_layout()
plt.savefig('lut_util_vs_cycles.png', dpi=300)

print("the number of valid data points is:", len(bram_utils_selected))
