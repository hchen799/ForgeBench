# Server data collection instructions

This document tells the server-side agent (or person) running Vitis HLS what to collect
after each run so the local analysis package can extract every PPA field.

Two flows are covered:

| Flow | Purpose |
|------|---------|
| **csynth** | Resources + latency (fully parsed, no stubs) |
| **impl** (`export_ip`) | Resources post-P&R + power + Fmax + timing closure |

Both flows also produce the **verification verdict** (CSIM only; cosim is optional).

---

## 1. Directory layout to produce

Run the generator in its normal mode; each design lands under `<results_dir>/`:

```
<results_dir>/
  <design_name>/
    run_hls.tcl
    *.cpp / *.h / *.golden.txt          ← inputs; keep as-is
    project_1/solution1/
      syn/report/
        csynth.xml                      ← REQUIRED for csynth PPA
      impl/report/verilog/
        export_impl.xml                 ← REQUIRED for impl area (post-P&R)
      impl/report/verilog/
        export_impl_power.rpt           ← REQUIRED for power (see §3)
      impl/report/verilog/
        *.timing_summary.rpt            ← REQUIRED for Fmax / timing closure (see §4)
      csim/build/csim.log              ← REQUIRED for VERIFICATION verdict (see §5)
      cosim/report/*.rpt               ← optional cosim pass/fail
```

Tarball or rsync the full `<results_dir>/` tree back — the parsers walk it automatically.

---

## 2. csynth report (`csynth.xml`)

**Path:** `project_1/solution1/syn/report/csynth.xml`

The extractor reads these XML tags — confirm they are present in your Vitis 2024.1 output:

```
HLSDesignSpace
└── AreaEstimates
    ├── Resources
    │   ├── BRAM_18K
    │   ├── DSP
    │   ├── FF
    │   └── LUT
    └── AvailableResources
        ├── BRAM_18K
        ├── DSP
        ├── FF
        └── LUT
└── PerformanceEstimates
    └── SummaryOfOverallLatency
    │   └── Worst-caseLatency          ← integer clock cycles
    └── SummaryOfTimingAnalysis
        └── EstimatedClockPeriod       ← float ns (used to derive latency_ns)
```

No action needed if this file is present — it is already fully parsed.

---

## 3. Impl area report (`export_impl.xml`)

**Path:** `project_1/solution1/impl/report/verilog/export_impl.xml`

The extractor reads:

```
<root>
└── AreaReport
    ├── Resources
    │   ├── BRAM_18K
    │   ├── DSP
    │   ├── FF
    │   └── LUT
    └── AvailableResources
        ├── BRAM_18K
        ├── DSP
        ├── FF
        └── LUT
└── PerformanceEstimates          ← Worst-caseLatency (cycles) if present
    └── SummaryOfOverallLatency
        └── Worst-caseLatency
```

No action needed if this file is present.

---

## 4. Power report (STUB — needed to fill `_parse_impl_ppa`)

**Expected path (confirm):**
`project_1/solution1/impl/report/verilog/<design>_power_routed.rpt`
or any `*_power*.rpt` under `impl/report/`.

From the Vivado power report, extract and note:

| Field | Location in .rpt | Target `PPAMetrics` field |
|-------|-----------------|--------------------------|
| Total on-chip power | `Total On-Chip Power (W)` row | `power_w` |
| Dynamic power | `Dynamic (W)` row | `power_dyn_w` |
| Static power | `Device Static (W)` row | `power_static_w` |

**Send:** the full `.rpt` file so the exact column/row format can be locked and the
parser in `analysis/extractors/vitis.py::_parse_impl_ppa` can be completed.

---

## 5. Timing / Fmax report (STUB — needed to fill `_parse_impl_ppa`)

**Expected path (confirm):**
`project_1/solution1/impl/report/verilog/<design>_timing_summary_routed.rpt`
or any `*timing*.rpt` under `impl/report/`.

From the Vivado timing summary, extract and note:

| Field | Location in .rpt | Target `PPAMetrics` field |
|-------|-----------------|--------------------------|
| Achieved clock period (ns) | `Data Path Delay` or WNS column | `achieved_clock_ns` |
| Worst Negative Slack (ns) | `WNS` column | used to compute `timing_met = WNS >= 0` |
| Fmax (MHz) | `1000 / achieved_clock_ns` | `fmax_mhz` |

**Send:** the full `.rpt` file so the column/row format can be locked.

---

## 6. Throughput (definition to confirm before parsing)

Two candidate definitions — confirm which the paper uses before the stub is filled:

| Option | Formula | Notes |
|--------|---------|-------|
| **End-to-end** | `1e9 / latency_ns` (designs/sec) | Simple; uses worst-case latency from csynth/impl |
| **II-based** | `target_clock_MHz / II` | Higher throughput for pipelined designs; requires II from csynth.xml `SummaryOfOverallLatency/Interval-min` |

Once confirmed, add to `analysis/extractors/vitis.py::_parse_impl_ppa`:
```python
m.throughput = 1e9 / m.latency_ns          # end-to-end variant
# or
m.throughput = (1000 / m.target_clock_ns) / ii   # II-based variant
```

---

## 7. Verification verdict (`csim.log`)

**Path:** `project_1/solution1/csim/build/csim.log`
(or any `.log`/`.rpt`/`.txt` under the design dir — `collect_results.py` walks all text files)

The testbench prints one of:

```
VERIFICATION: PASS (max_abs=<f>, max_rel=<f>, n=<N>)
VERIFICATION: FAIL (max_abs=<f>, max_rel=<f>, n_mismatch=<k>/<N>)
VERIFICATION: SKIP (no golden files found)
```

No action needed — `verification/collect_results.py` scans the whole tree automatically.

---

## 8. What to run on the local machine after receiving the data

```bash
# Parse csynth results from the full suite run:
uv run python -m analysis.collect <results_dir> \
    --tool vitis --flow csynth --domain gemm \
    --out analysis/metrics.csv

# Parse impl results from the 4-design impl run:
uv run python -m analysis.impl_runner \
    --parse-only --out analysis/_impl

# Aggregate verification verdicts:
uv run python -m verification.collect_results <results_dir> \
    --out verification/results.csv
```

Once power and timing .rpt samples arrive, `_parse_impl_ppa` in
`analysis/extractors/vitis.py` will be completed and `--parse-only` will populate
all `PPAMetrics` fields.

---

## 9. Minimal tarball checklist

For each design dir, ensure these files are present before sending:

- [ ] `project_1/solution1/syn/report/csynth.xml`
- [ ] `project_1/solution1/impl/report/verilog/export_impl.xml`
- [ ] `project_1/solution1/impl/report/verilog/*_power_routed.rpt` ← **new, needed for stub**
- [ ] `project_1/solution1/impl/report/verilog/*_timing_summary_routed.rpt` ← **new, needed for stub**
- [ ] Any `.log` file that contains the `VERIFICATION:` line (csim output)

Everything else (generated C++, Vitis project files) is optional — the parsers only
read the above paths.
