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
        csynth.xml                      ← REQUIRED for csynth PPA + impl latency (cycles)
      impl/report/verilog/
        export_impl.xml                 ← REQUIRED for impl area + Fmax/timing (see §3, §5)
      impl/verilog/project.runs/impl_1/
        *_power_routed.rpt              ← REQUIRED for power (see §4)
      csim/build/csim.log              ← REQUIRED for VERIFICATION verdict (see §7)
      cosim/report/*.rpt               ← optional cosim pass/fail
```

> Paths locked against a real Vitis 2024.1.2 / Vivado impl bundle (ZCU102,
> xczu9eg). Timing/Fmax come from `export_impl.xml`'s `<TimingReport>` — the
> routed `*_timing_*.rpt` files are NOT needed by the parser. Impl latency reuses
> the cycle count from `csynth.xml` (the impl XML has no PerformanceEstimates).

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

The extractor reads (root element is `<profile>`):

```
<profile>
└── AreaReport
    ├── Resources
    │   ├── BRAM        ← RAMB36 tile count (NOT BRAM_18K — different unit than csynth)
    │   ├── DSP
    │   ├── FF
    │   └── LUT
    └── AvailableResources
        ├── BRAM
        ├── DSP
        ├── FF
        └── LUT
└── TimingReport                  ← Fmax / timing closure (see §5)
```

No PerformanceEstimates/latency node exists in this XML — impl latency is derived
from the sibling `csynth.xml` cycle count (see §6). No action needed if present.

> **Unit note (reviewer B3):** csynth reports `BRAM_18K` (18-Kb blocks); impl
> reports `BRAM` as RAMB36 (36-Kb) tiles. DSP/FF/LUT are directly comparable.

---

## 4. Power report (DONE — parser locked)

**Path:** `project_1/solution1/impl/verilog/project.runs/impl_1/*_power_routed.rpt`
(Vivado `report_power` output, e.g. `bd_0_wrapper_power_routed.rpt`).

`_parse_power_rpt` reads these Summary rows:

| Field | Row in .rpt | `PPAMetrics` field |
|-------|-------------|--------------------|
| Total on-chip power | `Total On-Chip Power (W)` | `power_w` |
| Dynamic power | `Dynamic (W)` | `power_dyn_w` |
| Static power | `Device Static (W)` | `power_static_w` |

No action needed — just include the routed power report in the tarball.

---

## 5. Timing / Fmax (DONE — from `export_impl.xml`, no .rpt needed)

Timing closure and achieved clock come from the `<TimingReport>` node inside
`export_impl.xml` (§3) — the routed `*_timing_*.rpt` files are not parsed.

| Tag in `<TimingReport>` | `PPAMetrics` field |
|-------------------------|--------------------|
| `AchievedClockPeriod` (= target − WNS) | `achieved_clock_ns`; `fmax_mhz = 1000/achieved` |
| `TargetClockPeriod` / `CP_TARGET` | `target_clock_ns` |
| `TIMING_MET` (`TRUE`/`FALSE`) | `timing_met` (falls back to `WNS_FINAL >= 0`) |

No action needed.

---

## 6. Throughput (DONE — end-to-end definition)

Implemented as **end-to-end**: `throughput = 1e9 / latency_ns` (designs/sec),
where `latency_ns = cycles × target_clock_ns`. `cycles` is the worst-case latency
from the sibling `csynth.xml` (the impl XML has none); the design is clocked at
`target_clock_ns` (the constraint it meets). `II` is also recorded (from
`SummaryOfOverallLatency/Interval-max`) so an II-based throughput can be derived
later if the paper prefers it.

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

`_parse_impl_ppa` in `analysis/extractors/vitis.py` is now complete — `--parse-only`
populates all `PPAMetrics` fields (area, latency, throughput, power, Fmax, timing).

---

## 9. Minimal tarball checklist

For each design dir, ensure these files are present before sending:

- [ ] `project_1/solution1/syn/report/csynth.xml` (csynth PPA **and** impl latency/II)
- [ ] `project_1/solution1/impl/report/verilog/export_impl.xml` (impl area + timing/Fmax)
- [ ] `project_1/solution1/impl/verilog/project.runs/impl_1/*_power_routed.rpt` (power)
- [ ] Any `.log` file that contains the `VERIFICATION:` line (csim output)

The routed `*_timing_*.rpt` files are NOT needed (timing is read from
`export_impl.xml`). Everything else (generated C++, Vitis project files) is
optional — the parsers only read the above paths.
