# ForgeBench analysis (artifact reproducibility)

Unified, tool-agnostic PPA extraction, aggregation, and plotting. Replaces the
per-domain `parse_*_resource_utlization.py` scripts and the duplicated
`analyze_report_data_new_plot_final.py` plotting copies with one package, and is
structured so a Catapult backend drops in next to the Vitis one.

## Install / run

Uses the project's `uv` environment (numpy, pandas, matplotlib):

```bash
uv run python -m analysis.<module> ...
```

## Reproduce the paper figures/tables

| Paper item | Command | Data source |
|-----------|---------|-------------|
| **Fig. 4** (design-space coverage: BRAM/DSP/LUT util vs Cycles, BRAM-vs-DSP) | `uv run python -m analysis.plots fig4` | committed `plotting/{gemm,conv,llm}_data/*.csv` |
| Resource vs latency scatter | `uv run python -m analysis.plots resource_vs_latency` | same |
| Resource vs power scatter | `uv run python -m analysis.plots resource_vs_power` | **needs impl power data (stub)** |
| **Table 2** (full-scale min/max LUT/BRAM/DSP) | `uv run python -m analysis.tables table2` | `scale_models/hls_synth_utilization.csv` |
| **Table 3** (suite testcase counts) | `uv run python -m analysis.tables table3` | generated-config dirs (falls back to paper counts) |
| **Table 5** (modularization %reuse) | `uv run python -m analysis.tables table5` | `modular_data/modularization_results.csv` |

Figures land in `analysis/figures/`, tables in `analysis/tables_out/` (CSV + LaTeX).

## Parse raw HLS reports into a unified CSV

```bash
# post-synthesis (resources + latency)
uv run python -m analysis.collect <results_dir> --tool vitis --flow csynth --domain gemm --out analysis/metrics.csv
# post-implementation (resources; power/Fmax/timing/throughput once available)
uv run python -m analysis.collect <results_dir> --tool vitis --flow impl --out analysis/impl_metrics.csv
```

`<results_dir>` holds one subdir per design with the Vitis reports at
`project_1/solution1/syn/report/csynth.xml` (csynth) or
`project_1/solution1/impl/report/verilog/export_impl.xml` (impl).

## Implementation (place-and-route) PPA on the 4 base designs

```bash
uv run python -m analysis.impl_runner --out analysis/_impl        # generate + TCL (csynth,export_ip)
# then on the Vitis server, per generated design dir:
#   (cd <design_dir> && vitis_hls -f run_hls.tcl)
uv run python -m analysis.impl_runner --parse-only --out analysis/_impl
```

Designs: the three generation-case-study base designs (GEMM `x·A·B·y`, DNN
`Conv-BatchNorm-Activation`, LLM `Attention-Dropout-Norm`) + a ResNet. **The exact
config paths in `impl_runner.DESIGNS` are best guesses — confirm them** (or pass
`--config domain:path`). `export_ip` is the generator keyword that emits
`export_design -format ip_catalog -flow impl` (the P&R run).

## Package layout

```
analysis/
  schema.py            # PPAMetrics: the tool-agnostic record everything shares
  extractors/base.py   # Extractor ABC + registry (get_extractor)
  extractors/vitis.py  # Vitis csynth + impl (impl PPA fields stubbed — see below)
  extractors/catapult.py  # STUB (concern E1): same interface, TODO
  collect.py           # results tree -> unified metrics.csv
  io.py                # CSV load/normalize (unified + legacy per-domain schemas)
  plots.py             # Fig 4 + resource-vs-latency + resource-vs-power (stub)
  tables.py            # Table 2 / 3 / 5
  impl_runner.py       # generate + drive impl on the 4 base designs
```

## Adding Catapult (concern E1)

Implement `analysis/extractors/catapult.py` (`discover` + `parse` returning
`PPAMetrics`) and it plugs into `collect`, `plots`, and `tables` unchanged. Run
`uv run python -m analysis.extractors.catapult` to see the stub contract.

## Known gaps / flags

- **Impl PPA (power, Fmax, timing closure, throughput) is stubbed** in
  `extractors/vitis.py::_parse_impl_ppa` — the csynth report has none of these,
  and the exact impl/Vivado report tag paths need a sample bundle to lock. The
  `resource_vs_power` plot no-ops until then.
- **Throughput definition** to confirm (designs/sec via end-to-end latency vs
  II-based).
- **Testcase count discrepancy:** repo README claims 6000+ (1920/2304/1944); the
  paper (Table 3) claims 10,837 (3335/4143/3359). `tables.table3` reports the
  paper numbers when the generated suites aren't present — regenerate the suites
  and reconcile before finalizing artifact claims.
- **Modular reuse** (`modular_data/parse_synth_resourc_util.py`) is kept as-is:
  its P1/P2/P3/shared %reuse computation is bespoke (not a duplicate of the
  resource parsers). `tables.table5` formats its committed output CSV. Porting it
  onto the unified extractor is future work.
