# Repo cleanup, restructure, reproducibility and Zenodo plan (R2)

Updated 2026-10-15 (first drafted 2026-10-11). This is the hand-off for the session that does the cleanup. It merges
`REVISION_R2_AGENT_BRIEF.md` §A4–A6 (hygiene, large artifacts, layout), `VERIFICATION_HANDOFF.md` and the 2026-10-02
`forgebench/` package direction, and records the state of the results at the time of writing.

Nothing in §3–§7 has been executed. Each phase needs the user's go-ahead.

## 0. How to work

- **Separate clone.** The user clones this repo into a new directory and cuts a new branch there. The original checkout
  (`/usr/scratch/awanna3/ForgeBench`, branch `FPT_TRETS_2026`) keeps running jobs and their outputs. Do not edit, move or delete
  anything in it.
- **Bringing results across.** Committed results come over with `git fetch <original> FPT_TRETS_2026` and a merge or cherry-pick.
  Ignored artifacts (`checkpoints/`, `_sweeps/`, `verification/results_paper/`) are not in git. Copy only what a phase needs,
  or point the release scripts at the original paths (read only).
- **Git.** Commit locally only; the user pushes. No history rewrite unless D4 says so.
- **Standing rules.**
  - Do not touch the `scale_models/` layout without the user's explicit OK. The full-model sources, `contracts.json` hashes and
    verification paths depend on it. This conflicts with D1(a), see D1.
  - Do not delete or overwrite `verification/results_paper/`.
  - Do not kill other users' Vitis processes, and keep the job count modest.
  - Ask before any change with paper impact. Report failures honestly, including manual source edits.
  - Never `pkill -f` with a pattern that matches your own command line.
- **Freeze.** Freeze is 2026-10-21; resubmission is 2026-10-30. The paper links the released repo, so its paths and README must
  match what is released.

## 1. State of results (2026-10-15)

Everything below is collected and committed on `FPT_TRETS_2026` unless marked otherwise. See `CHANGELOG_R2.md`.

| Paper item | Data | Generator | State |
|---|---|---|---|
| Table 4 (sweeps), synth summary table | `manifest/designs/{gemm,conv,llm}.csv` | `manifest/build_manifest.py` | GEMM 3,072 / conv 5,184 / LLM 3,888 csynth; impl GEMM 999 + 1 FAIL, conv 1,000 (seeded, `conv_impl_selection.csv`), LLM 962 + 38 over capacity |
| Table 5, Figs 6–8 | `analysis/results_{csynth,impl}/metrics_{gemm,conv,llm}.csv` | `python3 -m analysis.section42` | done (impl 2,961) |
| Table 7 | `modular_data/modularization_{results.csv,table.tex}` | `modular_data/parse_synth_resourc_util.py --reports-root _sweeps/lean/modular_csynth_32_10_r2` | done, `<32,10>`, (LUT, DSP, BRAM) tuples |
| Table 3 (full models) | `analysis/tables_out/table3_fullmodel.{csv,tex}` | `python3 -m analysis.table3_fullmodel` | **ResNet-152 tiled `<16,5>` csynth still running** (`_sweeps/lean/fullmodel_csynth_r3_16_5`); everything else done |
| Table 2 / verification tables | `verification/paper_results/` (compact, tracked); raw runs in `verification/results_paper/` (66 GB, ignored) | `verification/paper_runs/make_tables.py`, `make_fullmodel_table.py` | R2 Table 7 runs (`C_32_10__table7r2_*`, merged) and full-model R2/R3 runs not yet copied into `paper_results/` |
| Table 6 (Catapult vs Vitis) | `analysis/catapult_vs_vitis.py` | | see `VERIFICATION_HANDOFF.md` |

Full-model sources:
- `scale_models/hls_files/*_32_10_`: 10 ResNets (new conv schedule) and 4 Llama, `<32,10>` RND/SAT. Verified and registered in
  `contracts.json`.
- Tiled ResNets at `<16,5>` with an 8×32 conv tile: only in `_sweeps/fullmodel_r3/{configs,hls_files}`, not yet in `scale_models/`.
  Their hashes are registered. Verified: bit-exact, but FP64 relative L2 > 5% for ResNet-34 (4/10 seeds), 101 (8/10) and 152
  (10/10). The user decided to report these and explain them. Copying them into `scale_models/hls_files` adds 5 project
  directories, so ask first.
- Full-model lean reports (`_sweeps/lean/fullmodel_csynth_{r2,r2_sched,r3_16_5}`) are not yet packaged into `checkpoints/r2/`.
  Package them like the other lean archives (`release/package_lean.sh`) and add them to `manifest/_common.py` SOURCES.

Archives (`checkpoints/`, ignored): `20260720/` (R1/July lean archives) and `r2/` (all R2 lean archives listed in
`manifest/_common.py` SOURCES; first match wins per design name).

## 2. Open decisions (user)

| # | Decision | Options | Notes |
|---|---|---|---|
| D1 | Generator layout | (a) a `forgebench/` package holding gemm, conv, llm, fullmodel, sweeps, backends and the CLI; (b) keep `gemm/ conv/ llm/ scale_models/` at top level | (a) was agreed 2026-10-02 (generator-first README). It needs package-relative template paths, a byte-compare regression on every domain, and moving `scale_models/`, which the standing rule protects. (b) is the safe option before the freeze. |
| D2 | When to do D1(a) | before the freeze / after resubmission | It is the only risky step. |
| D3 | Public repo? | | This decides whether `docs/revision_*` (planning notes, a OneDrive path; the PDF is ignored) leave the release branch. |
| D4 | History rewrite | none (default) / `git filter-repo` for old large blobs | `.git` is 3.6 GB. Ask first. |
| D5 | `modular_data/` | keep / move under `examples/` or the package | Read by `manifest/build_registry.py`, `verification/manual_designs.py`, `analysis/extractors/vitis.py` and `verification/merged_designs.py`. |
| D6 | `checkpoints/20260720/` (9 tracked CSVs) | keep as R1 provenance / move to `legacy/results_r1` | |
| D7 | Zenodo scope | reports + configs only / + lean archives / + raw verification runs | `verification/results_paper/` is 66 GB; the compact `paper_results/` is 14 MB. |
| D8 | Tiled `<16,5>` sources into `scale_models/hls_files` | yes / keep under `_sweeps` and ship via Zenodo | See §1. |

## 3. Phase H — hygiene (no generator paths change)

| Item | Action |
|---|---|
| `plotting/*_data/*.csv` (R1 exploratory sweep, not the paper dataset) | Move to `legacy/plotting/`. Drop the `analysis/plots.py` / `io.py` fallback that reads them: every figure now reads `analysis/results_*`. |
| `ModularHLS_testsuite.tar.gz` (tracked, 4.5 MB) | Move into the release bundle and delete it at the tip. |
| `ML_testsuite_part_*` (untracked) | Keep out of git, record in `release/bundle_files.csv`, remove README references. |
| One-off run scripts (`analysis/run_*_r2b.sh`, `run_conv_impl_r3.sh`, `_sweeps/*.sh`) | Keep the ones that reproduce a paper result as documented commands (`REPRODUCE.md`). Move the rest to `legacy/`. |
| `.gitattributes` | `* text=auto eol=lf` and mark binaries; renormalize in its own commit. |
| `.vscode/`, `uv.lock`, `.python-version` | Decide what a user needs. `pyproject.toml` must list the analysis dependencies (pandas, matplotlib, seaborn, scipy). The verifier needs torch (currently a private venv). |

## 4. Phase L — layout (D1)

Target under D1(b):
```
gemm/ conv/ llm/ scale_models/  backends/  verification/  analysis/
manifest/  paper_artifacts/  release/  docs/  legacy/  examples/
README.md  REPRODUCE.md  Makefile
```
- `examples/`: the Table 1 JSON examples (`<domain>/test_case_configs/` that are not Table 7 designs).
- `paper_artifacts/` (or `make paper`): one command per table and figure (§1 "Generator" column). Outputs go to a tracked
  directory, since `analysis/figures/` and `analysis/tables_out/` are ignored today.
- Any move of code-generating files needs a byte-compare regression first:
  - regenerate all sweep configs and a sample of `top.cpp` per domain before and after the move, and diff them;
  - regenerate one ResNet and one Llama full model and compare with `scale_models/hls_files` (the recipe in `CHANGELOG_R2`
    2026-10-15 reproduces the verified `<32,10>` tiled ResNet-18 byte for byte);
  - run `python -m verification.regression_check`.

## 5. Phase R — reproducibility

- `REPRODUCE.md`: environment (Vitis HLS 2024.1, Vivado 2024.1, Catapult version, Python env including torch for the full-model
  verifier), then one section per paper table and figure: inputs → command → output. Mark which steps need the EDA tools and which
  run from the released metrics alone.
- `Makefile` targets:
  - `manifest`: `rekey_results.py` + `build_manifest.py` + `build_registry.py`.
  - `tables`: Tables 3, 5, 7 and the synth summary.
  - `figures`: Figs 6–8.
  - `verify-tables`: the verification tables from `verification/paper_results/`.
- Check that every script runs from a fresh clone with no absolute paths:
  - `verification/paper_runs/run_fullmodel.sh` and `_sweeps/fullmodel_r3/verify.sh` hard-code a scratchpad torch venv;
  - grep for `/usr/scratch`, `/tmp/claude` and `OneDrive`.
- Seeds: GEMM 20261002, LLM 20261001, conv 20261014 (`analysis/impl_sampler.py`). Verification seeds are in the
  `paper_runs/*.json` configs.

## 6. Phase Z — Zenodo bundle (tooling exists: `release/`)

- `release/build_release.sh` → `assemble_reports.py` (configs/ + reports/ trees, per-domain tarballs, inner SHA256SUMS, checks
  every manifest path). `package_lean.sh`, `make_manifest.py` and `bundle_files.csv` already exist.
- To do:
  1. Package the full-model lean reports (§1).
  2. Add `manifest/designs/fullmodel.csv` (Table 3: 14 + 5 tiled `<16,5>` projects, sources, reports, verification runs).
  3. Add compact verification results: copy the R2 runs into `verification/paper_results/` the same way as the R1 runs.
  4. Add the Catapult runs.
  5. Dry run: every manifest path resolves, SHA256SUMS verify, no file over the Zenodo per-file limit (and the per-record file
     count, hence the per-domain tarballs).
- Acceptance: from a fresh clone plus the extracted bundle, `make tables figures` reproduces the committed outputs byte for byte
  (or within plotting nondeterminism).

## 7. Phase D — docs

- README rewrite: generator first, with counts that match the manifest (12,144 sweep designs; impl 2,961; Table 7 38 designs;
  14 full-model projects + 5 tiled `<16,5>`). Show the layout. Remove `anonymous.4open.science` and stale 6000+/5,400/12,912
  numbers.
- Keep `docs/TOOL_VERSIONS.md` and `docs/DATA_TYPES.md`. Handle `docs/revision_*` per D3.
- `manifest/README.md`: add `fullmodel.csv`, the conv seeded selection (done) and the verif tables.

## 8. Manifest — what is left

- `designs/fullmodel.csv` (Table 3).
- `verif/*.csv`: one row per design × datatype × sim level. The verification results exist.
- Brief B2 counts funnel, B4 duplicate check, B5 Table 4 from `SWEEP`: confirm whether they are still wanted.
