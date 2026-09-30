# ForgeBench — TRETS R2 Revision: Code-Agent Brief

**Read this whole file before touching the repo.** It is self-contained. It supersedes `HANDOFF.md`,
`VERIFICATION_FLOW_PLAN.md` and `PAPER_CHANGES.md` (those describe the July R1 round and are now history).

- Manuscript: TRETS-2026-0091.R1, "ForgeBench: An HLS Design Generator of Machine Learning Test Suites
  for Next-Generation HLS Tools" (Wanna, Chen, Hao — Georgia Tech).
- Status: final revision round. Reviewers 2 and 3 accept. **Only Reviewer 1 (R1) remains.**
- **Hard deadline: resubmission by 2026-10-30.** Code/results freeze target: **2026-10-21**.
- Reviews and manuscript (human-held, not in repo):
  `C:\Users\andyw\OneDrive - Georgia Institute of Technology\forgebench_revision\`
  (`forgebench_reviews_oct2026.txt` = the current review; read R1's comments in full before starting).
- Tooling split: you (the agent) write code and validate it **without Vitis/Catapult** (Python + g++).
  Every Vitis/Catapult run is a **[SERVER]** task: you prepare a one-command script, the human runs it
  and commits the outputs, you then parse/aggregate.

---

## 0. The one-sentence problem

R1's verdict: *the paper's claims (functionally verified, complete model-level inference, cross-tool
portable, reproducible) are stronger than the evidence, and the public artifact does not match the
manuscript.* Your job is to (a) make the repo exactly match what the paper will claim, and (b) build
the harnesses that generate the missing evidence. **Never invent, estimate or hand-type a result.**
If a number is not produced by a committed script from committed inputs, it does not go in a table;
leave a `TODO(result)` placeholder and tell the human.

## 1. Repo facts (verified 2026-09-30)

Remote: `https://github.com/hchen799/HLS_ML_benchmark` (also reachable as `hchen799/ForgeBench`).

| Branch | Tip | vs main | Notes |
|---|---|---|---|
| `main` | b24b2a8 (2026-04-30) | — | What R1 inspected on 2026-08-02 via the anonymous mirror. Old 5,400-config `ML_testsuite_part_*`, README says "6000+". |
| `feature/correctness-harness` | 7f9cbd7 (2026-07-27) | +32/−0 | All July work: `verification/`, `analysis/`, `backends/`, `checkpoints/20260720/`. |
| `catapult-dev` | afd74cc (2026-07-22) | +22/−0 | **Already fully merged into the harness branch (merge 43206dc). 0 unique commits.** |
| `FPT_experiment` | 7deafd2 (2025-07-09) | +2/−10 | Old FPT plotting commits. Inspect; probably archive only. |
| `updated_LLM_bench` | d2a26d3 (2025-03-28) | +0/−10 | Ancestor of main. Delete-safe. |

Known defects / gaps (each maps to a workstream below):

1. `scale_models/generate_code.py::generate_dram_txt_files` writes **all zeros** (`str(0)`), and the
   scale-model testbench has **no golden compare / pass-fail**. R1 found this.
2. No design manifest; no per-design status/failure table; no impl stratified-sampler script or seed.
3. Table 4 LLM row: code uses `vals_num_groups = [1, 2, 4]` (→ 3,888); paper prints {1,2,4,8} (→ 5,184).
4. `dropout` is now an identity at inference (fix 0f6ca97), so the 3 dropout-probability variants per
   LLM point may be **functionally/structurally identical** → possible duplicate designs.
5. csynth metrics: gemm 3,840/3,840, conv **5,183/5,184**, llm 3,888/3,888; 0 designs over ZCU102
   capacity (per `*_util` columns — double-check the definition). Impl: 2,716 staged, 2,711 finished,
   5 timeouts (3 gemm, 2 llm).
6. Verification numbers disagree: paper says 17 operators / 57 variants / 1,000 inputs; repo says 69
   variants, 6,900 (n=100 full) + 69,000 (n=1000 exe) trials. Reconcile and let one script own the count.
7. `verification/results.csv` (whole designs) is stale: shows gemm `mlp` + 2 llm FAILs from before fixes.
8. Catapult backend: wired into **gemm only**; validated only by `verification/catapult_syntax_check.py`
   (g++ `-fsyntax-only`). No Catapult synthesis run exists. `analysis/extractors/catapult.py` is a stub.
9. No HLSFactory integration files anywhere (paper Fig. 4c/4d are snippets only).
10. No source-to-source tool evaluation scripts/logs (Table 6) in the repo.
11. Table 3 (full-model min/max LUT/BRAM/DSP) has no backing data in the repo:
    `scale_models/hls_synth_utilization.csv` holds old LLaMA-3B/ViT/Longformer rows, and
    `analysis/results_impl/metrics_scale_csynth.csv` shows tiled ResNet-18 at ~1.12M LUT vs 331K in the
    paper. The generating configs live on Hanqiu's server (`*/auto_generated_configs` is gitignored).
12. Junk committed: `jgproject/` (JasperGold session), `.claude/`, root planning docs, duplicated
    legacy `parse_*`/`analyze_report_*` scripts, `gemm/out.txt`, PNGs in `plotting/`.
13. ~960 MB of binary archives in git (ML_testsuite parts, checkpoint tarballs). GitHub caps files at
    100 MB; history is already ~480 MB packed.

## 2. Ground rules

- **Branch:** create `revision-r2` from `origin/feature/correctness-harness`. All work there, small
  focused commits (`area: what`), then one PR to `main` at freeze. Never force-push `main`; never
  rewrite shared history.
- **Do not change emitted Vitis code for the 12,912 sweep designs** unless a workstream says so. Any
  generator change must pass a regression check: regenerate a fixed sample (≥50 per domain, fixed seed)
  and byte-compare `top.cpp`/`top.h`/`run_hls.tcl` against the pre-change output. If a change is
  unavoidable, stop and write a note in `docs/revision_r2/CHANGELOG_R2.md` listing which committed
  metrics became stale.
- **Data files > 50 MB do not go in git.** They go to the release archive (§A5). Commit small CSVs,
  JSON configs, manifests, scripts, logs (compressed if large), and checksum files.
- Every result table must be regenerable by one command from committed inputs (§I).
- Seeds: every random choice takes an explicit seed recorded in the output CSV.
- Tool versions: record `vitis_hls -version`, Vivado version, Catapult version, g++ version, Python
  package versions (from `uv.lock`) in `docs/TOOL_VERSIONS.md` and in each results CSV header/sidecar.
- Do not edit the manuscript. Produce LaTeX table fragments under `paper_artifacts/tables/` and
  figures under `paper_artifacts/figures/`; the human pastes them.
- When you hit a **[SERVER]** step: write the script, validate it locally as far as possible (dry-run,
  g++ path), then stop and print exact instructions (command, expected runtime, expected output files,
  what to commit). Continue with independent workstreams meanwhile.

## 2b. Agreed verification matrix (decided 2026-09-30 — this is the scope for C, D, E)

| Tier | Designs | Data types | CSIM | CO-SIM | Golden |
|---|---|---|---|---|---|
| 1. Core operators | all 69 operator variants (`verification/op_configs/`) | float, `ap_fixed<16,5>`, `ap_fixed<32,10>` | 1,000 seeds per variant×type (exe mode) | 20 seeds per variant×type, **looped inside one CO-SIM run** (207 runs) | NumPy float64 on the same quantized inputs |
| 2a. Modularization suite | all 13 test cases: every input program P1–P3, each shared module, each modularized design (~40 designs) | float, `<16,5>` | 100 seeds | 10 seeds | per-P golden; the modularized design must reproduce **every** P's outputs (equivalence check) |
| 2b. Example ML designs | the 50 whole-design `test_case_configs` + a seeded 30-per-domain sample of the 12,912 sweep (`manifest/verif_sample.csv`) | float, `<16,5>` | 100 seeds | 10 seeds on ~10 designs per domain | NumPy float64 |
| 3a. ResNet-18 e2e | full generated ResNet-18 | float (random weights); `<16,5>`, `<32,10>` (pretrained) | float: 3 images, logits vs torchvision; fixed: ~20 images, top-1 agreement vs float | if time: 1 image at reduced input (e.g. 32×32); LightningSim for latency only | torchvision |
| 3b. LLaMA e2e | reduced-width LLaMA-3, same code path, prefill 16 tokens + 8 decode steps | float; mixed `<16,5>`/`<32,10>` | 1 sequence, compared per step | not planned | HF `LlamaForCausalLM` |

Rules:
- Float = pass/fail functional check. Fixed-point = **error characterization** (max-abs, RMSE, SQNR,
  overflow count, input range, seed) — never labelled PASS/FAIL against the float tolerance.
- CO-SIM testbenches must loop N input sets inside a single `cosim_design` run (reload DRAM inputs and
  goldens per trial, one verdict per trial) so RTL build/elaboration is paid once. Add a `--trials`
  option to the tb generator; default 1 keeps old behaviour.
- The modularized reference designs currently ship **without testbenches or DRAM input files**
  (`*/hls_files/*_module/` has only `top.cpp/top.h/run_hls.tcl`). Build tb + inputs for them, work out
  from each `top.cpp` how the shared design is invoked for each P, and compare its outputs against the
  corresponding P's golden **and** against P's own CSIM output.
- LightningSim (Sarkar & Hao) reproduces cycle latency from the C trace; it is **not** RTL functional
  evidence. Check it supports Vitis 2024.1 before integrating; if not, skip it.
- **Ownership:** Tier 3 (ResNet-18 and LLaMA full-model generation, export and e2e runs) is owned by
  **Hanqiu Chen**. He either runs it or hands code to Andy. Your job in D is the shared infrastructure
  (D1, D2, the golden/compare harness and the run scripts). Don't restructure `scale_models/`
  generator internals without coordinating through the human.
- All csynth/impl results and the July impl driver are on Andy's server; Catapult access is confirmed.
  Table 6 was run by Hanqiu in July; his commits, case list and logs will be provided.

## 3. Workstreams (priority order)

Priorities: **P0 = must ship; P1 = should ship; P2 = stretch.**

### A. Consolidate and clean the repo (P0) — do first, day 1–2

A1. Tag before touching anything: `r1-public-snapshot` → `origin/main`; `r1-july-harness` →
    `origin/feature/correctness-harness`; `archive/catapult-dev`, `archive/FPT_experiment`,
    `archive/updated_LLM_bench` → their tips. Push tags.
A2. Confirm `git rev-list --count origin/feature/correctness-harness..origin/catapult-dev` is 0. Inspect
    the 2 unique `FPT_experiment` commits (`git show --stat 8186bbc 7deafd2`); if anything is used by the
    paper's current figures, cherry-pick it, else leave it archived. Report to the human which branches
    are safe to delete (do not delete remote branches yourself).
A3. Create `revision-r2`. Add `.gitattributes` (`* text=auto eol=lf`, binaries marked `binary`) and
    renormalize in a dedicated commit if needed.
A4. Hygiene commit(s): `git rm -r jgproject .claude`; move `HANDOFF.md`, `PAPER_CHANGES.md`,
    `VERIFICATION_FLOW_PLAN.md`, `ForgeBench_Review_Summary_and_Action_Plan.md` and this brief into
    `docs/revision_r1/` and `docs/revision_r2/`; move superseded per-domain `parse_*_resource_utlization.py`,
    `analyze_report_data_new_plot_final.py`, `display_synth.py`, `out.txt`, `plotting/*.png` into
    `legacy/` (or delete if `analysis/` fully replaces them — check imports first). Update `.gitignore`.
A5. Large artifacts: stop committing multi-GB archives to git. Create `release/` tooling
    (`release/build_release.sh`) that assembles, outside git: (i) all 12,912 sweep JSON configs,
    (ii) lean csynth report archives, (iii) lean impl report archive, (iv) full-model configs + reports,
    (v) verification logs, (vi) Catapult / HLSFactory / tool-eval logs — then writes `SHA256SUMS`.
    The human uploads the bundle to Zenodo (and a GitHub Release). The git repo keeps
    `release/MANIFEST_FILES.csv` (file, size, sha256, zenodo path). Remove `ML_testsuite_part_*` from
    the tip of the branch (history can keep them) and replace README references. Ask the human before
    any history rewrite (e.g. `git filter-repo`) — default is **no rewrite**.
A6. Top-level layout target:
    ```
    forgebench/ (or keep gemm/ conv/ llm/ scale_models/ as-is if a move breaks imports; prefer as-is)
    backends/  verification/  analysis/  integrations/hlsfactory/  tool_eval/
    manifest/  paper_artifacts/  release/  docs/  legacy/
    README.md  REPRODUCE.md  Makefile
    ```
    Prefer **not** moving the domain generator folders (paths are baked into scripts); if moved, fix
    every path and rerun the regression check.

Acceptance: `revision-r2` builds clean from a fresh clone; `git status` clean; no file > 50 MB at tip;
all tags pushed; README no longer mentions 6000+/5,400.

### B. Design manifest and counts funnel (P0) — addresses R1 M2, M4

B1. `manifest/build_manifest.py` → `manifest/designs.csv`, one row per generated sweep design:
    `design_id, domain, suite, <every sweep parameter as its own column>, data_type, config_path,
    config_sha256, generated (bool), csynth_status {ok, fail, timeout, invalid_report},
    csynth_fail_reason, over_capacity (bool + which resource), in_fig6, in_fig7,
    impl_selected, impl_stratum, impl_status {ok, timeout, fail, not_selected}, impl_fail_reason,
    in_table5, in_fig8, report_archive, report_path_in_archive, duplicate_of (design_id or empty)`.
    Build it by regenerating the configs from `*/auto_generate_json.py` (no HLS needed) and joining
    `analysis/results_csynth/*.csv`, `analysis/results_impl/*.csv` and the checkpoint archives.
B2. `manifest/counts.py` → `paper_artifacts/tables/counts_funnel.{csv,tex}`: per domain —
    generated, csynth ok, csynth fail (with reasons), timeouts, invalid reports, over-capacity, plotted in
    Fig 6/7; impl staged, ok, timeout, fail; plotted in Fig 8 / Table 5.
B3. Find the missing conv design (5,184 − 5,183): identify its ID and the failure reason from the
    checkpoint archive/logs. If the log is gone, mark `csynth_status=fail, reason=unknown (log not
    retained)` and add a [SERVER] one-design rerun script.
B4. Duplicate check: for LLM configs that differ only in `dropout_prob`, diff the generated `top.cpp`
    and compare csynth metrics. Report exactly how many designs are byte-identical / metric-identical,
    fill `duplicate_of`, and produce counts both "as generated" and "unique". **Do not delete configs;**
    the human decides how the paper reports this. Run the same identical-source check across all
    domains (hash normalized `top.cpp`).
B5. Table 4 regeneration: `paper_artifacts/tables/table4_sweep.tex` generated from the actual value
    lists in the three `auto_generate_json.py` files (single source of truth), including an explicit
    "how the product equals N" line per domain. Groups must print {1,2,4}.
B6. Impl sampling reproducibility: the sampler that chose the 2,716 impl designs is not committed.
    (a) Ask the human to recover it from the server (`_impl1000/` driver). (b) Meanwhile, infer the rule
    from the selected IDs vs the full manifest (per-domain counts per unroll factor / dims etc.), write
    `analysis/impl_sampler.py` that reproduces that exact selection from a seed, and assert
    set-equality with the committed list. If exact reproduction is impossible, document the actual
    selected list as the canonical definition (`manifest/impl_selection.csv`) and describe the strata
    empirically (per-stratum population vs selected counts table).
B7. Over-capacity: verify the `bram_util/dsp_util/lut_util/ff_util` denominators are ZCU102 totals
    (BRAM 1824, DSP 2520, LUT 274,080, FF 548,160). Report the count that exceed any.

Acceptance: `make manifest` recreates `designs.csv` and the funnel table deterministically; every
number in the paper's §4.2 is traceable to a row filter on `designs.csv`.

### C. Fixed-point verification (P0) — addresses R1 M1

The paper's claim "float-verified ⇒ holds at any fixed-point width" is withdrawn. Replace it with
measured fixed-point error characterization.

C1. Local path: vendor or fetch the open-source Xilinx arbitrary-precision headers
    (`github.com/Xilinx/HLS_arbitrary_Precision_Types`, record the commit) so `ap_fixed` designs compile
    with g++. Keep `verification/shim/` for float.
C2. Extend `verification/verify_operators.py` and `csim_runner.py` with `--dtype` accepting
    `float`, `ap_fixed<16,5>`, `ap_fixed<32,10>`, and the mixed LLaMA template setting (storage <16,5>,
    accumulate/normalize <32,10>) where the generator supports it.
C3. Golden: float64 NumPy (not float32) on the **same quantized inputs** the design receives
    (quantize inputs with the same AP_TRN/AP_WRAP semantics before computing golden), so the measured
    error is the design's internal quantization/overflow, not input rounding. Also report error vs the
    unquantized-input golden as a second column.
C4. Inputs: per-operator input ranges chosen inside the representable range of the tested type
    (<16,5> is [−16, 16) with 2⁻¹¹ resolution). Record `input_range`, distribution (uniform), seed
    base, N trials in the CSV. Add one deliberate **stress** range per operator family that provokes
    overflow, to show wraparound effects honestly (report separately; do not call it pass/fail).
C5. Metrics per operator variant × dtype: max_abs_err, mean_abs_err, RMSE, SQNR (dB), relative L2,
    fraction of elements within (atol, rtol) at a dtype-appropriate tolerance (document the tolerance
    derivation, e.g. k·2⁻ᶠ scaled by accumulation length), overflow/saturation event count (instrument
    via a debug build that checks each intermediate `ap_fixed` assignment against range — or compare
    against a wide-type reference build; document the method). Quantization mode is the Vitis default
    AP_TRN / AP_WRAP unless the generator sets otherwise — **verify from emitted code and state it.**
C6. Special attention (R1 names these): softmax, layernorm, rmsnorm, RoPE (sin/cos/pow), attention
    with long accumulation, GEMM with K up to 512. Report accumulation length alongside error.
C7. Outputs: `verification/results_fixed/ops_<dtype>_n<N>.csv`, per-trial logs compressed,
    `paper_artifacts/tables/verif_ops.tex` (float pass + fixed error columns).
C8. **[SERVER]** `bash verification/run_fixed.sh` — Vitis CSIM exe-mode, N=1000, all three dtypes
    (Tier 1). Then `bash verification/run_designs.sh` — Tier 2a + 2b CSIM (N=100, float and <16,5>),
    which also replaces the stale whole-design `verification/results.csv`.
C9. Reconcile operator/variant counts with the paper: one script prints "N distinct operators, M
    variants, K trials" and the table uses it.

### D. End-to-end full-model validation (P0) — addresses R1 M1, M3

Goal: show the **generated full-model code path** computes the same function as an independent
reference, end to end, at a scale CSIM can run. Same generator code, reduced dimensions.

D1. Fix `scale_models/generate_code.py::generate_dram_txt_files`: random values with explicit seed and
    per-DRAM `input_range` (mirror the conv/llm fix in `verification/`), or load provided tensors
    (weights/inputs) from files. Keep a flag to reproduce old behavior only if needed for regression.
D2. Scale-model testbench: add golden compare + `VERIFICATION: PASS/FAIL` + non-zero exit, reusing the
    verification tb snippet. Dump all output DRAMs, plus (debug flag) per-layer intermediates.
D3. Reduced-width configs, **no code-path changes**: expose CLI overrides in
    `scale_models/auto_generate_json.py` for LLaMA (`--layers --hidden --heads --kv-heads --head-dim
    --ffn --vocab --ctx`) and ResNet (`--input-hw`, channel width multiplier only if the generator
    supports it; otherwise use full-width ResNet-18 at reduced input resolution, e.g. 64×64 or 224×224
    if CSIM time allows). Defaults remain the real LLaMA-3.1-8B / ResNet dims. Keep: all 32 layers
    where feasible (else document), GQA ratio 4:1, RoPE θ = 500,000, RMSNorm eps, SwiGLU, KV cache.
    **Check whether Llama-3.1 RoPE frequency scaling ("llama3" rope_scaling) is implemented; if not,
    record that as an explicit assumption and configure the golden identically.**
D4. Weight/input export: `verification/e2e/export_resnet.py` (torchvision ResNet-18, eval-mode BN
    folded or not — match the generator; random weights with seed for float check; pretrained weights
    for fixed-point agreement) and `verification/e2e/export_llama.py` (HF `transformers`
    `LlamaForCausalLM` with a `LlamaConfig` mirroring the reduced config, `torch.manual_seed(0)`).
    Write tensors into the generator's DRAM layout (`[num_layers][dim1][dim2]` etc.). Verify layout by
    a round-trip test.
D5. Prefill → decode sequence: prefill and decode are separate generated tops. Write
    `verification/e2e/run_llama_sequence.py`: run PREFILL CSIM on P=16 prompt tokens, capture the KV-cache
    and hidden-state DRAM dumps, feed them to DECODE CSIM for steps 1..8 (greedy: next token = argmax
    of the golden, teacher-forced so both sides see the same token), and compare at every step:
    logits max_abs_err, cosine similarity, argmax agreement, KV-cache max_abs_err. Golden = HF model
    with `use_cache=True` run on the same token sequence.
D6. ResNet: compare logits and the final feature map; report max_abs_err, cosine, top-1/top-5 agreement.
    Fixed-point run on ~20 images with pretrained weights: top-1 agreement vs float.
D7. Validate everything locally with g++ + float first (small config) before handing to the server.
D8. **[SERVER]** `bash verification/e2e/run_e2e.sh {resnet,llama} {float,fixed16,fixed32}`.
    Stretch (P2): one full-width (hidden 4096) LLaMA decoder layer, float, vs HF.
D9. Commit: exact full-model and reduced configs (JSON), generated source for the reduced designs,
    testbenches, logs, `verification/e2e/results_e2e.csv`, `paper_artifacts/tables/e2e.tex`.

### E. CO-SIM and the stage-status table (P0/P1) — addresses R1 M1, M6

E1. `verification/run_cosim.sh` **[SERVER]** per the §2b matrix: Tier 1 (69 variants × 3 dtypes ×
    20 looped seeds), Tier 2a (~40 modularization designs × 2 dtypes × 10 seeds), Tier 2b (~10 designs
    per domain × 2 dtypes × 10 seeds). Parse per-trial verdicts, error stats and cosim latency into
    `verification/results_cosim.csv`; compare cosim latency with the csynth estimate per design.
    Tier 3a reduced-input ResNet-18 CO-SIM only if time allows (Hanqiu's design).
E2. `analysis/status_table.py` → `paper_artifacts/tables/status.tex`: one row per artifact class
    (operator variants; sweep designs per domain; modularization test cases; ResNet-18..152; LLaMA-3.1-8B
    prefill/decode; reduced e2e models) with columns: configs released · source generated · CSIM
    (float) · CSIM (fixed) · C-synth · CO-SIM · impl (P&R) · e2e numerical check · on-board run. Cells
    are counts "passed/attempted" or "—" (not attempted). Derived from result CSVs, never hand-typed.
E3. `analysis/model_support_table.py` → Table 1 replacement: per model family, operators available /
    expressible in principle / JSON released / full source generated / stages passed.

### F. Catapult backend evidence (P1) — addresses R1 M5

F1. Implement `analysis/extractors/catapult.py` (area, latency, throughput/II, clock) against real
    Catapult report files once one exists.
F2. `backends/catapult_run.py` + **[SERVER]** `bash backends/run_catapult_subset.sh`: stratified 96-design
    GEMM subset (all 6 loop orders × 8 dim triples × unroll {all-1, all-8}, with_bias alternating,
    fixed seed; write the selection to `manifest/catapult_selection.csv`). For each: generate with
    `tool="catapult"`, run Catapult (C-sim with golden compare using `data_type=float` and ac_fixed
    <16,5>, then synthesis to the configured target), record status, failure reason, manual edits
    (must be none — if any are needed, log them as a patch file and count them).
F3. Results: `results/catapult/metrics_gemm.csv`, logs, `paper_artifacts/tables/catapult.tex`
    (version, N attempted/passed per stage, area/latency ranges vs the same 96 designs under Vitis).
F4. Catapult access is confirmed (Andy's server). Record the exact Catapult version and target
    library in `docs/TOOL_VERSIONS.md`. If runs are still blocked on 2026-10-12, fall back to
    `docs/CATAPULT_STATUS.md` stating exactly what exists (GEMM backend + header syntax check).

### G. HLSFactory integration example (P1) — addresses R1 M5

G1. `integrations/hlsfactory/`: pin HLSFactory by commit (record it), `README.md` with exact commands,
    `forgebench_dataset.toml` and `opt_dsl.tcl` (the Fig 4c/4d content, fixed to be valid), a
    `run_example.py` that (1) generates 3 ForgeBench designs (one per domain) from committed configs,
    (2) registers them as an HLSFactory dataset, (3) runs OptDSL expansion to a small number of variants
    (e.g. ≤ 16 per design), (4) runs the Vitis synth flow, (5) writes the aggregated dataset CSV.
G2. Local dry-run up to (3); **[SERVER]** step (4). Commit `expected_output/` (the CSV + logs) so
    anyone can diff.

### H. Source-to-source tool evaluation (P1) — addresses R1 M5, m4

H1. Hanqiu ran the July evaluation and will provide scripts, tool commits, case list and logs (via the
    human). Commit them under `tool_eval/` unchanged, then build the results table from them.
H2. Only for gaps his material doesn't cover, build `tool_eval/` with one directory per tool (ScaleHLS, HIDA, AutoSA, StreamHLS):
    Dockerfile or install notes pinned to a commit, `run.sh`, a fixed case list
    (`tool_eval/cases.csv`: e.g. 20 GEMM, 20 DNN, 20 LLM design IDs from the manifest, seeded), and a
    driver recording for each case the furthest stage reached {parse, transform, codegen, csynth} and
    the error signature mapped to F1–F6.
H3. Output `tool_eval/results.csv` and `paper_artifacts/tables/table6.tex` with columns: tool, version/
    commit, cases per domain, edits required, stage reached per domain, F1–F6. **No Allo row.**

### I. Modularization table detail (P1) — addresses R1 m5

I1. Extend `modular_data/parse_synth_resourc_util.py` (or port to `analysis/`) to emit LUT, FF, DSP,
    BRAM (absolute and %), latency (cycles), II and throughput for P1–P3, shared module and overall
    design. Keep the old averaged metric as one labelled column. Output
    `paper_artifacts/tables/table7.tex`. If any reports are missing, write a [SERVER] csynth rerun script.

### J. Representativeness evidence (P2) — addresses R1 M4

J1. `analysis/shape_corpus.py`: hard-code (with citations in comments) layer shapes for ~8 models
    (ResNet-18/50, MobileNetV2, ViT-B/16, BERT-base, GPT-2, LLaMA-3-8B, Mistral-7B): GEMM (M,N,K);
    conv (H,W,Cin,Cout,k,stride,groups); attention (seq, d_model, heads, kv-heads, head_dim).
J2. `analysis/shape_coverage.py`: for each sweep design, list which corpus layers it is an exact tile
    of (dimensions divide the layer's), and report the fraction of corpus layers that are tile-covered
    by the sweep, per domain. Output a small table + one figure. Report honestly; the human decides the
    wording.

### K. Full-model data recovery (P0, human-dependent) — addresses R1 M2, M3; Table 3

K1. Ask Hanqiu (via the human) for: the exact `auto_generated_configs/*.json` used for Table 3 (ResNet
    18/34/50/101/152 min & max configs; LLaMA-3.1-8B prefill/decode min & max), their csynth reports,
    and the command lines used.
K2. Commit configs under `scale_models/paper_configs/`, lean reports into the release bundle, and a
    `analysis/tables.py table3` that regenerates Table 3 from them. If values differ from the paper,
    flag it loudly — do not "fix" the paper numbers.
K3. If unrecoverable: **[SERVER]** script to regenerate + csynth exactly the documented min/max configs.

### L. Reproducibility docs and release (P0) — addresses R1 M2, m6

L1. `README.md` rewrite: what ForgeBench is (generator), counts that match the manifest, layout,
    quick start, link to REPRODUCE.md, citation, license. No "ready-to-use", no "6000+".
L2. `REPRODUCE.md`: environment (uv, Python version, g++ version, Vitis 2024.1 [exact build], Vivado,
    Catapult), a **minimal clean-environment procedure** (fresh clone → `uv sync` → `make manifest`
    → `make selfcheck` [g++ float verification, no Vitis] → `make paper-artifacts` [all tables/figures
    from committed CSVs]), and the full server procedure per experiment with measured runtime and
    storage (from July: csynth sweep; impl ~17 h at 32 jobs, ~363 GB raw build trees; lean archives
    sizes; CSIM rates).
L3. `Makefile` targets: `manifest`, `selfcheck`, `paper-artifacts`, `release`, `test`.
L4. `paper_artifacts/INDEX.csv`: paper item (Table 2…7, Fig 6…8, counts in text) → script → inputs →
    output file. Every item in the manuscript must appear.
L5. Clean-room test: in a fresh container (`docker run python:3.12`), clone the tag, run the
    clean-environment procedure, and diff regenerated tables/figures against committed ones. Record the
    run in `docs/revision_r2/cleanroom_log.txt`.
L6. At freeze: bump version, write `CITATION.cff`, open PR `revision-r2 → main`, human merges, tag
    `v2.0-trets-r2`, build release bundle + `SHA256SUMS`, human uploads to Zenodo.
    **Decided 2026-09-30: no anonymous mirror.** Code lives on GitHub (tagged release, GitHub Release
    notes linking the DOI); all large data (sweep configs archive, lean csynth/impl report archives,
    full-model configs/reports, verification/CO-SIM/Catapult/HLSFactory/tool-eval logs) lives on Zenodo.
    Prepare `release/zenodo_metadata.json` (title, authors + ORCIDs as TODO, description, license,
    related identifiers: GitHub tag URL and the paper DOI placeholder) and a `.zenodo.json` in the repo
    root. README badges: DOI + release tag. Remove every `anonymous.4open.science` reference. Put the commit hash, tag and DOI in README and in
    `docs/revision_r2/RELEASE_NOTES.md`.

## 4. Schedule for the agent

| Dates | Agent work | Human/server |
|---|---|---|
| Oct 1–3 | A (all), B1–B5, B7, D1–D2 | Recover K1 data, impl sampler, tool-eval logs; confirm Catapult license |
| Oct 4–9 | C1–C7, D3–D7 (local float), E2–E3 skeletons, B6, G1 | Run C8 (fixed CSIM + refreshed whole-design CSIM), E1 CO-SIM |
| Oct 10–16 | D8 scripts, F1–F3, G2, H, I, J | Run E2E (D8), Catapult (F2), HLSFactory (G2), tool eval (H2 if needed) |
| Oct 17–21 | Parse all results; finish E2, L1–L5; regenerate every table/figure | Review numbers; decide dedup + wording |
| Oct 21 | **Freeze** results; PR to main | Merge, tag, Zenodo |
| Oct 22–30 | Only bug fixes that do not change numbers | Paper + response letter |

## 5. Definition of done (checklist for the PR description)

- [ ] `main` == what the paper describes; tag `v2.0-trets-r2`; Zenodo DOI; SHA256SUMS.
- [ ] `manifest/designs.csv` covers all 12,912 (or the final agreed count) with status + reasons.
- [ ] Counts funnel, Table 4, Table 6, Table 7, status table, verification tables, e2e table, Table 3
      all regenerate via `make paper-artifacts` from committed inputs.
- [ ] Fixed-point operator results (both dtypes), CO-SIM results, refreshed whole-design CSIM committed.
- [ ] E2E ResNet + LLaMA prefill/decode results vs independent golden committed with logs.
- [ ] Catapult: results committed **or** `docs/CATAPULT_STATUS.md` states prototype scope.
- [ ] HLSFactory example runs and has expected output committed.
- [ ] No junk (jgproject, .claude, stale docs at root); no file > 50 MB at tip; README counts match.
- [ ] Clean-room reproduction log committed.
- [ ] `docs/revision_r2/CHANGELOG_R2.md` lists every generator change and any metrics it made stale.

## 6. Things to ask the human (do not guess)

Answered 2026-09-30: Catapult access confirmed; csynth/impl results and the impl driver are on Andy's
server; Hanqiu owns Table 6 material and all full-model (Tier 3) work.

1. Exact Catapult version and target library (for F).
2. Paths on Andy's server to the impl driver/sampler and the raw csynth/impl trees (for B6, B3).
3. From Hanqiu: Table 3 configs/reports (K); whether he runs LLaMA e2e himself or hands over code (D).
4. Whether history rewrite to purge large blobs is acceptable (default: no).
5. How to report duplicate dropout variants (as generated vs unique) once B4 has numbers.
6. Whether pretrained weights may be downloaded on the server for D6 (torchvision) — otherwise random.
