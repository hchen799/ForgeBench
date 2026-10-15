# Full-model verification: answers for the restructure (2026-10-15)

Answers to the cleanup session's questions A1–E17. Paths are relative to the repo root unless noted. "Production" means the
designs the generator emits from JSON (`scale_models/auto_generate_json.py` → `gen_configs.run_hls_flow` →
`scale_models/hls_files/<project>/top.cpp`).

## A. Which verifier

1. **The paper relies on `scale_models/existing_model_verification/`.** `verify.py` compiles the saved, unchanged `top.cpp` of a
   production project in Vitis C simulation, against two references:
   - a fixed-point PyTorch reference (integer codes);
   - an FP64 PyTorch model.

   `verification/paper_runs/run_fullmodel.sh` calls it, and every full-model number in the paper comes from it.
   `scale_models/verification/` is a different thing, not an older or newer version. It verifies Hanqiu's dedicated
   "configurable" accelerator generators (its own ResNet-18 and Llama templates, `models/*/codegen.py` + `kernel.cpp.in`). Its
   README says so: "their PASS results do not establish correctness of the production JSON-to-HLS designs". "Generic" there means
   configurable dimensions and formats for those two dedicated models, not arbitrary configs. Both trees are on this branch
   (Hanqiu, 2026-10-04..06); neither is in an unmerged branch.
2. **Nothing in `existing_model_verification/` imports from `scale_models/verification/`** (`precision.py`, `reporting.py` and
   `generic.py` included). The only matches are path strings inside preserved log files. The configurable tree is not needed for
   the paper and can leave the release, or move to `legacy/` / Zenodo. It sits inside `scale_models/`, whose layout the user has
   asked not to touch, so confirm with the user first.
3. **The legacy trees and `results/historical_v1/` are not used by the paper:**
   - `scale_models/verification/legacy/{llama3_verification,tiled_resnet18_verification}` are Hanqiu's earliest dedicated
     verification implementations, run by `scale_models/verification/run_legacy_llama3.sh`.
   - `existing_model_verification/results/historical_v1/` holds results for reference-arithmetic version 1 (sources superseded by v2).

   Same caveat as 2: they are inside `scale_models/`, so confirm with the user.

## B. Scope

4. **Table 3 has 14 rows, all claimed as verified:**
   - ResNet-18/34/50/101/152 **full** at `<32,10>` RND/SAT (`scale_models/hls_files/RESNET*_config_ap_fixed_32_10_`, new conv
     schedule);
   - ResNet-18/34/50/101/152 **tiled** at `<16,5>` **truncate/wrap** with an 8×32 conv tile (`_sweeps/fullmodel_r4_trn/hls_files/RESNET*_TILED_ci8_co32_config_ap_fixed_16_5_`,
     **not yet in `scale_models/`**; see 6 and 17);
   - Llama-3-8B {prefill, decode} × {ctx2048, ctx8192} at `<32,10>` RND/SAT.

   Of the 29 projects in `scale_models/hls_files/`:
   - the 14 `*_32_10_` are the R2 sources: all registered, all verified;
   - the 14 `*_16_5_` are Hanqiu's earlier truncate/wrap sources: registered, verified by Hanqiu with one seed, and not in the
     paper;
   - `RESNET18_tiled_c128_h14_w14_ap_fixed_16_5_` is unregistered and unused.

   The 5 tiled `<32,10>` projects are verified (10 seeds) but are no longer Table 3 rows.
5. **Tracked results** (`verification/paper_results/fullmodel/`, 3.9 MB) are the pre-schedule `<32,10>` sources: 10 ResNets × seeds
   42–51, plus `LLAMA3_8B_ctx2048` × 42–44. The paper's current numbers come from newer runs in `verification/results_paper/`
   (ignored, raw):

   | Run | What | Size |
   |---|---|---|
   | `fullmodel_r2_sched/` | new-schedule `<32,10>` ResNets, 10 × 10 seeds, all PASS (identical to the tracked runs) | 26 GB |
   | `fullmodel_ctx8192_r2/` | Llama ctx8192 × 3 seeds, PASS | 31 GB |
   | `fullmodel_r4_16_5_trn/` | tiled `<16,5>` truncate/wrap 8×32 ResNets, 5 × 10 seeds, all PASS (**the paper's**) | |
   | `fullmodel_r3_16_5/` | the same at round/saturate (superseded: FP64 failures, see 6) | 9.1 GB |
   | `fullmodel_check_r1src_16_5/` | seed-42 isolation runs for 6 and 17 | |

   These need their compact parts (`summary.txt`, `summary.json`, `manifest.json`, `inputs_manifest`) copied into
   `paper_results/fullmodel/` the way the tracked runs were.

   **`LLAMA3_8B_ctx2048` is one combined run:** `--project` is the prefill project and `--decode-project` the decode project, with
   4 prefill tokens then 2 decode calls. Table 3 reports the prefill call (`call0000`) on the prefill row and the two decode calls
   (`call0001`, `call0002`) on the decode row. ctx2048 and ctx8192 give identical numbers: same seed, same 6 tokens, and only the
   cache capacity differs.
6. **The tiled `<16,5>` ci8/co32 sources are verified:** 50 runs, all bit-exact (0 mismatching codes). Against FP64, ResNet-18
   (10/10) and ResNet-50 (10/10) pass at ≤ 0.03% relative L2; ResNet-34 passes 6/10, ResNet-101 2/10 and ResNet-152 0/10 (up to
   81%). Seed-42 isolation runs show the cause is the **round/saturate** storage mode, not the 8×32 schedule:

   | ResNet-152 tiled `<16,5>`, seed 42 | rel. L2 at logits |
   |---|---|
   | Hanqiu's source (truncate/wrap, 8×64) | 2.046% PASS |
   | 8×32 schedule, truncate/wrap | 2.046% PASS (identical) |
   | 8×32 schedule, round/saturate (Table 3 row) | 71% FAIL |

   **User decision (2026-10-15): truncate/wrap.** Re-run `_sweeps/fullmodel_r4_trn` → `results_paper/fullmodel_r4_16_5_trn`: 50/50 PASS,
   0 mismatching codes, relative L2 0.44–2.42%. The round/saturate runs (`fullmodel_r3_16_5`) are superseded.
7. **The 24 variants** (`existing_model_verification/results/validation_summary.json`, README "all 24") are 12 models × 2 formats:
   10 ResNets, plus Llama ctx2048 and ctx8192 (each a prefill+decode pair), at `<16,5>` and `<32,10>`. They are Hanqiu's single
   seed (42), on his earlier sources: truncate/wrap `<16,5>`, and the `<32,10>` revision before R2's round/saturate one. They are
   **not** the paper's results. The paper's are the R2 runs in 5 (and the tracked `paper_results/fullmodel/`).

## C. Testbenches and goldens

8. **There is one verifier per family** (`--family resnet18 … resnet152 | llama3`), run against any project directory.
   - `verify.py` snapshots the project (`source/`), emits an external testbench (`case/`) and compiles it with the unchanged
     `top.cpp`.
   - The golden is computed at run time: seeded synthetic parameters and input → fixed-point PyTorch reference (integer codes)
     and FP64 PyTorch. No stored expected-output files are used.
9. **The goldens are too large to ship.** One ResNet-152 run directory is about 0.5 GB (`inputs/` 231 MB, `case/` 271 MB), and 100
   ResNet runs are 26 GB. A Llama run writes a ~33 GB parameter store, which the script deletes afterwards.

   The reproducible substitute is what each run already records:
   - the seed;
   - the reference version, plus the `reference_sources` hashes in `summary.json`;
   - `manifest.json` / `inputs_manifest` (SHA-256 of every generated input tensor);
   - `summary.txt` / `summary.json` (verdicts, per-tensor errors).

   Shipping those makes every run re-checkable bit for bit.
10. **`references/contracts.json` pins, per variant** (e.g. `resnet18-tiled`, `llama3-prefill`), a list of revisions. Each revision
    is the SHA-256 of `top.cpp` and `top.h` plus the reference-arithmetic version it implements (2 = production wide
    accumulators). The project path is a command-line argument and is not pinned.
    - **Moving a project directory needs no change to `contracts.json`.** Only changing the bytes of `top.cpp` or `top.h` does.
    - New sources are added with `existing_model_verification/register_sources.py --family <f> --reference-version 2 <dir>`.
    - `verify.py` refuses unregistered sources, and fails if `contracts.json` changes while it runs.
11. **Acceptance per run** (`verify.py`) has two parts, and the run PASSes only if both hold at **every observed tensor**:
    - **implementation:** 0 mismatching integer codes against the fixed-point reference;
    - **mathematical:** relative L2 against FP64 ≤ 5%, or max abs ≤ 0.1 where the FP64 tensor is all zero.

    Table 3 reports the final-logit max abs error and relative L2, each the **maximum over seeds**, with no threshold applied.
    Mismatching codes are 0 everywhere.

## D. Generator dependencies

12. **The verifier imports nothing from the generator.** `verify.py` and `references/*.py` import only torch, numpy and their own
    package. The only links are:
    - `existing_model_verification/reproduce.py`, a regenerate-then-verify driver that calls `auto_generate_json.py` and
      `gen_configs`;
    - `register_sources.py`, which mentions `gen_configs` in its docstring.

    **Generation (paper):** `auto_generate_json.py`, `gen_configs.py`, `generate_code.py`, `production_types.py`,
    `*_template.cpp`.

    **Configurable / legacy (not the paper):**
    - `generate_tiled_resnet18.py` (imported by `verification/models/resnet18/` and the legacy tree);
    - `generate_tiled_llama3.py`, `golden_tiled_llama3.py` and `verify_tiled_llama3.sh` (used by
      `verification/run_legacy_llama3.sh`);
    - `golden_tiled_resnet18.py` (no users found).

    `cli_llama.py` and `cli_resnet.py` live in `scale_models/verification/legacy/`, not at top level.
13. **The symlinks are not needed by the production verifier:**
    - `scale_models/llama3_verification` and `tiled_resnet18_verification` point into the configurable tree's `legacy/`;
    - `verification_runs → verification/runs` is dangling.
14. **Tests** (not run in this session):
    - `existing_model_verification/tests/test_{existing,production,production_graph,production_wide}.py` test the production
      verifier. Keep them.
    - `scale_models/verification/tests/test_{configurable,resnet_formats}.py` test the configurable tree. They go wherever it goes.
    - `scale_models/verification/results/regression_tests.log` records 77 passing tests for that tree.

## E. Running it

15. **End to end** (from `scale_models/`, Python env with torch + numpy):
    1. **Generate:**
       ```
       python auto_generate_json.py --data-types "ap_fixed<32,10>"
       ```
       Optionally add `--conv-co-factors 32` for the tiled 8×32 designs. Then set `"data_type": "fixed<W,I,rnd,sat>"`, and
       `"schedule": "pipeline_inner"` on every `conv_tile` op (`_sweeps/fullmodel_r3/post.py`), and run
       `gen_configs.run_hls_flow(config, out)`. This reproduces the verified `<32,10>` tiled ResNet-18 `top.cpp` / `top.h` /
       `tb_top.cpp` byte for byte.
    2. **Register** new sources (if they changed): `register_sources.py`.
    3. **Verify:**
       ```
       SOURCES=<dir> OUT=<dir> PY=<python> bash verification/paper_runs/run_fullmodel.sh
       ```
       It only selects `RESNET*32_10*` projects and the ctx2048 Llama pair. ctx8192 used `_sweeps/llama_ctx8192_verify.sh`, and
       the tiled `<16,5>` ResNets used `_sweeps/fullmodel_r3/verify.sh`. These should become one parameterized script.
    4. **Table:**
       ```
       python3 -m analysis.table3_fullmodel
       ```
       It reads the run summaries and the csynth lean reports.

    **Time:**
    - ResNet: about 1–2 min per seed (ResNet-152 120 s). 100 runs took 26 min at 6 in parallel; 50 tiled runs took 13 min.
    - Llama: about 17 min per seed, sequential, plus a ~33 GB temporary parameter store per run.

    **Memory:** not measured. Hanqiu's README quotes about 60 GiB for the Llama-3-8B matrix cache plus activations.

    **Synthesis** (separate, `analysis.run_prebuilt`): ResNet full 18 min–29 h, tiled `<16,5>` 10–80 min, Llama 45–90 min.
16. **`verification/paper_runs/make_fullmodel_table.py` → `verif_fullmodel.tex`** is built from Hanqiu's `validation_summary.json`:
    a single seed on his earlier sources (the 24 variants in 7). It is **not consistent** with Table 3, which computes its own
    error columns from the R2 runs. Drop it, or rebuild it from the R2 run summaries; I don't know whether the current paper draft
    still includes it.
17. **Running:** csynth of the 5 tiled `<16,5>` truncate/wrap ResNets (`_sweeps/lean/fullmodel_csynth_r4_16_5_trn`); Table 3
    regenerates when it finishes.

    Decided: 6 (truncate/wrap), verified.

    No other full-model verification is planned before the freeze.
