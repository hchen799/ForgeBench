# Repo cleanup and restructure plan (R2)

Drafted 2026-10-11. This file merges plans that were spread over three places:
- `REVISION_R2_AGENT_BRIEF.md` §A4–A6: hygiene, large artifacts and layout.
- `VERIFICATION_HANDOFF.md` §2 item 5 and §4 step 7: ordering.
- The working decisions from the 2026-10-02 session: the `forgebench/` package direction.

Nothing here has been executed. Each phase needs the user's go-ahead.

## 0. Constraints and ordering

- **Running sweeps.** Do not move `backends/`, `gemm/`, `conv/`, `llm/` or `analysis/` while a sweep is running. `analysis.sweep_runner` imports `backends` from the repo root for each design. The LLM re-run (`analysis/run_llm_r2b.sh`, csynth then impl) is in progress as of 2026-10-11.
- **Full models first.** `scale_models/` layout changes and the package move come after the full-model consolidation, regeneration and synthesis (HANDOFF §4 steps 1–6).
- **Freeze.** Freeze is 2026-10-21; resubmission is 2026-10-30. The package move is the only risky step, so it either lands before the freeze with a regression check or waits until after resubmission.
- **Git.** No history rewrite by default. The user pushes; the agent commits locally only.

## 1. Open decisions (user)

| # | Decision | Options | Notes |
|---|---|---|---|
| D1 | Generator layout | (a) `forgebench/` package holding gemm, conv, llm, fullmodel, sweeps, backends and the CLI; (b) keep `gemm/ conv/ llm/ scale_models/` at top level | Brief A6 prefers (b), because paths are baked into scripts. The 2026-10-02 session agreed on (a), the generator-first README. (a) requires package-relative template paths and a byte-compare regression on every domain. |
| D2 | When to do D1(a) | before the freeze, after the LLM re-run and full-model synthesis / after resubmission | The paper's repo link and paths have to match whatever is released. |
| D3 | Public repo? | | This decides whether `docs/revision_*` leave the release branch, since they contain planning notes, a OneDrive path and the paper PDF (the PDF is gitignored). |
| D4 | History rewrite | none (default) / `git filter-repo` for old large blobs | Brief A5: ask first. |
| D5 | `modular_data/` (235 tracked files, 12 MB) | keep / move under `examples/` or the generator package | It is used by `manifest/build_registry.py`, `verification/manual_designs.py` and `analysis/extractors/vitis.py`; it holds the Table 7 designs. |
| D6 | `checkpoints/20260720/` (9 tracked metric CSVs) | keep as R1 provenance / move to `legacy/results_r1` | `legacy/results_r1` already exists. |

## 2. Phase H — hygiene (safe any time; no generator paths change)

Status of brief A4 as of today:

| Item | State | Action |
|---|---|---|
| `jgproject/` and `.claude/` | untracked and ignored; local only | delete locally if unwanted. Nothing to commit. |
| Planning `.md` files moved to `docs/revision_r1/`, `docs/revision_r2/` | done | — |
| Old parse/plot scripts moved to `legacy/` | done (`legacy/{conv,gemm,llm,plotting}`) | — |
| `plotting/*_data/*.csv` (17 tracked) | still at top level | These are the R1 metric inputs that `analysis/io.py` and `analysis/plots.py` read as a fallback. Move them to `legacy/plotting/` and update the two readers, or drop the fallback once every figure reads `analysis/results_*`. |
| `ModularHLS_testsuite.tar.gz` (4.5 MB, tracked) | still tracked at the tip | Move it into the release bundle (A5) and delete it at the tip. History keeps it. |
| `ML_testsuite_part_00..04` (321 MB) | untracked and ignored | Not in git. Keep them out of the repo, record them in `release/bundle_files.csv`, and remove the README references. |
| `_run_logs/`, `_sweeps/`, `tmux-*.log`, `vitis_hls.log` | ignored local scratch | Delete after the runs are archived into `checkpoints/r2/` and `manifest/evidence/`. |
| `.gitattributes` (brief A3) | check | Add `* text=auto eol=lf` and mark binaries. Renormalize in its own commit if needed. |

## 3. Phase R — release bundle (brief A5; tooling exists)

`release/build_release.sh`, `assemble_reports.py`, `package_lean.sh`, `make_manifest.py` and `bundle_files.csv` already exist. Remaining work:
- Add the R2 LLM re-run lean archives to `manifest/_common.py` SOURCES after collection.
- Add the verification results (`verification/results_paper/`, `verification/paper_results/`), the Catapult runs and the Catapult-vs-Vitis lean trees to the bundle.
- Do a dry run: check `SHA256SUMS` and that every manifest path resolves.
- Acceptance: no file over 50 MB at the tip, and a fresh clone builds.

## 4. Phase L — layout (brief A6 + D1)

Target, from brief A6. It changes to (a) if D1 says so.
```
gemm/ conv/ llm/ scale_models/      (or forgebench/{gemm,conv,llm,fullmodel,sweeps,backends,cli} under D1(a))
backends/  verification/  analysis/  integrations/hlsfactory/  tool_eval/
manifest/  paper_artifacts/  release/  docs/  legacy/  examples/
README.md  REPRODUCE.md  Makefile
```
- `examples/`: the Table 1 JSON examples (`<domain>/test_case_configs/` that are not Table 7 designs), per `manifest/README.md`.
- `paper_artifacts/`: each paper table and figure with the command that makes it. Today these are spread across `analysis/` and `verification/paper_results/`.
- Moving anything that generates code requires:
  - the regression `python -m verification.regression_check` (or the byte-compare used for the LLM configs);
  - regenerating 3,888 LLM + 3,072 GEMM + 5,184 conv configs and a sample of `top.cpp`, compared byte for byte against the pre-move output;
  - for D1(a), also fixing every path baked into scripts, `pyproject.toml` and the docs.

## 5. Phase D — docs

- README rewrite (brief L1): generator first, counts that match the manifest, the layout, and no `anonymous.4open.science` or 6000+/5,400 references.
- `REPRODUCE.md` and `Makefile` targets (`make manifest`, tables, figures).
- Keep `docs/TOOL_VERSIONS.md` and `docs/DATA_TYPES.md`. Handle `docs/revision_*` per D3.

## 6. Manifest — where it is documented and what is left

Documented in:
- `manifest/README.md`: the implemented schema. Columns of `designs/{gemm,conv,llm}.csv`, `ops.csv`, `modular.csv`, `modular_cases.csv`, the impl selection rules, the regeneration commands, and `internal/id_map.csv`.
- `REVISION_R2_AGENT_BRIEF.md` §B (B1–B7): the original specification. The implemented schema replaced B1's single `designs.csv` with per-domain files and YES/NO/FAIL columns.
- Agent memory from 2026-10-02 (not in the repo): the decision to register only paper-referenced designs, and the bundle path layout.

Open, as far as the repo shows:
- `designs/fullmodel.csv` (Table 3). Waits on the full-model consolidation.
- `verif/*.csv`: one row per design × datatype × sim level. The verification results now exist, so this can be built.
- Rebuild after the LLM re-run: collect, `rekey_results.py`, `build_manifest.py`.
- Brief B2 counts funnel, B4 duplicate check, B5 Table 4 from `SWEEP`. I don't know whether these were done or dropped. Confirm against the chat where the manifest was discussed.
