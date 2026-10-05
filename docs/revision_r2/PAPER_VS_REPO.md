# Paper (Forgebench_FPT_Journal_2026_R1.pdf) vs repo — where they disagree

Compiled 2026-10-01 from the manuscript text and the repo state. For the human who edits the paper; nothing here
edits it. "Repo" means branch FPT_TRETS_2026.

## Numbers the repo changes (generator/sweep changes in CHANGELOG_R2.md)
| Paper | Repo now | Why |
|---|---|---|
| Abstract/§3.4/§5 "over 12,000"; §4.2 "12,912 designs" (3,840 / 5,184 / 3,888) | 12,144 (3,072 / 5,184 / 3,888) | GEMM: loop-order axis was dead for options 1/4/5 (1,536 byte-identical copies); now per-op orders |
| Table 4 GEMM rows "Loop Order: 6 permutations", "Computation Order: 5 parenthesizations" | options 2/3: 6 gemm orders; options 1/4/5: each vm op picks ij/ji (4) | same |
| Table 4 LLM "Groups {1,2,4,8}" | `{1,2,4}` in the generator (3 values gives 3,888; 4 values would give 5,184) | prose/code mismatch from R1 |
| Table 4 LLM "Dropout Probability {False,0.1,0.3,0.5}" | head-dim unroll `hd_unroll` {1,2,4,8} | dropout is an identity at inference, sweep axis did nothing |
| Table 5 "GEMM (817) DNN (896) LLM (998)", "2,711 designs", Fig. 8 | to be regenerated | LLM and GEMM impl re-run/re-sampled after the sweep changes |
| Figs 6-7 | to be regenerated | LLM csynth re-run; GEMM +768 designs |
| §4.2 "all of which meet timing" (2,711) | re-check after rerun | July: 5 of 2,716 did not finish (3 Vivado 2024.1.2 segfaults in opt_design, 2 timeouts) |
| LLM sweep "Attention-Dropout-Norm" base design | Attention-Norm (dropout removed from the sweep) | same as above |

## Verification (§3.2.5, Table 2)
* Table 2: 17 operators / 57 variants. Repo `verification/op_configs`: same 17 operators, 69 variants. The gap is
  activations (paper "2 x 15" = 30; repo 13 gemm + 13 llm + 15 conv = 41) and `matrix_add` (paper counts it once, repo has
  it in both conv and llm). Either update the paper to 69 or restrict `designs/ops.csv`.
* Table 2 / text: "1,000 randomized inputs" per variant; repo July results exist at n=100 (full) and n=1000 (exe). R2 plan: float N=1000.
* §3.2.5 "an operator that meets the tolerance in float is taken to hold at any fixed-point width" — claim withdrawn in the R2 brief (replace with measured fixed-point error).
* §3.2.5 "26 of the designs are bit-exact" — recompute from the final results CSV.

## Modularization (§4.4, Table 7)
* Table 7 has 12 test cases (matches `modular_cases.csv`); older docs/README say 13.
* Program dims in the text match the configs (GEMM (96,512,128)/(128,256,64)/(256,128,192); conv (64,64,14,14)/(128,128,7,7)/(128,128,14,14); tiled attention 16 vs 4 heads).
* No csynth report exists in the repo for any of the 38 designs behind Table 7; the numbers have no local backing reports.
* Modularized designs are hand-written: no configs, testbenches or inputs ship with them (relevant to the R2 verification of modules).
* 8 of the 26 program designs (activation_op1-3, conv_block_op1-3, attn_breakdown_op1-2) have no config in the repo.

## Full models (§4.1, Table 3)
* No data in the repo backs Table 3 (min/max LUT/BRAM/DSP); configs and reports must come from Hanqiu (workstream K).
* §4.1 says tile sizes/unrolls: ResNet min = 7x7 tile, channels 8, no unroll; max = 28x28, 64, unroll 16; LLaMA min = ctx 2048, 64x64 tiles, unroll 4; max = ctx 8192, 256x256, unroll 16. These are the parameters to confirm in the recovered configs.
* §4.1 says prefill and decode blocks are "tested through CSIM and C-synthesis" and "complete 32-layer project"; the scale-model testbench currently has no golden compare (workstream D).

## Artifact claims (§3.5)
* "manifest mapping each design identifier to its parameters and extracted results" and "lean report archives" — now provided by `manifest/` and `release/`; the Zenodo DOI and anonymous link text must be replaced (decision: no anonymous mirror).
* "documents the approximate runtime and storage to regenerate" — to be written (REPRODUCE.md).

## Full-model verification (Hanqiu, commits f25fd39 / 3ca82fd, `scale_models/verification/`) — reviewed 2026-10-05
What exists: a configurable harness (`verification/verify.py`, configs in `verification/configs/`) with C/C++ emitters for a tiled
ResNet-18 and a Llama-3 family model, PyTorch references, exact comparison at every checkpoint, FP64 error characterization, 77 passing
unit tests, and a machine-readable inventory (`verification/results/validation_summary.json`, 13 architecture runs, Vitis HLS 2024.1.2 C-sim).
Results: bit-exact (0 mismatching integer codes) against the fixed-point PyTorch reference in all 8 headline runs: ResNet-18 <16,5>/<32,10>/<24,8>
(65 checkpoints, 8,054,760 codes, all 1,000 logits), tiny Llama <16,5>/<32,10>/<24,8>, and (frozen earlier implementation) full Llama-3 8B <16,5>
4+2 tokens (1,737 checkpoints) and 2048+2 tokens (12,740 checkpoints, 674,368,000 codes, 262,924,800 logits). FP64 errors are DIAGNOSTIC_ONLY
(e.g. ResNet <16,5> max abs logit error 6.1e-4; full 8B 2048+2 <16,5> max abs 7.0e-2).
Open points against R1 and against the paper:
* Which design is verified vs which design Table 3 reports. The verified C++ comes from dedicated emitters (`generate_tiled_resnet18.py`,
  `verification/models/*/codegen.py`), not from the operator-template flow (`auto_generate_json.py` -> `gen_configs.py` -> `generate_code.py`) whose
  JSON configs the README documents for the Table 3 knobs. Must confirm with Hanqiu that they are the same code path (workstream K1).
* References are the authors' own PyTorch models of the same graphs (custom ResNet without biases/shortcut BN; custom Llama): no torchvision or
  HuggingFace cross-check, synthetic random weights (`pretrained_weights: false`), no Llama-3.1 `rope_scaling` handling or discussion.
* No CO-SIM / RTL / board; full 8B at <32,10> and 8192 tokens not run; the new configurable generator was not run at full 8B (only the frozen earlier one was).
* Per-run logs (`summary.json`, `comparison.json`, `verification.log`) are not in git (`verification/runs` is absent; `verification_runs` symlink dangles);
  needed for the release bundle.
* `scale_models/README.md` contains an absolute path under `/usr/scratch/hchen799/...` (scrub before release).
* §4.1 text "tested through CSIM and C-synthesis" is supported by the 4+2 and 2048+2 full-8B runs if the verified design is the Table 3 design.
