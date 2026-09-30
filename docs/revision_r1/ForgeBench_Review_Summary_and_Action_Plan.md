# ForgeBench (TRETS-2026-0091) — Review Summary & Revision Plan

**Decision:** Major Revision · **Revised manuscript + point-by-point response due:** July 30, 2026
**Budget:** ~10 hrs/week (≈40 hrs over 4 weeks) · **Prepared:** July 1, 2026

The editor's meta-review distills three reviews into two headline asks: (1) sharpen the **novelty / positioning** vs. existing benchmark and dataset frameworks, and (2) **strengthen the evaluation**. Everything below rolls up under those two themes.

---

## Part 1 — Reviewer Concerns, Grouped by Key Point

Legend for reviewer columns: **R1** (most detailed, 7 major + 7 minor), **R2** (5 concerns), **R3** (higher-level). ★ = editor explicitly flagged this theme in the meta-review.

### A. Novelty & positioning ★
| # | Concern | Reviewers |
|---|---------|-----------|
| A1 | **Undisclosed prior version.** An earlier FCCM poster/abstract with the same title + authors and a DOI exists but is not cited or differentiated. Must cite it and add a paragraph on new technical contributions, benchmarks, and results. | **R2** |
| A2 | **Limited technical novelty beyond "automated benchmark generation."** Clarify the novel methodological contribution vs. existing benchmark frameworks; expand related work. | **R3** (R1 #6 related) |
| A3 | **Benchmark vs. dataset identity.** Paper reads ambiguously between "design generator" and "dataset benchmark." Pick one framing and commit to it. If dataset: needs schema, splits, standardized tasks, artifact hashes, full design/label list. If generator: state so explicitly. | **R1 #1** |
| A4 | **Positioning vs. recent dataset frameworks.** Separate two axes: (i) workload modernity and (ii) dataset/benchmark protocol completeness. Acknowledge ForgeBench leads on (i) but lacks the cross-platform/cross-stage/fixed-task structure of recent FPGA-HLS dataset efforts. Avoid overclaiming. | **R1 #6**, R3 |
| A5 | **Missing comparison to Allo in Table 4.** Allo is discussed in-text and supports ML/LLM-style design — include it or justify exclusion. | **R2 #4** |

### B. "Full-scale" model claim — precision & feasibility ★
| # | Concern | Reviewers |
|---|---------|-----------|
| B1 | **"Full-scale" is overclaimed / imprecise.** Define exactly what is generated, synthesized, and executable. Consider replacing with e.g. "full-model graph-level HLS design with tiled off-chip execution." | **R1 #2**, R1-minor #3, **R2 #2** |
| B2 | **Memory feasibility contradiction.** LLaMA-3.1-8B in FP16 ≈ 16 GB weights, but ZCU102 has only 4 GB PS DDR4. Explain precision/quantization, weight representation, loading, tiling, storage. | **R2 #2** |
| B3 | **Resource numbers exceed the device.** Table 2 shows up to 835K LUT (ResNet-152) and 519K LUT (LLaMA); ZU9EG has ~274K LUTs. Explain units, whether designs are filtered before/after synth, and interpretation of over-capacity numbers. | **R1 #2**, R1-minor #6, **R2 #3** |
| B4 | **Coverage of full model pipeline.** For LLaMA: does it cover embeddings, all decoder blocks, KV-cache handling, output projection/LM head? Are prefill AND decode both generated and tested? Full inference pass or representative modules only? | **R1 #2**, R2 #2 |
| B5 | **Board choice justification.** If some generated benchmarks can't fit on ZCU102, justify it or move to a larger FPGA. | **R2 #3** |

### C. Correctness validation ★
| # | Concern | Reviewers |
|---|---------|-----------|
| C1 | **Functional correctness barely described.** Need: per-design testbenches; how inputs/expected outputs are generated; golden reference (PyTorch/NumPy/ONNX); numerical tolerance (float & fixed); end-to-end correctness for full ResNet/LLaMA; whether CSIM and CO-SIM pass for all released designs. | **R1 #3** (R2 #2, R3 touch on numerical correctness/accuracy) |

### D. Evaluation rigor & richer metrics (PPA) ★
| # | Concern | Reviewers |
|---|---------|-----------|
| D1 | **Evaluation is feasibility-oriented, not a rigorous protocol.** Establish a proper benchmark protocol rather than "we can generate designs." | **R1 #4**, R3 |
| D2 | **PPA claim unsupported.** Sec 3.1 promises performance/power/area reports, but results are mostly resource utilization. Add end-to-end latency, throughput, power, timing closure, post-route/implementation results, achieved Fmax, runtime validation. | **R1 #2/#4**, **R2 #5**, **R3** |
| D3 | **Case Study 2 (design-space coverage).** Justify parameter-range choices; show the 10,837 points are *representative*, not just numerous. Add distributions by operator type, resource, latency, and success/failure of synthesis. Quantitatively show advantage over existing suites. | **R1 #4**, R3 |
| D4 | **Case Study 3 (Table 4 tool survey) too coarse.** Define what "support" means (parse / transform / codegen / synth / optimize). Give failure categories + examples (unsupported constructs, memory interfaces, function hierarchy, attention ops, pragmas). Add tool versions, input assumptions, required manual mods. | **R1 #4**, R1-minor #4, **R3** |
| D5 | **Case Study 4 (modularization) mostly qualitative & manual.** Discuss whether shared modules are optimal, how chosen, whether multiple valid baselines exist. Metrics only capture resource reuse — acknowledge missing latency/throughput/power/scheduling. Address generality/scalability of manual construction. | **R1 #4**, **R3** |
| D6 | **Table 5 metric oversimplified.** Reporting the *average of LUT and DSP %* hides BRAM/latency effects. Justify or add metrics. | **R1-minor #5** |

### E. Cross-tool / cross-platform generality
| # | Concern | Reviewers |
|---|---------|-----------|
| E1 | **Single toolchain + single board.** Only Vitis HLS 2024.1 on ZCU102. Clarify: which generator parts are Vitis-specific; which constructs rely on Xilinx pragmas/interfaces; what's needed for Catapult/Intel HLS/Dynamatic; whether any subset was tested outside Vitis. Cross-tool evidence would substantially strengthen the paper. | **R1 #5** (E-platform overlaps B5/R2 #3) |

### F. Artifact & reproducibility ★
| # | Concern | Reviewers |
|---|---------|-----------|
| F1 | **Specify artifact contents.** Are all 10,837 designs released, or only generator scripts/configs? Are all JSON configs available? Synthesis reports + PPA labels? A manifest mapping design ID → params → results? Can Tables 2–5 and Fig 4 be reproduced from released scripts? | **R1 #1, #7** |
| F2 | **Reproduction instructions + resource cost.** Add manifest, clear repro steps, and approximate runtime/storage to regenerate the suite. | **R1 #7** |
| F3 | **HLSFactory integration is asserted, not shown.** Provide a runnable example or small integration case. | **R1 #6** |

### G. Writing, citations & presentation cleanup
| # | Concern | Reviewers |
|---|---------|-----------|
| G1 | **Figure 1 citation numbers are wrong/inconsistent** (e.g., Vitis shown as [31] but is [29]; PolyBench, MachSuite, Rosetta, ScaleHLS also inconsistent). Audit every figure citation. Note in-text ref mismatches too (e.g., MachSuite/PolyBench/Rosetta swapped around lines 77–78, Intar [11] vs [13]). | **R1-minor #1** |
| G2 | **Typos / inconsistent terminology.** "custmozied"→"customized"; "LLama-3.1-8B"→"LLaMA-3.1-8B"; "python scripting"→"Python scripting"; unify testcases/test cases/testsuite/test suite; fix conclusion "on eiwth full scale…". | **R1-minor #2** |
| G3 | **Tone down claims** ("full-scale LLaMA-3.1-8B", "ready-to-use benchmark suite") until backed by correctness + artifact evidence. | **R1-minor #3** |
| G4 | **General proofreading pass** — grammar and consistency throughout. | **R1-minor #7** |

### Concern → Reviewer coverage matrix (quick view)
| Theme | R1 | R2 | R3 |
|-------|:--:|:--:|:--:|
| A. Novelty & positioning | ● | ● | ● |
| B. "Full-scale" precision/feasibility | ● | ● | ○ |
| C. Correctness validation | ● | ○ | ○ |
| D. Evaluation rigor / PPA | ● | ● | ● |
| E. Cross-tool / cross-platform | ● | ○ | — |
| F. Artifact & reproducibility | ● | — | — |
| G. Writing & citations | ● | — | — |

● = raised directly · ○ = touched/implied · — = not raised

---

## Part 2 — Action Plan (July 1 → July 30)

### Guiding strategy
Two things decide whether this gets accepted: **honest reframing** (stop overclaiming "full-scale"; state clearly whether ForgeBench is a *generator + generated suites* — R1 #1 strongly implies this is the correct framing) and **more evidence** (correctness + real PPA/implementation numbers). Do the cheap, high-impact reframing and writing fixes first so that even if experiments run long, the paper is coherent and defensible. Then layer in as much new experimental data as the compute budget allows.

Effort tags: **[Write]** = manuscript/prose, no compute · **[Exp]** = needs synthesis/implementation runs (delegate to claude-code agents on the repo) · **[Repo]** = artifact/scripts/manifest work.

### Triage — what moves the needle most
- **Must-do, low effort, high impact:** A1 (cite prior version), B1/B3 (fix/explain "full-scale" + LUT-vs-device numbers), G1–G4 (citations/typos), F1 (state artifact contents), A3/A4 (reframe generator-vs-dataset + positioning). These alone address the two loudest complaints and cost mostly writing time.
- **Must-do, higher effort:** C1 (correctness methodology + CSIM/CO-SIM evidence), D2 (post-implementation PPA), D4 (define "support" + failure taxonomy), F1/F2 (manifest + repro).
- **Strong-if-time:** D3 (coverage distributions), D5/D6 (modularization metrics), E1 (cross-tool discussion + small non-Vitis probe), A5 (Allo in Table 4), F3 (HLSFactory runnable example).

### Week-by-week (≈10 hrs each)

**Week 1 (Jul 1–7) — Framing, writing wins, and kick off long experiments.**
- [Write] A1: add prior-FCCM-version citation + a "Relationship to prior version" paragraph listing new contributions/benchmarks/results.
- [Write] A3/A4: rewrite abstract, intro contributions, and related work to commit to the **generator + generated-suites** framing; add the two-axis positioning (workload modernity vs. protocol completeness) and explicitly scope out dataset-only claims.
- [Write] B1/B3/G3: replace "full-scale" with a precise phrase; add a paragraph explaining Table 2 units, that over-capacity configs are *generatable design points* (and how filtering works), and tone down claims.
- [Write] G1–G2: audit Figure 1 + all figure/in-text citation numbers; fix typos and unify terminology.
- [Exp — kick off early] Start the longest-pole runs so they finish while you write: **post-implementation (place & route) runs** for a representative subset to get power/Fmax/timing (D2), and **CSIM/CO-SIM** passes for the released designs (C1). *Do these first — they dominate wall-clock time.*
- Deliverable: cleaned manuscript draft + running experiments; start the response-to-reviewers document skeleton (one heading per concern A1…G4).

**Week 2 (Jul 8–14) — Correctness & PPA evidence.**
- [Write+Exp] C1: document the correctness methodology (testbench generation, golden reference PyTorch/NumPy, tolerances for float/fixed); collect CSIM/CO-SIM pass results into a table; state end-to-end status for ResNet and LLaMA honestly.
- [Write+Exp] B2/B4: write the precise LLaMA description — precision/data type, weight tiling/streaming from DRAM, which components are covered (embeddings, decoder blocks, KV cache, LM head), prefill vs decode status. Reconcile the 16 GB-vs-4 GB feasibility point explicitly.
- [Exp] D2: assemble post-implementation PPA table (latency, throughput, power, Fmax) from Week-1 runs; add to Case Study 1/2.
- [Write] B5: justify ZCU102 (or note results are device-agnostic design points; consider adding a larger board if a run is cheap).
- Deliverable: correctness section + PPA results integrated; response doc updated for B*, C1, D2.

**Week 3 (Jul 15–21) — Tool survey, coverage, modularization depth.**
- [Write+Exp] D4: redo Table 4 — define "support" levels, add tool versions/input assumptions/manual mods, and a failure taxonomy with concrete examples per tool (ScaleHLS, HIDA, AutoSA, StreamHLS). Add **Allo** (A5) or justify exclusion.
- [Exp] D3: generate distribution plots for the 10,837-point suite (by operator type, resource, latency, synth success/failure); add a sentence on how ranges were chosen and representativeness; add a quantitative contrast vs. legacy suites.
- [Write] D5/D6: add limitations paragraph on modularization (manual selection, optimality, generality/scalability); justify or augment the LUT/DSP-average metric in Table 5 (report BRAM separately if feasible).
- [Write] E1: add cross-tool/cross-platform discussion — Vitis-specific parts, Xilinx pragma dependencies, what Catapult/Intel/Dynamatic would need; note any small non-Vitis probe if you managed one.
- Deliverable: all case studies strengthened; response doc updated for D*, E1, A5.

**Week 4 (Jul 22–30) — Artifact, reproducibility, final polish, submit.**
- [Repo] F1/F2: add a manifest (design ID → params → results), decide/state exactly what's released (scripts + configs + reports + labels), write repro instructions, report runtime/storage to regenerate.
- [Repo] F3: add a small runnable HLSFactory integration example.
- [Write] G4: full proofreading pass; verify every reviewer point has a matching change.
- [Write] Finalize the **point-by-point response letter** — quote each concern, state the change, cite the new section/table/figure/line. Cross-check nothing is unanswered (even "we respectfully disagree, because…" where appropriate — e.g., if you keep ZCU102).
- Buffer: Jul 28–30 for overflow, co-author review, Overleaf compile, and submission. **Submit a few days before Jul 30, not on it.**

### Delegating to claude-code agents (once the repo is connected)
Good candidates to hand off as scoped tasks:
1. **Citation/typo sweep** — fix Figure 1 numbers, cross-check every `\cite` against `\ref` order, apply the terminology unifications (G1, G2, G4).
2. **CSIM/CO-SIM harness + run matrix** — script per-design testbench generation vs. a NumPy/PyTorch golden reference, collect pass/tolerance results into a CSV (C1).
3. **Post-implementation PPA collection** — extend the synthesis flow to run place & route on a representative subset, scrape latency/power/Fmax/timing into a table (D2).
4. **Coverage analysis plots** — parse the 10,837-design reports and produce distribution figures + a summary table (D3).
5. **Tool-survey reproduction** — re-run ScaleHLS/HIDA/AutoSA/StreamHLS (+Allo) on the same inputs, log exact failure messages to build the support-level + failure-taxonomy table (D4, A5).
6. **Artifact manifest + repro script** — generate the ID→params→results manifest and a one-command regeneration script; measure runtime/storage (F1, F2, F3).

Keep prose/reframing tasks (A1, A3, A4, B1, G3) for yourself — they need authorial judgment; hand the mechanical, compute-heavy, and scripting tasks to agents.

### Two deliverables due July 30
1. **Revised manuscript** (Overleaf) — all changes above, claims aligned to evidence.
2. **Response-to-reviewers document** — point-by-point, one entry per concern in Part 1 (A1…G4), each quoting the reviewer, describing the change, and pointing to the exact revised location.
