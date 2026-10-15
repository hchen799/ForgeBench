# Brief for rewriting Section 4.1 (Case Study 1: full model-level designs), R2

For the writing agent. The current text is R1's section with R2 `\new{}` patches. It describes the resource-*bound* table
(`tab:ml_model_suite`, min/max tiling configurations), which R2 replaces with **Table 3: one row per synthesized design, with
synthesis resources and C-simulation error**. Below: what R2 actually built and measured, the numbers, what in the old text is
now wrong, and what still holds. Every number here comes from the repo (`analysis/tables_out/table3_fullmodel.{csv,tex}`,
written by `python3 -m analysis.table3_fullmodel`; history in `docs/revision_r2/CHANGELOG_R2.md`).

## 1. What R2 built (14 designs = 14 rows of Table 3)

| Group | Designs | Format (storage / accumulator) | Architecture |
|---|---|---|---|
| ResNet **Full** | ResNet-18, -34, -50, -101, -152 | `ap_fixed<32,10>` round/saturate / `ap_fixed<80,36>` | **Untiled:** every layer's weights and feature maps are held in on-chip buffers (weights are loaded once from DRAM). Conv unroll 8 input × 64 output channels (the stem: all 3 input channels). "Full" means not tiled, **not** fully unrolled. |
| ResNet **Tiled** | same five | `ap_fixed<16,5>` truncate/wrap / `ap_fixed<64,42>` | Layer by layer, tile by tile: feature-map tiles 14×14 spatial, 32 input × 128 output channels, streamed from/to DRAM. Each tile runs conv → quantize → batch norm → activation; the residual add is a separate step. Conv unroll 8 input × 32 output channels. |
| Llama-3.1-8B **Prefill** | KV-cache capacity 2048 and 8192 | `ap_fixed<32,10>` round/saturate / `ap_fixed<80,36>` | 32 layers emitted in full. Weights in DRAM as `[32][d1][d2]`, streamed per layer. Token tile 16, hidden chunk 128, ping-pong buffers. Ops: embedding, RMSNorm, Q/K/V projections, RoPE, attention (score, row max, softmax, context), output projection, SwiGLU FFN, LM head, KV-cache load/store. |
| Llama-3.1-8B **Decode** | 2048 and 8192 | same | as prefill, token tile 1 |

**Common settings:**
- **Tool and target:** Vitis HLS 2024.1, C-synthesis only, ZCU102 (`xczu9eg-ffvb1156-2-e`), 10 ns clock.
- **Accumulator:** the default the generator derives from the storage type. Its fraction is twice the storage fraction (so
  products are exact) and its width is max(64, 2W + 16) bits. Rounding and saturation apply when results are stored back.
- **Math functions:** `exp`, division and square root are table- or integer-based, computed in the accumulator type.

**Why these formats** (decisions this revision):
- **Tiled ResNets at 16 bits:** at `<32,10>` the tiled ResNets exceeded the ZCU102 (133–174% BRAM, 363–444% DSP, 277–544% LUT).
  Moving to 16-bit storage and halving the conv unroll (8×64 → 8×32) brings DSP and BRAM within the device.
- **Llama at `<32,10>`:** `<16,5>` is not accurate enough for Llama-3-8B (≈123% relative L2 at the logits, from the earlier
  `<16,5>` run). The verifier only supports `<16,5>` and `<32,10>`, so Llama stays `<32,10>`.
- **Truncate/wrap for the tiled ResNets:** at `<16,5>`, round/saturate failed the FP64 check for the deep ResNets (up to 81%
  relative L2 for ResNet-152), while truncate/wrap passes for all five. An isolation run with the same design and seed showed the
  storage mode, not the tiling, is the cause. Optional for the text; one sentence if precision trade-offs are discussed.

## 2. Verification (new in R2; the old section has none)

- **Method:** every design is run in Vitis C simulation exactly as generated (`top.cpp` unchanged). Its outputs are compared
  against two PyTorch references built from the same seeded synthetic parameters and input:
  - **(a) a fixed-point reference** that models the design's arithmetic. **Requirement: identical integer codes** (bit-exact) at
    every observed tensor: layer outputs, logits, and for Llama every populated KV-cache entry.
  - **(b) an FP64 model.** **Requirement: relative L2 error ≤ 5%** at every observed tensor.
- **Seeds:** ResNets 10 (42–51). Llama 3, each run = a 4-token prefill and 2 decode calls; the prefill and decode designs run
  together.
- **Result:**
  - All 14 designs pass every seed, with **0 mismatching integer codes** across all runs (bit-exact to the fixed-point reference).
  - Final-logit error vs FP64 (Table 3 reports the maximum over seeds):
    - `<32,10>` designs: max abs ≤ 1.3×10⁻⁶ (ResNets) and 4.9×10⁻⁶ (Llama), relative L2 ≤ 1.0×10⁻⁶.
    - Tiled `<16,5>`: relative L2 0.6% (ResNet-18) to 2.4% (ResNet-152), growing with depth.
- **Inputs:** synthetic, seeded parameters (conv/linear weights uniform ±√(3/fan_in), BN gamma/var in [0.9, 1.1], BN beta/mean
  ±0.05, input ±1; Llama embedding ±0.5, norm weights [0.9, 1.1]). **Not pretrained weights**: no claim about model accuracy.
- **Scope limits to state honestly:**
  - C simulation only: no RTL co-simulation, no board run.
  - Llama runs use 6 tokens. ctx2048 and ctx8192 differ in cache capacity, and the 6-token runs give identical numbers; they do not
    exercise 2048/8192-token sequences.

## 3. Table 3 (current values; generated LaTeX in `analysis/tables_out/table3_fullmodel.tex`)

Percent of ZCU102 (LUT 274,080; DSP 2,520; BRAM18K 1,824).

| Variant | Model | LUT | DSP | BRAM18K | Latency (cycles) | Max abs. err. | Rel. L2 |
|---|---|---|---|---|---|---|---|
| Full `<32,10>` | ResNet-18 | 1,910K (697%) | 32,049 (1,272%) | 46,638 (2,557%) | 494M | 4.0e-7 | 1.2e-7 |
| | ResNet-34 | 2,313K (844%) | 32,049 (1,272%) | 81,954 (4,493%) | 796M | 4.8e-7 | 4.1e-8 |
| | ResNet-50 | 3,468K (1,265%) | 50,914 (2,020%) | 98,300 (5,389%) | 1,973M | 4.0e-7 | 1.6e-7 |
| | ResNet-101 | 4,707K (1,717%) | 50,914 (2,020%) | 157,732 (8,648%) | 3,078M | 6.5e-7 | 5.6e-8 |
| | ResNet-152 | 5,919K (2,160%) | 50,914 (2,020%) | 212,412 (11,645%) | 4,428M | 1.3e-6 | 1.9e-8 |
| Tiled `<16,5>` | ResNet-18 | 426K (155%) | 1,191 (47%) | 878 (48%) | 110M | 7.2e-3 | 6.3e-3 |
| | ResNet-34 | 459K (168%) | 1,187 (47%) | 926 (51%) | 173M | 1.4e-2 | 8.4e-3 |
| | ResNet-50 | 725K (265%) | 1,461 (58%) | 977 (54%) | 249M | 2.1e-2 | 1.4e-2 |
| | ResNet-101 | 844K (308%) | 1,461 (58%) | 1,113 (61%) | 361M | 6.2e-2 | 2.2e-2 |
| | ResNet-152 | **pending** (synthesis finishing) | | | | 1.3e-1 | 2.4e-2 |
| Prefill `<32,10>` | Llama-3.1-8B (2048) | 286K (104%) | 302 (12%) | 375 (21%) | n/a* | 4.7e-6 | 9.0e-7 |
| | Llama-3.1-8B (8192) | 286K (104%) | 302 (12%) | 375 (21%) | n/a* | 4.7e-6 | 9.0e-7 |
| Decode `<32,10>` | Llama-3.1-8B (2048) | 152K (56%) | 296 (12%) | 250 (14%) | n/a* | 4.9e-6 | 1.0e-6 |
| | Llama-3.1-8B (8192) | 152K (56%) | 296 (12%) | 250 (14%) | n/a* | 4.9e-6 | 1.0e-6 |

\* Vitis reports the Llama latency as undefined: loop bounds depend on run-time inputs (prefill length, cache position).

**Observations the text can use:**
- **Tiling cuts resources by an order of magnitude:**
  - ResNet-18: LUT 4.5×, DSP 27×, BRAM 53×.
  - ResNet-152: BRAM ~190×, with the remaining value pending.
  - It also cuts latency 4.5× for ResNet-18, from fewer and narrower operations per cycle.
- **What fits on the ZCU102:**
  - Tiled ResNets: DSP and BRAM fit (47–61%), LUT does not (1.6–3.1×).
  - Full ResNets: exceed the device on every resource, by 7–116×. Holding every layer's weights on chip cannot fit: ResNet-18's
    11.7M parameters alone exceed the BRAM capacity at any bit width ≥ 8.
  - Llama: decode fits; prefill is 4% over on LUT.
- **Context length barely matters for resources:** ctx8192 matches ctx2048 because the KV cache lives in DRAM.
- **Verification is independent of resources:** every design matches its fixed-point reference exactly, including the ones that
  don't fit.

## 4. What in the old section is now wrong or must change

1. **The "resource utilization bounds" framing:**
   - The DNN "lower bound 7×7 tile, channels 8, no unroll / upper bound 28×28, channels 64, unroll 16" and the LLM "2048 / 64×64 /
     unroll 4 vs 8192 / 256×256 / unroll 16" were configuration sweeps for `tab:ml_model_suite`. They are not what Table 3 reports.
   - Replace them with the actual configurations in §1 and the per-design rows in §3.
   - The generator still exposes tiling and unroll factors as command-line options (`scale_models/auto_generate_json.py`:
     `--resnet-tiled-{oc,ic,h,w}`, `--conv-{ci,co}-factors`, `--llama3-8b-{token-tile-prefill,token-tile-decode,hidden-chunk,
     in-chunk,ffn-chunk,linear-in-factors,...}`). Keep the claim that they are configurable.
2. **The precision paragraph**, "`<16,5>` for weights/KV-cache/embedding storage and `<32,10>` for multiply-accumulate and
   normalization", no longer describes the designs. Now:
   - Llama and full ResNets store everything as `<32,10>` round/saturate, with an 80-bit accumulator.
   - Tiled ResNets store `<16,5>` truncate/wrap, with a 64-bit accumulator.
   - Keep "precision is a configurable parameter".
3. **"Unroll factors set to 4 / 16" for Llama:** the R2 Llama designs use no linear-layer unrolling (factor 1; parallelism comes
   from the tiles). The ResNet conv unroll is 8×64 (full) and 8×32 (tiled).
4. **The DNN paragraph describes only the tiled architecture.** R2 has two ResNet variants; describe "Full" (untiled) as well, and
   say what "Full" means.
5. **"Prefilling and decoding blocks are tested through CSIM and C-synthesis":** now true and quantified. Point to the method and
   result in §2.
6. **"Resource utilization range shown in Tab. ml_model_suite"** → Table 3 (`tab:fullmodel` or whatever label the paper uses).
   The generated table has no `\label`.
7. **The disclaimer paragraph** (designs over ZCU102 limits are benchmark points, not deployed designs): keep it. It is now more
   precise:
   - every full ResNet exceeds the device;
   - the tiled ResNets exceed only LUT;
   - Llama decode fits.
8. **Old R1 numbers** (Hanqiu's single-seed `<16,5>` results and the `verif_fullmodel.tex` table) are superseded by Table 3. Do not
   cite them.

## 5. What still holds (keep)

- **Motivation** (first two paragraphs) and the extensibility / retargeting paragraph (U50/U280 with larger memory).
- **Tiled ResNet:** layer-by-layer, tile-by-tile execution; feature-map tiles and the current layer's parameters fetched from DRAM;
  results written back for the next layer; buffers reused across layers; the conv / BN / activation sequence per tile.
- **Llama:** a complete 32-layer project with embedding, prefill and decode, KV cache, Q/K/V, attention, output projection, LM
  head and SwiGLU. Weights stored as `[num_layers][d1][d2]` in DRAM and streamed per layer. Tiling over embedding and sequence,
  with defaults embedding tile 128 and sequence tile 16. Ping-pong buffers overlapping transfer and compute.
- **A Python script with command-line options** generates all of them (`scale_models/auto_generate_json.py` → `gen_configs`).

## 6. Suggested structure

1. Motivation (keep).
2. What is generated: ResNet-18…152 in two variants (Full, Tiled) and Llama-3.1-8B prefill/decode at two cache sizes; the
   configurable knobs; the formats.
3. DNN architecture: Full vs Tiled (§1).
4. LLM architecture (keep, fix the precision and unroll statements).
5. Verification: method and result (§2), two to four sentences.
6. Results: Table 3 and the observations in §3.
7. Disclaimer and retargeting (keep, updated per §4.7).

## 7. Pending

- **ResNet-152 tiled synthesis** finishes tonight, 2026-10-15. Its LUT, DSP, BRAM and latency then appear in
  `analysis/tables_out/table3_fullmodel.tex` (rerun `python3 -m analysis.table3_fullmodel`). Its error values are final.
- **The tiled-LUT range** in the text (155–308%) may widen when ResNet-152 lands.
