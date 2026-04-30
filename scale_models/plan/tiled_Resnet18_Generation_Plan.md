# Tiled ResNet-18 HLS Generation With Shared 128x14x14 Buffers and Shared-Exponent Reduction

  ## Summary

  Build a new parallel ResNet-18 generation path that replaces the current layer-sized BRAM design with a tiled scheduler
  and a reusable tiled operator library.

  This version should:

  - Keep the current full-buffer ResNet-18 flow unchanged
  - Add a separate tiled implementation for standard ImageNet ResNet-18
  - Use fixed on-chip feature buffers sized around 128 x 14 x 14
  - Reuse one tiled 3x3 kernel and one tiled 1x1 kernel across the whole network
  - Keep full intermediate feature maps in DRAM ping/pong scratch buffers
  - Respect the hard math cap: no internal numeric type wider than ap_fixed<32,10>
  - Avoid precision loss during long reductions by using shared-exponent partial sums rather than repeated requantization

  ## Public Interfaces

  Add a new tiled ResNet-18 generator under scale_models that emits a dedicated HLS project, not the existing generic
  JSON/full-buffer path.

  Generated top-level memory model:

  - Model weights, BN params, input, and logits are DRAM ports
  - External scratch feature-map ports:
      - DRAM_stem[64][112][112]
      - DRAM_feat_ping[512][56][56]
      - DRAM_feat_pong[512][56][56]

  Generated numeric interface:

  - data_t = ap_fixed<16,5>
  - acc_t = ap_fixed<32,10>
  - No type wider than acc_t
  - Shared-exponent metadata for reduction-owning operators:
      - one exponent per output-channel tile or per output tile, depending on the operator

  Reusable tiled operators:

  - conv3x3_tiled_runtime
  - conv1x1_tiled_runtime
  - batchnorm_tiled_runtime
  - relu_tiled_runtime
  - maxpool_pad_tiled_runtime
  - residual_add_tiled_runtime
  - global_avgpool_runtime
  - fc_tiled_runtime
  - Tile load/store helpers for activations, skip tiles, weight tiles, BN tiles, and logits

  ## Key Changes

  ### 1. Dedicated tiled codegen path

  Do not force this into the current generate_code.py per-shape specialization path.
  Reason:

  - The current path assumes full BRAM tensors per layer
  - It generates shape-specialized functions instead of reusable tiled kernels
  - That conflicts with fixed tile buffers and runtime scheduling

  Use merged_block_design.cpp only as algorithmic reference:

  - Reuse its tiling structure and operator boundaries
  - Reimplement the production path cleanly in a new tiled emitter

  ### 2. Fixed on-chip buffer architecture

  Use a fixed BRAM set for the entire network:

  - Feature tiles:
      - FM_IN[128][14][14]
      - FM_OUT[128][14][14]
      - FM_SKIP[128][14][14]
  - Convolution scratch:
      - PATCH_IN[128][33][33]
      - W3_TILE[128][128][3][3]
      - W1_TILE[128][128][1][1]
  - Param scratch:
      - BN_TILE[4][128]
      - BIAS_TILE[128]
      - POOL_VEC[512]
      - FC_OUT_TILE[128]
  - Exponent scratch:
      - EXP_TILE[128] or one scalar exponent per output tile, depending on selected granularity

  Use 33x33 as the max patch buffer so the stem 7x7/2, pad=3 over a 14x14 output tile is covered.

  ### 3. Precision and reduction policy

  Lock the math model explicitly.

  Stored tensors:

  - Activations, weights, BN params, and committed outputs use data_t = ap_fixed<16,5>

  Internal math:

  - All products and partial sums use acc_t = ap_fixed<32,10>
  - No operator may accumulate directly in data_t
  - No operator may use a type wider than acc_t

  Shared-exponent accumulation:

  - Reduction-owning operators keep:
      - mantissas in acc_t
      - a shared exponent/shift alongside them
  - During accumulation, if a mantissa tile approaches saturation:
      - arithmetic right-shift the mantissa tile
      - increment its shared exponent
  - Continue accumulation in the shifted domain
  - Only at operator output commit is the shared exponent folded into the castback / requantization path

  Why this is the chosen approach:

  - Better than tree reduction alone, which does not solve overflow
  - Better than chunk+requantize, which loses precision repeatedly
  - Compatible with the ap_fixed<32,10> cap

  Castback rule:

  - Operator outputs are committed to data_t only after the full reduction is complete
  - Rounding/saturation must be explicit and uniform across operators

  ### 4. Reduction handling by operator

  conv3x3 / conv1x1:

  - Own one output tile in acc_t
  - Accumulate over Cin chunks into the same mantissa tile
  - Maintain shared exponent state for that tile
  - Flush only after all Cin contributions are merged

  Global average pool:

  - Keep one widened running sum vector plus shared exponent metadata
  - Sweep spatial tiles and accumulate channelwise
  - Normalize only after the full spatial reduction is complete

  FC:

  - Keep one output-vector tile in acc_t
  - Accumulate across all input-channel chunks with shared exponent control
  - Cast to data_t only after full reduction is complete

  BN / ReLU / residual add / maxpool:

  - BN arithmetic still uses acc_t intermediates but is not a long reduction
  - Residual add should use widened intermediate math before castback
  - Maxpool remains local-window reduction and does not need shared exponent tracking

  ### 5. ResNet-18 tiled scheduler

  Implement a fixed scheduler for:

  - Stem:
      - tiled 7x7/2 conv from input to DRAM_stem
      - tiled padded maxpool from DRAM_stem to DRAM_feat_ping
  - Residual stages:
      - Stage 1: 64x56x56, 2 basic blocks
      - Stage 2: 128x28x28, 2 basic blocks, first block downsample
      - Stage 3: 256x14x14, 2 basic blocks, first block downsample
      - Stage 4: 512x7x7, 2 basic blocks, first block downsample
  - Head:
      - global average pool into POOL_VEC
      - tiled FC from 512 -> 1000
      - store logits

  Per block:

  - Read activation tiles from current feature DRAM into FM_IN
  - For downsample blocks, compute skip tiles into FM_SKIP
  - Execute conv1 -> bn -> relu -> conv2 -> bn -> add -> relu
  - Write output tiles to the opposite feature DRAM
  - Swap DRAM ping/pong at block boundaries

  Channel tiling:

  - Use fixed 128-channel groups
  - 64-channel stages use partial occupancy of the same buffers
  - 256 and 512 channel stages loop over channel tiles and reduction chunks

  ### 6. Calibration and scale policy

  Use offline calibration for final output scaling, not for every reduction step.

  Calibration output should provide:

  - per-layer output requantization shifts
  - optional per-layer guard thresholds for shared-exponent renormalization if needed

  Do not make the shared exponent itself runtime-learned or data-driven beyond overflow-avoidance logic.
  The exponent is a local numerical control mechanism; calibration is only for stable operator output scaling.

  ### 7. Generated outputs

  Emit a dedicated project directory such as:

  - scale_models/hls_files/RESNET18_tiled_c128_h14_w14_ap_fixed_16_5_/

  Generate:

  - top.cpp
  - top.h
  - run_hls.tcl
  Keep the existing gen_configs.py and current JSON/full-buffer path unchanged.
  ## Test Plan
      - Verify long reductions use mantissa + shared exponent accumulation instead of chunk requantization
      - Verify final operator outputs use explicit castback to data_t
  - Schedule correctness:
      - Confirm all 8 basic blocks are scheduled
      - Confirm downsample appears only in stage2/3/4 block0
      - Confirm stage outputs are 56 -> 28 -> 14 -> 7
      - Confirm FC emits 1000 logits
  - Acceptance metric:
      - BRAM footprint is fixed and independent of layer count
      - Kernel reuse is structural, not one-kernel-per-layer
      - Precision stays within the ap_fixed<32,10> cap everywhere in generated code
      - Reduction precision loss is controlled by shared-exponent renormalization rather than repeated requantization

  ## Assumptions

  - The tiled implementation is a new ResNet-18 variant, not a replacement for the current full-buffer flow
  - Off-chip feature-map scratch buffers are acceptable to reduce BRAM
  - Per-layer weight DRAM ports are acceptable in v1; packed shared weight storage is deferred
  - ap_fixed<16,5> remains the storage format for external tensors
  - The preferred overflow-control mechanism is shared-exponent accumulation, not a wider accumulator and not repeated
    chunk requantization