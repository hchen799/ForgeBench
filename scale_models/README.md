# `scale_models`

This directory generates JSON model configs and HLS projects for the full-model benchmarks in ForgeBench.

Supported model families covered in this README:

- ResNet family: `18`, `34`, `50`, `101`, `152`
- Llama 3 8B: separate `PREFILL` and `DECODE` projects

The generation flow is:

```text
auto_generate_json.py -> auto_generated_configs/*.json -> gen_configs.py -> hls_files/<config_name>/
```

## Directory Layout

- `auto_generate_json.py`: emits JSON configs
- `gen_configs.py`: converts one or more JSON configs into HLS projects
- `auto_generated_configs/`: generated JSON files
- `hls_files/`: generated HLS projects

## Important Notes

- Run all commands from this directory:

```bash
cd /usr/scratch/hchen799/FPT_journal_ForgeBench/ForgeBench/scale_models
```

- `python auto_generate_json.py` generates all benchmark JSONs handled by the script, not only ResNet or Llama. If you only care about one model, just run HLS on the target JSON file afterward.
- `python gen_configs.py` generates HLS projects for every JSON file currently present in `auto_generated_configs/`.
- Llama 3 8B is generated as two separate kernels:
  - `PREFILL`: prompt processing
  - `DECODE`: one-token autoregressive step

## 1. Generate JSON Configs

### ResNet Family

Generate the default ResNet family JSONs:

```bash
python auto_generate_json.py --resnet-depths 18,34,50,101,152
```

This generates:

- Full-buffer configs such as:
  - `RESNET18_config_ap_fixed_16_5_.json`
  - `RESNET50_config_ap_fixed_16_5_.json`
- Tiled configs such as:
  - `RESNET18_TILED_config_ap_fixed_16_5_.json`
  - `RESNET50_TILED_config_ap_fixed_16_5_.json`

Generate only selected depths:

```bash
python auto_generate_json.py --resnet-depths 18,50
```

### ResNet Conv-Factor Sweep

The full-buffer conv path and the active tiled `conv_tile` path both support global `ci/co` tuning sweeps:

```bash
python auto_generate_json.py \
  --resnet-depths 18 \
  --conv-ci-factors 1,4 \
  --conv-co-factors 1,4
```

This generates Cartesian-product variants such as:

- `RESNET18_ci1_co1_config_ap_fixed_16_5_.json`
- `RESNET18_ci1_co4_config_ap_fixed_16_5_.json`
- `RESNET18_ci4_co1_config_ap_fixed_16_5_.json`
- `RESNET18_ci4_co4_config_ap_fixed_16_5_.json`

### ResNet Tiled Sweep

The tiled ResNet path also supports sweeping tile shapes:

```bash
python auto_generate_json.py \
  --resnet-depths 18 \
  --conv-ci-factors 1,4 \
  --conv-co-factors 1,4 \
  --resnet-tiled-oc 32,64 \
  --resnet-tiled-ic 8,16 \
  --resnet-tiled-h 7,14 \
  --resnet-tiled-w 7,14
```

This generates tiled variants such as:

- `RESNET18_TILED_oc32_ic8_h7_w7_ci1_co1_config_ap_fixed_16_5_.json`
- `RESNET18_TILED_oc64_ic16_h14_w14_ci4_co4_config_ap_fixed_16_5_.json`

## 2. Generate Full Llama 3 8B JSON Configs

Generate the default Llama 3 8B configs for context lengths `2048` and `8192`:

```bash
python auto_generate_json.py --llama3-8b-contexts 2048,8192
```

This generates:

- `LLAMA3_8B_PREFILL_ctx2048_config_ap_fixed_16_5_.json`
- `LLAMA3_8B_DECODE_ctx2048_config_ap_fixed_16_5_.json`
- `LLAMA3_8B_PREFILL_ctx8192_config_ap_fixed_16_5_.json`
- `LLAMA3_8B_DECODE_ctx8192_config_ap_fixed_16_5_.json`

### Llama Tile Knobs

The Llama generator supports compile-time tile sweeps for:

- `--llama3-8b-token-tile-prefill`
- `--llama3-8b-token-tile-decode`
- `--llama3-8b-hidden-chunk`
- `--llama3-8b-in-chunk`
- `--llama3-8b-max-out-chunk`
- `--llama3-8b-ffn-chunk`
- `--llama3-8b-k-token-tile`
- `--llama3-8b-q-head-tile`

Example:

```bash
python auto_generate_json.py \
  --llama3-8b-contexts 2048 \
  --llama3-8b-hidden-chunk 64 \
  --llama3-8b-in-chunk 128 \
  --llama3-8b-max-out-chunk 256 \
  --llama3-8b-ffn-chunk 128 \
  --llama3-8b-k-token-tile 64 \
  --llama3-8b-q-head-tile 8
```

This generates suffixed filenames such as:

- `LLAMA3_8B_PREFILL_ctx2048_tp16_td1_hc64_ic128_oc256_ffn128_kt64_qh8_config_ap_fixed_16_5_.json`

### Llama Pragma Knobs

The Llama tile-local operators also support HLS pragma sweeps:

- `--llama3-8b-linear-in-factors`
- `--llama3-8b-linear-out-factors`
- `--llama3-8b-chunk-col-factors`
- `--llama3-8b-attn-head-factors`
- `--llama3-8b-attn-dim-factors`

Example:

```bash
python auto_generate_json.py \
  --llama3-8b-contexts 2048 \
  --llama3-8b-linear-in-factors 4 \
  --llama3-8b-linear-out-factors 4 \
  --llama3-8b-chunk-col-factors 4 \
  --llama3-8b-attn-head-factors 4 \
  --llama3-8b-attn-dim-factors 8
```

This generates filenames such as:

- `LLAMA3_8B_PREFILL_ctx2048_tp16_td1_hc128_ic128_oc256_ffn128_kt128_qh4_li4_lo4_cc4_ah4_ad8_config_ap_fixed_16_5_.json`

### Llama Multi-Sweep Example

All Llama tile and pragma lists use Cartesian-product sweeps, similar to ResNet:

```bash
python auto_generate_json.py \
  --llama3-8b-contexts 2048,8192 \
  --llama3-8b-hidden-chunk 64,128 \
  --llama3-8b-k-token-tile 64,128 \
  --llama3-8b-q-head-tile 4,8 \
  --llama3-8b-linear-in-factors 1,4 \
  --llama3-8b-linear-out-factors 1,4 \
  --llama3-8b-chunk-col-factors 1,4 \
  --llama3-8b-attn-head-factors 1,4 \
  --llama3-8b-attn-dim-factors 1,8
```

Invalid Llama combinations are skipped and reported by `auto_generate_json.py`.

## 3. Generate HLS Projects

### Generate HLS for One JSON

ResNet example:

```bash
python - <<'PY'
import gen_configs
gen_configs.run_hls_flow(
    "auto_generated_configs/RESNET50_config_ap_fixed_16_5_.json",
    "hls_files",
)
PY
```

Llama prefill example:

```bash
python - <<'PY'
import gen_configs
gen_configs.run_hls_flow(
    "auto_generated_configs/LLAMA3_8B_PREFILL_ctx2048_config_ap_fixed_16_5_.json",
    "hls_files",
)
PY
```

Llama decode example:

```bash
python - <<'PY'
import gen_configs
gen_configs.run_hls_flow(
    "auto_generated_configs/LLAMA3_8B_DECODE_ctx2048_config_ap_fixed_16_5_.json",
    "hls_files",
)
PY
```

### Generate HLS for All JSONs

```bash
python gen_configs.py
```

This scans `auto_generated_configs/` and generates one HLS project directory per JSON config under `hls_files/`.

## 4. Output Naming Rules

### ResNet

- Single default conv config keeps legacy names:
  - `RESNET18_config_ap_fixed_16_5_.json`
- Non-default or multi-sweep conv configs add:
  - `_ci{...}_co{...}`
- Tiled configs add:
  - `_oc{...}_ic{...}_h{...}_w{...}`

### Llama 3 8B

- Single default tile + default pragma config keeps legacy names:
  - `LLAMA3_8B_PREFILL_ctx2048_config_ap_fixed_16_5_.json`
- Non-default tile configs add:
  - `_tp{...}_td{...}_hc{...}_ic{...}_oc{...}_ffn{...}_kt{...}_qh{...}`
- Non-default pragma configs add:
  - `_li{...}_lo{...}_cc{...}_ah{...}_ad{...}`

## 5. What Gets Generated

For each JSON config, `gen_configs.run_hls_flow(...)` generates:

- `top.cpp`
- `top.h`
- `tb_top.cpp`
- `run_hls.tcl`

under:

```text
hls_files/<json_basename>/
```

## 6. Current Scope

- ResNet family:
  - full-buffer path
  - explicit tiled path
- Llama 3 8B:
  - full-model `PREFILL`
  - full-model `DECODE`
  - explicit DRAM/BRAM tile scheduling
  - configurable tile sizes
  - configurable array-partition / unroll pragma sweeps

This README covers generation and HLS-project emission only. It does not run Vitis HLS synthesis automatically beyond generating the TCL script unless you execute the generated `run_hls.tcl` in your HLS environment.
