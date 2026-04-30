import os
import re
from dataclasses import dataclass


OUTPUT_DIR_NAME = "RESNET18_tiled_c128_h14_w14_ap_fixed_16_5_"
FPGA_NAME = "xczu9eg-ffvb1156-2-e"
CLOCK_PERIOD = 10
DATA_TYPE = "ap_fixed<16,5,AP_RND,AP_SAT>"
ACC_TYPE = "ap_fixed<32,10,AP_RND,AP_SAT>"

TILE_C = 128
TILE_H = 14
TILE_W = 14
MAX_PATCH = 33
MAX_FEAT_C = 512
MAX_FEAT_H = 56
MAX_FEAT_W = 56


@dataclass(frozen=True)
class ParamSpec:
    name: str
    dims: tuple[int, ...]


@dataclass(frozen=True)
class BlockSpec:
    stage: int
    block: int
    c_in: int
    c_out: int
    h_in: int
    w_in: int
    downsample: bool

    @property
    def prefix(self) -> str:
        return f"s{self.stage}_b{self.block}"

    @property
    def h_out(self) -> int:
        return self.h_in // (2 if self.downsample else 1)

    @property
    def w_out(self) -> int:
        return self.w_in // (2 if self.downsample else 1)


def sanitize_dtype(data_type: str) -> str:
    return re.sub(r"[<>, ]", "_", data_type).replace("__", "_")


def array_suffix(dims: tuple[int, ...]) -> str:
    return "".join(f"[{d}]" for d in dims)


def build_blocks() -> list[BlockSpec]:
    specs: list[BlockSpec] = []
    stage_defs = [
        (1, 64, 56, 56),
        (2, 128, 28, 28),
        (3, 256, 14, 14),
        (4, 512, 7, 7),
    ]
    c_prev = 64
    h_prev = 56
    w_prev = 56
    for stage, c_out, h_out, w_out in stage_defs:
        for block in range(2):
            downsample = stage > 1 and block == 0
            specs.append(
                BlockSpec(
                    stage=stage,
                    block=block,
                    c_in=c_prev,
                    c_out=c_out,
                    h_in=h_prev,
                    w_in=w_prev,
                    downsample=downsample,
                )
            )
            c_prev = c_out
            h_prev = h_out
            w_prev = w_out
    return specs


def build_params(blocks: list[BlockSpec]) -> list[ParamSpec]:
    params = [
        ParamSpec("DRAM_input", (3, 224, 224)),
        ParamSpec("DRAM_stem", (64, 112, 112)),
        ParamSpec("DRAM_feat_ping", (MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W)),
        ParamSpec("DRAM_feat_pong", (MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W)),
        ParamSpec("DRAM_feat_mid", (MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W)),
        ParamSpec("DRAM_w_stem", (64, 3, 7, 7)),
        ParamSpec("DRAM_bn_stem", (4, 64)),
    ]
    for block in blocks:
        prefix = block.prefix
        params.extend(
            [
                ParamSpec(f"DRAM_w_{prefix}_1", (block.c_out, block.c_in, 3, 3)),
                ParamSpec(f"DRAM_bn_{prefix}_1", (4, block.c_out)),
                ParamSpec(f"DRAM_w_{prefix}_2", (block.c_out, block.c_out, 3, 3)),
                ParamSpec(f"DRAM_bn_{prefix}_2", (4, block.c_out)),
            ]
        )
        if block.downsample:
            params.append(ParamSpec(f"DRAM_w_{prefix}_down", (block.c_out, block.c_in, 1, 1)))
    params.extend(
        [
            ParamSpec("DRAM_fc", (1000, 512)),
            ParamSpec("DRAM_out", (1000,)),
        ]
    )
    return params


def build_shift_names(blocks: list[BlockSpec]) -> list[str]:
    names = ["SHIFT_STEM"]
    for block in blocks:
        names.append(f"SHIFT_{block.prefix.upper()}_1")
        names.append(f"SHIFT_{block.prefix.upper()}_2")
        if block.downsample:
            names.append(f"SHIFT_{block.prefix.upper()}_DOWN")
    names.append("SHIFT_FC")
    return names


def make_scale_header(blocks: list[BlockSpec]) -> str:
    shift_names = build_shift_names(blocks)
    enum_lines = ["enum ConvShiftIndex {"]
    for idx, name in enumerate(shift_names):
        comma = "," if idx < len(shift_names) - 1 else ""
        enum_lines.append(f"    {name} = {idx}{comma}")
    enum_lines.append("};")
    zero_shifts = ", ".join("0" for _ in shift_names)
    comments = "\n".join(f"// {idx}: {name}" for idx, name in enumerate(shift_names))
    return f"""#pragma once
#include <ap_fixed.h>

// Shared-exponent output shifts for tiled reductions.
// Defaults are conservative placeholders and can be tuned per layer later.
{comments}
{chr(10).join(enum_lines)}

static const int kConvOutputShift[{len(shift_names)}] = {{{zero_shifts}}};
static const int kRenormGuard = 256;
"""


def make_top_h(params: list[ParamSpec]) -> str:
    param_lines = []
    for idx, param in enumerate(params):
        comma = "," if idx < len(params) - 1 else ""
        param_lines.append(f"    data_t {param.name}{array_suffix(param.dims)}{comma}")
    return f"""#pragma once
#include <ap_fixed.h>

typedef {DATA_TYPE} data_t;

void top(
{chr(10).join(param_lines)}
);
"""


def make_top_cpp(blocks: list[BlockSpec], params: list[ParamSpec]) -> str:
    top_params = []
    for idx, param in enumerate(params):
        comma = "," if idx < len(params) - 1 else ""
        top_params.append(f"    data_t {param.name}{array_suffix(param.dims)}{comma}")

    top_calls = [
        "    stem_conv7x7_tiled_runtime(DRAM_input, DRAM_w_stem, kConvOutputShift[SHIFT_STEM], DRAM_stem);",
        "    batchnorm_tiled_runtime<64, 64, 112, 112>(112, 112, DRAM_stem, DRAM_bn_stem, DRAM_stem);",
        "    relu_tiled_runtime<64, 64, 112, 112>(112, 112, DRAM_stem, DRAM_stem);",
        "    maxpool_pad_tiled_runtime(DRAM_stem, DRAM_feat_ping);",
    ]

    current = "DRAM_feat_ping"
    next_buf = "DRAM_feat_pong"
    mid = "DRAM_feat_mid"
    for block in blocks:
        if block.downsample:
            top_calls.append(
                f"    run_downsample_block<{block.c_in}, {block.c_out}>("
                f"{block.h_in}, {block.w_in}, {current}, DRAM_w_{block.prefix}_1, DRAM_bn_{block.prefix}_1, "
                f"DRAM_w_{block.prefix}_2, DRAM_bn_{block.prefix}_2, DRAM_w_{block.prefix}_down, {mid}, {next_buf}, "
                f"kConvOutputShift[SHIFT_{block.prefix.upper()}_1], "
                f"kConvOutputShift[SHIFT_{block.prefix.upper()}_2], "
                f"kConvOutputShift[SHIFT_{block.prefix.upper()}_DOWN]);"
            )
        else:
            top_calls.append(
                f"    run_identity_block<{block.c_out}>("
                f"{block.h_in}, {block.w_in}, {current}, DRAM_w_{block.prefix}_1, DRAM_bn_{block.prefix}_1, "
                f"DRAM_w_{block.prefix}_2, DRAM_bn_{block.prefix}_2, {mid}, {next_buf}, "
                f"kConvOutputShift[SHIFT_{block.prefix.upper()}_1], "
                f"kConvOutputShift[SHIFT_{block.prefix.upper()}_2]);"
            )
        current, next_buf = next_buf, current

    top_calls.extend(
        [
            f"    global_avgpool_runtime({current}, 7, 7, POOL_VEC);",
            "    fc_tiled_runtime(POOL_VEC, DRAM_fc, kConvOutputShift[SHIFT_FC], DRAM_out);",
        ]
    )

    return f"""#include <ap_fixed.h>
#include <hls_math.h>
#include <ap_int.h>
#include "top.h"
#include "resnet18_tiled_scales.h"

typedef {ACC_TYPE} acc_t;

static const int TILE_C = {TILE_C};
static const int TILE_H = {TILE_H};
static const int TILE_W = {TILE_W};
static const int MAX_PATCH = {MAX_PATCH};
static const int MAX_FEAT_C = {MAX_FEAT_C};
static const int MAX_FEAT_H = {MAX_FEAT_H};
static const int MAX_FEAT_W = {MAX_FEAT_W};

data_t PATCH_IN[TILE_C][MAX_PATCH][MAX_PATCH];
data_t FM_IN[TILE_C][TILE_H][TILE_W];
data_t FM_OUT[TILE_C][TILE_H][TILE_W];
data_t FM_SKIP[TILE_C][TILE_H][TILE_W];
data_t BN_TILE[4][TILE_C];
data_t W3_TILE[TILE_C][TILE_C][3][3];
data_t W1_TILE[TILE_C][TILE_C];
data_t W7_TILE[64][3][7][7];
data_t FC_W_TILE[TILE_C][TILE_C];
data_t POOL_VEC[512];
acc_t ACC_TILE[TILE_C][TILE_H][TILE_W];
acc_t FC_ACC[TILE_C];

inline int ceil_div(int x, int y) {{
    return (x + y - 1) / y;
}}

inline int out_dim(int size, int pad, int stride, int kernel) {{
    return ((size + 2 * pad - kernel) / stride) + 1;
}}

inline acc_t abs_acc(acc_t value) {{
    return value < 0 ? -value : value;
}}

inline acc_t apply_power_of_two_shift(acc_t value, int shift) {{
    if (shift > 0) {{
        for (int i = 0; i < shift; ++i) {{
            value *= (acc_t)2;
        }}
    }} else if (shift < 0) {{
        for (int i = 0; i < -shift; ++i) {{
            value /= (acc_t)2;
        }}
    }}
    return value;
}}

inline data_t quantize_acc(acc_t value, int tile_exp, int output_shift) {{
    return (data_t)apply_power_of_two_shift(value, tile_exp - output_shift);
}}

void clear_acc_tile(int valid_oc, int valid_h, int valid_w) {{
    for (int oc = 0; oc < valid_oc; ++oc) {{
        for (int h = 0; h < valid_h; ++h) {{
            for (int w = 0; w < valid_w; ++w) {{
                ACC_TILE[oc][h][w] = (acc_t)0;
            }}
        }}
    }}
}}

void clear_fc_acc(int valid_oc) {{
    for (int oc = 0; oc < valid_oc; ++oc) {{
        FC_ACC[oc] = (acc_t)0;
    }}
}}

void maybe_rescale_acc_tile(int valid_oc, int valid_h, int valid_w, int &tile_exp) {{
    while (true) {{
        acc_t max_abs = (acc_t)0;
        for (int oc = 0; oc < valid_oc; ++oc) {{
            for (int h = 0; h < valid_h; ++h) {{
                for (int w = 0; w < valid_w; ++w) {{
                    acc_t current = abs_acc(ACC_TILE[oc][h][w]);
                    if (current > max_abs) {{
                        max_abs = current;
                    }}
                }}
            }}
        }}
        if (max_abs <= (acc_t)kRenormGuard) {{
            return;
        }}
        for (int oc = 0; oc < valid_oc; ++oc) {{
            for (int h = 0; h < valid_h; ++h) {{
                for (int w = 0; w < valid_w; ++w) {{
                    ACC_TILE[oc][h][w] /= (acc_t)2;
                }}
            }}
        }}
        ++tile_exp;
    }}
}}

void maybe_rescale_fc_acc(int valid_oc, int &tile_exp) {{
    while (true) {{
        acc_t max_abs = (acc_t)0;
        for (int oc = 0; oc < valid_oc; ++oc) {{
            acc_t current = abs_acc(FC_ACC[oc]);
            if (current > max_abs) {{
                max_abs = current;
            }}
        }}
        if (max_abs <= (acc_t)kRenormGuard) {{
            return;
        }}
        for (int oc = 0; oc < valid_oc; ++oc) {{
            FC_ACC[oc] /= (acc_t)2;
        }}
        ++tile_exp;
    }}
}}

template <int OUT_CMAX, int OUT_HMAX, int OUT_WMAX>
void commit_acc_tile(
    data_t output[OUT_CMAX][OUT_HMAX][OUT_WMAX],
    int co_offset,
    int row_offset,
    int col_offset,
    int valid_oc,
    int valid_h,
    int valid_w,
    int tile_exp,
    int output_shift
) {{
    for (int oc = 0; oc < valid_oc; ++oc) {{
        for (int h = 0; h < valid_h; ++h) {{
            for (int w = 0; w < valid_w; ++w) {{
                output[co_offset + oc][row_offset + h][col_offset + w] =
                    quantize_acc(ACC_TILE[oc][h][w], tile_exp, output_shift);
            }}
        }}
    }}
}}

template <int C>
void load_bn_tile(const data_t weights[4][C], int channel_offset, int valid_c) {{
    for (int stat = 0; stat < 4; ++stat) {{
        for (int c = 0; c < valid_c; ++c) {{
            BN_TILE[stat][c] = weights[stat][channel_offset + c];
        }}
    }}
}}

template <int C_OUT, int C_IN>
void load_w3_tile(const data_t weights[C_OUT][C_IN][3][3], int co_offset, int ci_offset, int valid_oc, int valid_ci) {{
    for (int oc = 0; oc < valid_oc; ++oc) {{
        for (int ic = 0; ic < valid_ci; ++ic) {{
            for (int kh = 0; kh < 3; ++kh) {{
                for (int kw = 0; kw < 3; ++kw) {{
                    W3_TILE[oc][ic][kh][kw] = weights[co_offset + oc][ci_offset + ic][kh][kw];
                }}
            }}
        }}
    }}
}}

template <int C_OUT, int C_IN>
void load_w1_tile(const data_t weights[C_OUT][C_IN][1][1], int co_offset, int ci_offset, int valid_oc, int valid_ci) {{
    for (int oc = 0; oc < valid_oc; ++oc) {{
        for (int ic = 0; ic < valid_ci; ++ic) {{
            W1_TILE[oc][ic] = weights[co_offset + oc][ci_offset + ic][0][0];
        }}
    }}
}}

void load_stem_w7_tile(const data_t weights[64][3][7][7], int co_offset, int valid_oc) {{
    for (int oc = 0; oc < valid_oc; ++oc) {{
        for (int ic = 0; ic < 3; ++ic) {{
            for (int kh = 0; kh < 7; ++kh) {{
                for (int kw = 0; kw < 7; ++kw) {{
                    W7_TILE[oc][ic][kh][kw] = weights[co_offset + oc][ic][kh][kw];
                }}
            }}
        }}
    }}
}}

template <int IN_CMAX, int IN_HMAX, int IN_WMAX>
void load_patch_from_tensor(
    const data_t input[IN_CMAX][IN_HMAX][IN_WMAX],
    int ci_offset,
    int valid_ci,
    int h_in,
    int w_in,
    int row_offset,
    int col_offset,
    int kernel,
    int stride,
    int pad,
    int tile_h,
    int tile_w
) {{
    int patch_h = (tile_h - 1) * stride + kernel;
    int patch_w = (tile_w - 1) * stride + kernel;
    for (int ic = 0; ic < valid_ci; ++ic) {{
        for (int h = 0; h < patch_h; ++h) {{
            for (int w = 0; w < patch_w; ++w) {{
                int in_row = row_offset * stride - pad + h;
                int in_col = col_offset * stride - pad + w;
                if (in_row >= 0 && in_row < h_in && in_col >= 0 && in_col < w_in) {{
                    PATCH_IN[ic][h][w] = input[ci_offset + ic][in_row][in_col];
                }} else {{
                    PATCH_IN[ic][h][w] = (data_t)0;
                }}
            }}
        }}
    }}
}}

void run_conv3x3_tile(int valid_oc, int valid_ci, int tile_h, int tile_w, int stride) {{
    for (int oc = 0; oc < valid_oc; ++oc) {{
        for (int oh = 0; oh < tile_h; ++oh) {{
            for (int ow = 0; ow < tile_w; ++ow) {{
                acc_t sum = ACC_TILE[oc][oh][ow];
                for (int ic = 0; ic < valid_ci; ++ic) {{
                    for (int kh = 0; kh < 3; ++kh) {{
                        for (int kw = 0; kw < 3; ++kw) {{
                            sum += (acc_t)PATCH_IN[ic][oh * stride + kh][ow * stride + kw] *
                                   (acc_t)W3_TILE[oc][ic][kh][kw];
                        }}
                    }}
                }}
                ACC_TILE[oc][oh][ow] = sum;
            }}
        }}
    }}
}}

void run_conv1x1_tile(int valid_oc, int valid_ci, int tile_h, int tile_w, int stride) {{
    for (int oc = 0; oc < valid_oc; ++oc) {{
        for (int oh = 0; oh < tile_h; ++oh) {{
            for (int ow = 0; ow < tile_w; ++ow) {{
                acc_t sum = ACC_TILE[oc][oh][ow];
                for (int ic = 0; ic < valid_ci; ++ic) {{
                    sum += (acc_t)PATCH_IN[ic][oh * stride][ow * stride] * (acc_t)W1_TILE[oc][ic];
                }}
                ACC_TILE[oc][oh][ow] = sum;
            }}
        }}
    }}
}}

void run_stem_conv7x7_tile(int valid_oc, int tile_h, int tile_w) {{
    for (int oc = 0; oc < valid_oc; ++oc) {{
        for (int oh = 0; oh < tile_h; ++oh) {{
            for (int ow = 0; ow < tile_w; ++ow) {{
                acc_t sum = ACC_TILE[oc][oh][ow];
                for (int ic = 0; ic < 3; ++ic) {{
                    for (int kh = 0; kh < 7; ++kh) {{
                        for (int kw = 0; kw < 7; ++kw) {{
                            sum += (acc_t)PATCH_IN[ic][oh * 2 + kh][ow * 2 + kw] *
                                   (acc_t)W7_TILE[oc][ic][kh][kw];
                        }}
                    }}
                }}
                ACC_TILE[oc][oh][ow] = sum;
            }}
        }}
    }}
}}

template <int C_IN, int C_OUT>
void conv3x3_tiled_runtime(
    int h_in,
    int w_in,
    int stride,
    const data_t input[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    const data_t weights[C_OUT][C_IN][3][3],
    int output_shift,
    data_t output[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W]
) {{
    int h_out = out_dim(h_in, 1, stride, 3);
    int w_out = out_dim(w_in, 1, stride, 3);
    for (int co = 0; co < C_OUT; co += TILE_C) {{
        int valid_oc = ((co + TILE_C) <= C_OUT) ? TILE_C : (C_OUT - co);
        for (int oh = 0; oh < h_out; oh += TILE_H) {{
            int valid_h = ((oh + TILE_H) <= h_out) ? TILE_H : (h_out - oh);
            for (int ow = 0; ow < w_out; ow += TILE_W) {{
                int valid_w = ((ow + TILE_W) <= w_out) ? TILE_W : (w_out - ow);
                int tile_exp = 0;
                clear_acc_tile(valid_oc, valid_h, valid_w);
                for (int ci = 0; ci < C_IN; ci += TILE_C) {{
                    int valid_ci = ((ci + TILE_C) <= C_IN) ? TILE_C : (C_IN - ci);
                    load_patch_from_tensor(input, ci, valid_ci, h_in, w_in, oh, ow, 3, stride, 1, valid_h, valid_w);
                    load_w3_tile(weights, co, ci, valid_oc, valid_ci);
                    run_conv3x3_tile(valid_oc, valid_ci, valid_h, valid_w, stride);
                    maybe_rescale_acc_tile(valid_oc, valid_h, valid_w, tile_exp);
                }}
                commit_acc_tile(output, co, oh, ow, valid_oc, valid_h, valid_w, tile_exp, output_shift);
            }}
        }}
    }}
}}

template <int C_IN, int C_OUT>
void conv1x1_tiled_runtime(
    int h_in,
    int w_in,
    int stride,
    const data_t input[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    const data_t weights[C_OUT][C_IN][1][1],
    int output_shift,
    data_t output[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W]
) {{
    int h_out = out_dim(h_in, 0, stride, 1);
    int w_out = out_dim(w_in, 0, stride, 1);
    for (int co = 0; co < C_OUT; co += TILE_C) {{
        int valid_oc = ((co + TILE_C) <= C_OUT) ? TILE_C : (C_OUT - co);
        for (int oh = 0; oh < h_out; oh += TILE_H) {{
            int valid_h = ((oh + TILE_H) <= h_out) ? TILE_H : (h_out - oh);
            for (int ow = 0; ow < w_out; ow += TILE_W) {{
                int valid_w = ((ow + TILE_W) <= w_out) ? TILE_W : (w_out - ow);
                int tile_exp = 0;
                clear_acc_tile(valid_oc, valid_h, valid_w);
                for (int ci = 0; ci < C_IN; ci += TILE_C) {{
                    int valid_ci = ((ci + TILE_C) <= C_IN) ? TILE_C : (C_IN - ci);
                    load_patch_from_tensor(input, ci, valid_ci, h_in, w_in, oh, ow, 1, stride, 0, valid_h, valid_w);
                    load_w1_tile(weights, co, ci, valid_oc, valid_ci);
                    run_conv1x1_tile(valid_oc, valid_ci, valid_h, valid_w, stride);
                    maybe_rescale_acc_tile(valid_oc, valid_h, valid_w, tile_exp);
                }}
                commit_acc_tile(output, co, oh, ow, valid_oc, valid_h, valid_w, tile_exp, output_shift);
            }}
        }}
    }}
}}

void stem_conv7x7_tiled_runtime(
    const data_t input[3][224][224],
    const data_t weights[64][3][7][7],
    int output_shift,
    data_t output[64][112][112]
) {{
    for (int co = 0; co < 64; co += 64) {{
        int valid_oc = 64 - co;
        for (int oh = 0; oh < 112; oh += TILE_H) {{
            int valid_h = ((oh + TILE_H) <= 112) ? TILE_H : (112 - oh);
            for (int ow = 0; ow < 112; ow += TILE_W) {{
                int valid_w = ((ow + TILE_W) <= 112) ? TILE_W : (112 - ow);
                int tile_exp = 0;
                clear_acc_tile(valid_oc, valid_h, valid_w);
                load_patch_from_tensor(input, 0, 3, 224, 224, oh, ow, 7, 2, 3, valid_h, valid_w);
                load_stem_w7_tile(weights, co, valid_oc);
                run_stem_conv7x7_tile(valid_oc, valid_h, valid_w);
                maybe_rescale_acc_tile(valid_oc, valid_h, valid_w, tile_exp);
                commit_acc_tile(output, co, oh, ow, valid_oc, valid_h, valid_w, tile_exp, output_shift);
            }}
        }}
    }}
}}

template <int C, int MAX_C, int MAX_H, int MAX_W>
void batchnorm_tiled_runtime(
    int h,
    int w,
    data_t input[MAX_C][MAX_H][MAX_W],
    const data_t weights[4][C],
    data_t output[MAX_C][MAX_H][MAX_W]
) {{
    for (int co = 0; co < C; co += TILE_C) {{
        int valid_c = ((co + TILE_C) <= C) ? TILE_C : (C - co);
        load_bn_tile(weights, co, valid_c);
        for (int oh = 0; oh < h; oh += TILE_H) {{
            int valid_h = ((oh + TILE_H) <= h) ? TILE_H : (h - oh);
            for (int ow = 0; ow < w; ow += TILE_W) {{
                int valid_w = ((ow + TILE_W) <= w) ? TILE_W : (w - ow);
                for (int c = 0; c < valid_c; ++c) {{
                    acc_t gamma = (acc_t)BN_TILE[0][c];
                    acc_t beta = (acc_t)BN_TILE[1][c];
                    acc_t mean = (acc_t)BN_TILE[2][c];
                    acc_t var = (acc_t)BN_TILE[3][c];
                    for (int ih = 0; ih < valid_h; ++ih) {{
                        for (int iw = 0; iw < valid_w; ++iw) {{
                            acc_t value = (acc_t)input[co + c][oh + ih][ow + iw];
                            acc_t norm = (value - mean) / hls::sqrt(var + (acc_t)1e-5);
                            output[co + c][oh + ih][ow + iw] = (data_t)(gamma * norm + beta);
                        }}
                    }}
                }}
            }}
        }}
    }}
}}

template <int C, int MAX_C, int MAX_H, int MAX_W>
void relu_tiled_runtime(int h, int w, data_t input[MAX_C][MAX_H][MAX_W], data_t output[MAX_C][MAX_H][MAX_W]) {{
    for (int co = 0; co < C; co += TILE_C) {{
        int valid_c = ((co + TILE_C) <= C) ? TILE_C : (C - co);
        for (int oh = 0; oh < h; oh += TILE_H) {{
            int valid_h = ((oh + TILE_H) <= h) ? TILE_H : (h - oh);
            for (int ow = 0; ow < w; ow += TILE_W) {{
                int valid_w = ((ow + TILE_W) <= w) ? TILE_W : (w - ow);
                for (int c = 0; c < valid_c; ++c) {{
                    for (int ih = 0; ih < valid_h; ++ih) {{
                        for (int iw = 0; iw < valid_w; ++iw) {{
                            data_t value = input[co + c][oh + ih][ow + iw];
                            output[co + c][oh + ih][ow + iw] = value > (data_t)0 ? value : (data_t)0;
                        }}
                    }}
                }}
            }}
        }}
    }}
}}

void maxpool_pad_tiled_runtime(
    data_t input[64][112][112],
    data_t output[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W]
) {{
    for (int co = 0; co < 64; co += TILE_C) {{
        int valid_c = 64 - co;
        for (int oh = 0; oh < 56; oh += TILE_H) {{
            int valid_h = ((oh + TILE_H) <= 56) ? TILE_H : (56 - oh);
            for (int ow = 0; ow < 56; ow += TILE_W) {{
                int valid_w = ((ow + TILE_W) <= 56) ? TILE_W : (56 - ow);
                int patch_h = (valid_h - 1) * 2 + 3;
                int patch_w = (valid_w - 1) * 2 + 3;
                for (int c = 0; c < valid_c; ++c) {{
                    for (int ph = 0; ph < patch_h; ++ph) {{
                        for (int pw = 0; pw < patch_w; ++pw) {{
                            int in_row = oh * 2 - 1 + ph;
                            int in_col = ow * 2 - 1 + pw;
                            if (in_row >= 0 && in_row < 112 && in_col >= 0 && in_col < 112) {{
                                PATCH_IN[c][ph][pw] = input[co + c][in_row][in_col];
                            }} else {{
                                PATCH_IN[c][ph][pw] = (data_t)0;
                            }}
                        }}
                    }}
                    for (int ih = 0; ih < valid_h; ++ih) {{
                        for (int iw = 0; iw < valid_w; ++iw) {{
                            data_t max_value = PATCH_IN[c][ih * 2][iw * 2];
                            for (int kh = 0; kh < 3; ++kh) {{
                                for (int kw = 0; kw < 3; ++kw) {{
                                    data_t candidate = PATCH_IN[c][ih * 2 + kh][iw * 2 + kw];
                                    if (candidate > max_value) {{
                                        max_value = candidate;
                                    }}
                                }}
                            }}
                            output[co + c][oh + ih][ow + iw] = max_value;
                        }}
                    }}
                }}
            }}
        }}
    }}
}}

template <int C>
void residual_add_tiled_runtime(
    int h,
    int w,
    data_t lhs[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    data_t rhs[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    data_t output[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W]
) {{
    for (int co = 0; co < C; co += TILE_C) {{
        int valid_c = ((co + TILE_C) <= C) ? TILE_C : (C - co);
        for (int oh = 0; oh < h; oh += TILE_H) {{
            int valid_h = ((oh + TILE_H) <= h) ? TILE_H : (h - oh);
            for (int ow = 0; ow < w; ow += TILE_W) {{
                int valid_w = ((ow + TILE_W) <= w) ? TILE_W : (w - ow);
                for (int c = 0; c < valid_c; ++c) {{
                    for (int ih = 0; ih < valid_h; ++ih) {{
                        for (int iw = 0; iw < valid_w; ++iw) {{
                            acc_t value = (acc_t)lhs[co + c][oh + ih][ow + iw] + (acc_t)rhs[co + c][oh + ih][ow + iw];
                            output[co + c][oh + ih][ow + iw] = (data_t)value;
                        }}
                    }}
                }}
            }}
        }}
    }}
}}

template <int C>
void global_avgpool_runtime(
    data_t input[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    int h,
    int w,
    data_t output[C]
) {{
    for (int co = 0; co < C; co += TILE_C) {{
        int valid_c = ((co + TILE_C) <= C) ? TILE_C : (C - co);
        clear_fc_acc(valid_c);
        int tile_exp = 0;
        for (int c = 0; c < valid_c; ++c) {{
            for (int ih = 0; ih < h; ++ih) {{
                for (int iw = 0; iw < w; ++iw) {{
                    FC_ACC[c] += (acc_t)input[co + c][ih][iw];
                }}
            }}
        }}
        maybe_rescale_fc_acc(valid_c, tile_exp);
        for (int c = 0; c < valid_c; ++c) {{
            acc_t value = apply_power_of_two_shift(FC_ACC[c], tile_exp) / (acc_t)(h * w);
            output[co + c] = (data_t)value;
        }}
    }}
}}

template <int C_OUT, int C_IN>
void fc_tiled_runtime(
    const data_t input[C_IN],
    const data_t weights[C_OUT][C_IN],
    int output_shift,
    data_t output[C_OUT]
) {{
    for (int co = 0; co < C_OUT; co += TILE_C) {{
        int valid_oc = ((co + TILE_C) <= C_OUT) ? TILE_C : (C_OUT - co);
        clear_fc_acc(valid_oc);
        int tile_exp = 0;
        for (int ci = 0; ci < C_IN; ci += TILE_C) {{
            int valid_ci = ((ci + TILE_C) <= C_IN) ? TILE_C : (C_IN - ci);
            for (int oc = 0; oc < valid_oc; ++oc) {{
                for (int ic = 0; ic < valid_ci; ++ic) {{
                    FC_W_TILE[oc][ic] = weights[co + oc][ci + ic];
                }}
            }}
            for (int oc = 0; oc < valid_oc; ++oc) {{
                acc_t sum = FC_ACC[oc];
                for (int ic = 0; ic < valid_ci; ++ic) {{
                    sum += (acc_t)input[ci + ic] * (acc_t)FC_W_TILE[oc][ic];
                }}
                FC_ACC[oc] = sum;
            }}
            maybe_rescale_fc_acc(valid_oc, tile_exp);
        }}
        for (int oc = 0; oc < valid_oc; ++oc) {{
            output[co + oc] = quantize_acc(FC_ACC[oc], tile_exp, output_shift);
        }}
    }}
}}

template <int C>
void run_identity_block(
    int h,
    int w,
    data_t input[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    const data_t w1[C][C][3][3],
    const data_t bn1[4][C],
    const data_t w2[C][C][3][3],
    const data_t bn2[4][C],
    data_t mid[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    data_t output[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    int shift1,
    int shift2
) {{
    conv3x3_tiled_runtime<C, C>(h, w, 1, input, w1, shift1, mid);
    batchnorm_tiled_runtime<C, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h, w, mid, bn1, mid);
    relu_tiled_runtime<C, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h, w, mid, mid);
    conv3x3_tiled_runtime<C, C>(h, w, 1, mid, w2, shift2, output);
    batchnorm_tiled_runtime<C, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h, w, output, bn2, output);
    residual_add_tiled_runtime<C>(h, w, output, input, output);
    relu_tiled_runtime<C, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h, w, output, output);
}}

template <int C_IN, int C_OUT>
void run_downsample_block(
    int h_in,
    int w_in,
    data_t input[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    const data_t w1[C_OUT][C_IN][3][3],
    const data_t bn1[4][C_OUT],
    const data_t w2[C_OUT][C_OUT][3][3],
    const data_t bn2[4][C_OUT],
    const data_t wdown[C_OUT][C_IN][1][1],
    data_t mid[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    data_t output[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    int shift1,
    int shift2,
    int shift_down
) {{
    int h_out = out_dim(h_in, 1, 2, 3);
    int w_out = out_dim(w_in, 1, 2, 3);
    conv3x3_tiled_runtime<C_IN, C_OUT>(h_in, w_in, 2, input, w1, shift1, mid);
    batchnorm_tiled_runtime<C_OUT, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h_out, w_out, mid, bn1, mid);
    relu_tiled_runtime<C_OUT, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h_out, w_out, mid, mid);
    conv3x3_tiled_runtime<C_OUT, C_OUT>(h_out, w_out, 1, mid, w2, shift2, output);
    batchnorm_tiled_runtime<C_OUT, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h_out, w_out, output, bn2, output);
    conv1x1_tiled_runtime<C_IN, C_OUT>(h_in, w_in, 2, input, wdown, shift_down, mid);
    residual_add_tiled_runtime<C_OUT>(h_out, w_out, output, mid, output);
    relu_tiled_runtime<C_OUT, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h_out, w_out, output, output);
}}

void top(
{chr(10).join(top_params)}
)
{{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem_input
    #pragma HLS interface m_axi port=DRAM_stem offset=slave bundle=mem_stem
    #pragma HLS interface m_axi port=DRAM_feat_ping offset=slave bundle=mem_feat_ping
    #pragma HLS interface m_axi port=DRAM_feat_pong offset=slave bundle=mem_feat_pong
    #pragma HLS interface m_axi port=DRAM_feat_mid offset=slave bundle=mem_feat_mid
    #pragma HLS interface m_axi port=DRAM_w_stem offset=slave bundle=mem_w_stem
    #pragma HLS interface m_axi port=DRAM_bn_stem offset=slave bundle=mem_bn_stem
{chr(10).join(make_weight_pragmas(blocks))}
    #pragma HLS interface m_axi port=DRAM_fc offset=slave bundle=mem_fc
    #pragma HLS interface m_axi port=DRAM_out offset=slave bundle=mem_out

{chr(10).join(top_calls)}
}}
"""


def make_weight_pragmas(blocks: list[BlockSpec]) -> list[str]:
    pragmas: list[str] = []
    for block in blocks:
        prefix = block.prefix
        pragmas.append(f"    #pragma HLS interface m_axi port=DRAM_w_{prefix}_1 offset=slave bundle=mem_w_{prefix}_1")
        pragmas.append(f"    #pragma HLS interface m_axi port=DRAM_bn_{prefix}_1 offset=slave bundle=mem_bn_{prefix}_1")
        pragmas.append(f"    #pragma HLS interface m_axi port=DRAM_w_{prefix}_2 offset=slave bundle=mem_w_{prefix}_2")
        pragmas.append(f"    #pragma HLS interface m_axi port=DRAM_bn_{prefix}_2 offset=slave bundle=mem_bn_{prefix}_2")
        if block.downsample:
            pragmas.append(f"    #pragma HLS interface m_axi port=DRAM_w_{prefix}_down offset=slave bundle=mem_w_{prefix}_down")
    return pragmas


def make_tcl() -> str:
    return f"""# Auto-generated TCL file for tiled ResNet-18 HLS
open_project -reset project_1

set_top top

add_files top.cpp
add_files top.h
add_files resnet18_tiled_scales.h

open_solution "solution1"

set_part {FPGA_NAME}

create_clock -period {CLOCK_PERIOD} -name default

csynth_design

exit
"""


def emit_project(base_dir: str) -> str:
    blocks = build_blocks()
    params = build_params(blocks)
    output_dir = os.path.join(base_dir, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    files = {
        "top.h": make_top_h(params),
        "top.cpp": make_top_cpp(blocks, params),
        "resnet18_tiled_scales.h": make_scale_header(blocks),
        "run_hls.tcl": make_tcl(),
    }
    for filename, content in files.items():
        with open(os.path.join(output_dir, filename), "w") as handle:
            handle.write(content)
    return output_dir


def main() -> None:
    base_dir = os.path.join(os.path.dirname(__file__), "hls_files")
    output_dir = emit_project(base_dir)
    print(f"Generated tiled ResNet-18 HLS project in {output_dir}")


if __name__ == "__main__":
    main()
