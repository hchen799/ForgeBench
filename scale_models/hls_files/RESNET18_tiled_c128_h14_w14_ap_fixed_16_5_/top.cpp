#include <ap_fixed.h>
#include <hls_math.h>
#include <ap_int.h>
#include "top.h"
#include "resnet18_tiled_scales.h"

typedef ap_fixed<32,10,AP_RND,AP_SAT> acc_t;

static const int TILE_C = 128;
static const int TILE_H = 14;
static const int TILE_W = 14;
static const int MAX_PATCH = 33;
static const int MAX_FEAT_C = 512;
static const int MAX_FEAT_H = 56;
static const int MAX_FEAT_W = 56;

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

inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

inline int out_dim(int size, int pad, int stride, int kernel) {
    return ((size + 2 * pad - kernel) / stride) + 1;
}

inline acc_t abs_acc(acc_t value) {
    return value < 0 ? -value : value;
}

inline acc_t apply_power_of_two_shift(acc_t value, int shift) {
    if (shift > 0) {
        for (int i = 0; i < shift; ++i) {
            value *= (acc_t)2;
        }
    } else if (shift < 0) {
        for (int i = 0; i < -shift; ++i) {
            value /= (acc_t)2;
        }
    }
    return value;
}

inline data_t quantize_acc(acc_t value, int tile_exp, int output_shift) {
    return (data_t)apply_power_of_two_shift(value, tile_exp - output_shift);
}

void clear_acc_tile(int valid_oc, int valid_h, int valid_w) {
    for (int oc = 0; oc < valid_oc; ++oc) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                ACC_TILE[oc][h][w] = (acc_t)0;
            }
        }
    }
}

void clear_fc_acc(int valid_oc) {
    for (int oc = 0; oc < valid_oc; ++oc) {
        FC_ACC[oc] = (acc_t)0;
    }
}

void maybe_rescale_acc_tile(int valid_oc, int valid_h, int valid_w, int &tile_exp) {
    while (true) {
        acc_t max_abs = (acc_t)0;
        for (int oc = 0; oc < valid_oc; ++oc) {
            for (int h = 0; h < valid_h; ++h) {
                for (int w = 0; w < valid_w; ++w) {
                    acc_t current = abs_acc(ACC_TILE[oc][h][w]);
                    if (current > max_abs) {
                        max_abs = current;
                    }
                }
            }
        }
        if (max_abs <= (acc_t)kRenormGuard) {
            return;
        }
        for (int oc = 0; oc < valid_oc; ++oc) {
            for (int h = 0; h < valid_h; ++h) {
                for (int w = 0; w < valid_w; ++w) {
                    ACC_TILE[oc][h][w] /= (acc_t)2;
                }
            }
        }
        ++tile_exp;
    }
}

void maybe_rescale_fc_acc(int valid_oc, int &tile_exp) {
    while (true) {
        acc_t max_abs = (acc_t)0;
        for (int oc = 0; oc < valid_oc; ++oc) {
            acc_t current = abs_acc(FC_ACC[oc]);
            if (current > max_abs) {
                max_abs = current;
            }
        }
        if (max_abs <= (acc_t)kRenormGuard) {
            return;
        }
        for (int oc = 0; oc < valid_oc; ++oc) {
            FC_ACC[oc] /= (acc_t)2;
        }
        ++tile_exp;
    }
}

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
) {
    for (int oc = 0; oc < valid_oc; ++oc) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[co_offset + oc][row_offset + h][col_offset + w] =
                    quantize_acc(ACC_TILE[oc][h][w], tile_exp, output_shift);
            }
        }
    }
}

template <int C>
void load_bn_tile(const data_t weights[4][C], int channel_offset, int valid_c) {
    for (int stat = 0; stat < 4; ++stat) {
        for (int c = 0; c < valid_c; ++c) {
            BN_TILE[stat][c] = weights[stat][channel_offset + c];
        }
    }
}

template <int C_OUT, int C_IN>
void load_w3_tile(const data_t weights[C_OUT][C_IN][3][3], int co_offset, int ci_offset, int valid_oc, int valid_ci) {
    for (int oc = 0; oc < valid_oc; ++oc) {
        for (int ic = 0; ic < valid_ci; ++ic) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    W3_TILE[oc][ic][kh][kw] = weights[co_offset + oc][ci_offset + ic][kh][kw];
                }
            }
        }
    }
}

template <int C_OUT, int C_IN>
void load_w1_tile(const data_t weights[C_OUT][C_IN][1][1], int co_offset, int ci_offset, int valid_oc, int valid_ci) {
    for (int oc = 0; oc < valid_oc; ++oc) {
        for (int ic = 0; ic < valid_ci; ++ic) {
            W1_TILE[oc][ic] = weights[co_offset + oc][ci_offset + ic][0][0];
        }
    }
}

void load_stem_w7_tile(const data_t weights[64][3][7][7], int co_offset, int valid_oc) {
    for (int oc = 0; oc < valid_oc; ++oc) {
        for (int ic = 0; ic < 3; ++ic) {
            for (int kh = 0; kh < 7; ++kh) {
                for (int kw = 0; kw < 7; ++kw) {
                    W7_TILE[oc][ic][kh][kw] = weights[co_offset + oc][ic][kh][kw];
                }
            }
        }
    }
}

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
) {
    int patch_h = (tile_h - 1) * stride + kernel;
    int patch_w = (tile_w - 1) * stride + kernel;
    for (int ic = 0; ic < valid_ci; ++ic) {
        for (int h = 0; h < patch_h; ++h) {
            for (int w = 0; w < patch_w; ++w) {
                int in_row = row_offset * stride - pad + h;
                int in_col = col_offset * stride - pad + w;
                if (in_row >= 0 && in_row < h_in && in_col >= 0 && in_col < w_in) {
                    PATCH_IN[ic][h][w] = input[ci_offset + ic][in_row][in_col];
                } else {
                    PATCH_IN[ic][h][w] = (data_t)0;
                }
            }
        }
    }
}

void run_conv3x3_tile(int valid_oc, int valid_ci, int tile_h, int tile_w, int stride) {
    for (int oc = 0; oc < valid_oc; ++oc) {
        for (int oh = 0; oh < tile_h; ++oh) {
            for (int ow = 0; ow < tile_w; ++ow) {
                acc_t sum = ACC_TILE[oc][oh][ow];
                for (int ic = 0; ic < valid_ci; ++ic) {
                    for (int kh = 0; kh < 3; ++kh) {
                        for (int kw = 0; kw < 3; ++kw) {
                            sum += (acc_t)PATCH_IN[ic][oh * stride + kh][ow * stride + kw] *
                                   (acc_t)W3_TILE[oc][ic][kh][kw];
                        }
                    }
                }
                ACC_TILE[oc][oh][ow] = sum;
            }
        }
    }
}

void run_conv1x1_tile(int valid_oc, int valid_ci, int tile_h, int tile_w, int stride) {
    for (int oc = 0; oc < valid_oc; ++oc) {
        for (int oh = 0; oh < tile_h; ++oh) {
            for (int ow = 0; ow < tile_w; ++ow) {
                acc_t sum = ACC_TILE[oc][oh][ow];
                for (int ic = 0; ic < valid_ci; ++ic) {
                    sum += (acc_t)PATCH_IN[ic][oh * stride][ow * stride] * (acc_t)W1_TILE[oc][ic];
                }
                ACC_TILE[oc][oh][ow] = sum;
            }
        }
    }
}

void run_stem_conv7x7_tile(int valid_oc, int tile_h, int tile_w) {
    for (int oc = 0; oc < valid_oc; ++oc) {
        for (int oh = 0; oh < tile_h; ++oh) {
            for (int ow = 0; ow < tile_w; ++ow) {
                acc_t sum = ACC_TILE[oc][oh][ow];
                for (int ic = 0; ic < 3; ++ic) {
                    for (int kh = 0; kh < 7; ++kh) {
                        for (int kw = 0; kw < 7; ++kw) {
                            sum += (acc_t)PATCH_IN[ic][oh * 2 + kh][ow * 2 + kw] *
                                   (acc_t)W7_TILE[oc][ic][kh][kw];
                        }
                    }
                }
                ACC_TILE[oc][oh][ow] = sum;
            }
        }
    }
}

template <int C_IN, int C_OUT>
void conv3x3_tiled_runtime(
    int h_in,
    int w_in,
    int stride,
    const data_t input[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    const data_t weights[C_OUT][C_IN][3][3],
    int output_shift,
    data_t output[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W]
) {
    int h_out = out_dim(h_in, 1, stride, 3);
    int w_out = out_dim(w_in, 1, stride, 3);
    for (int co = 0; co < C_OUT; co += TILE_C) {
        int valid_oc = ((co + TILE_C) <= C_OUT) ? TILE_C : (C_OUT - co);
        for (int oh = 0; oh < h_out; oh += TILE_H) {
            int valid_h = ((oh + TILE_H) <= h_out) ? TILE_H : (h_out - oh);
            for (int ow = 0; ow < w_out; ow += TILE_W) {
                int valid_w = ((ow + TILE_W) <= w_out) ? TILE_W : (w_out - ow);
                int tile_exp = 0;
                clear_acc_tile(valid_oc, valid_h, valid_w);
                for (int ci = 0; ci < C_IN; ci += TILE_C) {
                    int valid_ci = ((ci + TILE_C) <= C_IN) ? TILE_C : (C_IN - ci);
                    load_patch_from_tensor(input, ci, valid_ci, h_in, w_in, oh, ow, 3, stride, 1, valid_h, valid_w);
                    load_w3_tile(weights, co, ci, valid_oc, valid_ci);
                    run_conv3x3_tile(valid_oc, valid_ci, valid_h, valid_w, stride);
                    maybe_rescale_acc_tile(valid_oc, valid_h, valid_w, tile_exp);
                }
                commit_acc_tile(output, co, oh, ow, valid_oc, valid_h, valid_w, tile_exp, output_shift);
            }
        }
    }
}

template <int C_IN, int C_OUT>
void conv1x1_tiled_runtime(
    int h_in,
    int w_in,
    int stride,
    const data_t input[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    const data_t weights[C_OUT][C_IN][1][1],
    int output_shift,
    data_t output[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W]
) {
    int h_out = out_dim(h_in, 0, stride, 1);
    int w_out = out_dim(w_in, 0, stride, 1);
    for (int co = 0; co < C_OUT; co += TILE_C) {
        int valid_oc = ((co + TILE_C) <= C_OUT) ? TILE_C : (C_OUT - co);
        for (int oh = 0; oh < h_out; oh += TILE_H) {
            int valid_h = ((oh + TILE_H) <= h_out) ? TILE_H : (h_out - oh);
            for (int ow = 0; ow < w_out; ow += TILE_W) {
                int valid_w = ((ow + TILE_W) <= w_out) ? TILE_W : (w_out - ow);
                int tile_exp = 0;
                clear_acc_tile(valid_oc, valid_h, valid_w);
                for (int ci = 0; ci < C_IN; ci += TILE_C) {
                    int valid_ci = ((ci + TILE_C) <= C_IN) ? TILE_C : (C_IN - ci);
                    load_patch_from_tensor(input, ci, valid_ci, h_in, w_in, oh, ow, 1, stride, 0, valid_h, valid_w);
                    load_w1_tile(weights, co, ci, valid_oc, valid_ci);
                    run_conv1x1_tile(valid_oc, valid_ci, valid_h, valid_w, stride);
                    maybe_rescale_acc_tile(valid_oc, valid_h, valid_w, tile_exp);
                }
                commit_acc_tile(output, co, oh, ow, valid_oc, valid_h, valid_w, tile_exp, output_shift);
            }
        }
    }
}

void stem_conv7x7_tiled_runtime(
    const data_t input[3][224][224],
    const data_t weights[64][3][7][7],
    int output_shift,
    data_t output[64][112][112]
) {
    for (int co = 0; co < 64; co += 64) {
        int valid_oc = 64 - co;
        for (int oh = 0; oh < 112; oh += TILE_H) {
            int valid_h = ((oh + TILE_H) <= 112) ? TILE_H : (112 - oh);
            for (int ow = 0; ow < 112; ow += TILE_W) {
                int valid_w = ((ow + TILE_W) <= 112) ? TILE_W : (112 - ow);
                int tile_exp = 0;
                clear_acc_tile(valid_oc, valid_h, valid_w);
                load_patch_from_tensor(input, 0, 3, 224, 224, oh, ow, 7, 2, 3, valid_h, valid_w);
                load_stem_w7_tile(weights, co, valid_oc);
                run_stem_conv7x7_tile(valid_oc, valid_h, valid_w);
                maybe_rescale_acc_tile(valid_oc, valid_h, valid_w, tile_exp);
                commit_acc_tile(output, co, oh, ow, valid_oc, valid_h, valid_w, tile_exp, output_shift);
            }
        }
    }
}

template <int C, int MAX_C, int MAX_H, int MAX_W>
void batchnorm_tiled_runtime(
    int h,
    int w,
    data_t input[MAX_C][MAX_H][MAX_W],
    const data_t weights[4][C],
    data_t output[MAX_C][MAX_H][MAX_W]
) {
    for (int co = 0; co < C; co += TILE_C) {
        int valid_c = ((co + TILE_C) <= C) ? TILE_C : (C - co);
        load_bn_tile(weights, co, valid_c);
        for (int oh = 0; oh < h; oh += TILE_H) {
            int valid_h = ((oh + TILE_H) <= h) ? TILE_H : (h - oh);
            for (int ow = 0; ow < w; ow += TILE_W) {
                int valid_w = ((ow + TILE_W) <= w) ? TILE_W : (w - ow);
                for (int c = 0; c < valid_c; ++c) {
                    acc_t gamma = (acc_t)BN_TILE[0][c];
                    acc_t beta = (acc_t)BN_TILE[1][c];
                    acc_t mean = (acc_t)BN_TILE[2][c];
                    acc_t var = (acc_t)BN_TILE[3][c];
                    for (int ih = 0; ih < valid_h; ++ih) {
                        for (int iw = 0; iw < valid_w; ++iw) {
                            acc_t value = (acc_t)input[co + c][oh + ih][ow + iw];
                            acc_t norm = (value - mean) / hls::sqrt(var + (acc_t)1e-5);
                            output[co + c][oh + ih][ow + iw] = (data_t)(gamma * norm + beta);
                        }
                    }
                }
            }
        }
    }
}

template <int C, int MAX_C, int MAX_H, int MAX_W>
void relu_tiled_runtime(int h, int w, data_t input[MAX_C][MAX_H][MAX_W], data_t output[MAX_C][MAX_H][MAX_W]) {
    for (int co = 0; co < C; co += TILE_C) {
        int valid_c = ((co + TILE_C) <= C) ? TILE_C : (C - co);
        for (int oh = 0; oh < h; oh += TILE_H) {
            int valid_h = ((oh + TILE_H) <= h) ? TILE_H : (h - oh);
            for (int ow = 0; ow < w; ow += TILE_W) {
                int valid_w = ((ow + TILE_W) <= w) ? TILE_W : (w - ow);
                for (int c = 0; c < valid_c; ++c) {
                    for (int ih = 0; ih < valid_h; ++ih) {
                        for (int iw = 0; iw < valid_w; ++iw) {
                            data_t value = input[co + c][oh + ih][ow + iw];
                            output[co + c][oh + ih][ow + iw] = value > (data_t)0 ? value : (data_t)0;
                        }
                    }
                }
            }
        }
    }
}

void maxpool_pad_tiled_runtime(
    data_t input[64][112][112],
    data_t output[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W]
) {
    for (int co = 0; co < 64; co += TILE_C) {
        int valid_c = 64 - co;
        for (int oh = 0; oh < 56; oh += TILE_H) {
            int valid_h = ((oh + TILE_H) <= 56) ? TILE_H : (56 - oh);
            for (int ow = 0; ow < 56; ow += TILE_W) {
                int valid_w = ((ow + TILE_W) <= 56) ? TILE_W : (56 - ow);
                int patch_h = (valid_h - 1) * 2 + 3;
                int patch_w = (valid_w - 1) * 2 + 3;
                for (int c = 0; c < valid_c; ++c) {
                    for (int ph = 0; ph < patch_h; ++ph) {
                        for (int pw = 0; pw < patch_w; ++pw) {
                            int in_row = oh * 2 - 1 + ph;
                            int in_col = ow * 2 - 1 + pw;
                            if (in_row >= 0 && in_row < 112 && in_col >= 0 && in_col < 112) {
                                PATCH_IN[c][ph][pw] = input[co + c][in_row][in_col];
                            } else {
                                PATCH_IN[c][ph][pw] = (data_t)0;
                            }
                        }
                    }
                    for (int ih = 0; ih < valid_h; ++ih) {
                        for (int iw = 0; iw < valid_w; ++iw) {
                            data_t max_value = PATCH_IN[c][ih * 2][iw * 2];
                            for (int kh = 0; kh < 3; ++kh) {
                                for (int kw = 0; kw < 3; ++kw) {
                                    data_t candidate = PATCH_IN[c][ih * 2 + kh][iw * 2 + kw];
                                    if (candidate > max_value) {
                                        max_value = candidate;
                                    }
                                }
                            }
                            output[co + c][oh + ih][ow + iw] = max_value;
                        }
                    }
                }
            }
        }
    }
}

template <int C>
void residual_add_tiled_runtime(
    int h,
    int w,
    data_t lhs[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    data_t rhs[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    data_t output[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W]
) {
    for (int co = 0; co < C; co += TILE_C) {
        int valid_c = ((co + TILE_C) <= C) ? TILE_C : (C - co);
        for (int oh = 0; oh < h; oh += TILE_H) {
            int valid_h = ((oh + TILE_H) <= h) ? TILE_H : (h - oh);
            for (int ow = 0; ow < w; ow += TILE_W) {
                int valid_w = ((ow + TILE_W) <= w) ? TILE_W : (w - ow);
                for (int c = 0; c < valid_c; ++c) {
                    for (int ih = 0; ih < valid_h; ++ih) {
                        for (int iw = 0; iw < valid_w; ++iw) {
                            acc_t value = (acc_t)lhs[co + c][oh + ih][ow + iw] + (acc_t)rhs[co + c][oh + ih][ow + iw];
                            output[co + c][oh + ih][ow + iw] = (data_t)value;
                        }
                    }
                }
            }
        }
    }
}

template <int C>
void global_avgpool_runtime(
    data_t input[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W],
    int h,
    int w,
    data_t output[C]
) {
    for (int co = 0; co < C; co += TILE_C) {
        int valid_c = ((co + TILE_C) <= C) ? TILE_C : (C - co);
        clear_fc_acc(valid_c);
        int tile_exp = 0;
        for (int c = 0; c < valid_c; ++c) {
            for (int ih = 0; ih < h; ++ih) {
                for (int iw = 0; iw < w; ++iw) {
                    FC_ACC[c] += (acc_t)input[co + c][ih][iw];
                }
            }
        }
        maybe_rescale_fc_acc(valid_c, tile_exp);
        for (int c = 0; c < valid_c; ++c) {
            acc_t value = apply_power_of_two_shift(FC_ACC[c], tile_exp) / (acc_t)(h * w);
            output[co + c] = (data_t)value;
        }
    }
}

template <int C_OUT, int C_IN>
void fc_tiled_runtime(
    const data_t input[C_IN],
    const data_t weights[C_OUT][C_IN],
    int output_shift,
    data_t output[C_OUT]
) {
    for (int co = 0; co < C_OUT; co += TILE_C) {
        int valid_oc = ((co + TILE_C) <= C_OUT) ? TILE_C : (C_OUT - co);
        clear_fc_acc(valid_oc);
        int tile_exp = 0;
        for (int ci = 0; ci < C_IN; ci += TILE_C) {
            int valid_ci = ((ci + TILE_C) <= C_IN) ? TILE_C : (C_IN - ci);
            for (int oc = 0; oc < valid_oc; ++oc) {
                for (int ic = 0; ic < valid_ci; ++ic) {
                    FC_W_TILE[oc][ic] = weights[co + oc][ci + ic];
                }
            }
            for (int oc = 0; oc < valid_oc; ++oc) {
                acc_t sum = FC_ACC[oc];
                for (int ic = 0; ic < valid_ci; ++ic) {
                    sum += (acc_t)input[ci + ic] * (acc_t)FC_W_TILE[oc][ic];
                }
                FC_ACC[oc] = sum;
            }
            maybe_rescale_fc_acc(valid_oc, tile_exp);
        }
        for (int oc = 0; oc < valid_oc; ++oc) {
            output[co + oc] = quantize_acc(FC_ACC[oc], tile_exp, output_shift);
        }
    }
}

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
) {
    conv3x3_tiled_runtime<C, C>(h, w, 1, input, w1, shift1, mid);
    batchnorm_tiled_runtime<C, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h, w, mid, bn1, mid);
    relu_tiled_runtime<C, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h, w, mid, mid);
    conv3x3_tiled_runtime<C, C>(h, w, 1, mid, w2, shift2, output);
    batchnorm_tiled_runtime<C, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h, w, output, bn2, output);
    residual_add_tiled_runtime<C>(h, w, output, input, output);
    relu_tiled_runtime<C, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(h, w, output, output);
}

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
) {
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
}

void top(
    data_t DRAM_input[3][224][224],
    data_t DRAM_stem[64][112][112],
    data_t DRAM_feat_ping[512][56][56],
    data_t DRAM_feat_pong[512][56][56],
    data_t DRAM_feat_mid[512][56][56],
    data_t DRAM_w_stem[64][3][7][7],
    data_t DRAM_bn_stem[4][64],
    data_t DRAM_w_s1_b0_1[64][64][3][3],
    data_t DRAM_bn_s1_b0_1[4][64],
    data_t DRAM_w_s1_b0_2[64][64][3][3],
    data_t DRAM_bn_s1_b0_2[4][64],
    data_t DRAM_w_s1_b1_1[64][64][3][3],
    data_t DRAM_bn_s1_b1_1[4][64],
    data_t DRAM_w_s1_b1_2[64][64][3][3],
    data_t DRAM_bn_s1_b1_2[4][64],
    data_t DRAM_w_s2_b0_1[128][64][3][3],
    data_t DRAM_bn_s2_b0_1[4][128],
    data_t DRAM_w_s2_b0_2[128][128][3][3],
    data_t DRAM_bn_s2_b0_2[4][128],
    data_t DRAM_w_s2_b0_down[128][64][1][1],
    data_t DRAM_w_s2_b1_1[128][128][3][3],
    data_t DRAM_bn_s2_b1_1[4][128],
    data_t DRAM_w_s2_b1_2[128][128][3][3],
    data_t DRAM_bn_s2_b1_2[4][128],
    data_t DRAM_w_s3_b0_1[256][128][3][3],
    data_t DRAM_bn_s3_b0_1[4][256],
    data_t DRAM_w_s3_b0_2[256][256][3][3],
    data_t DRAM_bn_s3_b0_2[4][256],
    data_t DRAM_w_s3_b0_down[256][128][1][1],
    data_t DRAM_w_s3_b1_1[256][256][3][3],
    data_t DRAM_bn_s3_b1_1[4][256],
    data_t DRAM_w_s3_b1_2[256][256][3][3],
    data_t DRAM_bn_s3_b1_2[4][256],
    data_t DRAM_w_s4_b0_1[512][256][3][3],
    data_t DRAM_bn_s4_b0_1[4][512],
    data_t DRAM_w_s4_b0_2[512][512][3][3],
    data_t DRAM_bn_s4_b0_2[4][512],
    data_t DRAM_w_s4_b0_down[512][256][1][1],
    data_t DRAM_w_s4_b1_1[512][512][3][3],
    data_t DRAM_bn_s4_b1_1[4][512],
    data_t DRAM_w_s4_b1_2[512][512][3][3],
    data_t DRAM_bn_s4_b1_2[4][512],
    data_t DRAM_fc[1000][512],
    data_t DRAM_out[1000]
)
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem_input
    #pragma HLS interface m_axi port=DRAM_stem offset=slave bundle=mem_stem
    #pragma HLS interface m_axi port=DRAM_feat_ping offset=slave bundle=mem_feat_ping
    #pragma HLS interface m_axi port=DRAM_feat_pong offset=slave bundle=mem_feat_pong
    #pragma HLS interface m_axi port=DRAM_feat_mid offset=slave bundle=mem_feat_mid
    #pragma HLS interface m_axi port=DRAM_w_stem offset=slave bundle=mem_w_stem
    #pragma HLS interface m_axi port=DRAM_bn_stem offset=slave bundle=mem_bn_stem
    #pragma HLS interface m_axi port=DRAM_w_s1_b0_1 offset=slave bundle=mem_w_s1_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s1_b0_1 offset=slave bundle=mem_bn_s1_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s1_b0_2 offset=slave bundle=mem_w_s1_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s1_b0_2 offset=slave bundle=mem_bn_s1_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s1_b1_1 offset=slave bundle=mem_w_s1_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s1_b1_1 offset=slave bundle=mem_bn_s1_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s1_b1_2 offset=slave bundle=mem_w_s1_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s1_b1_2 offset=slave bundle=mem_bn_s1_b1_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_1 offset=slave bundle=mem_w_s2_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b0_1 offset=slave bundle=mem_bn_s2_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_2 offset=slave bundle=mem_w_s2_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b0_2 offset=slave bundle=mem_bn_s2_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_down offset=slave bundle=mem_w_s2_b0_down
    #pragma HLS interface m_axi port=DRAM_w_s2_b1_1 offset=slave bundle=mem_w_s2_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b1_1 offset=slave bundle=mem_bn_s2_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b1_2 offset=slave bundle=mem_w_s2_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b1_2 offset=slave bundle=mem_bn_s2_b1_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_1 offset=slave bundle=mem_w_s3_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b0_1 offset=slave bundle=mem_bn_s3_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_2 offset=slave bundle=mem_w_s3_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b0_2 offset=slave bundle=mem_bn_s3_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_down offset=slave bundle=mem_w_s3_b0_down
    #pragma HLS interface m_axi port=DRAM_w_s3_b1_1 offset=slave bundle=mem_w_s3_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b1_1 offset=slave bundle=mem_bn_s3_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b1_2 offset=slave bundle=mem_w_s3_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b1_2 offset=slave bundle=mem_bn_s3_b1_2
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_1 offset=slave bundle=mem_w_s4_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s4_b0_1 offset=slave bundle=mem_bn_s4_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_2 offset=slave bundle=mem_w_s4_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s4_b0_2 offset=slave bundle=mem_bn_s4_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_down offset=slave bundle=mem_w_s4_b0_down
    #pragma HLS interface m_axi port=DRAM_w_s4_b1_1 offset=slave bundle=mem_w_s4_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s4_b1_1 offset=slave bundle=mem_bn_s4_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s4_b1_2 offset=slave bundle=mem_w_s4_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s4_b1_2 offset=slave bundle=mem_bn_s4_b1_2
    #pragma HLS interface m_axi port=DRAM_fc offset=slave bundle=mem_fc
    #pragma HLS interface m_axi port=DRAM_out offset=slave bundle=mem_out

    stem_conv7x7_tiled_runtime(DRAM_input, DRAM_w_stem, kConvOutputShift[SHIFT_STEM], DRAM_stem);
    batchnorm_tiled_runtime<64, 64, 112, 112>(112, 112, DRAM_stem, DRAM_bn_stem, DRAM_stem);
    relu_tiled_runtime<64, 64, 112, 112>(112, 112, DRAM_stem, DRAM_stem);
    maxpool_pad_tiled_runtime(DRAM_stem, DRAM_feat_ping);
    run_identity_block<64>(56, 56, DRAM_feat_ping, DRAM_w_s1_b0_1, DRAM_bn_s1_b0_1, DRAM_w_s1_b0_2, DRAM_bn_s1_b0_2, DRAM_feat_mid, DRAM_feat_pong, kConvOutputShift[SHIFT_S1_B0_1], kConvOutputShift[SHIFT_S1_B0_2]);
    run_identity_block<64>(56, 56, DRAM_feat_pong, DRAM_w_s1_b1_1, DRAM_bn_s1_b1_1, DRAM_w_s1_b1_2, DRAM_bn_s1_b1_2, DRAM_feat_mid, DRAM_feat_ping, kConvOutputShift[SHIFT_S1_B1_1], kConvOutputShift[SHIFT_S1_B1_2]);
    run_downsample_block<64, 128>(56, 56, DRAM_feat_ping, DRAM_w_s2_b0_1, DRAM_bn_s2_b0_1, DRAM_w_s2_b0_2, DRAM_bn_s2_b0_2, DRAM_w_s2_b0_down, DRAM_feat_mid, DRAM_feat_pong, kConvOutputShift[SHIFT_S2_B0_1], kConvOutputShift[SHIFT_S2_B0_2], kConvOutputShift[SHIFT_S2_B0_DOWN]);
    run_identity_block<128>(28, 28, DRAM_feat_pong, DRAM_w_s2_b1_1, DRAM_bn_s2_b1_1, DRAM_w_s2_b1_2, DRAM_bn_s2_b1_2, DRAM_feat_mid, DRAM_feat_ping, kConvOutputShift[SHIFT_S2_B1_1], kConvOutputShift[SHIFT_S2_B1_2]);
    run_downsample_block<128, 256>(28, 28, DRAM_feat_ping, DRAM_w_s3_b0_1, DRAM_bn_s3_b0_1, DRAM_w_s3_b0_2, DRAM_bn_s3_b0_2, DRAM_w_s3_b0_down, DRAM_feat_mid, DRAM_feat_pong, kConvOutputShift[SHIFT_S3_B0_1], kConvOutputShift[SHIFT_S3_B0_2], kConvOutputShift[SHIFT_S3_B0_DOWN]);
    run_identity_block<256>(14, 14, DRAM_feat_pong, DRAM_w_s3_b1_1, DRAM_bn_s3_b1_1, DRAM_w_s3_b1_2, DRAM_bn_s3_b1_2, DRAM_feat_mid, DRAM_feat_ping, kConvOutputShift[SHIFT_S3_B1_1], kConvOutputShift[SHIFT_S3_B1_2]);
    run_downsample_block<256, 512>(14, 14, DRAM_feat_ping, DRAM_w_s4_b0_1, DRAM_bn_s4_b0_1, DRAM_w_s4_b0_2, DRAM_bn_s4_b0_2, DRAM_w_s4_b0_down, DRAM_feat_mid, DRAM_feat_pong, kConvOutputShift[SHIFT_S4_B0_1], kConvOutputShift[SHIFT_S4_B0_2], kConvOutputShift[SHIFT_S4_B0_DOWN]);
    run_identity_block<512>(7, 7, DRAM_feat_pong, DRAM_w_s4_b1_1, DRAM_bn_s4_b1_1, DRAM_w_s4_b1_2, DRAM_bn_s4_b1_2, DRAM_feat_mid, DRAM_feat_ping, kConvOutputShift[SHIFT_S4_B1_1], kConvOutputShift[SHIFT_S4_B1_2]);
    global_avgpool_runtime(DRAM_feat_ping, 7, 7, POOL_VEC);
    fc_tiled_runtime(POOL_VEC, DRAM_fc, kConvOutputShift[SHIFT_FC], DRAM_out);
}
