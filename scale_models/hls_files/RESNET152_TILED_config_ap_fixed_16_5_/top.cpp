
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>
using namespace std;

typedef ap_fixed<16,5> data_t;
typedef ap_fixed<32,10> acc_t;

data_t BRAM_in_patch_stem[32][33][33];
data_t BRAM_in_patch_stride2_k3[32][29][29];
data_t BRAM_in_patch_stride2_k1[32][27][27];
data_t BRAM_in_patch_stride1_k3[32][16][16];
data_t BRAM_in_patch_1[32][1][1];
data_t BRAM_pool_patch[128][29][29];
data_t BRAM_out_tile[128][14][14];
data_t BRAM_skip_tile[128][14][14];
data_t BRAM_weight_tile_7[128][32][7][7];
data_t BRAM_weight_tile_3[128][32][3][3];
data_t BRAM_weight_tile_1[128][32][1][1];
data_t BRAM_bn_tile[4][128];
data_t BRAM_gap_tile[128][7][7];
data_t BRAM_gap_out[128][1][1];
data_t BRAM_fc_out[128][1][1];

void clear_tile_128_14_14_ap_fixed_16_5_(data_t tile[128][14][14])
{
    for (int c = 0; c < 128; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                tile[c][h][w] = (data_t)0;
            }
        }
    }
}

void load_fmap_patch_3_224_224_32_33_33_2_3_ap_fixed_16_5_(
    data_t input[3][224][224],
    data_t output[32][33][33],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 33; ++h) {
            for (int w = 0; w < 33; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 3 + h;
                int src_w = ow_base * 2 - 3 + w;
                if (src_c < 3 && src_h >= 0 && src_h < 224 && src_w >= 0 && src_w < 224) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_64_3_7_128_32_ap_fixed_16_5_(
    data_t input[64][3][7][7],
    data_t output[128][32][7][7],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 7; ++kh) {
                for (int kw = 0; kw < 7; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 64 && src_ci < 3) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void conv_tile_32_33_33_128_32_7_128_14_14_2_8_64_ap_fixed_16_5_(
    data_t input[32][33][33],
    data_t weight[128][32][7][7],
    data_t output[128][14][14],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=8 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=64 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=8 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=64 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 7; ++kh) {
                for (int kw = 0; kw < 7; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=8
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=64
                            output[co][oh][ow] += input[ci][oh * 2 + kh][ow * 2 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void load_bn_tile_64_128_ap_fixed_16_5_(
    data_t input[4][64],
    data_t output[4][128],
    int co_base
)
{
    for (int stat = 0; stat < 4; ++stat) {
        for (int co = 0; co < 128; ++co) {
            int src_co = co_base + co;
            if (src_co < 64) {
                output[stat][co] = input[stat][src_co];
            } else {
                output[stat][co] = (data_t)0;
            }
        }
    }
}

void batchnorm_tile_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t weights[4][128],
    data_t output[128][14][14],
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                data_t norm = (input[c][h][w] - weights[2][c]) / hls::sqrt(weights[3][c] + (data_t)0.00001);
                output[c][h][w] = weights[0][c] * norm + weights[1][c];
            }
        }
    }
}

void relu_tile_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[128][14][14],
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c][h][w] = input[c][h][w] > (data_t)0 ? input[c][h][w] : (data_t)0;
            }
        }
    }
}

void store_fmap_tile_64_112_112_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[64][112][112],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_64_112_112_128_29_29_2_1_ap_fixed_16_5_(
    data_t input[64][112][112],
    data_t output[128][29][29],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 128; ++c) {
        for (int h = 0; h < 29; ++h) {
            for (int w = 0; w < 29; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 1 + h;
                int src_w = ow_base * 2 - 1 + w;
                if (src_c < 64 && src_h >= 0 && src_h < 112 && src_w >= 0 && src_w < 112) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void maxpool_tile_128_29_29_128_14_14_3_3_2_2_ap_fixed_16_5_(
    data_t input[128][29][29],
    data_t output[128][14][14],
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int oh = 0; oh < valid_h; ++oh) {
            for (int ow = 0; ow < valid_w; ++ow) {
                data_t max_val = input[c][oh * 2][ow * 2];
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        data_t cand = input[c][oh * 2 + kh][ow * 2 + kw];
                        if (cand > max_val) {
                            max_val = cand;
                        }
                    }
                }
                output[c][oh][ow] = max_val;
            }
        }
    }
}

void store_fmap_tile_64_56_56_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[64][56][56],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_64_56_56_32_14_14_1_0_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t output[32][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 0 + h;
                int src_w = ow_base * 1 - 0 + w;
                if (src_c < 64 && src_h >= 0 && src_h < 56 && src_w >= 0 && src_w < 56) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_64_64_1_128_32_ap_fixed_16_5_(
    data_t input[64][64][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 64 && src_ci < 64) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(
    data_t input[32][14][14],
    data_t weight[128][32][1][1],
    data_t output[128][14][14],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=8 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=64 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=8 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=64 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=8
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=64
                            output[co][oh][ow] += input[ci][oh * 1 + kh][ow * 1 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void load_fmap_patch_64_56_56_32_16_16_1_1_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t output[32][16][16],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 16; ++h) {
            for (int w = 0; w < 16; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 1 + h;
                int src_w = ow_base * 1 - 1 + w;
                if (src_c < 64 && src_h >= 0 && src_h < 56 && src_w >= 0 && src_w < 56) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_64_64_3_128_32_ap_fixed_16_5_(
    data_t input[64][64][3][3],
    data_t output[128][32][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 64 && src_ci < 64) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(
    data_t input[32][16][16],
    data_t weight[128][32][3][3],
    data_t output[128][14][14],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=8 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=64 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=8 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=64 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=8
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=64
                            output[co][oh][ow] += input[ci][oh * 1 + kh][ow * 1 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void load_weight_tile_256_64_1_128_32_ap_fixed_16_5_(
    data_t input[256][64][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 256 && src_ci < 64) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_bn_tile_256_128_ap_fixed_16_5_(
    data_t input[4][256],
    data_t output[4][128],
    int co_base
)
{
    for (int stat = 0; stat < 4; ++stat) {
        for (int co = 0; co < 128; ++co) {
            int src_co = co_base + co;
            if (src_co < 256) {
                output[stat][co] = input[stat][src_co];
            } else {
                output[stat][co] = (data_t)0;
            }
        }
    }
}

void store_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[256][56][56],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(
    data_t input[256][56][56],
    data_t output[128][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 128; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base + h;
                int src_w = ow_base + w;
                if (src_c < 256 && src_h >= 0 && src_h < 56 && src_w >= 0 && src_w < 56) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void matrix_add_tile_128_14_14_ap_fixed_16_5_(
    data_t lhs[128][14][14],
    data_t rhs[128][14][14],
    data_t output[128][14][14],
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c][h][w] = lhs[c][h][w] + rhs[c][h][w];
            }
        }
    }
}

void load_fmap_patch_256_56_56_32_14_14_1_0_ap_fixed_16_5_(
    data_t input[256][56][56],
    data_t output[32][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 0 + h;
                int src_w = ow_base * 1 - 0 + w;
                if (src_c < 256 && src_h >= 0 && src_h < 56 && src_w >= 0 && src_w < 56) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_64_256_1_128_32_ap_fixed_16_5_(
    data_t input[64][256][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 64 && src_ci < 256) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_weight_tile_128_256_1_128_32_ap_fixed_16_5_(
    data_t input[128][256][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 128 && src_ci < 256) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_bn_tile_128_128_ap_fixed_16_5_(
    data_t input[4][128],
    data_t output[4][128],
    int co_base
)
{
    for (int stat = 0; stat < 4; ++stat) {
        for (int co = 0; co < 128; ++co) {
            int src_co = co_base + co;
            if (src_co < 128) {
                output[stat][co] = input[stat][src_co];
            } else {
                output[stat][co] = (data_t)0;
            }
        }
    }
}

void store_fmap_tile_128_56_56_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[128][56][56],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_128_56_56_32_29_29_2_1_ap_fixed_16_5_(
    data_t input[128][56][56],
    data_t output[32][29][29],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 29; ++h) {
            for (int w = 0; w < 29; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 1 + h;
                int src_w = ow_base * 2 - 1 + w;
                if (src_c < 128 && src_h >= 0 && src_h < 56 && src_w >= 0 && src_w < 56) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_128_128_3_128_32_ap_fixed_16_5_(
    data_t input[128][128][3][3],
    data_t output[128][32][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 128 && src_ci < 128) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void conv_tile_32_29_29_128_32_3_128_14_14_2_8_64_ap_fixed_16_5_(
    data_t input[32][29][29],
    data_t weight[128][32][3][3],
    data_t output[128][14][14],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=8 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=64 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=8 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=64 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=8
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=64
                            output[co][oh][ow] += input[ci][oh * 2 + kh][ow * 2 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[128][28][28],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_128_28_28_32_14_14_1_0_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t output[32][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 0 + h;
                int src_w = ow_base * 1 - 0 + w;
                if (src_c < 128 && src_h >= 0 && src_h < 28 && src_w >= 0 && src_w < 28) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_512_128_1_128_32_ap_fixed_16_5_(
    data_t input[512][128][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 512 && src_ci < 128) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_bn_tile_512_128_ap_fixed_16_5_(
    data_t input[4][512],
    data_t output[4][128],
    int co_base
)
{
    for (int stat = 0; stat < 4; ++stat) {
        for (int co = 0; co < 128; ++co) {
            int src_co = co_base + co;
            if (src_co < 512) {
                output[stat][co] = input[stat][src_co];
            } else {
                output[stat][co] = (data_t)0;
            }
        }
    }
}

void store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[512][28][28],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_256_56_56_32_27_27_2_0_ap_fixed_16_5_(
    data_t input[256][56][56],
    data_t output[32][27][27],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 27; ++h) {
            for (int w = 0; w < 27; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 0 + h;
                int src_w = ow_base * 2 - 0 + w;
                if (src_c < 256 && src_h >= 0 && src_h < 56 && src_w >= 0 && src_w < 56) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_512_256_1_128_32_ap_fixed_16_5_(
    data_t input[512][256][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 512 && src_ci < 256) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void conv_tile_32_27_27_128_32_1_128_14_14_2_8_64_ap_fixed_16_5_(
    data_t input[32][27][27],
    data_t weight[128][32][1][1],
    data_t output[128][14][14],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=8 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=64 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=8 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=64 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=8
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=64
                            output[co][oh][ow] += input[ci][oh * 2 + kh][ow * 2 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(
    data_t input[512][28][28],
    data_t output[128][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 128; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base + h;
                int src_w = ow_base + w;
                if (src_c < 512 && src_h >= 0 && src_h < 28 && src_w >= 0 && src_w < 28) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_fmap_patch_512_28_28_32_14_14_1_0_ap_fixed_16_5_(
    data_t input[512][28][28],
    data_t output[32][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 0 + h;
                int src_w = ow_base * 1 - 0 + w;
                if (src_c < 512 && src_h >= 0 && src_h < 28 && src_w >= 0 && src_w < 28) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_128_512_1_128_32_ap_fixed_16_5_(
    data_t input[128][512][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 128 && src_ci < 512) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_fmap_patch_128_28_28_32_16_16_1_1_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t output[32][16][16],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 16; ++h) {
            for (int w = 0; w < 16; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 1 + h;
                int src_w = ow_base * 1 - 1 + w;
                if (src_c < 128 && src_h >= 0 && src_h < 28 && src_w >= 0 && src_w < 28) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_256_512_1_128_32_ap_fixed_16_5_(
    data_t input[256][512][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 256 && src_ci < 512) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void store_fmap_tile_256_28_28_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[256][28][28],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_256_28_28_32_29_29_2_1_ap_fixed_16_5_(
    data_t input[256][28][28],
    data_t output[32][29][29],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 29; ++h) {
            for (int w = 0; w < 29; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 1 + h;
                int src_w = ow_base * 2 - 1 + w;
                if (src_c < 256 && src_h >= 0 && src_h < 28 && src_w >= 0 && src_w < 28) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(
    data_t input[256][256][3][3],
    data_t output[128][32][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 256 && src_ci < 256) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[256][14][14],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t output[32][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 0 + h;
                int src_w = ow_base * 1 - 0 + w;
                if (src_c < 256 && src_h >= 0 && src_h < 14 && src_w >= 0 && src_w < 14) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(
    data_t input[1024][256][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 1024 && src_ci < 256) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_bn_tile_1024_128_ap_fixed_16_5_(
    data_t input[4][1024],
    data_t output[4][128],
    int co_base
)
{
    for (int stat = 0; stat < 4; ++stat) {
        for (int co = 0; co < 128; ++co) {
            int src_co = co_base + co;
            if (src_co < 1024) {
                output[stat][co] = input[stat][src_co];
            } else {
                output[stat][co] = (data_t)0;
            }
        }
    }
}

void store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[1024][14][14],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_512_28_28_32_27_27_2_0_ap_fixed_16_5_(
    data_t input[512][28][28],
    data_t output[32][27][27],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 27; ++h) {
            for (int w = 0; w < 27; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 0 + h;
                int src_w = ow_base * 2 - 0 + w;
                if (src_c < 512 && src_h >= 0 && src_h < 28 && src_w >= 0 && src_w < 28) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_1024_512_1_128_32_ap_fixed_16_5_(
    data_t input[1024][512][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 1024 && src_ci < 512) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(
    data_t input[1024][14][14],
    data_t output[128][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 128; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base + h;
                int src_w = ow_base + w;
                if (src_c < 1024 && src_h >= 0 && src_h < 14 && src_w >= 0 && src_w < 14) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(
    data_t input[1024][14][14],
    data_t output[32][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 0 + h;
                int src_w = ow_base * 1 - 0 + w;
                if (src_c < 1024 && src_h >= 0 && src_h < 14 && src_w >= 0 && src_w < 14) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(
    data_t input[256][1024][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 256 && src_ci < 1024) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t output[32][16][16],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 16; ++h) {
            for (int w = 0; w < 16; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 1 + h;
                int src_w = ow_base * 1 - 1 + w;
                if (src_c < 256 && src_h >= 0 && src_h < 14 && src_w >= 0 && src_w < 14) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_512_1024_1_128_32_ap_fixed_16_5_(
    data_t input[512][1024][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 512 && src_ci < 1024) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void store_fmap_tile_512_14_14_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[512][14][14],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_512_14_14_32_29_29_2_1_ap_fixed_16_5_(
    data_t input[512][14][14],
    data_t output[32][29][29],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 29; ++h) {
            for (int w = 0; w < 29; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 1 + h;
                int src_w = ow_base * 2 - 1 + w;
                if (src_c < 512 && src_h >= 0 && src_h < 14 && src_w >= 0 && src_w < 14) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_512_512_3_128_32_ap_fixed_16_5_(
    data_t input[512][512][3][3],
    data_t output[128][32][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 512 && src_ci < 512) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void store_fmap_tile_512_7_7_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[512][7][7],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_512_7_7_32_14_14_1_0_ap_fixed_16_5_(
    data_t input[512][7][7],
    data_t output[32][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 0 + h;
                int src_w = ow_base * 1 - 0 + w;
                if (src_c < 512 && src_h >= 0 && src_h < 7 && src_w >= 0 && src_w < 7) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_2048_512_1_128_32_ap_fixed_16_5_(
    data_t input[2048][512][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 2048 && src_ci < 512) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_bn_tile_2048_128_ap_fixed_16_5_(
    data_t input[4][2048],
    data_t output[4][128],
    int co_base
)
{
    for (int stat = 0; stat < 4; ++stat) {
        for (int co = 0; co < 128; ++co) {
            int src_co = co_base + co;
            if (src_co < 2048) {
                output[stat][co] = input[stat][src_co];
            } else {
                output[stat][co] = (data_t)0;
            }
        }
    }
}

void store_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(
    data_t input[128][14][14],
    data_t output[2048][7][7],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_1024_14_14_32_27_27_2_0_ap_fixed_16_5_(
    data_t input[1024][14][14],
    data_t output[32][27][27],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 27; ++h) {
            for (int w = 0; w < 27; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 0 + h;
                int src_w = ow_base * 2 - 0 + w;
                if (src_c < 1024 && src_h >= 0 && src_h < 14 && src_w >= 0 && src_w < 14) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_2048_1024_1_128_32_ap_fixed_16_5_(
    data_t input[2048][1024][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 2048 && src_ci < 1024) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(
    data_t input[2048][7][7],
    data_t output[128][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 128; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base + h;
                int src_w = ow_base + w;
                if (src_c < 2048 && src_h >= 0 && src_h < 7 && src_w >= 0 && src_w < 7) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_fmap_patch_2048_7_7_32_14_14_1_0_ap_fixed_16_5_(
    data_t input[2048][7][7],
    data_t output[32][14][14],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 14; ++h) {
            for (int w = 0; w < 14; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 0 + h;
                int src_w = ow_base * 1 - 0 + w;
                if (src_c < 2048 && src_h >= 0 && src_h < 7 && src_w >= 0 && src_w < 7) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_512_2048_1_128_32_ap_fixed_16_5_(
    data_t input[512][2048][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 512 && src_ci < 2048) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_fmap_patch_512_7_7_32_16_16_1_1_ap_fixed_16_5_(
    data_t input[512][7][7],
    data_t output[32][16][16],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 16; ++h) {
            for (int w = 0; w < 16; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 1 + h;
                int src_w = ow_base * 1 - 1 + w;
                if (src_c < 512 && src_h >= 0 && src_h < 7 && src_w >= 0 && src_w < 7) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void clear_tile_128_1_1_ap_fixed_16_5_(data_t tile[128][1][1])
{
    for (int c = 0; c < 128; ++c) {
        for (int h = 0; h < 1; ++h) {
            for (int w = 0; w < 1; ++w) {
                tile[c][h][w] = (data_t)0;
            }
        }
    }
}

void load_fmap_tile_2048_7_7_128_7_7_ap_fixed_16_5_(
    data_t input[2048][7][7],
    data_t output[128][7][7],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 128; ++c) {
        for (int h = 0; h < 7; ++h) {
            for (int w = 0; w < 7; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base + h;
                int src_w = ow_base + w;
                if (src_c < 2048 && src_h >= 0 && src_h < 7 && src_w >= 0 && src_w < 7) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void avgpool_accumulate_tile_128_7_7_ap_fixed_16_5_(
    data_t input[128][7][7],
    data_t output[128][1][1],
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c][0][0] += input[c][h][w];
            }
        }
    }
}

void avgpool_finalize_tile_128_49_ap_fixed_16_5_(
    data_t input[128][1][1],
    data_t output[128][1][1],
    int valid_c
)
{
    for (int c = 0; c < valid_c; ++c) {
        output[c][0][0] = input[c][0][0] / (data_t)49;
    }
}

void store_fmap_tile_2048_1_1_128_1_1_ap_fixed_16_5_(
    data_t input[128][1][1],
    data_t output[2048][1][1],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void load_fmap_patch_2048_1_1_32_1_1_1_0_ap_fixed_16_5_(
    data_t input[2048][1][1],
    data_t output[32][1][1],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 32; ++c) {
        for (int h = 0; h < 1; ++h) {
            for (int w = 0; w < 1; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 0 + h;
                int src_w = ow_base * 1 - 0 + w;
                if (src_c < 2048 && src_h >= 0 && src_h < 1 && src_w >= 0 && src_w < 1) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_1000_2048_1_128_32_ap_fixed_16_5_(
    data_t input[1000][2048][1][1],
    data_t output[128][32][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 128; ++co) {
        for (int ci = 0; ci < 32; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 1000 && src_ci < 2048) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void conv_tile_32_1_1_128_32_1_128_1_1_1_8_64_ap_fixed_16_5_(
    data_t input[32][1][1],
    data_t weight[128][32][1][1],
    data_t output[128][1][1],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=8 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=64 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=8 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=64 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=8
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=64
                            output[co][oh][ow] += input[ci][oh * 1 + kh][ow * 1 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void store_fmap_tile_1000_1_1_128_1_1_ap_fixed_16_5_(
    data_t input[128][1][1],
    data_t output[1000][1][1],
    int c_base,
    int oh_base,
    int ow_base,
    int valid_c,
    int valid_h,
    int valid_w
)
{
    for (int c = 0; c < valid_c; ++c) {
        for (int h = 0; h < valid_h; ++h) {
            for (int w = 0; w < valid_w; ++w) {
                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];
            }
        }
    }
}

void top(data_t DRAM_input[3][224][224], data_t DRAM_w_stem[64][3][7][7], data_t DRAM_bn_stem[4][64], data_t DRAM_stem_feat[64][112][112], data_t DRAM_stem_pool[64][56][56], data_t DRAM_s1_mid1[64][56][56], data_t DRAM_s1_mid2[64][56][56], data_t DRAM_s1_skip[256][56][56], data_t DRAM_s1_b0[256][56][56], data_t DRAM_w_s1_b0_1[64][64][1][1], data_t DRAM_bn_s1_b0_1[4][64], data_t DRAM_w_s1_b0_2[64][64][3][3], data_t DRAM_bn_s1_b0_2[4][64], data_t DRAM_w_s1_b0_3[256][64][1][1], data_t DRAM_bn_s1_b0_3[4][256], data_t DRAM_w_s1_b0_down[256][64][1][1], data_t DRAM_s1_b1[256][56][56], data_t DRAM_w_s1_b1_1[64][256][1][1], data_t DRAM_bn_s1_b1_1[4][64], data_t DRAM_w_s1_b1_2[64][64][3][3], data_t DRAM_bn_s1_b1_2[4][64], data_t DRAM_w_s1_b1_3[256][64][1][1], data_t DRAM_bn_s1_b1_3[4][256], data_t DRAM_s1_b2[256][56][56], data_t DRAM_w_s1_b2_1[64][256][1][1], data_t DRAM_bn_s1_b2_1[4][64], data_t DRAM_w_s1_b2_2[64][64][3][3], data_t DRAM_bn_s1_b2_2[4][64], data_t DRAM_w_s1_b2_3[256][64][1][1], data_t DRAM_bn_s1_b2_3[4][256], data_t DRAM_s2_mid1[128][28][28], data_t DRAM_s2_mid2[128][28][28], data_t DRAM_s2_skip[512][28][28], data_t DRAM_s2_b0[512][28][28], data_t DRAM_w_s2_b0_1[128][256][1][1], data_t DRAM_bn_s2_b0_1[4][128], data_t DRAM_w_s2_b0_2[128][128][3][3], data_t DRAM_bn_s2_b0_2[4][128], data_t DRAM_w_s2_b0_3[512][128][1][1], data_t DRAM_bn_s2_b0_3[4][512], data_t DRAM_w_s2_b0_down[512][256][1][1], data_t DRAM_s2_b1[512][28][28], data_t DRAM_w_s2_b1_1[128][512][1][1], data_t DRAM_bn_s2_b1_1[4][128], data_t DRAM_w_s2_b1_2[128][128][3][3], data_t DRAM_bn_s2_b1_2[4][128], data_t DRAM_w_s2_b1_3[512][128][1][1], data_t DRAM_bn_s2_b1_3[4][512], data_t DRAM_s2_b2[512][28][28], data_t DRAM_w_s2_b2_1[128][512][1][1], data_t DRAM_bn_s2_b2_1[4][128], data_t DRAM_w_s2_b2_2[128][128][3][3], data_t DRAM_bn_s2_b2_2[4][128], data_t DRAM_w_s2_b2_3[512][128][1][1], data_t DRAM_bn_s2_b2_3[4][512], data_t DRAM_s2_b3[512][28][28], data_t DRAM_w_s2_b3_1[128][512][1][1], data_t DRAM_bn_s2_b3_1[4][128], data_t DRAM_w_s2_b3_2[128][128][3][3], data_t DRAM_bn_s2_b3_2[4][128], data_t DRAM_w_s2_b3_3[512][128][1][1], data_t DRAM_bn_s2_b3_3[4][512], data_t DRAM_s2_b4[512][28][28], data_t DRAM_w_s2_b4_1[128][512][1][1], data_t DRAM_bn_s2_b4_1[4][128], data_t DRAM_w_s2_b4_2[128][128][3][3], data_t DRAM_bn_s2_b4_2[4][128], data_t DRAM_w_s2_b4_3[512][128][1][1], data_t DRAM_bn_s2_b4_3[4][512], data_t DRAM_s2_b5[512][28][28], data_t DRAM_w_s2_b5_1[128][512][1][1], data_t DRAM_bn_s2_b5_1[4][128], data_t DRAM_w_s2_b5_2[128][128][3][3], data_t DRAM_bn_s2_b5_2[4][128], data_t DRAM_w_s2_b5_3[512][128][1][1], data_t DRAM_bn_s2_b5_3[4][512], data_t DRAM_s2_b6[512][28][28], data_t DRAM_w_s2_b6_1[128][512][1][1], data_t DRAM_bn_s2_b6_1[4][128], data_t DRAM_w_s2_b6_2[128][128][3][3], data_t DRAM_bn_s2_b6_2[4][128], data_t DRAM_w_s2_b6_3[512][128][1][1], data_t DRAM_bn_s2_b6_3[4][512], data_t DRAM_s2_b7[512][28][28], data_t DRAM_w_s2_b7_1[128][512][1][1], data_t DRAM_bn_s2_b7_1[4][128], data_t DRAM_w_s2_b7_2[128][128][3][3], data_t DRAM_bn_s2_b7_2[4][128], data_t DRAM_w_s2_b7_3[512][128][1][1], data_t DRAM_bn_s2_b7_3[4][512], data_t DRAM_s3_mid1[256][14][14], data_t DRAM_s3_mid2[256][14][14], data_t DRAM_s3_skip[1024][14][14], data_t DRAM_s3_b0[1024][14][14], data_t DRAM_w_s3_b0_1[256][512][1][1], data_t DRAM_bn_s3_b0_1[4][256], data_t DRAM_w_s3_b0_2[256][256][3][3], data_t DRAM_bn_s3_b0_2[4][256], data_t DRAM_w_s3_b0_3[1024][256][1][1], data_t DRAM_bn_s3_b0_3[4][1024], data_t DRAM_w_s3_b0_down[1024][512][1][1], data_t DRAM_s3_b1[1024][14][14], data_t DRAM_w_s3_b1_1[256][1024][1][1], data_t DRAM_bn_s3_b1_1[4][256], data_t DRAM_w_s3_b1_2[256][256][3][3], data_t DRAM_bn_s3_b1_2[4][256], data_t DRAM_w_s3_b1_3[1024][256][1][1], data_t DRAM_bn_s3_b1_3[4][1024], data_t DRAM_s3_b2[1024][14][14], data_t DRAM_w_s3_b2_1[256][1024][1][1], data_t DRAM_bn_s3_b2_1[4][256], data_t DRAM_w_s3_b2_2[256][256][3][3], data_t DRAM_bn_s3_b2_2[4][256], data_t DRAM_w_s3_b2_3[1024][256][1][1], data_t DRAM_bn_s3_b2_3[4][1024], data_t DRAM_s3_b3[1024][14][14], data_t DRAM_w_s3_b3_1[256][1024][1][1], data_t DRAM_bn_s3_b3_1[4][256], data_t DRAM_w_s3_b3_2[256][256][3][3], data_t DRAM_bn_s3_b3_2[4][256], data_t DRAM_w_s3_b3_3[1024][256][1][1], data_t DRAM_bn_s3_b3_3[4][1024], data_t DRAM_s3_b4[1024][14][14], data_t DRAM_w_s3_b4_1[256][1024][1][1], data_t DRAM_bn_s3_b4_1[4][256], data_t DRAM_w_s3_b4_2[256][256][3][3], data_t DRAM_bn_s3_b4_2[4][256], data_t DRAM_w_s3_b4_3[1024][256][1][1], data_t DRAM_bn_s3_b4_3[4][1024], data_t DRAM_s3_b5[1024][14][14], data_t DRAM_w_s3_b5_1[256][1024][1][1], data_t DRAM_bn_s3_b5_1[4][256], data_t DRAM_w_s3_b5_2[256][256][3][3], data_t DRAM_bn_s3_b5_2[4][256], data_t DRAM_w_s3_b5_3[1024][256][1][1], data_t DRAM_bn_s3_b5_3[4][1024], data_t DRAM_s3_b6[1024][14][14], data_t DRAM_w_s3_b6_1[256][1024][1][1], data_t DRAM_bn_s3_b6_1[4][256], data_t DRAM_w_s3_b6_2[256][256][3][3], data_t DRAM_bn_s3_b6_2[4][256], data_t DRAM_w_s3_b6_3[1024][256][1][1], data_t DRAM_bn_s3_b6_3[4][1024], data_t DRAM_s3_b7[1024][14][14], data_t DRAM_w_s3_b7_1[256][1024][1][1], data_t DRAM_bn_s3_b7_1[4][256], data_t DRAM_w_s3_b7_2[256][256][3][3], data_t DRAM_bn_s3_b7_2[4][256], data_t DRAM_w_s3_b7_3[1024][256][1][1], data_t DRAM_bn_s3_b7_3[4][1024], data_t DRAM_s3_b8[1024][14][14], data_t DRAM_w_s3_b8_1[256][1024][1][1], data_t DRAM_bn_s3_b8_1[4][256], data_t DRAM_w_s3_b8_2[256][256][3][3], data_t DRAM_bn_s3_b8_2[4][256], data_t DRAM_w_s3_b8_3[1024][256][1][1], data_t DRAM_bn_s3_b8_3[4][1024], data_t DRAM_s3_b9[1024][14][14], data_t DRAM_w_s3_b9_1[256][1024][1][1], data_t DRAM_bn_s3_b9_1[4][256], data_t DRAM_w_s3_b9_2[256][256][3][3], data_t DRAM_bn_s3_b9_2[4][256], data_t DRAM_w_s3_b9_3[1024][256][1][1], data_t DRAM_bn_s3_b9_3[4][1024], data_t DRAM_s3_b10[1024][14][14], data_t DRAM_w_s3_b10_1[256][1024][1][1], data_t DRAM_bn_s3_b10_1[4][256], data_t DRAM_w_s3_b10_2[256][256][3][3], data_t DRAM_bn_s3_b10_2[4][256], data_t DRAM_w_s3_b10_3[1024][256][1][1], data_t DRAM_bn_s3_b10_3[4][1024], data_t DRAM_s3_b11[1024][14][14], data_t DRAM_w_s3_b11_1[256][1024][1][1], data_t DRAM_bn_s3_b11_1[4][256], data_t DRAM_w_s3_b11_2[256][256][3][3], data_t DRAM_bn_s3_b11_2[4][256], data_t DRAM_w_s3_b11_3[1024][256][1][1], data_t DRAM_bn_s3_b11_3[4][1024], data_t DRAM_s3_b12[1024][14][14], data_t DRAM_w_s3_b12_1[256][1024][1][1], data_t DRAM_bn_s3_b12_1[4][256], data_t DRAM_w_s3_b12_2[256][256][3][3], data_t DRAM_bn_s3_b12_2[4][256], data_t DRAM_w_s3_b12_3[1024][256][1][1], data_t DRAM_bn_s3_b12_3[4][1024], data_t DRAM_s3_b13[1024][14][14], data_t DRAM_w_s3_b13_1[256][1024][1][1], data_t DRAM_bn_s3_b13_1[4][256], data_t DRAM_w_s3_b13_2[256][256][3][3], data_t DRAM_bn_s3_b13_2[4][256], data_t DRAM_w_s3_b13_3[1024][256][1][1], data_t DRAM_bn_s3_b13_3[4][1024], data_t DRAM_s3_b14[1024][14][14], data_t DRAM_w_s3_b14_1[256][1024][1][1], data_t DRAM_bn_s3_b14_1[4][256], data_t DRAM_w_s3_b14_2[256][256][3][3], data_t DRAM_bn_s3_b14_2[4][256], data_t DRAM_w_s3_b14_3[1024][256][1][1], data_t DRAM_bn_s3_b14_3[4][1024], data_t DRAM_s3_b15[1024][14][14], data_t DRAM_w_s3_b15_1[256][1024][1][1], data_t DRAM_bn_s3_b15_1[4][256], data_t DRAM_w_s3_b15_2[256][256][3][3], data_t DRAM_bn_s3_b15_2[4][256], data_t DRAM_w_s3_b15_3[1024][256][1][1], data_t DRAM_bn_s3_b15_3[4][1024], data_t DRAM_s3_b16[1024][14][14], data_t DRAM_w_s3_b16_1[256][1024][1][1], data_t DRAM_bn_s3_b16_1[4][256], data_t DRAM_w_s3_b16_2[256][256][3][3], data_t DRAM_bn_s3_b16_2[4][256], data_t DRAM_w_s3_b16_3[1024][256][1][1], data_t DRAM_bn_s3_b16_3[4][1024], data_t DRAM_s3_b17[1024][14][14], data_t DRAM_w_s3_b17_1[256][1024][1][1], data_t DRAM_bn_s3_b17_1[4][256], data_t DRAM_w_s3_b17_2[256][256][3][3], data_t DRAM_bn_s3_b17_2[4][256], data_t DRAM_w_s3_b17_3[1024][256][1][1], data_t DRAM_bn_s3_b17_3[4][1024], data_t DRAM_s3_b18[1024][14][14], data_t DRAM_w_s3_b18_1[256][1024][1][1], data_t DRAM_bn_s3_b18_1[4][256], data_t DRAM_w_s3_b18_2[256][256][3][3], data_t DRAM_bn_s3_b18_2[4][256], data_t DRAM_w_s3_b18_3[1024][256][1][1], data_t DRAM_bn_s3_b18_3[4][1024], data_t DRAM_s3_b19[1024][14][14], data_t DRAM_w_s3_b19_1[256][1024][1][1], data_t DRAM_bn_s3_b19_1[4][256], data_t DRAM_w_s3_b19_2[256][256][3][3], data_t DRAM_bn_s3_b19_2[4][256], data_t DRAM_w_s3_b19_3[1024][256][1][1], data_t DRAM_bn_s3_b19_3[4][1024], data_t DRAM_s3_b20[1024][14][14], data_t DRAM_w_s3_b20_1[256][1024][1][1], data_t DRAM_bn_s3_b20_1[4][256], data_t DRAM_w_s3_b20_2[256][256][3][3], data_t DRAM_bn_s3_b20_2[4][256], data_t DRAM_w_s3_b20_3[1024][256][1][1], data_t DRAM_bn_s3_b20_3[4][1024], data_t DRAM_s3_b21[1024][14][14], data_t DRAM_w_s3_b21_1[256][1024][1][1], data_t DRAM_bn_s3_b21_1[4][256], data_t DRAM_w_s3_b21_2[256][256][3][3], data_t DRAM_bn_s3_b21_2[4][256], data_t DRAM_w_s3_b21_3[1024][256][1][1], data_t DRAM_bn_s3_b21_3[4][1024], data_t DRAM_s3_b22[1024][14][14], data_t DRAM_w_s3_b22_1[256][1024][1][1], data_t DRAM_bn_s3_b22_1[4][256], data_t DRAM_w_s3_b22_2[256][256][3][3], data_t DRAM_bn_s3_b22_2[4][256], data_t DRAM_w_s3_b22_3[1024][256][1][1], data_t DRAM_bn_s3_b22_3[4][1024], data_t DRAM_s3_b23[1024][14][14], data_t DRAM_w_s3_b23_1[256][1024][1][1], data_t DRAM_bn_s3_b23_1[4][256], data_t DRAM_w_s3_b23_2[256][256][3][3], data_t DRAM_bn_s3_b23_2[4][256], data_t DRAM_w_s3_b23_3[1024][256][1][1], data_t DRAM_bn_s3_b23_3[4][1024], data_t DRAM_s3_b24[1024][14][14], data_t DRAM_w_s3_b24_1[256][1024][1][1], data_t DRAM_bn_s3_b24_1[4][256], data_t DRAM_w_s3_b24_2[256][256][3][3], data_t DRAM_bn_s3_b24_2[4][256], data_t DRAM_w_s3_b24_3[1024][256][1][1], data_t DRAM_bn_s3_b24_3[4][1024], data_t DRAM_s3_b25[1024][14][14], data_t DRAM_w_s3_b25_1[256][1024][1][1], data_t DRAM_bn_s3_b25_1[4][256], data_t DRAM_w_s3_b25_2[256][256][3][3], data_t DRAM_bn_s3_b25_2[4][256], data_t DRAM_w_s3_b25_3[1024][256][1][1], data_t DRAM_bn_s3_b25_3[4][1024], data_t DRAM_s3_b26[1024][14][14], data_t DRAM_w_s3_b26_1[256][1024][1][1], data_t DRAM_bn_s3_b26_1[4][256], data_t DRAM_w_s3_b26_2[256][256][3][3], data_t DRAM_bn_s3_b26_2[4][256], data_t DRAM_w_s3_b26_3[1024][256][1][1], data_t DRAM_bn_s3_b26_3[4][1024], data_t DRAM_s3_b27[1024][14][14], data_t DRAM_w_s3_b27_1[256][1024][1][1], data_t DRAM_bn_s3_b27_1[4][256], data_t DRAM_w_s3_b27_2[256][256][3][3], data_t DRAM_bn_s3_b27_2[4][256], data_t DRAM_w_s3_b27_3[1024][256][1][1], data_t DRAM_bn_s3_b27_3[4][1024], data_t DRAM_s3_b28[1024][14][14], data_t DRAM_w_s3_b28_1[256][1024][1][1], data_t DRAM_bn_s3_b28_1[4][256], data_t DRAM_w_s3_b28_2[256][256][3][3], data_t DRAM_bn_s3_b28_2[4][256], data_t DRAM_w_s3_b28_3[1024][256][1][1], data_t DRAM_bn_s3_b28_3[4][1024], data_t DRAM_s3_b29[1024][14][14], data_t DRAM_w_s3_b29_1[256][1024][1][1], data_t DRAM_bn_s3_b29_1[4][256], data_t DRAM_w_s3_b29_2[256][256][3][3], data_t DRAM_bn_s3_b29_2[4][256], data_t DRAM_w_s3_b29_3[1024][256][1][1], data_t DRAM_bn_s3_b29_3[4][1024], data_t DRAM_s3_b30[1024][14][14], data_t DRAM_w_s3_b30_1[256][1024][1][1], data_t DRAM_bn_s3_b30_1[4][256], data_t DRAM_w_s3_b30_2[256][256][3][3], data_t DRAM_bn_s3_b30_2[4][256], data_t DRAM_w_s3_b30_3[1024][256][1][1], data_t DRAM_bn_s3_b30_3[4][1024], data_t DRAM_s3_b31[1024][14][14], data_t DRAM_w_s3_b31_1[256][1024][1][1], data_t DRAM_bn_s3_b31_1[4][256], data_t DRAM_w_s3_b31_2[256][256][3][3], data_t DRAM_bn_s3_b31_2[4][256], data_t DRAM_w_s3_b31_3[1024][256][1][1], data_t DRAM_bn_s3_b31_3[4][1024], data_t DRAM_s3_b32[1024][14][14], data_t DRAM_w_s3_b32_1[256][1024][1][1], data_t DRAM_bn_s3_b32_1[4][256], data_t DRAM_w_s3_b32_2[256][256][3][3], data_t DRAM_bn_s3_b32_2[4][256], data_t DRAM_w_s3_b32_3[1024][256][1][1], data_t DRAM_bn_s3_b32_3[4][1024], data_t DRAM_s3_b33[1024][14][14], data_t DRAM_w_s3_b33_1[256][1024][1][1], data_t DRAM_bn_s3_b33_1[4][256], data_t DRAM_w_s3_b33_2[256][256][3][3], data_t DRAM_bn_s3_b33_2[4][256], data_t DRAM_w_s3_b33_3[1024][256][1][1], data_t DRAM_bn_s3_b33_3[4][1024], data_t DRAM_s3_b34[1024][14][14], data_t DRAM_w_s3_b34_1[256][1024][1][1], data_t DRAM_bn_s3_b34_1[4][256], data_t DRAM_w_s3_b34_2[256][256][3][3], data_t DRAM_bn_s3_b34_2[4][256], data_t DRAM_w_s3_b34_3[1024][256][1][1], data_t DRAM_bn_s3_b34_3[4][1024], data_t DRAM_s3_b35[1024][14][14], data_t DRAM_w_s3_b35_1[256][1024][1][1], data_t DRAM_bn_s3_b35_1[4][256], data_t DRAM_w_s3_b35_2[256][256][3][3], data_t DRAM_bn_s3_b35_2[4][256], data_t DRAM_w_s3_b35_3[1024][256][1][1], data_t DRAM_bn_s3_b35_3[4][1024], data_t DRAM_s4_mid1[512][7][7], data_t DRAM_s4_mid2[512][7][7], data_t DRAM_s4_skip[2048][7][7], data_t DRAM_s4_b0[2048][7][7], data_t DRAM_w_s4_b0_1[512][1024][1][1], data_t DRAM_bn_s4_b0_1[4][512], data_t DRAM_w_s4_b0_2[512][512][3][3], data_t DRAM_bn_s4_b0_2[4][512], data_t DRAM_w_s4_b0_3[2048][512][1][1], data_t DRAM_bn_s4_b0_3[4][2048], data_t DRAM_w_s4_b0_down[2048][1024][1][1], data_t DRAM_s4_b1[2048][7][7], data_t DRAM_w_s4_b1_1[512][2048][1][1], data_t DRAM_bn_s4_b1_1[4][512], data_t DRAM_w_s4_b1_2[512][512][3][3], data_t DRAM_bn_s4_b1_2[4][512], data_t DRAM_w_s4_b1_3[2048][512][1][1], data_t DRAM_bn_s4_b1_3[4][2048], data_t DRAM_s4_b2[2048][7][7], data_t DRAM_w_s4_b2_1[512][2048][1][1], data_t DRAM_bn_s4_b2_1[4][512], data_t DRAM_w_s4_b2_2[512][512][3][3], data_t DRAM_bn_s4_b2_2[4][512], data_t DRAM_w_s4_b2_3[2048][512][1][1], data_t DRAM_bn_s4_b2_3[4][2048], data_t DRAM_gap[2048][1][1], data_t DRAM_fc[1000][2048][1][1], data_t DRAM_out[1000][1][1])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem_input
    #pragma HLS interface m_axi port=DRAM_w_stem offset=slave bundle=mem_w_stem
    #pragma HLS interface m_axi port=DRAM_bn_stem offset=slave bundle=mem_bn_stem
    #pragma HLS interface m_axi port=DRAM_stem_feat offset=slave bundle=mem_stem_feat
    #pragma HLS interface m_axi port=DRAM_stem_pool offset=slave bundle=mem_stem_pool
    #pragma HLS interface m_axi port=DRAM_s1_mid1 offset=slave bundle=mem_s1_mid1
    #pragma HLS interface m_axi port=DRAM_s1_mid2 offset=slave bundle=mem_s1_mid2
    #pragma HLS interface m_axi port=DRAM_s1_skip offset=slave bundle=mem_s1_skip
    #pragma HLS interface m_axi port=DRAM_s1_b0 offset=slave bundle=mem_s1_b0
    #pragma HLS interface m_axi port=DRAM_w_s1_b0_1 offset=slave bundle=mem_w_s1_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s1_b0_1 offset=slave bundle=mem_bn_s1_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s1_b0_2 offset=slave bundle=mem_w_s1_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s1_b0_2 offset=slave bundle=mem_bn_s1_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s1_b0_3 offset=slave bundle=mem_w_s1_b0_3
    #pragma HLS interface m_axi port=DRAM_bn_s1_b0_3 offset=slave bundle=mem_bn_s1_b0_3
    #pragma HLS interface m_axi port=DRAM_w_s1_b0_down offset=slave bundle=mem_w_s1_b0_down
    #pragma HLS interface m_axi port=DRAM_s1_b1 offset=slave bundle=mem_s1_b1
    #pragma HLS interface m_axi port=DRAM_w_s1_b1_1 offset=slave bundle=mem_w_s1_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s1_b1_1 offset=slave bundle=mem_bn_s1_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s1_b1_2 offset=slave bundle=mem_w_s1_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s1_b1_2 offset=slave bundle=mem_bn_s1_b1_2
    #pragma HLS interface m_axi port=DRAM_w_s1_b1_3 offset=slave bundle=mem_w_s1_b1_3
    #pragma HLS interface m_axi port=DRAM_bn_s1_b1_3 offset=slave bundle=mem_bn_s1_b1_3
    #pragma HLS interface m_axi port=DRAM_s1_b2 offset=slave bundle=mem_s1_b2
    #pragma HLS interface m_axi port=DRAM_w_s1_b2_1 offset=slave bundle=mem_w_s1_b2_1
    #pragma HLS interface m_axi port=DRAM_bn_s1_b2_1 offset=slave bundle=mem_bn_s1_b2_1
    #pragma HLS interface m_axi port=DRAM_w_s1_b2_2 offset=slave bundle=mem_w_s1_b2_2
    #pragma HLS interface m_axi port=DRAM_bn_s1_b2_2 offset=slave bundle=mem_bn_s1_b2_2
    #pragma HLS interface m_axi port=DRAM_w_s1_b2_3 offset=slave bundle=mem_w_s1_b2_3
    #pragma HLS interface m_axi port=DRAM_bn_s1_b2_3 offset=slave bundle=mem_bn_s1_b2_3
    #pragma HLS interface m_axi port=DRAM_s2_mid1 offset=slave bundle=mem_s2_mid1
    #pragma HLS interface m_axi port=DRAM_s2_mid2 offset=slave bundle=mem_s2_mid2
    #pragma HLS interface m_axi port=DRAM_s2_skip offset=slave bundle=mem_s2_skip
    #pragma HLS interface m_axi port=DRAM_s2_b0 offset=slave bundle=mem_s2_b0
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_1 offset=slave bundle=mem_w_s2_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b0_1 offset=slave bundle=mem_bn_s2_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_2 offset=slave bundle=mem_w_s2_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b0_2 offset=slave bundle=mem_bn_s2_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_3 offset=slave bundle=mem_w_s2_b0_3
    #pragma HLS interface m_axi port=DRAM_bn_s2_b0_3 offset=slave bundle=mem_bn_s2_b0_3
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_down offset=slave bundle=mem_w_s2_b0_down
    #pragma HLS interface m_axi port=DRAM_s2_b1 offset=slave bundle=mem_s2_b1
    #pragma HLS interface m_axi port=DRAM_w_s2_b1_1 offset=slave bundle=mem_w_s2_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b1_1 offset=slave bundle=mem_bn_s2_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b1_2 offset=slave bundle=mem_w_s2_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b1_2 offset=slave bundle=mem_bn_s2_b1_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b1_3 offset=slave bundle=mem_w_s2_b1_3
    #pragma HLS interface m_axi port=DRAM_bn_s2_b1_3 offset=slave bundle=mem_bn_s2_b1_3
    #pragma HLS interface m_axi port=DRAM_s2_b2 offset=slave bundle=mem_s2_b2
    #pragma HLS interface m_axi port=DRAM_w_s2_b2_1 offset=slave bundle=mem_w_s2_b2_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b2_1 offset=slave bundle=mem_bn_s2_b2_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b2_2 offset=slave bundle=mem_w_s2_b2_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b2_2 offset=slave bundle=mem_bn_s2_b2_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b2_3 offset=slave bundle=mem_w_s2_b2_3
    #pragma HLS interface m_axi port=DRAM_bn_s2_b2_3 offset=slave bundle=mem_bn_s2_b2_3
    #pragma HLS interface m_axi port=DRAM_s2_b3 offset=slave bundle=mem_s2_b3
    #pragma HLS interface m_axi port=DRAM_w_s2_b3_1 offset=slave bundle=mem_w_s2_b3_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b3_1 offset=slave bundle=mem_bn_s2_b3_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b3_2 offset=slave bundle=mem_w_s2_b3_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b3_2 offset=slave bundle=mem_bn_s2_b3_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b3_3 offset=slave bundle=mem_w_s2_b3_3
    #pragma HLS interface m_axi port=DRAM_bn_s2_b3_3 offset=slave bundle=mem_bn_s2_b3_3
    #pragma HLS interface m_axi port=DRAM_s2_b4 offset=slave bundle=mem_s2_b4
    #pragma HLS interface m_axi port=DRAM_w_s2_b4_1 offset=slave bundle=mem_w_s2_b4_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b4_1 offset=slave bundle=mem_bn_s2_b4_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b4_2 offset=slave bundle=mem_w_s2_b4_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b4_2 offset=slave bundle=mem_bn_s2_b4_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b4_3 offset=slave bundle=mem_w_s2_b4_3
    #pragma HLS interface m_axi port=DRAM_bn_s2_b4_3 offset=slave bundle=mem_bn_s2_b4_3
    #pragma HLS interface m_axi port=DRAM_s2_b5 offset=slave bundle=mem_s2_b5
    #pragma HLS interface m_axi port=DRAM_w_s2_b5_1 offset=slave bundle=mem_w_s2_b5_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b5_1 offset=slave bundle=mem_bn_s2_b5_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b5_2 offset=slave bundle=mem_w_s2_b5_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b5_2 offset=slave bundle=mem_bn_s2_b5_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b5_3 offset=slave bundle=mem_w_s2_b5_3
    #pragma HLS interface m_axi port=DRAM_bn_s2_b5_3 offset=slave bundle=mem_bn_s2_b5_3
    #pragma HLS interface m_axi port=DRAM_s2_b6 offset=slave bundle=mem_s2_b6
    #pragma HLS interface m_axi port=DRAM_w_s2_b6_1 offset=slave bundle=mem_w_s2_b6_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b6_1 offset=slave bundle=mem_bn_s2_b6_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b6_2 offset=slave bundle=mem_w_s2_b6_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b6_2 offset=slave bundle=mem_bn_s2_b6_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b6_3 offset=slave bundle=mem_w_s2_b6_3
    #pragma HLS interface m_axi port=DRAM_bn_s2_b6_3 offset=slave bundle=mem_bn_s2_b6_3
    #pragma HLS interface m_axi port=DRAM_s2_b7 offset=slave bundle=mem_s2_b7
    #pragma HLS interface m_axi port=DRAM_w_s2_b7_1 offset=slave bundle=mem_w_s2_b7_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b7_1 offset=slave bundle=mem_bn_s2_b7_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b7_2 offset=slave bundle=mem_w_s2_b7_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b7_2 offset=slave bundle=mem_bn_s2_b7_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b7_3 offset=slave bundle=mem_w_s2_b7_3
    #pragma HLS interface m_axi port=DRAM_bn_s2_b7_3 offset=slave bundle=mem_bn_s2_b7_3
    #pragma HLS interface m_axi port=DRAM_s3_mid1 offset=slave bundle=mem_s3_mid1
    #pragma HLS interface m_axi port=DRAM_s3_mid2 offset=slave bundle=mem_s3_mid2
    #pragma HLS interface m_axi port=DRAM_s3_skip offset=slave bundle=mem_s3_skip
    #pragma HLS interface m_axi port=DRAM_s3_b0 offset=slave bundle=mem_s3_b0
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_1 offset=slave bundle=mem_w_s3_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b0_1 offset=slave bundle=mem_bn_s3_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_2 offset=slave bundle=mem_w_s3_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b0_2 offset=slave bundle=mem_bn_s3_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_3 offset=slave bundle=mem_w_s3_b0_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b0_3 offset=slave bundle=mem_bn_s3_b0_3
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_down offset=slave bundle=mem_w_s3_b0_down
    #pragma HLS interface m_axi port=DRAM_s3_b1 offset=slave bundle=mem_s3_b1
    #pragma HLS interface m_axi port=DRAM_w_s3_b1_1 offset=slave bundle=mem_w_s3_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b1_1 offset=slave bundle=mem_bn_s3_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b1_2 offset=slave bundle=mem_w_s3_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b1_2 offset=slave bundle=mem_bn_s3_b1_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b1_3 offset=slave bundle=mem_w_s3_b1_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b1_3 offset=slave bundle=mem_bn_s3_b1_3
    #pragma HLS interface m_axi port=DRAM_s3_b2 offset=slave bundle=mem_s3_b2
    #pragma HLS interface m_axi port=DRAM_w_s3_b2_1 offset=slave bundle=mem_w_s3_b2_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b2_1 offset=slave bundle=mem_bn_s3_b2_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b2_2 offset=slave bundle=mem_w_s3_b2_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b2_2 offset=slave bundle=mem_bn_s3_b2_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b2_3 offset=slave bundle=mem_w_s3_b2_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b2_3 offset=slave bundle=mem_bn_s3_b2_3
    #pragma HLS interface m_axi port=DRAM_s3_b3 offset=slave bundle=mem_s3_b3
    #pragma HLS interface m_axi port=DRAM_w_s3_b3_1 offset=slave bundle=mem_w_s3_b3_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b3_1 offset=slave bundle=mem_bn_s3_b3_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b3_2 offset=slave bundle=mem_w_s3_b3_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b3_2 offset=slave bundle=mem_bn_s3_b3_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b3_3 offset=slave bundle=mem_w_s3_b3_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b3_3 offset=slave bundle=mem_bn_s3_b3_3
    #pragma HLS interface m_axi port=DRAM_s3_b4 offset=slave bundle=mem_s3_b4
    #pragma HLS interface m_axi port=DRAM_w_s3_b4_1 offset=slave bundle=mem_w_s3_b4_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b4_1 offset=slave bundle=mem_bn_s3_b4_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b4_2 offset=slave bundle=mem_w_s3_b4_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b4_2 offset=slave bundle=mem_bn_s3_b4_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b4_3 offset=slave bundle=mem_w_s3_b4_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b4_3 offset=slave bundle=mem_bn_s3_b4_3
    #pragma HLS interface m_axi port=DRAM_s3_b5 offset=slave bundle=mem_s3_b5
    #pragma HLS interface m_axi port=DRAM_w_s3_b5_1 offset=slave bundle=mem_w_s3_b5_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b5_1 offset=slave bundle=mem_bn_s3_b5_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b5_2 offset=slave bundle=mem_w_s3_b5_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b5_2 offset=slave bundle=mem_bn_s3_b5_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b5_3 offset=slave bundle=mem_w_s3_b5_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b5_3 offset=slave bundle=mem_bn_s3_b5_3
    #pragma HLS interface m_axi port=DRAM_s3_b6 offset=slave bundle=mem_s3_b6
    #pragma HLS interface m_axi port=DRAM_w_s3_b6_1 offset=slave bundle=mem_w_s3_b6_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b6_1 offset=slave bundle=mem_bn_s3_b6_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b6_2 offset=slave bundle=mem_w_s3_b6_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b6_2 offset=slave bundle=mem_bn_s3_b6_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b6_3 offset=slave bundle=mem_w_s3_b6_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b6_3 offset=slave bundle=mem_bn_s3_b6_3
    #pragma HLS interface m_axi port=DRAM_s3_b7 offset=slave bundle=mem_s3_b7
    #pragma HLS interface m_axi port=DRAM_w_s3_b7_1 offset=slave bundle=mem_w_s3_b7_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b7_1 offset=slave bundle=mem_bn_s3_b7_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b7_2 offset=slave bundle=mem_w_s3_b7_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b7_2 offset=slave bundle=mem_bn_s3_b7_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b7_3 offset=slave bundle=mem_w_s3_b7_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b7_3 offset=slave bundle=mem_bn_s3_b7_3
    #pragma HLS interface m_axi port=DRAM_s3_b8 offset=slave bundle=mem_s3_b8
    #pragma HLS interface m_axi port=DRAM_w_s3_b8_1 offset=slave bundle=mem_w_s3_b8_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b8_1 offset=slave bundle=mem_bn_s3_b8_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b8_2 offset=slave bundle=mem_w_s3_b8_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b8_2 offset=slave bundle=mem_bn_s3_b8_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b8_3 offset=slave bundle=mem_w_s3_b8_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b8_3 offset=slave bundle=mem_bn_s3_b8_3
    #pragma HLS interface m_axi port=DRAM_s3_b9 offset=slave bundle=mem_s3_b9
    #pragma HLS interface m_axi port=DRAM_w_s3_b9_1 offset=slave bundle=mem_w_s3_b9_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b9_1 offset=slave bundle=mem_bn_s3_b9_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b9_2 offset=slave bundle=mem_w_s3_b9_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b9_2 offset=slave bundle=mem_bn_s3_b9_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b9_3 offset=slave bundle=mem_w_s3_b9_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b9_3 offset=slave bundle=mem_bn_s3_b9_3
    #pragma HLS interface m_axi port=DRAM_s3_b10 offset=slave bundle=mem_s3_b10
    #pragma HLS interface m_axi port=DRAM_w_s3_b10_1 offset=slave bundle=mem_w_s3_b10_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b10_1 offset=slave bundle=mem_bn_s3_b10_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b10_2 offset=slave bundle=mem_w_s3_b10_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b10_2 offset=slave bundle=mem_bn_s3_b10_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b10_3 offset=slave bundle=mem_w_s3_b10_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b10_3 offset=slave bundle=mem_bn_s3_b10_3
    #pragma HLS interface m_axi port=DRAM_s3_b11 offset=slave bundle=mem_s3_b11
    #pragma HLS interface m_axi port=DRAM_w_s3_b11_1 offset=slave bundle=mem_w_s3_b11_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b11_1 offset=slave bundle=mem_bn_s3_b11_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b11_2 offset=slave bundle=mem_w_s3_b11_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b11_2 offset=slave bundle=mem_bn_s3_b11_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b11_3 offset=slave bundle=mem_w_s3_b11_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b11_3 offset=slave bundle=mem_bn_s3_b11_3
    #pragma HLS interface m_axi port=DRAM_s3_b12 offset=slave bundle=mem_s3_b12
    #pragma HLS interface m_axi port=DRAM_w_s3_b12_1 offset=slave bundle=mem_w_s3_b12_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b12_1 offset=slave bundle=mem_bn_s3_b12_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b12_2 offset=slave bundle=mem_w_s3_b12_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b12_2 offset=slave bundle=mem_bn_s3_b12_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b12_3 offset=slave bundle=mem_w_s3_b12_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b12_3 offset=slave bundle=mem_bn_s3_b12_3
    #pragma HLS interface m_axi port=DRAM_s3_b13 offset=slave bundle=mem_s3_b13
    #pragma HLS interface m_axi port=DRAM_w_s3_b13_1 offset=slave bundle=mem_w_s3_b13_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b13_1 offset=slave bundle=mem_bn_s3_b13_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b13_2 offset=slave bundle=mem_w_s3_b13_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b13_2 offset=slave bundle=mem_bn_s3_b13_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b13_3 offset=slave bundle=mem_w_s3_b13_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b13_3 offset=slave bundle=mem_bn_s3_b13_3
    #pragma HLS interface m_axi port=DRAM_s3_b14 offset=slave bundle=mem_s3_b14
    #pragma HLS interface m_axi port=DRAM_w_s3_b14_1 offset=slave bundle=mem_w_s3_b14_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b14_1 offset=slave bundle=mem_bn_s3_b14_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b14_2 offset=slave bundle=mem_w_s3_b14_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b14_2 offset=slave bundle=mem_bn_s3_b14_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b14_3 offset=slave bundle=mem_w_s3_b14_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b14_3 offset=slave bundle=mem_bn_s3_b14_3
    #pragma HLS interface m_axi port=DRAM_s3_b15 offset=slave bundle=mem_s3_b15
    #pragma HLS interface m_axi port=DRAM_w_s3_b15_1 offset=slave bundle=mem_w_s3_b15_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b15_1 offset=slave bundle=mem_bn_s3_b15_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b15_2 offset=slave bundle=mem_w_s3_b15_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b15_2 offset=slave bundle=mem_bn_s3_b15_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b15_3 offset=slave bundle=mem_w_s3_b15_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b15_3 offset=slave bundle=mem_bn_s3_b15_3
    #pragma HLS interface m_axi port=DRAM_s3_b16 offset=slave bundle=mem_s3_b16
    #pragma HLS interface m_axi port=DRAM_w_s3_b16_1 offset=slave bundle=mem_w_s3_b16_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b16_1 offset=slave bundle=mem_bn_s3_b16_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b16_2 offset=slave bundle=mem_w_s3_b16_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b16_2 offset=slave bundle=mem_bn_s3_b16_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b16_3 offset=slave bundle=mem_w_s3_b16_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b16_3 offset=slave bundle=mem_bn_s3_b16_3
    #pragma HLS interface m_axi port=DRAM_s3_b17 offset=slave bundle=mem_s3_b17
    #pragma HLS interface m_axi port=DRAM_w_s3_b17_1 offset=slave bundle=mem_w_s3_b17_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b17_1 offset=slave bundle=mem_bn_s3_b17_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b17_2 offset=slave bundle=mem_w_s3_b17_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b17_2 offset=slave bundle=mem_bn_s3_b17_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b17_3 offset=slave bundle=mem_w_s3_b17_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b17_3 offset=slave bundle=mem_bn_s3_b17_3
    #pragma HLS interface m_axi port=DRAM_s3_b18 offset=slave bundle=mem_s3_b18
    #pragma HLS interface m_axi port=DRAM_w_s3_b18_1 offset=slave bundle=mem_w_s3_b18_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b18_1 offset=slave bundle=mem_bn_s3_b18_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b18_2 offset=slave bundle=mem_w_s3_b18_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b18_2 offset=slave bundle=mem_bn_s3_b18_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b18_3 offset=slave bundle=mem_w_s3_b18_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b18_3 offset=slave bundle=mem_bn_s3_b18_3
    #pragma HLS interface m_axi port=DRAM_s3_b19 offset=slave bundle=mem_s3_b19
    #pragma HLS interface m_axi port=DRAM_w_s3_b19_1 offset=slave bundle=mem_w_s3_b19_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b19_1 offset=slave bundle=mem_bn_s3_b19_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b19_2 offset=slave bundle=mem_w_s3_b19_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b19_2 offset=slave bundle=mem_bn_s3_b19_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b19_3 offset=slave bundle=mem_w_s3_b19_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b19_3 offset=slave bundle=mem_bn_s3_b19_3
    #pragma HLS interface m_axi port=DRAM_s3_b20 offset=slave bundle=mem_s3_b20
    #pragma HLS interface m_axi port=DRAM_w_s3_b20_1 offset=slave bundle=mem_w_s3_b20_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b20_1 offset=slave bundle=mem_bn_s3_b20_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b20_2 offset=slave bundle=mem_w_s3_b20_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b20_2 offset=slave bundle=mem_bn_s3_b20_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b20_3 offset=slave bundle=mem_w_s3_b20_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b20_3 offset=slave bundle=mem_bn_s3_b20_3
    #pragma HLS interface m_axi port=DRAM_s3_b21 offset=slave bundle=mem_s3_b21
    #pragma HLS interface m_axi port=DRAM_w_s3_b21_1 offset=slave bundle=mem_w_s3_b21_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b21_1 offset=slave bundle=mem_bn_s3_b21_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b21_2 offset=slave bundle=mem_w_s3_b21_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b21_2 offset=slave bundle=mem_bn_s3_b21_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b21_3 offset=slave bundle=mem_w_s3_b21_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b21_3 offset=slave bundle=mem_bn_s3_b21_3
    #pragma HLS interface m_axi port=DRAM_s3_b22 offset=slave bundle=mem_s3_b22
    #pragma HLS interface m_axi port=DRAM_w_s3_b22_1 offset=slave bundle=mem_w_s3_b22_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b22_1 offset=slave bundle=mem_bn_s3_b22_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b22_2 offset=slave bundle=mem_w_s3_b22_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b22_2 offset=slave bundle=mem_bn_s3_b22_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b22_3 offset=slave bundle=mem_w_s3_b22_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b22_3 offset=slave bundle=mem_bn_s3_b22_3
    #pragma HLS interface m_axi port=DRAM_s3_b23 offset=slave bundle=mem_s3_b23
    #pragma HLS interface m_axi port=DRAM_w_s3_b23_1 offset=slave bundle=mem_w_s3_b23_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b23_1 offset=slave bundle=mem_bn_s3_b23_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b23_2 offset=slave bundle=mem_w_s3_b23_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b23_2 offset=slave bundle=mem_bn_s3_b23_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b23_3 offset=slave bundle=mem_w_s3_b23_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b23_3 offset=slave bundle=mem_bn_s3_b23_3
    #pragma HLS interface m_axi port=DRAM_s3_b24 offset=slave bundle=mem_s3_b24
    #pragma HLS interface m_axi port=DRAM_w_s3_b24_1 offset=slave bundle=mem_w_s3_b24_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b24_1 offset=slave bundle=mem_bn_s3_b24_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b24_2 offset=slave bundle=mem_w_s3_b24_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b24_2 offset=slave bundle=mem_bn_s3_b24_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b24_3 offset=slave bundle=mem_w_s3_b24_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b24_3 offset=slave bundle=mem_bn_s3_b24_3
    #pragma HLS interface m_axi port=DRAM_s3_b25 offset=slave bundle=mem_s3_b25
    #pragma HLS interface m_axi port=DRAM_w_s3_b25_1 offset=slave bundle=mem_w_s3_b25_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b25_1 offset=slave bundle=mem_bn_s3_b25_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b25_2 offset=slave bundle=mem_w_s3_b25_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b25_2 offset=slave bundle=mem_bn_s3_b25_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b25_3 offset=slave bundle=mem_w_s3_b25_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b25_3 offset=slave bundle=mem_bn_s3_b25_3
    #pragma HLS interface m_axi port=DRAM_s3_b26 offset=slave bundle=mem_s3_b26
    #pragma HLS interface m_axi port=DRAM_w_s3_b26_1 offset=slave bundle=mem_w_s3_b26_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b26_1 offset=slave bundle=mem_bn_s3_b26_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b26_2 offset=slave bundle=mem_w_s3_b26_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b26_2 offset=slave bundle=mem_bn_s3_b26_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b26_3 offset=slave bundle=mem_w_s3_b26_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b26_3 offset=slave bundle=mem_bn_s3_b26_3
    #pragma HLS interface m_axi port=DRAM_s3_b27 offset=slave bundle=mem_s3_b27
    #pragma HLS interface m_axi port=DRAM_w_s3_b27_1 offset=slave bundle=mem_w_s3_b27_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b27_1 offset=slave bundle=mem_bn_s3_b27_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b27_2 offset=slave bundle=mem_w_s3_b27_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b27_2 offset=slave bundle=mem_bn_s3_b27_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b27_3 offset=slave bundle=mem_w_s3_b27_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b27_3 offset=slave bundle=mem_bn_s3_b27_3
    #pragma HLS interface m_axi port=DRAM_s3_b28 offset=slave bundle=mem_s3_b28
    #pragma HLS interface m_axi port=DRAM_w_s3_b28_1 offset=slave bundle=mem_w_s3_b28_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b28_1 offset=slave bundle=mem_bn_s3_b28_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b28_2 offset=slave bundle=mem_w_s3_b28_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b28_2 offset=slave bundle=mem_bn_s3_b28_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b28_3 offset=slave bundle=mem_w_s3_b28_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b28_3 offset=slave bundle=mem_bn_s3_b28_3
    #pragma HLS interface m_axi port=DRAM_s3_b29 offset=slave bundle=mem_s3_b29
    #pragma HLS interface m_axi port=DRAM_w_s3_b29_1 offset=slave bundle=mem_w_s3_b29_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b29_1 offset=slave bundle=mem_bn_s3_b29_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b29_2 offset=slave bundle=mem_w_s3_b29_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b29_2 offset=slave bundle=mem_bn_s3_b29_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b29_3 offset=slave bundle=mem_w_s3_b29_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b29_3 offset=slave bundle=mem_bn_s3_b29_3
    #pragma HLS interface m_axi port=DRAM_s3_b30 offset=slave bundle=mem_s3_b30
    #pragma HLS interface m_axi port=DRAM_w_s3_b30_1 offset=slave bundle=mem_w_s3_b30_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b30_1 offset=slave bundle=mem_bn_s3_b30_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b30_2 offset=slave bundle=mem_w_s3_b30_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b30_2 offset=slave bundle=mem_bn_s3_b30_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b30_3 offset=slave bundle=mem_w_s3_b30_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b30_3 offset=slave bundle=mem_bn_s3_b30_3
    #pragma HLS interface m_axi port=DRAM_s3_b31 offset=slave bundle=mem_s3_b31
    #pragma HLS interface m_axi port=DRAM_w_s3_b31_1 offset=slave bundle=mem_w_s3_b31_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b31_1 offset=slave bundle=mem_bn_s3_b31_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b31_2 offset=slave bundle=mem_w_s3_b31_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b31_2 offset=slave bundle=mem_bn_s3_b31_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b31_3 offset=slave bundle=mem_w_s3_b31_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b31_3 offset=slave bundle=mem_bn_s3_b31_3
    #pragma HLS interface m_axi port=DRAM_s3_b32 offset=slave bundle=mem_s3_b32
    #pragma HLS interface m_axi port=DRAM_w_s3_b32_1 offset=slave bundle=mem_w_s3_b32_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b32_1 offset=slave bundle=mem_bn_s3_b32_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b32_2 offset=slave bundle=mem_w_s3_b32_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b32_2 offset=slave bundle=mem_bn_s3_b32_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b32_3 offset=slave bundle=mem_w_s3_b32_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b32_3 offset=slave bundle=mem_bn_s3_b32_3
    #pragma HLS interface m_axi port=DRAM_s3_b33 offset=slave bundle=mem_s3_b33
    #pragma HLS interface m_axi port=DRAM_w_s3_b33_1 offset=slave bundle=mem_w_s3_b33_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b33_1 offset=slave bundle=mem_bn_s3_b33_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b33_2 offset=slave bundle=mem_w_s3_b33_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b33_2 offset=slave bundle=mem_bn_s3_b33_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b33_3 offset=slave bundle=mem_w_s3_b33_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b33_3 offset=slave bundle=mem_bn_s3_b33_3
    #pragma HLS interface m_axi port=DRAM_s3_b34 offset=slave bundle=mem_s3_b34
    #pragma HLS interface m_axi port=DRAM_w_s3_b34_1 offset=slave bundle=mem_w_s3_b34_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b34_1 offset=slave bundle=mem_bn_s3_b34_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b34_2 offset=slave bundle=mem_w_s3_b34_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b34_2 offset=slave bundle=mem_bn_s3_b34_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b34_3 offset=slave bundle=mem_w_s3_b34_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b34_3 offset=slave bundle=mem_bn_s3_b34_3
    #pragma HLS interface m_axi port=DRAM_s3_b35 offset=slave bundle=mem_s3_b35
    #pragma HLS interface m_axi port=DRAM_w_s3_b35_1 offset=slave bundle=mem_w_s3_b35_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b35_1 offset=slave bundle=mem_bn_s3_b35_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b35_2 offset=slave bundle=mem_w_s3_b35_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b35_2 offset=slave bundle=mem_bn_s3_b35_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b35_3 offset=slave bundle=mem_w_s3_b35_3
    #pragma HLS interface m_axi port=DRAM_bn_s3_b35_3 offset=slave bundle=mem_bn_s3_b35_3
    #pragma HLS interface m_axi port=DRAM_s4_mid1 offset=slave bundle=mem_s4_mid1
    #pragma HLS interface m_axi port=DRAM_s4_mid2 offset=slave bundle=mem_s4_mid2
    #pragma HLS interface m_axi port=DRAM_s4_skip offset=slave bundle=mem_s4_skip
    #pragma HLS interface m_axi port=DRAM_s4_b0 offset=slave bundle=mem_s4_b0
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_1 offset=slave bundle=mem_w_s4_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s4_b0_1 offset=slave bundle=mem_bn_s4_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_2 offset=slave bundle=mem_w_s4_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s4_b0_2 offset=slave bundle=mem_bn_s4_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_3 offset=slave bundle=mem_w_s4_b0_3
    #pragma HLS interface m_axi port=DRAM_bn_s4_b0_3 offset=slave bundle=mem_bn_s4_b0_3
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_down offset=slave bundle=mem_w_s4_b0_down
    #pragma HLS interface m_axi port=DRAM_s4_b1 offset=slave bundle=mem_s4_b1
    #pragma HLS interface m_axi port=DRAM_w_s4_b1_1 offset=slave bundle=mem_w_s4_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s4_b1_1 offset=slave bundle=mem_bn_s4_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s4_b1_2 offset=slave bundle=mem_w_s4_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s4_b1_2 offset=slave bundle=mem_bn_s4_b1_2
    #pragma HLS interface m_axi port=DRAM_w_s4_b1_3 offset=slave bundle=mem_w_s4_b1_3
    #pragma HLS interface m_axi port=DRAM_bn_s4_b1_3 offset=slave bundle=mem_bn_s4_b1_3
    #pragma HLS interface m_axi port=DRAM_s4_b2 offset=slave bundle=mem_s4_b2
    #pragma HLS interface m_axi port=DRAM_w_s4_b2_1 offset=slave bundle=mem_w_s4_b2_1
    #pragma HLS interface m_axi port=DRAM_bn_s4_b2_1 offset=slave bundle=mem_bn_s4_b2_1
    #pragma HLS interface m_axi port=DRAM_w_s4_b2_2 offset=slave bundle=mem_w_s4_b2_2
    #pragma HLS interface m_axi port=DRAM_bn_s4_b2_2 offset=slave bundle=mem_bn_s4_b2_2
    #pragma HLS interface m_axi port=DRAM_w_s4_b2_3 offset=slave bundle=mem_w_s4_b2_3
    #pragma HLS interface m_axi port=DRAM_bn_s4_b2_3 offset=slave bundle=mem_bn_s4_b2_3
    #pragma HLS interface m_axi port=DRAM_gap offset=slave bundle=mem_gap
    #pragma HLS interface m_axi port=DRAM_fc offset=slave bundle=mem_fc
    #pragma HLS interface m_axi port=DRAM_out offset=slave bundle=mem_out

    for (int co_base = 0; co_base < 64; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (64) ? (128) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 112; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (112) ? (14) : ((112) - (oh_base)));
            for (int ow_base = 0; ow_base < 112; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (112) ? (14) : ((112) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 3; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (3) ? (32) : ((3) - (ci_base)));
                    load_fmap_patch_3_224_224_32_33_33_2_3_ap_fixed_16_5_(DRAM_input, BRAM_in_patch_stem, ci_base, oh_base, ow_base);
                    load_weight_tile_64_3_7_128_32_ap_fixed_16_5_(DRAM_w_stem, BRAM_weight_tile_7, co_base, ci_base);
                    conv_tile_32_33_33_128_32_7_128_14_14_2_8_64_ap_fixed_16_5_(BRAM_in_patch_stem, BRAM_weight_tile_7, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_128_ap_fixed_16_5_(DRAM_bn_stem, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_112_112_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_stem_feat, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (64) ? (128) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                load_fmap_patch_64_112_112_128_29_29_2_1_ap_fixed_16_5_(DRAM_stem_feat, BRAM_pool_patch, co_base, oh_base, ow_base);
                maxpool_tile_128_29_29_128_14_14_3_3_2_2_ap_fixed_16_5_(BRAM_pool_patch, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_stem_pool, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (64) ? (128) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (64) ? (32) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_32_14_14_1_0_ap_fixed_16_5_(DRAM_stem_pool, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_64_64_1_128_32_ap_fixed_16_5_(DRAM_w_s1_b0_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_128_ap_fixed_16_5_(DRAM_bn_s1_b0_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (64) ? (128) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (64) ? (32) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_32_16_16_1_1_ap_fixed_16_5_(DRAM_s1_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_64_64_3_128_32_ap_fixed_16_5_(DRAM_w_s1_b0_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_128_ap_fixed_16_5_(DRAM_bn_s1_b0_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (64) ? (32) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_32_14_14_1_0_ap_fixed_16_5_(DRAM_s1_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_64_1_128_32_ap_fixed_16_5_(DRAM_w_s1_b0_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s1_b0_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (64) ? (32) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_32_14_14_1_0_ap_fixed_16_5_(DRAM_stem_pool, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_64_1_128_32_ap_fixed_16_5_(DRAM_w_s1_b0_down, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                store_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_skip, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                load_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(DRAM_s1_b0, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(DRAM_s1_skip, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (64) ? (128) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_56_56_32_14_14_1_0_ap_fixed_16_5_(DRAM_s1_b0, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_64_256_1_128_32_ap_fixed_16_5_(DRAM_w_s1_b1_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_128_ap_fixed_16_5_(DRAM_bn_s1_b1_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (64) ? (128) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (64) ? (32) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_32_16_16_1_1_ap_fixed_16_5_(DRAM_s1_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_64_64_3_128_32_ap_fixed_16_5_(DRAM_w_s1_b1_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_128_ap_fixed_16_5_(DRAM_bn_s1_b1_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (64) ? (32) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_32_14_14_1_0_ap_fixed_16_5_(DRAM_s1_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_64_1_128_32_ap_fixed_16_5_(DRAM_w_s1_b1_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s1_b1_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                load_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(DRAM_s1_b1, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(DRAM_s1_b0, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (64) ? (128) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_56_56_32_14_14_1_0_ap_fixed_16_5_(DRAM_s1_b1, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_64_256_1_128_32_ap_fixed_16_5_(DRAM_w_s1_b2_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_128_ap_fixed_16_5_(DRAM_bn_s1_b2_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (64) ? (128) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (64) ? (32) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_32_16_16_1_1_ap_fixed_16_5_(DRAM_s1_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_64_64_3_128_32_ap_fixed_16_5_(DRAM_w_s1_b2_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_128_ap_fixed_16_5_(DRAM_bn_s1_b2_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (64) ? (32) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_32_14_14_1_0_ap_fixed_16_5_(DRAM_s1_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_64_1_128_32_ap_fixed_16_5_(DRAM_w_s1_b2_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s1_b2_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_b2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                load_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(DRAM_s1_b2, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(DRAM_s1_b1, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_b2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (56) ? (14) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (56) ? (14) : ((56) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_56_56_32_14_14_1_0_ap_fixed_16_5_(DRAM_s1_b2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_128_256_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b0_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b0_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_56_56_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_56_56_32_29_29_2_1_ap_fixed_16_5_(DRAM_s2_mid1, BRAM_in_patch_stride2_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_128_32_ap_fixed_16_5_(DRAM_w_s2_b0_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_29_29_128_32_3_128_14_14_2_8_64_ap_fixed_16_5_(BRAM_in_patch_stride2_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b0_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_128_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b0_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s2_b0_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_56_56_32_27_27_2_0_ap_fixed_16_5_(DRAM_s1_b2, BRAM_in_patch_stride2_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_256_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b0_down, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_27_27_128_32_1_128_14_14_2_8_64_ap_fixed_16_5_(BRAM_in_patch_stride2_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_skip, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b0, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_skip, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_b0, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_128_512_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b1_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b1_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_16_16_1_1_ap_fixed_16_5_(DRAM_s2_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_128_32_ap_fixed_16_5_(DRAM_w_s2_b1_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b1_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_128_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b1_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s2_b1_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b1, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b0, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_b1, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_128_512_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b2_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b2_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_16_16_1_1_ap_fixed_16_5_(DRAM_s2_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_128_32_ap_fixed_16_5_(DRAM_w_s2_b2_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b2_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_128_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b2_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s2_b2_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b2, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b1, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_b2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_128_512_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b3_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b3_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_16_16_1_1_ap_fixed_16_5_(DRAM_s2_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_128_32_ap_fixed_16_5_(DRAM_w_s2_b3_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b3_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_128_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b3_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s2_b3_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b3, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b3, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b2, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b3, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_b3, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_128_512_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b4_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b4_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_16_16_1_1_ap_fixed_16_5_(DRAM_s2_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_128_32_ap_fixed_16_5_(DRAM_w_s2_b4_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b4_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_128_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b4_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s2_b4_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b4, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b4, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b3, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b4, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_b4, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_128_512_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b5_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b5_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_16_16_1_1_ap_fixed_16_5_(DRAM_s2_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_128_32_ap_fixed_16_5_(DRAM_w_s2_b5_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b5_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_128_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b5_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s2_b5_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b5, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b5, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b4, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b5, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_b5, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_128_512_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b6_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b6_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_16_16_1_1_ap_fixed_16_5_(DRAM_s2_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_128_32_ap_fixed_16_5_(DRAM_w_s2_b6_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b6_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_128_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b6_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s2_b6_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b6, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b6, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b5, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b6, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_b6, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_128_512_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b7_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b7_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (128) ? (128) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_16_16_1_1_ap_fixed_16_5_(DRAM_s2_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_128_32_ap_fixed_16_5_(DRAM_w_s2_b7_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_128_ap_fixed_16_5_(DRAM_bn_s2_b7_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (128) ? (32) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_128_1_128_32_ap_fixed_16_5_(DRAM_w_s2_b7_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s2_b7_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b7, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b7, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(DRAM_s2_b6, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b7, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (28) ? (14) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (28) ? (14) : ((28) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_28_28_32_14_14_1_0_ap_fixed_16_5_(DRAM_s2_b7, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_512_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b0_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b0_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_28_28_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_28_28_32_29_29_2_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride2_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b0_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_29_29_128_32_3_128_14_14_2_8_64_ap_fixed_16_5_(BRAM_in_patch_stride2_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b0_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b0_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b0_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_28_28_32_27_27_2_0_ap_fixed_16_5_(DRAM_s2_b7, BRAM_in_patch_stride2_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_512_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b0_down, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_27_27_128_32_1_128_14_14_2_8_64_ap_fixed_16_5_(BRAM_in_patch_stride2_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_skip, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b0, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_skip, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b0, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b1_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b1_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b1_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b1_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b1_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b1_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b1, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b0, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b1, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b2_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b2_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b2_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b2_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b2_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b2_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b2, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b1, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b3_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b3_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b3_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b3_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b3_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b3_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b3, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b3, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b2, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b3, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b3, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b4_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b4_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b4_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b4_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b4_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b4_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b4, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b4, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b3, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b4, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b4, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b5_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b5_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b5_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b5_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b5_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b5_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b5, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b5, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b4, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b5, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b5, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b6_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b6_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b6_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b6_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b6_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b6_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b6, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b6, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b5, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b6, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b6, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b7_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b7_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b7_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b7_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b7_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b7_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b7, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b7, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b6, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b7, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b7, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b8_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b8_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b8_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b8_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b8_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b8_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b8, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b8, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b7, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b8, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b8, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b9_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b9_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b9_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b9_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b9_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b9_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b9, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b9, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b8, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b9, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b9, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b10_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b10_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b10_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b10_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b10_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b10_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b10, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b10, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b9, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b10, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b10, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b11_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b11_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b11_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b11_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b11_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b11_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b11, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b11, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b10, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b11, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b11, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b12_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b12_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b12_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b12_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b12_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b12_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b12, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b12, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b11, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b12, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b12, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b13_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b13_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b13_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b13_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b13_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b13_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b13, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b13, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b12, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b13, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b13, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b14_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b14_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b14_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b14_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b14_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b14_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b14, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b14, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b13, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b14, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b14, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b15_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b15_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b15_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b15_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b15_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b15_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b15, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b15, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b14, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b15, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b15, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b16_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b16_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b16_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b16_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b16_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b16_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b16, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b16, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b15, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b16, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b16, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b17_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b17_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b17_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b17_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b17_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b17_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b17, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b17, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b16, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b17, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b17, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b18_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b18_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b18_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b18_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b18_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b18_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b18, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b18, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b17, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b18, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b18, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b19_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b19_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b19_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b19_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b19_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b19_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b19, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b19, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b18, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b19, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b19, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b20_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b20_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b20_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b20_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b20_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b20_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b20, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b20, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b19, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b20, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b20, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b21_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b21_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b21_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b21_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b21_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b21_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b21, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b21, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b20, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b21, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b21, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b22_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b22_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b22_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b22_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b22_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b22_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b22, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b22, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b21, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b22, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b22, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b23_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b23_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b23_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b23_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b23_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b23_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b23, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b23, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b22, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b23, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b23, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b24_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b24_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b24_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b24_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b24_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b24_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b24, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b24, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b23, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b24, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b24, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b25_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b25_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b25_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b25_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b25_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b25_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b25, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b25, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b24, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b25, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b25, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b26_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b26_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b26_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b26_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b26_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b26_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b26, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b26, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b25, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b26, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b26, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b27_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b27_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b27_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b27_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b27_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b27_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b27, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b27, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b26, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b27, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b27, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b28_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b28_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b28_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b28_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b28_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b28_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b28, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b28, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b27, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b28, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b28, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b29_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b29_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b29_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b29_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b29_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b29_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b29, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b29, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b28, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b29, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b29, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b30_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b30_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b30_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b30_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b30_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b30_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b30, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b30, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b29, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b30, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b30, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b31_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b31_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b31_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b31_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b31_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b31_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b31, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b31, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b30, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b31, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b31, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b32_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b32_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b32_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b32_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b32_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b32_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b32, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b32, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b31, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b32, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b32, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b33_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b33_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b33_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b33_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b33_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b33_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b33, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b33, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b32, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b33, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b33, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b34_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b34_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b34_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b34_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b34_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b34_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b34, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b34, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b33, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b34, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b34, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b35_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b35_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (256) ? (128) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_16_16_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_128_32_ap_fixed_16_5_(DRAM_w_s3_b35_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_128_ap_fixed_16_5_(DRAM_bn_s3_b35_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (256) ? (32) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_1024_256_1_128_32_ap_fixed_16_5_(DRAM_w_s3_b35_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_1024_128_ap_fixed_16_5_(DRAM_bn_s3_b35_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b35, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 1024; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1024) ? (128) : ((1024) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b35, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(DRAM_s3_b34, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_1024_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b35, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (14) ? (14) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (14) ? (14) : ((14) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_14_14_1_0_ap_fixed_16_5_(DRAM_s3_b35, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s4_b0_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s4_b0_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_14_14_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_14_14_32_29_29_2_1_ap_fixed_16_5_(DRAM_s4_mid1, BRAM_in_patch_stride2_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_512_512_3_128_32_ap_fixed_16_5_(DRAM_w_s4_b0_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_29_29_128_32_3_128_14_14_2_8_64_ap_fixed_16_5_(BRAM_in_patch_stride2_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s4_b0_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 2048; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (2048) ? (128) : ((2048) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_7_7_32_14_14_1_0_ap_fixed_16_5_(DRAM_s4_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_2048_512_1_128_32_ap_fixed_16_5_(DRAM_w_s4_b0_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_2048_128_ap_fixed_16_5_(DRAM_bn_s4_b0_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 2048; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (2048) ? (128) : ((2048) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 1024; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (1024) ? (32) : ((1024) - (ci_base)));
                    load_fmap_patch_1024_14_14_32_27_27_2_0_ap_fixed_16_5_(DRAM_s3_b35, BRAM_in_patch_stride2_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_2048_1024_1_128_32_ap_fixed_16_5_(DRAM_w_s4_b0_down, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_27_27_128_32_1_128_14_14_2_8_64_ap_fixed_16_5_(BRAM_in_patch_stride2_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                store_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_skip, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 2048; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (2048) ? (128) : ((2048) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                load_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(DRAM_s4_b0, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(DRAM_s4_skip, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 2048; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (2048) ? (32) : ((2048) - (ci_base)));
                    load_fmap_patch_2048_7_7_32_14_14_1_0_ap_fixed_16_5_(DRAM_s4_b0, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_2048_1_128_32_ap_fixed_16_5_(DRAM_w_s4_b1_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s4_b1_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_7_7_32_16_16_1_1_ap_fixed_16_5_(DRAM_s4_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_512_512_3_128_32_ap_fixed_16_5_(DRAM_w_s4_b1_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s4_b1_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 2048; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (2048) ? (128) : ((2048) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_7_7_32_14_14_1_0_ap_fixed_16_5_(DRAM_s4_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_2048_512_1_128_32_ap_fixed_16_5_(DRAM_w_s4_b1_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_2048_128_ap_fixed_16_5_(DRAM_bn_s4_b1_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 2048; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (2048) ? (128) : ((2048) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                load_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(DRAM_s4_b1, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(DRAM_s4_b0, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 2048; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (2048) ? (32) : ((2048) - (ci_base)));
                    load_fmap_patch_2048_7_7_32_14_14_1_0_ap_fixed_16_5_(DRAM_s4_b1, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_2048_1_128_32_ap_fixed_16_5_(DRAM_w_s4_b2_1, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s4_b2_1, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (512) ? (128) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_7_7_32_16_16_1_1_ap_fixed_16_5_(DRAM_s4_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_512_512_3_128_32_ap_fixed_16_5_(DRAM_w_s4_b2_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_32_16_16_128_32_3_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_128_ap_fixed_16_5_(DRAM_bn_s4_b2_2, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_mid2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 2048; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (2048) ? (128) : ((2048) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                clear_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (512) ? (32) : ((512) - (ci_base)));
                    load_fmap_patch_512_7_7_32_14_14_1_0_ap_fixed_16_5_(DRAM_s4_mid2, BRAM_in_patch_stride1_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_2048_512_1_128_32_ap_fixed_16_5_(DRAM_w_s4_b2_3, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_14_14_128_32_1_128_14_14_1_8_64_ap_fixed_16_5_(BRAM_in_patch_stride1_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_2048_128_ap_fixed_16_5_(DRAM_bn_s4_b2_3, BRAM_bn_tile, co_base);
                batchnorm_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_b2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 2048; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (2048) ? (128) : ((2048) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 14) {
            int valid_oh = (((oh_base) + (14)) <= (7) ? (14) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 14) {
                int valid_ow = (((ow_base) + (14)) <= (7) ? (14) : ((7) - (ow_base)));
                load_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(DRAM_s4_b2, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(DRAM_s4_b1, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_128_14_14_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_2048_7_7_128_14_14_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_b2, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 2048; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (2048) ? (128) : ((2048) - (co_base)));
        clear_tile_128_1_1_ap_fixed_16_5_(BRAM_gap_out);
        for (int gh_base = 0; gh_base < 7; gh_base += 7) {
            int valid_gh = (((gh_base) + (7)) <= (7) ? (7) : ((7) - (gh_base)));
            for (int gw_base = 0; gw_base < 7; gw_base += 7) {
                int valid_gw = (((gw_base) + (7)) <= (7) ? (7) : ((7) - (gw_base)));
                load_fmap_tile_2048_7_7_128_7_7_ap_fixed_16_5_(DRAM_s4_b2, BRAM_gap_tile, co_base, gh_base, gw_base);
                avgpool_accumulate_tile_128_7_7_ap_fixed_16_5_(BRAM_gap_tile, BRAM_gap_out, valid_co, valid_gh, valid_gw);
            }
        }
        avgpool_finalize_tile_128_49_ap_fixed_16_5_(BRAM_gap_out, BRAM_gap_out, valid_co);
        store_fmap_tile_2048_1_1_128_1_1_ap_fixed_16_5_(BRAM_gap_out, DRAM_gap, co_base, 0, 0, valid_co, 1, 1);
    }
    for (int co_base = 0; co_base < 1000; co_base += 128) {
        int valid_co = (((co_base) + (128)) <= (1000) ? (128) : ((1000) - (co_base)));
        for (int oh_base = 0; oh_base < 1; oh_base += 1) {
            int valid_oh = (((oh_base) + (1)) <= (1) ? (1) : ((1) - (oh_base)));
            for (int ow_base = 0; ow_base < 1; ow_base += 1) {
                int valid_ow = (((ow_base) + (1)) <= (1) ? (1) : ((1) - (ow_base)));
                clear_tile_128_1_1_ap_fixed_16_5_(BRAM_fc_out);
                for (int ci_base = 0; ci_base < 2048; ci_base += 32) {
                    int valid_ci = (((ci_base) + (32)) <= (2048) ? (32) : ((2048) - (ci_base)));
                    load_fmap_patch_2048_1_1_32_1_1_1_0_ap_fixed_16_5_(DRAM_gap, BRAM_in_patch_1, ci_base, oh_base, ow_base);
                    load_weight_tile_1000_2048_1_128_32_ap_fixed_16_5_(DRAM_fc, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_32_1_1_128_32_1_128_1_1_1_8_64_ap_fixed_16_5_(BRAM_in_patch_1, BRAM_weight_tile_1, BRAM_fc_out, valid_co, valid_ci, valid_oh, valid_ow);
                }
                store_fmap_tile_1000_1_1_128_1_1_ap_fixed_16_5_(BRAM_fc_out, DRAM_out, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
}