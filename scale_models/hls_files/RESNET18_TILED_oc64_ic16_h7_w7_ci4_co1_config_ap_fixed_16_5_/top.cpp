
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

data_t BRAM_in_patch_stem[16][19][19];
data_t BRAM_in_patch_stride2_k3[16][15][15];
data_t BRAM_in_patch_stride2_k1[16][13][13];
data_t BRAM_in_patch_stride1_k3[16][9][9];
data_t BRAM_in_patch_1[16][1][1];
data_t BRAM_pool_patch[64][15][15];
data_t BRAM_out_tile[64][7][7];
data_t BRAM_skip_tile[64][7][7];
data_t BRAM_weight_tile_7[64][16][7][7];
data_t BRAM_weight_tile_3[64][16][3][3];
data_t BRAM_weight_tile_1[64][16][1][1];
data_t BRAM_bn_tile[4][64];
data_t BRAM_gap_tile[64][7][7];
data_t BRAM_gap_out[64][1][1];
data_t BRAM_fc_out[64][1][1];

void clear_tile_64_7_7_ap_fixed_16_5_(data_t tile[64][7][7])
{
    for (int c = 0; c < 64; ++c) {
        for (int h = 0; h < 7; ++h) {
            for (int w = 0; w < 7; ++w) {
                tile[c][h][w] = (data_t)0;
            }
        }
    }
}

void load_fmap_patch_3_224_224_16_19_19_2_3_ap_fixed_16_5_(
    data_t input[3][224][224],
    data_t output[16][19][19],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 19; ++h) {
            for (int w = 0; w < 19; ++w) {
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

void load_weight_tile_64_3_7_64_16_ap_fixed_16_5_(
    data_t input[64][3][7][7],
    data_t output[64][16][7][7],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
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

void conv_tile_16_19_19_64_16_7_64_7_7_2_4_1_ap_fixed_16_5_(
    data_t input[16][19][19],
    data_t weight[64][16][7][7],
    data_t output[64][7][7],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=1 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=4 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=1 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 7; ++kh) {
                for (int kw = 0; kw < 7; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=4
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=1
                            output[co][oh][ow] += input[ci][oh * 2 + kh][ow * 2 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void load_bn_tile_64_64_ap_fixed_16_5_(
    data_t input[4][64],
    data_t output[4][64],
    int co_base
)
{
    for (int stat = 0; stat < 4; ++stat) {
        for (int co = 0; co < 64; ++co) {
            int src_co = co_base + co;
            if (src_co < 64) {
                output[stat][co] = input[stat][src_co];
            } else {
                output[stat][co] = (data_t)0;
            }
        }
    }
}

void batchnorm_tile_64_7_7_ap_fixed_16_5_(
    data_t input[64][7][7],
    data_t weights[4][64],
    data_t output[64][7][7],
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

void relu_tile_64_7_7_ap_fixed_16_5_(
    data_t input[64][7][7],
    data_t output[64][7][7],
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

void store_fmap_tile_64_112_112_64_7_7_ap_fixed_16_5_(
    data_t input[64][7][7],
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

void load_fmap_patch_64_112_112_64_15_15_2_1_ap_fixed_16_5_(
    data_t input[64][112][112],
    data_t output[64][15][15],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 64; ++c) {
        for (int h = 0; h < 15; ++h) {
            for (int w = 0; w < 15; ++w) {
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

void maxpool_tile_64_15_15_64_7_7_3_3_2_2_ap_fixed_16_5_(
    data_t input[64][15][15],
    data_t output[64][7][7],
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

void store_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(
    data_t input[64][7][7],
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

void load_fmap_patch_64_56_56_16_9_9_1_1_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t output[16][9][9],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 9; ++h) {
            for (int w = 0; w < 9; ++w) {
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

void load_weight_tile_64_64_3_64_16_ap_fixed_16_5_(
    data_t input[64][64][3][3],
    data_t output[64][16][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
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

void conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(
    data_t input[16][9][9],
    data_t weight[64][16][3][3],
    data_t output[64][7][7],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=1 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=4 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=1 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=4
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=1
                            output[co][oh][ow] += input[ci][oh * 1 + kh][ow * 1 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void load_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t output[64][7][7],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 64; ++c) {
        for (int h = 0; h < 7; ++h) {
            for (int w = 0; w < 7; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base + h;
                int src_w = ow_base + w;
                if (src_c < 64 && src_h >= 0 && src_h < 56 && src_w >= 0 && src_w < 56) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void matrix_add_tile_64_7_7_ap_fixed_16_5_(
    data_t lhs[64][7][7],
    data_t rhs[64][7][7],
    data_t output[64][7][7],
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

void load_fmap_patch_64_56_56_16_15_15_2_1_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t output[16][15][15],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 15; ++h) {
            for (int w = 0; w < 15; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 1 + h;
                int src_w = ow_base * 2 - 1 + w;
                if (src_c < 64 && src_h >= 0 && src_h < 56 && src_w >= 0 && src_w < 56) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_128_64_3_64_16_ap_fixed_16_5_(
    data_t input[128][64][3][3],
    data_t output[64][16][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 128 && src_ci < 64) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void conv_tile_16_15_15_64_16_3_64_7_7_2_4_1_ap_fixed_16_5_(
    data_t input[16][15][15],
    data_t weight[64][16][3][3],
    data_t output[64][7][7],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=1 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=4 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=1 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=4
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=1
                            output[co][oh][ow] += input[ci][oh * 2 + kh][ow * 2 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void load_bn_tile_128_64_ap_fixed_16_5_(
    data_t input[4][128],
    data_t output[4][64],
    int co_base
)
{
    for (int stat = 0; stat < 4; ++stat) {
        for (int co = 0; co < 64; ++co) {
            int src_co = co_base + co;
            if (src_co < 128) {
                output[stat][co] = input[stat][src_co];
            } else {
                output[stat][co] = (data_t)0;
            }
        }
    }
}

void store_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(
    data_t input[64][7][7],
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

void load_fmap_patch_128_28_28_16_9_9_1_1_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t output[16][9][9],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 9; ++h) {
            for (int w = 0; w < 9; ++w) {
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

void load_weight_tile_128_128_3_64_16_ap_fixed_16_5_(
    data_t input[128][128][3][3],
    data_t output[64][16][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
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

void load_fmap_patch_64_56_56_16_13_13_2_0_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t output[16][13][13],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 0 + h;
                int src_w = ow_base * 2 - 0 + w;
                if (src_c < 64 && src_h >= 0 && src_h < 56 && src_w >= 0 && src_w < 56) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_128_64_1_64_16_ap_fixed_16_5_(
    data_t input[128][64][1][1],
    data_t output[64][16][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 128 && src_ci < 64) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void conv_tile_16_13_13_64_16_1_64_7_7_2_4_1_ap_fixed_16_5_(
    data_t input[16][13][13],
    data_t weight[64][16][1][1],
    data_t output[64][7][7],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=1 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=4 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=1 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=4
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=1
                            output[co][oh][ow] += input[ci][oh * 2 + kh][ow * 2 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void load_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t output[64][7][7],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 64; ++c) {
        for (int h = 0; h < 7; ++h) {
            for (int w = 0; w < 7; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base + h;
                int src_w = ow_base + w;
                if (src_c < 128 && src_h >= 0 && src_h < 28 && src_w >= 0 && src_w < 28) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_fmap_patch_128_28_28_16_15_15_2_1_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t output[16][15][15],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 15; ++h) {
            for (int w = 0; w < 15; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 1 + h;
                int src_w = ow_base * 2 - 1 + w;
                if (src_c < 128 && src_h >= 0 && src_h < 28 && src_w >= 0 && src_w < 28) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_256_128_3_64_16_ap_fixed_16_5_(
    data_t input[256][128][3][3],
    data_t output[64][16][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 256 && src_ci < 128) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_bn_tile_256_64_ap_fixed_16_5_(
    data_t input[4][256],
    data_t output[4][64],
    int co_base
)
{
    for (int stat = 0; stat < 4; ++stat) {
        for (int co = 0; co < 64; ++co) {
            int src_co = co_base + co;
            if (src_co < 256) {
                output[stat][co] = input[stat][src_co];
            } else {
                output[stat][co] = (data_t)0;
            }
        }
    }
}

void store_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(
    data_t input[64][7][7],
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

void load_fmap_patch_256_14_14_16_9_9_1_1_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t output[16][9][9],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 9; ++h) {
            for (int w = 0; w < 9; ++w) {
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

void load_weight_tile_256_256_3_64_16_ap_fixed_16_5_(
    data_t input[256][256][3][3],
    data_t output[64][16][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
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

void load_fmap_patch_128_28_28_16_13_13_2_0_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t output[16][13][13],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 0 + h;
                int src_w = ow_base * 2 - 0 + w;
                if (src_c < 128 && src_h >= 0 && src_h < 28 && src_w >= 0 && src_w < 28) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_256_128_1_64_16_ap_fixed_16_5_(
    data_t input[256][128][1][1],
    data_t output[64][16][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 256 && src_ci < 128) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void load_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t output[64][7][7],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 64; ++c) {
        for (int h = 0; h < 7; ++h) {
            for (int w = 0; w < 7; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base + h;
                int src_w = ow_base + w;
                if (src_c < 256 && src_h >= 0 && src_h < 14 && src_w >= 0 && src_w < 14) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_fmap_patch_256_14_14_16_15_15_2_1_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t output[16][15][15],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 15; ++h) {
            for (int w = 0; w < 15; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 1 + h;
                int src_w = ow_base * 2 - 1 + w;
                if (src_c < 256 && src_h >= 0 && src_h < 14 && src_w >= 0 && src_w < 14) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_512_256_3_64_16_ap_fixed_16_5_(
    data_t input[512][256][3][3],
    data_t output[64][16][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
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

void load_bn_tile_512_64_ap_fixed_16_5_(
    data_t input[4][512],
    data_t output[4][64],
    int co_base
)
{
    for (int stat = 0; stat < 4; ++stat) {
        for (int co = 0; co < 64; ++co) {
            int src_co = co_base + co;
            if (src_co < 512) {
                output[stat][co] = input[stat][src_co];
            } else {
                output[stat][co] = (data_t)0;
            }
        }
    }
}

void store_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(
    data_t input[64][7][7],
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

void load_fmap_patch_512_7_7_16_9_9_1_1_ap_fixed_16_5_(
    data_t input[512][7][7],
    data_t output[16][9][9],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 9; ++h) {
            for (int w = 0; w < 9; ++w) {
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

void load_weight_tile_512_512_3_64_16_ap_fixed_16_5_(
    data_t input[512][512][3][3],
    data_t output[64][16][3][3],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
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

void load_fmap_patch_256_14_14_16_13_13_2_0_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t output[16][13][13],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 13; ++h) {
            for (int w = 0; w < 13; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 2 - 0 + h;
                int src_w = ow_base * 2 - 0 + w;
                if (src_c < 256 && src_h >= 0 && src_h < 14 && src_w >= 0 && src_w < 14) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_512_256_1_64_16_ap_fixed_16_5_(
    data_t input[512][256][1][1],
    data_t output[64][16][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
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

void load_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(
    data_t input[512][7][7],
    data_t output[64][7][7],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 64; ++c) {
        for (int h = 0; h < 7; ++h) {
            for (int w = 0; w < 7; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base + h;
                int src_w = ow_base + w;
                if (src_c < 512 && src_h >= 0 && src_h < 7 && src_w >= 0 && src_w < 7) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void clear_tile_64_1_1_ap_fixed_16_5_(data_t tile[64][1][1])
{
    for (int c = 0; c < 64; ++c) {
        for (int h = 0; h < 1; ++h) {
            for (int w = 0; w < 1; ++w) {
                tile[c][h][w] = (data_t)0;
            }
        }
    }
}

void avgpool_accumulate_tile_64_7_7_ap_fixed_16_5_(
    data_t input[64][7][7],
    data_t output[64][1][1],
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

void avgpool_finalize_tile_64_49_ap_fixed_16_5_(
    data_t input[64][1][1],
    data_t output[64][1][1],
    int valid_c
)
{
    for (int c = 0; c < valid_c; ++c) {
        output[c][0][0] = input[c][0][0] / (data_t)49;
    }
}

void store_fmap_tile_512_1_1_64_1_1_ap_fixed_16_5_(
    data_t input[64][1][1],
    data_t output[512][1][1],
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

void load_fmap_patch_512_1_1_16_1_1_1_0_ap_fixed_16_5_(
    data_t input[512][1][1],
    data_t output[16][1][1],
    int c_base,
    int oh_base,
    int ow_base
)
{
    for (int c = 0; c < 16; ++c) {
        for (int h = 0; h < 1; ++h) {
            for (int w = 0; w < 1; ++w) {
                int src_c = c_base + c;
                int src_h = oh_base * 1 - 0 + h;
                int src_w = ow_base * 1 - 0 + w;
                if (src_c < 512 && src_h >= 0 && src_h < 1 && src_w >= 0 && src_w < 1) {
                    output[c][h][w] = input[src_c][src_h][src_w];
                } else {
                    output[c][h][w] = (data_t)0;
                }
            }
        }
    }
}

void load_weight_tile_1000_512_1_64_16_ap_fixed_16_5_(
    data_t input[1000][512][1][1],
    data_t output[64][16][1][1],
    int co_base,
    int ci_base
)
{
    for (int co = 0; co < 64; ++co) {
        for (int ci = 0; ci < 16; ++ci) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    int src_co = co_base + co;
                    int src_ci = ci_base + ci;
                    if (src_co < 1000 && src_ci < 512) {
                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];
                    } else {
                        output[co][ci][kh][kw] = (data_t)0;
                    }
                }
            }
        }
    }
}

void conv_tile_16_1_1_64_16_1_64_1_1_1_4_1_ap_fixed_16_5_(
    data_t input[16][1][1],
    data_t weight[64][16][1][1],
    data_t output[64][1][1],
    int valid_oc,
    int valid_ic,
    int valid_h,
    int valid_w
)
{
    #pragma HLS array_partition variable=input type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=1 dim=1
    #pragma HLS array_partition variable=weight type=cyclic factor=4 dim=2
    #pragma HLS array_partition variable=output type=cyclic factor=1 dim=1
    for (int oh = 0; oh < valid_h; ++oh) {
        for (int ow = 0; ow < valid_w; ++ow) {
            for (int kh = 0; kh < 1; ++kh) {
                for (int kw = 0; kw < 1; ++kw) {
                    for (int ci = 0; ci < valid_ic; ++ci) {
                        #pragma HLS unroll factor=4
                        for (int co = 0; co < valid_oc; ++co) {
                            #pragma HLS unroll factor=1
                            output[co][oh][ow] += input[ci][oh * 1 + kh][ow * 1 + kw] * weight[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void store_fmap_tile_1000_1_1_64_1_1_ap_fixed_16_5_(
    data_t input[64][1][1],
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

void top(data_t DRAM_input[3][224][224], data_t DRAM_w_stem[64][3][7][7], data_t DRAM_bn_stem[4][64], data_t DRAM_stem_feat[64][112][112], data_t DRAM_stem_pool[64][56][56], data_t DRAM_s1_mid1[64][56][56], data_t DRAM_s1_b0[64][56][56], data_t DRAM_w_s1_b0_1[64][64][3][3], data_t DRAM_bn_s1_b0_1[4][64], data_t DRAM_w_s1_b0_2[64][64][3][3], data_t DRAM_bn_s1_b0_2[4][64], data_t DRAM_s1_b1[64][56][56], data_t DRAM_w_s1_b1_1[64][64][3][3], data_t DRAM_bn_s1_b1_1[4][64], data_t DRAM_w_s1_b1_2[64][64][3][3], data_t DRAM_bn_s1_b1_2[4][64], data_t DRAM_s2_mid1[128][28][28], data_t DRAM_s2_skip[128][28][28], data_t DRAM_s2_b0[128][28][28], data_t DRAM_w_s2_b0_1[128][64][3][3], data_t DRAM_bn_s2_b0_1[4][128], data_t DRAM_w_s2_b0_2[128][128][3][3], data_t DRAM_bn_s2_b0_2[4][128], data_t DRAM_w_s2_b0_down[128][64][1][1], data_t DRAM_s2_b1[128][28][28], data_t DRAM_w_s2_b1_1[128][128][3][3], data_t DRAM_bn_s2_b1_1[4][128], data_t DRAM_w_s2_b1_2[128][128][3][3], data_t DRAM_bn_s2_b1_2[4][128], data_t DRAM_s3_mid1[256][14][14], data_t DRAM_s3_skip[256][14][14], data_t DRAM_s3_b0[256][14][14], data_t DRAM_w_s3_b0_1[256][128][3][3], data_t DRAM_bn_s3_b0_1[4][256], data_t DRAM_w_s3_b0_2[256][256][3][3], data_t DRAM_bn_s3_b0_2[4][256], data_t DRAM_w_s3_b0_down[256][128][1][1], data_t DRAM_s3_b1[256][14][14], data_t DRAM_w_s3_b1_1[256][256][3][3], data_t DRAM_bn_s3_b1_1[4][256], data_t DRAM_w_s3_b1_2[256][256][3][3], data_t DRAM_bn_s3_b1_2[4][256], data_t DRAM_s4_mid1[512][7][7], data_t DRAM_s4_skip[512][7][7], data_t DRAM_s4_b0[512][7][7], data_t DRAM_w_s4_b0_1[512][256][3][3], data_t DRAM_bn_s4_b0_1[4][512], data_t DRAM_w_s4_b0_2[512][512][3][3], data_t DRAM_bn_s4_b0_2[4][512], data_t DRAM_w_s4_b0_down[512][256][1][1], data_t DRAM_s4_b1[512][7][7], data_t DRAM_w_s4_b1_1[512][512][3][3], data_t DRAM_bn_s4_b1_1[4][512], data_t DRAM_w_s4_b1_2[512][512][3][3], data_t DRAM_bn_s4_b1_2[4][512], data_t DRAM_gap[512][1][1], data_t DRAM_fc[1000][512][1][1], data_t DRAM_out[1000][1][1])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem_input
    #pragma HLS interface m_axi port=DRAM_w_stem offset=slave bundle=mem_w_stem
    #pragma HLS interface m_axi port=DRAM_bn_stem offset=slave bundle=mem_bn_stem
    #pragma HLS interface m_axi port=DRAM_stem_feat offset=slave bundle=mem_stem_feat
    #pragma HLS interface m_axi port=DRAM_stem_pool offset=slave bundle=mem_stem_pool
    #pragma HLS interface m_axi port=DRAM_s1_mid1 offset=slave bundle=mem_s1_mid1
    #pragma HLS interface m_axi port=DRAM_s1_b0 offset=slave bundle=mem_s1_b0
    #pragma HLS interface m_axi port=DRAM_w_s1_b0_1 offset=slave bundle=mem_w_s1_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s1_b0_1 offset=slave bundle=mem_bn_s1_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s1_b0_2 offset=slave bundle=mem_w_s1_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s1_b0_2 offset=slave bundle=mem_bn_s1_b0_2
    #pragma HLS interface m_axi port=DRAM_s1_b1 offset=slave bundle=mem_s1_b1
    #pragma HLS interface m_axi port=DRAM_w_s1_b1_1 offset=slave bundle=mem_w_s1_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s1_b1_1 offset=slave bundle=mem_bn_s1_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s1_b1_2 offset=slave bundle=mem_w_s1_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s1_b1_2 offset=slave bundle=mem_bn_s1_b1_2
    #pragma HLS interface m_axi port=DRAM_s2_mid1 offset=slave bundle=mem_s2_mid1
    #pragma HLS interface m_axi port=DRAM_s2_skip offset=slave bundle=mem_s2_skip
    #pragma HLS interface m_axi port=DRAM_s2_b0 offset=slave bundle=mem_s2_b0
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_1 offset=slave bundle=mem_w_s2_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b0_1 offset=slave bundle=mem_bn_s2_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_2 offset=slave bundle=mem_w_s2_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b0_2 offset=slave bundle=mem_bn_s2_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_down offset=slave bundle=mem_w_s2_b0_down
    #pragma HLS interface m_axi port=DRAM_s2_b1 offset=slave bundle=mem_s2_b1
    #pragma HLS interface m_axi port=DRAM_w_s2_b1_1 offset=slave bundle=mem_w_s2_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b1_1 offset=slave bundle=mem_bn_s2_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b1_2 offset=slave bundle=mem_w_s2_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b1_2 offset=slave bundle=mem_bn_s2_b1_2
    #pragma HLS interface m_axi port=DRAM_s3_mid1 offset=slave bundle=mem_s3_mid1
    #pragma HLS interface m_axi port=DRAM_s3_skip offset=slave bundle=mem_s3_skip
    #pragma HLS interface m_axi port=DRAM_s3_b0 offset=slave bundle=mem_s3_b0
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_1 offset=slave bundle=mem_w_s3_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b0_1 offset=slave bundle=mem_bn_s3_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_2 offset=slave bundle=mem_w_s3_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b0_2 offset=slave bundle=mem_bn_s3_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_down offset=slave bundle=mem_w_s3_b0_down
    #pragma HLS interface m_axi port=DRAM_s3_b1 offset=slave bundle=mem_s3_b1
    #pragma HLS interface m_axi port=DRAM_w_s3_b1_1 offset=slave bundle=mem_w_s3_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b1_1 offset=slave bundle=mem_bn_s3_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b1_2 offset=slave bundle=mem_w_s3_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b1_2 offset=slave bundle=mem_bn_s3_b1_2
    #pragma HLS interface m_axi port=DRAM_s4_mid1 offset=slave bundle=mem_s4_mid1
    #pragma HLS interface m_axi port=DRAM_s4_skip offset=slave bundle=mem_s4_skip
    #pragma HLS interface m_axi port=DRAM_s4_b0 offset=slave bundle=mem_s4_b0
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_1 offset=slave bundle=mem_w_s4_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s4_b0_1 offset=slave bundle=mem_bn_s4_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_2 offset=slave bundle=mem_w_s4_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s4_b0_2 offset=slave bundle=mem_bn_s4_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_down offset=slave bundle=mem_w_s4_b0_down
    #pragma HLS interface m_axi port=DRAM_s4_b1 offset=slave bundle=mem_s4_b1
    #pragma HLS interface m_axi port=DRAM_w_s4_b1_1 offset=slave bundle=mem_w_s4_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s4_b1_1 offset=slave bundle=mem_bn_s4_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s4_b1_2 offset=slave bundle=mem_w_s4_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s4_b1_2 offset=slave bundle=mem_bn_s4_b1_2
    #pragma HLS interface m_axi port=DRAM_gap offset=slave bundle=mem_gap
    #pragma HLS interface m_axi port=DRAM_fc offset=slave bundle=mem_fc
    #pragma HLS interface m_axi port=DRAM_out offset=slave bundle=mem_out

    for (int co_base = 0; co_base < 64; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (64) ? (64) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 112; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (112) ? (7) : ((112) - (oh_base)));
            for (int ow_base = 0; ow_base < 112; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (112) ? (7) : ((112) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 3; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (3) ? (16) : ((3) - (ci_base)));
                    load_fmap_patch_3_224_224_16_19_19_2_3_ap_fixed_16_5_(DRAM_input, BRAM_in_patch_stem, ci_base, oh_base, ow_base);
                    load_weight_tile_64_3_7_64_16_ap_fixed_16_5_(DRAM_w_stem, BRAM_weight_tile_7, co_base, ci_base);
                    conv_tile_16_19_19_64_16_7_64_7_7_2_4_1_ap_fixed_16_5_(BRAM_in_patch_stem, BRAM_weight_tile_7, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_64_ap_fixed_16_5_(DRAM_bn_stem, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_112_112_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_stem_feat, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (64) ? (64) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (56) ? (7) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (56) ? (7) : ((56) - (ow_base)));
                load_fmap_patch_64_112_112_64_15_15_2_1_ap_fixed_16_5_(DRAM_stem_feat, BRAM_pool_patch, co_base, oh_base, ow_base);
                maxpool_tile_64_15_15_64_7_7_3_3_2_2_ap_fixed_16_5_(BRAM_pool_patch, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_stem_pool, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (64) ? (64) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (56) ? (7) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (56) ? (7) : ((56) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (64) ? (16) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_16_9_9_1_1_ap_fixed_16_5_(DRAM_stem_pool, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_64_64_3_64_16_ap_fixed_16_5_(DRAM_w_s1_b0_1, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_64_ap_fixed_16_5_(DRAM_bn_s1_b0_1, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (64) ? (64) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (56) ? (7) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (56) ? (7) : ((56) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (64) ? (16) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_16_9_9_1_1_ap_fixed_16_5_(DRAM_s1_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_64_64_3_64_16_ap_fixed_16_5_(DRAM_w_s1_b0_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_64_ap_fixed_16_5_(DRAM_bn_s1_b0_2, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (64) ? (64) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (56) ? (7) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (56) ? (7) : ((56) - (ow_base)));
                load_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(DRAM_s1_b0, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(DRAM_stem_pool, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (64) ? (64) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (56) ? (7) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (56) ? (7) : ((56) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (64) ? (16) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_16_9_9_1_1_ap_fixed_16_5_(DRAM_s1_b0, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_64_64_3_64_16_ap_fixed_16_5_(DRAM_w_s1_b1_1, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_64_ap_fixed_16_5_(DRAM_bn_s1_b1_1, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (64) ? (64) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (56) ? (7) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (56) ? (7) : ((56) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (64) ? (16) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_16_9_9_1_1_ap_fixed_16_5_(DRAM_s1_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_64_64_3_64_16_ap_fixed_16_5_(DRAM_w_s1_b1_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_64_64_ap_fixed_16_5_(DRAM_bn_s1_b1_2, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 64; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (64) ? (64) : ((64) - (co_base)));
        for (int oh_base = 0; oh_base < 56; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (56) ? (7) : ((56) - (oh_base)));
            for (int ow_base = 0; ow_base < 56; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (56) ? (7) : ((56) - (ow_base)));
                load_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(DRAM_s1_b1, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(DRAM_s1_b0, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_64_56_56_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s1_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (128) ? (64) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (28) ? (7) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (28) ? (7) : ((28) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (64) ? (16) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_16_15_15_2_1_ap_fixed_16_5_(DRAM_s1_b1, BRAM_in_patch_stride2_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_64_3_64_16_ap_fixed_16_5_(DRAM_w_s2_b0_1, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_15_15_64_16_3_64_7_7_2_4_1_ap_fixed_16_5_(BRAM_in_patch_stride2_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_64_ap_fixed_16_5_(DRAM_bn_s2_b0_1, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (128) ? (64) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (28) ? (7) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (28) ? (7) : ((28) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (128) ? (16) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_16_9_9_1_1_ap_fixed_16_5_(DRAM_s2_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_64_16_ap_fixed_16_5_(DRAM_w_s2_b0_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_64_ap_fixed_16_5_(DRAM_bn_s2_b0_2, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (128) ? (64) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (28) ? (7) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (28) ? (7) : ((28) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 64; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (64) ? (16) : ((64) - (ci_base)));
                    load_fmap_patch_64_56_56_16_13_13_2_0_ap_fixed_16_5_(DRAM_s1_b1, BRAM_in_patch_stride2_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_128_64_1_64_16_ap_fixed_16_5_(DRAM_w_s2_b0_down, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_16_13_13_64_16_1_64_7_7_2_4_1_ap_fixed_16_5_(BRAM_in_patch_stride2_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                store_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_skip, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (128) ? (64) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (28) ? (7) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (28) ? (7) : ((28) - (ow_base)));
                load_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(DRAM_s2_b0, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(DRAM_s2_skip, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (128) ? (64) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (28) ? (7) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (28) ? (7) : ((28) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (128) ? (16) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_16_9_9_1_1_ap_fixed_16_5_(DRAM_s2_b0, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_64_16_ap_fixed_16_5_(DRAM_w_s2_b1_1, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_64_ap_fixed_16_5_(DRAM_bn_s2_b1_1, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (128) ? (64) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (28) ? (7) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (28) ? (7) : ((28) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (128) ? (16) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_16_9_9_1_1_ap_fixed_16_5_(DRAM_s2_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_128_128_3_64_16_ap_fixed_16_5_(DRAM_w_s2_b1_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_128_64_ap_fixed_16_5_(DRAM_bn_s2_b1_2, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 128; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (128) ? (64) : ((128) - (co_base)));
        for (int oh_base = 0; oh_base < 28; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (28) ? (7) : ((28) - (oh_base)));
            for (int ow_base = 0; ow_base < 28; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (28) ? (7) : ((28) - (ow_base)));
                load_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(DRAM_s2_b1, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(DRAM_s2_b0, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_128_28_28_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s2_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (256) ? (64) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (14) ? (7) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (14) ? (7) : ((14) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (128) ? (16) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_16_15_15_2_1_ap_fixed_16_5_(DRAM_s2_b1, BRAM_in_patch_stride2_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_128_3_64_16_ap_fixed_16_5_(DRAM_w_s3_b0_1, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_15_15_64_16_3_64_7_7_2_4_1_ap_fixed_16_5_(BRAM_in_patch_stride2_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_64_ap_fixed_16_5_(DRAM_bn_s3_b0_1, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (256) ? (64) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (14) ? (7) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (14) ? (7) : ((14) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (256) ? (16) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_16_9_9_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_64_16_ap_fixed_16_5_(DRAM_w_s3_b0_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_64_ap_fixed_16_5_(DRAM_bn_s3_b0_2, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (256) ? (64) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (14) ? (7) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (14) ? (7) : ((14) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 128; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (128) ? (16) : ((128) - (ci_base)));
                    load_fmap_patch_128_28_28_16_13_13_2_0_ap_fixed_16_5_(DRAM_s2_b1, BRAM_in_patch_stride2_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_256_128_1_64_16_ap_fixed_16_5_(DRAM_w_s3_b0_down, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_16_13_13_64_16_1_64_7_7_2_4_1_ap_fixed_16_5_(BRAM_in_patch_stride2_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                store_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_skip, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (256) ? (64) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (14) ? (7) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (14) ? (7) : ((14) - (ow_base)));
                load_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(DRAM_s3_b0, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(DRAM_s3_skip, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (256) ? (64) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (14) ? (7) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (14) ? (7) : ((14) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (256) ? (16) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_16_9_9_1_1_ap_fixed_16_5_(DRAM_s3_b0, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_64_16_ap_fixed_16_5_(DRAM_w_s3_b1_1, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_64_ap_fixed_16_5_(DRAM_bn_s3_b1_1, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (256) ? (64) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (14) ? (7) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (14) ? (7) : ((14) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (256) ? (16) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_16_9_9_1_1_ap_fixed_16_5_(DRAM_s3_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_256_256_3_64_16_ap_fixed_16_5_(DRAM_w_s3_b1_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_256_64_ap_fixed_16_5_(DRAM_bn_s3_b1_2, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 256; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (256) ? (64) : ((256) - (co_base)));
        for (int oh_base = 0; oh_base < 14; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (14) ? (7) : ((14) - (oh_base)));
            for (int ow_base = 0; ow_base < 14; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (14) ? (7) : ((14) - (ow_base)));
                load_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(DRAM_s3_b1, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(DRAM_s3_b0, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_256_14_14_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s3_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (512) ? (64) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (7) ? (7) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (7) ? (7) : ((7) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (256) ? (16) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_16_15_15_2_1_ap_fixed_16_5_(DRAM_s3_b1, BRAM_in_patch_stride2_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_512_256_3_64_16_ap_fixed_16_5_(DRAM_w_s4_b0_1, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_15_15_64_16_3_64_7_7_2_4_1_ap_fixed_16_5_(BRAM_in_patch_stride2_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_64_ap_fixed_16_5_(DRAM_bn_s4_b0_1, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (512) ? (64) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (7) ? (7) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (7) ? (7) : ((7) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (512) ? (16) : ((512) - (ci_base)));
                    load_fmap_patch_512_7_7_16_9_9_1_1_ap_fixed_16_5_(DRAM_s4_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_512_512_3_64_16_ap_fixed_16_5_(DRAM_w_s4_b0_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_64_ap_fixed_16_5_(DRAM_bn_s4_b0_2, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (512) ? (64) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (7) ? (7) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (7) ? (7) : ((7) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 256; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (256) ? (16) : ((256) - (ci_base)));
                    load_fmap_patch_256_14_14_16_13_13_2_0_ap_fixed_16_5_(DRAM_s3_b1, BRAM_in_patch_stride2_k1, ci_base, oh_base, ow_base);
                    load_weight_tile_512_256_1_64_16_ap_fixed_16_5_(DRAM_w_s4_b0_down, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_16_13_13_64_16_1_64_7_7_2_4_1_ap_fixed_16_5_(BRAM_in_patch_stride2_k1, BRAM_weight_tile_1, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                store_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_skip, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (512) ? (64) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (7) ? (7) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (7) ? (7) : ((7) - (ow_base)));
                load_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(DRAM_s4_b0, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(DRAM_s4_skip, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_b0, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (512) ? (64) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (7) ? (7) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (7) ? (7) : ((7) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (512) ? (16) : ((512) - (ci_base)));
                    load_fmap_patch_512_7_7_16_9_9_1_1_ap_fixed_16_5_(DRAM_s4_b0, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_512_512_3_64_16_ap_fixed_16_5_(DRAM_w_s4_b1_1, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_64_ap_fixed_16_5_(DRAM_bn_s4_b1_1, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_mid1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (512) ? (64) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (7) ? (7) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (7) ? (7) : ((7) - (ow_base)));
                clear_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile);
                for (int ci_base = 0; ci_base < 512; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (512) ? (16) : ((512) - (ci_base)));
                    load_fmap_patch_512_7_7_16_9_9_1_1_ap_fixed_16_5_(DRAM_s4_mid1, BRAM_in_patch_stride1_k3, ci_base, oh_base, ow_base);
                    load_weight_tile_512_512_3_64_16_ap_fixed_16_5_(DRAM_w_s4_b1_2, BRAM_weight_tile_3, co_base, ci_base);
                    conv_tile_16_9_9_64_16_3_64_7_7_1_4_1_ap_fixed_16_5_(BRAM_in_patch_stride1_k3, BRAM_weight_tile_3, BRAM_out_tile, valid_co, valid_ci, valid_oh, valid_ow);
                }
                load_bn_tile_512_64_ap_fixed_16_5_(DRAM_bn_s4_b1_2, BRAM_bn_tile, co_base);
                batchnorm_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_bn_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (512) ? (64) : ((512) - (co_base)));
        for (int oh_base = 0; oh_base < 7; oh_base += 7) {
            int valid_oh = (((oh_base) + (7)) <= (7) ? (7) : ((7) - (oh_base)));
            for (int ow_base = 0; ow_base < 7; ow_base += 7) {
                int valid_ow = (((ow_base) + (7)) <= (7) ? (7) : ((7) - (ow_base)));
                load_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(DRAM_s4_b1, BRAM_out_tile, co_base, oh_base, ow_base);
                load_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(DRAM_s4_b0, BRAM_skip_tile, co_base, oh_base, ow_base);
                matrix_add_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_skip_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                relu_tile_64_7_7_ap_fixed_16_5_(BRAM_out_tile, BRAM_out_tile, valid_co, valid_oh, valid_ow);
                store_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(BRAM_out_tile, DRAM_s4_b1, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
    for (int co_base = 0; co_base < 512; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (512) ? (64) : ((512) - (co_base)));
        clear_tile_64_1_1_ap_fixed_16_5_(BRAM_gap_out);
        for (int gh_base = 0; gh_base < 7; gh_base += 7) {
            int valid_gh = (((gh_base) + (7)) <= (7) ? (7) : ((7) - (gh_base)));
            for (int gw_base = 0; gw_base < 7; gw_base += 7) {
                int valid_gw = (((gw_base) + (7)) <= (7) ? (7) : ((7) - (gw_base)));
                load_fmap_tile_512_7_7_64_7_7_ap_fixed_16_5_(DRAM_s4_b1, BRAM_gap_tile, co_base, gh_base, gw_base);
                avgpool_accumulate_tile_64_7_7_ap_fixed_16_5_(BRAM_gap_tile, BRAM_gap_out, valid_co, valid_gh, valid_gw);
            }
        }
        avgpool_finalize_tile_64_49_ap_fixed_16_5_(BRAM_gap_out, BRAM_gap_out, valid_co);
        store_fmap_tile_512_1_1_64_1_1_ap_fixed_16_5_(BRAM_gap_out, DRAM_gap, co_base, 0, 0, valid_co, 1, 1);
    }
    for (int co_base = 0; co_base < 1000; co_base += 64) {
        int valid_co = (((co_base) + (64)) <= (1000) ? (64) : ((1000) - (co_base)));
        for (int oh_base = 0; oh_base < 1; oh_base += 1) {
            int valid_oh = (((oh_base) + (1)) <= (1) ? (1) : ((1) - (oh_base)));
            for (int ow_base = 0; ow_base < 1; ow_base += 1) {
                int valid_ow = (((ow_base) + (1)) <= (1) ? (1) : ((1) - (ow_base)));
                clear_tile_64_1_1_ap_fixed_16_5_(BRAM_fc_out);
                for (int ci_base = 0; ci_base < 512; ci_base += 16) {
                    int valid_ci = (((ci_base) + (16)) <= (512) ? (16) : ((512) - (ci_base)));
                    load_fmap_patch_512_1_1_16_1_1_1_0_ap_fixed_16_5_(DRAM_gap, BRAM_in_patch_1, ci_base, oh_base, ow_base);
                    load_weight_tile_1000_512_1_64_16_ap_fixed_16_5_(DRAM_fc, BRAM_weight_tile_1, co_base, ci_base);
                    conv_tile_16_1_1_64_16_1_64_1_1_1_4_1_ap_fixed_16_5_(BRAM_in_patch_1, BRAM_weight_tile_1, BRAM_fc_out, valid_co, valid_ci, valid_oh, valid_ow);
                }
                store_fmap_tile_1000_1_1_64_1_1_ap_fixed_16_5_(BRAM_fc_out, DRAM_out, co_base, oh_base, ow_base, valid_co, valid_oh, valid_ow);
            }
        }
    }
}