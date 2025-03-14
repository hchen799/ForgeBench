#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include "common.h"
using namespace std;

void conv_via_tiling(
    int in_ch,
    int out_ch,
    int H,
    int W,
    data_t input[MAX_CH][MAX_SPATIAL],
    data_t weight[MAX_CH][MAX_CH][KSIZE][KSIZE],
    data_t bias[MAX_CH],
    data_t output[MAX_CH][MAX_SPATIAL],
    int stride,
    int pad
);

void conv_ref(
    int in_ch,
    int out_ch,
    int H,
    int W,
    data_t input[MAX_CH][MAX_SPATIAL],
    data_t weight[MAX_CH][MAX_CH][KSIZE][KSIZE],
    data_t bias[MAX_CH],
    data_t output[MAX_CH][MAX_SPATIAL],
    int stride,
    int pad
)
{
    int out_h = out_dim(H, pad, stride, KSIZE);
    int out_w = out_dim(W, pad, stride, KSIZE);
    // Initialize output with bias.
    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                output[oc][oh * out_w + ow] = bias[oc];
            }
        }
    }
    // Convolution accumulate.
    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    data_t sum_val = (data_t) 0;
                    int in_r0 = oh * stride - pad;
                    int in_c0 = ow * stride - pad;
                    for (int kh = 0; kh < KSIZE; kh++) {
                        for (int kw = 0; kw < KSIZE; kw++) {
                            int in_r = in_r0 + kh;
                            int in_c = in_c0 + kw;
                            if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W) {
                                sum_val += input[ic][in_r * W + in_c] *
                                           weight[oc][ic][kh][kw];
                            }
                        }
                    }
                    output[oc][oh * out_w + ow] += sum_val;
                }
            }
        }
    }
}

int main()
{
    srand((unsigned)time(NULL));

    static data_t input[MAX_CH][MAX_SPATIAL];          // [in_ch][H*W]
    static data_t weight[MAX_CH][MAX_CH][KSIZE][KSIZE];  // [out_ch][in_ch][5][5]
    static data_t bias[MAX_CH];
    static data_t out_ref[MAX_CH][MAX_SPATIAL];          // [out_ch][out_h*out_w]
    static data_t out_block[MAX_CH][MAX_SPATIAL];        // same

    auto runTest = [&](int inC, int outC, int H, int W, int stride, int pad)
    {
        std::cout << "\n=== Test inC=" << inC << ", outC=" << outC
                  << ", H=" << H << ", W=" << W
                  << ", stride=" << stride << ", pad=" << pad << " ===\n";
        int out_h = out_dim(H, pad, stride, KSIZE);
        int out_w = out_dim(W, pad, stride, KSIZE);
        if (out_h <= 0 || out_w <= 0) {
            std::cout << "  [Invalid output size => skipping]\n";
            return;
        }
        // Initialize input and weights randomly.
        for (int ic = 0; ic < inC; ic++) {
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    input[ic][r * W + c] = (data_t)(float(rand() % 201 - 100) / 25.f);
                }
            }
        }
        for (int oc = 0; oc < outC; oc++) {
            bias[oc] = (data_t)(float(rand() % 101 - 50) / 25.f);
            for (int ic = 0; ic < inC; ic++) {
                for (int kr = 0; kr < KSIZE; kr++) {
                    for (int kc = 0; kc < KSIZE; kc++) {
                        weight[oc][ic][kr][kc] = (data_t)(float(rand() % 101 - 50) / 25.f);
                    }
                }
            }
        }
        // Zero out unused portions.
        for (int ic = inC; ic < MAX_CH; ic++) {
            for (int i = 0; i < MAX_SPATIAL; i++) {
                input[ic][i] = (data_t) 0;
            }
        }
        for (int oc = outC; oc < MAX_CH; oc++) {
            bias[oc] = (data_t) 0;
            for (int ic = 0; ic < MAX_CH; ic++) {
                for (int kr = 0; kr < KSIZE; kr++) {
                    for (int kc = 0; kc < KSIZE; kc++) {
                        weight[oc][ic][kr][kc] = (data_t) 0;
                    }
                }
            }
        }

        conv_ref(inC, outC, H, W, input, weight, bias, out_ref, stride, pad);
        conv_via_tiling(inC, outC, H, W, input, weight, bias, out_block, stride, pad);

        data_t max_diff = 0.0;
        for (int oc = 0; oc < outC; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    data_t rVal = out_ref[oc][oh * out_w + ow];
                    data_t bVal = out_block[oc][oh * out_w + ow];
                    data_t diff = hls::fabs(rVal - bVal);
                    if (diff > max_diff)
                        max_diff = diff;
                }
            }
        }
        std::cout << "  Output shape: " << out_h << "x" << out_w
                  << ", max diff = " << max_diff << "\n";
        std::cout << "  Output ch0, top-left 4x4 => ref | block\n";
        for (int r = 0; r < 4 && r < out_h; r++) {
            for (int c = 0; c < 4 && c < out_w; c++) {
                std::cout << out_ref[0][r * out_w + c] << " ";
            }
            std::cout << " | ";
            for (int c = 0; c < 4 && c < out_w; c++) {
                std::cout << out_block[0][r * out_w + c] << " ";
            }
            std::cout << "\n";
        }
    };

    // -------------------------------------------------------------------------
    // Run test cases.
    // -------------------------------------------------------------------------
    // (1) 64->64, 14x14, stride=1, pad=1
    runTest(64, 64, 14, 14, 1, 1);
    // (2) 128->128, 7x7, stride=1, pad=1
    runTest(128, 128, 7, 7, 1, 1);
    // (3) 128->128, 14x14, stride=1, pad=1
    runTest(128, 128, 14, 14, 1, 1);
    // (4) 64->128, 10x10, stride=2, pad=2
    runTest(64, 128, 10, 10, 2, 2);
    // (5) 128->128, input 28x28, stride=2, pad=1 => output 14x14
    runTest(128, 128, 28, 28, 2, 1);
    // (6) 256->256, 14x14, stride=1, pad=1
    runTest(256, 256, 14, 14, 1, 1);
    // (7) Non-square input: 128->128, input 28x14, stride=3, pad=2 => output will be (out_dim(28,..) x out_dim(14,..))
    runTest(128, 128, 28, 14, 3, 2);

    runTest(64, 128, 14, 14, 1, 0);
    runTest(128, 64, 14, 14, 1, 0);

    return 0;
}

