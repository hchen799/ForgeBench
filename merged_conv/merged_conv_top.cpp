#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>
#include "common.h"
using namespace std;



int out_dim(int size, int pad, int stride, int kernel) {
    return (size + 2 * pad - kernel) / stride + 1;
}
int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

int min_value (int x, int y)
{
    if (x < y)
    {
        return x;
    }
    else
    {
        return y;
    }
}

void conv_kernel(
    data_t inData[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE],
    data_t weight[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE],
    data_t outData[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W],
    int outTileH, int outTileW,
    int localInH, int localInW,
    int stride
)
{
    // Loop only over the valid output tile region.
    for (int ic = 0; ic < BLOCK_IN_CH; ic++) {
        for (int oc = 0; oc < BLOCK_OUT_CH; oc++) {
            for (int oh = 0; oh < outTileH; oh++) {
                for (int ow = 0; ow < outTileW; ow++) {
                    data_t sum_val = (data_t) 0;
                    for (int kh = 0; kh < KSIZE; kh++) {
                        for (int kw = 0; kw < KSIZE; kw++) {
                            int in_r = oh * stride + kh;
                            int in_c = ow * stride + kw;
                            sum_val += inData[ic][in_r][in_c] * weight[oc][ic][kh][kw];
                        }
                    }
                    outData[oc][oh][ow] += sum_val;
                }
            }
        }
    }
}


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
)

#pragma HLS INTERFACE s_axilite port=in_ch
#pragma HLS INTERFACE s_axilite port=out_ch
#pragma HLS INTERFACE s_axilite port=H
#pragma HLS INTERFACE s_axilite port=W
#pragma HLS interface m_axi port=input offset=slave bundle=mem1
#pragma HLS interface m_axi port=weight offset=slave bundle=mem1
#pragma HLS interface m_axi port=bias offset=slave bundle=mem1
#pragma HLS interface m_axi port=output offset=slave bundle=mem2
#pragma HLS INTERFACE s_axilite port=stride
#pragma HLS INTERFACE s_axilite port=pad

{
    int out_h = out_dim(H, pad, stride, KSIZE);
    int out_w = out_dim(W, pad, stride, KSIZE);

    // (A) Initialize the full output with bias.
    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                output[oc][oh * out_w + ow] = bias[oc];
            }
        }
    }

    // Tiling parameters.
    int tile_rows = ceil_div(out_h, BLOCK_OUT_H);
    int tile_cols = ceil_div(out_w, BLOCK_OUT_W);
    int out_ch_tiles = ceil_div(out_ch, BLOCK_OUT_CH);
    int in_ch_tiles  = ceil_div(in_ch, BLOCK_IN_CH);

    // (B) Loop over spatial tiles.
    for (int tr = 0; tr < tile_rows; tr++) {
        for (int tc = 0; tc < tile_cols; tc++) {
            int out_row0 = tr * BLOCK_OUT_H;
            int out_col0 = tc * BLOCK_OUT_W;
            int tileH = min_value(BLOCK_OUT_H, out_h - out_row0);
            int tileW = min_value(BLOCK_OUT_W, out_w - out_col0);

            // Compute local input patch size for this tile.
            int localInH = (tileH - 1) * stride + KSIZE;
            int localInW = (tileW - 1) * stride + KSIZE;

            // (C) Loop over output-channel tiles.
            for (int oct = 0; oct < out_ch_tiles; oct++) {
                int oc_offset = oct * BLOCK_OUT_CH;
                int cur_out_ch = (oc_offset + BLOCK_OUT_CH <= out_ch) ? BLOCK_OUT_CH : (out_ch - oc_offset);

                // Copy current partial sums for this spatial tile from global output
                // into a local buffer.
                data_t localOut[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W];
                for (int block_out_ch_i = 0; block_out_ch_i < BLOCK_OUT_CH; block_out_ch_i++)
                {
                    for (int block_out_h_i = 0; block_out_h_i < BLOCK_OUT_H; block_out_h_i++)
                    {
                        for (int block_out_w_i = 0; block_out_w_i < BLOCK_OUT_W; block_out_w_i++)
                        {
                            localOut[block_out_ch_i][block_out_h_i][block_out_w_i] = (data_t) 0;
                        }
                    }
                }
                for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                    for (int i = 0; i < tileH * tileW; i++) {
                        int r_local = i / tileW;
                        int c_local = i % tileW;
                        localOut[oc2][r_local][c_local] =
                            output[oc_offset + oc2][(out_row0 + r_local) * out_w + (out_col0 + c_local)];
                    }
                }

                // (D) Loop over input-channel tiles.
                for (int ict = 0; ict < in_ch_tiles; ict++) {
                    int ic_offset = ict * BLOCK_IN_CH;
                    int cur_in_ch = (ic_offset + BLOCK_IN_CH <= in_ch) ? BLOCK_IN_CH : (in_ch - ic_offset);

                    // Build localW for the current slice.
                    data_t localW[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE];

                    for (int block_out_ch_j = 0; block_out_ch_j < BLOCK_OUT_CH; block_out_ch_j++)
                    {
                        for (int block_in_ch_j = 0; block_in_ch_j < BLOCK_IN_CH; block_in_ch_j++)
                        {
                            for (int k_i = 0; k_i < KSIZE; k_i++)
                            {
                                for (int k_j = 0; k_j < KSIZE; k_j++)
                                {
                                    localW[block_out_ch_j][block_in_ch_j][k_i][k_j] = (data_t) 0;
                                }
                            }
                        }
                    }

                    for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                        for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
                            for (int kr = 0; kr < KSIZE; kr++) {
                                for (int kc = 0; kc < KSIZE; kc++) {
                                    localW[oc2][ic2][kr][kc] =
                                        weight[oc_offset + oc2][ic_offset + ic2][kr][kc];
                                }
                            }
                        }
                    }

                    // Build localIn for the current input-channel tile.
                    data_t localIn[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE];
                    
                    for (int block_in_ch_k = 0; block_in_ch_k < BLOCK_IN_CH; block_in_ch_k++)
                    {
                        for (int block_in_h_k = 0; block_in_h_k < MAX_LOCAL_SIZE; block_in_h_k++)
                        {
                            for (int block_in_w_k = 0; block_in_w_k < MAX_LOCAL_SIZE; block_in_w_k++)
                            {
                                localIn[block_in_ch_k][block_in_h_k][block_in_w_k] = (data_t) 0;
                            }
                        }
                    }
                    for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
                        for (int rr = 0; rr < localInH; rr++) {
                            for (int cc = 0; cc < localInW; cc++) {
                                int global_r = out_row0 * stride - pad + rr;
                                int global_c = out_col0 * stride - pad + cc;
                                if (global_r >= 0 && global_r < H && global_c >= 0 && global_c < W) {
                                    localIn[ic2][rr][cc] =
                                        input[ic_offset + ic2][global_r * W + global_c];
                                }
                            }
                        }
                    }

                    // (E) Call conv_kernel for this tile.
                    conv_kernel(localIn, localW, localOut, tileH, tileW, localInH, localInW, stride);
                } // end input-channel tile loop

                // (F) Write the updated localOut back to global output.
                for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                    for (int r = 0; r < tileH; r++) {
                        for (int c = 0; c < tileW; c++) {
                            output[oc_offset + oc2][(out_row0 + r) * out_w + (out_col0 + c)] =
                                localOut[oc2][r][c];
                        }
                    }
                }
            } // end output-channel tile loop
        } // end spatial tile loop (tc)
    } // end spatial tile loop (tr)
}