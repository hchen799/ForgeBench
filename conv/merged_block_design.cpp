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

#define MAX_OUT_H 56   // Maximum height
#define MAX_OUT_W 56   // Maximum width

void conv_kernel_3x3(
    data_t inData[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE],
    data_t weight[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE],
    data_t outData[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W],
    int outTileH, int outTileW,
    int localInH, int localInW,
    int stride
)
{
    #pragma HLS inline off
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


void conv_via_tiling_3x3(
    int in_ch,
    int out_ch,
    int H, int W,  // input spatial dimensions
    data_t input[MAX_CH][MAX_H][MAX_W],
    data_t weight[MAX_CH][MAX_CH][KSIZE][KSIZE],
    data_t bias[MAX_CH],
    data_t output[MAX_CH][MAX_OUT_H][MAX_OUT_W],
    int stride,
    int pad
)
{
    int out_h = out_dim(H, pad, stride, KSIZE);
    int out_w = out_dim(W, pad, stride, KSIZE);

    // (A) Initialize the full 3‑D output with bias.
    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                output[oc][oh][ow] = bias[oc];
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
            int tileH = min(BLOCK_OUT_H, out_h - out_row0);
            int tileW = min(BLOCK_OUT_W, out_w - out_col0);

            // Compute local input patch size for this tile.
            int localInH = (tileH - 1) * stride + KSIZE;
            int localInW = (tileW - 1) * stride + KSIZE;

            // (C) Loop over output-channel tiles.
            for (int oct = 0; oct < out_ch_tiles; oct++) {
                int oc_offset = oct * BLOCK_OUT_CH;
                int cur_out_ch = (oc_offset + BLOCK_OUT_CH <= out_ch) ? BLOCK_OUT_CH : (out_ch - oc_offset);

                // Copy current partial sums from global output into a local buffer.
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
                    for (int r_local = 0; r_local < tileH; r_local++) {
                        for (int c_local = 0; c_local < tileW; c_local++) {
                            localOut[oc2][r_local][c_local] =
                                output[oc_offset + oc2][out_row0 + r_local][out_col0 + c_local];
                        }
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
                                    localIn[ic2][rr][cc] = input[ic_offset + ic2][global_r][global_c];
                                }
                            }
                        }
                    }

                    // (E) Call conv_kernel for this tile.
                    conv_kernel_3x3(localIn, localW, localOut, tileH, tileW, localInH, localInW, stride);
                } // end input-channel tile loop

                // (F) Write the updated localOut back to the global 3‑D output.
                for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                    for (int r = 0; r < tileH; r++) {
                        for (int c = 0; c < tileW; c++) {
                            output[oc_offset + oc2][out_row0 + r][out_col0 + c] = localOut[oc2][r][c];
                        }
                    }
                }
            } // end output-channel tile loop
        } // end spatial tile loop (tc)
    } // end spatial tile loop (tr)
}




// void conv_via_tiling(
//     int in_ch,
//     int out_ch,
//     int H,
//     int W,
//     data_t input[MAX_CH][MAX_SPATIAL],
//     data_t weight[MAX_CH][MAX_CH][KSIZE][KSIZE],
//     data_t bias[MAX_CH],
//     data_t output[MAX_CH][MAX_SPATIAL],
//     int stride,
//     int pad
// )

// {

// // #pragma HLS INTERFACE s_axilite port=in_ch
// // #pragma HLS INTERFACE s_axilite port=out_ch
// // #pragma HLS INTERFACE s_axilite port=H
// // #pragma HLS INTERFACE s_axilite port=W
// // #pragma HLS interface m_axi port=input offset=slave bundle=mem1
// // #pragma HLS interface m_axi port=weight offset=slave bundle=mem1
// // #pragma HLS interface m_axi port=bias offset=slave bundle=mem1
// // #pragma HLS interface m_axi port=output offset=slave bundle=mem2
// // #pragma HLS INTERFACE s_axilite port=stride
// // #pragma HLS INTERFACE s_axilite port=pad
// // #pragma HLS allocation function instances=conv_kernel limit=1
//     int out_h = out_dim(H, pad, stride, KSIZE);
//     int out_w = out_dim(W, pad, stride, KSIZE);

//     // (A) Initialize the full output with bias.
//     for (int oc = 0; oc < out_ch; oc++) {
//         for (int oh = 0; oh < out_h; oh++) {
//             for (int ow = 0; ow < out_w; ow++) {
//                 output[oc][oh * out_w + ow] = bias[oc];
//             }
//         }
//     }

//     // Tiling parameters.
//     int tile_rows = ceil_div(out_h, BLOCK_OUT_H);
//     int tile_cols = ceil_div(out_w, BLOCK_OUT_W);
//     int out_ch_tiles = ceil_div(out_ch, BLOCK_OUT_CH);
//     int in_ch_tiles  = ceil_div(in_ch, BLOCK_IN_CH);

//     // (B) Loop over spatial tiles.
//     for (int tr = 0; tr < tile_rows; tr++) {
//         for (int tc = 0; tc < tile_cols; tc++) {
//             int out_row0 = tr * BLOCK_OUT_H;
//             int out_col0 = tc * BLOCK_OUT_W;
//             int tileH = min_value(BLOCK_OUT_H, out_h - out_row0);
//             int tileW = min_value(BLOCK_OUT_W, out_w - out_col0);

//             // Compute local input patch size for this tile.
//             int localInH = (tileH - 1) * stride + KSIZE;
//             int localInW = (tileW - 1) * stride + KSIZE;

//             // (C) Loop over output-channel tiles.
//             for (int oct = 0; oct < out_ch_tiles; oct++) {
//                 int oc_offset = oct * BLOCK_OUT_CH;
//                 int cur_out_ch = (oc_offset + BLOCK_OUT_CH <= out_ch) ? BLOCK_OUT_CH : (out_ch - oc_offset);

//                 // Copy current partial sums for this spatial tile from global output
//                 // into a local buffer.
//                 data_t localOut[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W];
//                 for (int block_out_ch_i = 0; block_out_ch_i < BLOCK_OUT_CH; block_out_ch_i++)
//                 {
//                     for (int block_out_h_i = 0; block_out_h_i < BLOCK_OUT_H; block_out_h_i++)
//                     {
//                         for (int block_out_w_i = 0; block_out_w_i < BLOCK_OUT_W; block_out_w_i++)
//                         {
//                             localOut[block_out_ch_i][block_out_h_i][block_out_w_i] = (data_t) 0;
//                         }
//                     }
//                 }
//                 for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
//                     for (int i = 0; i < tileH * tileW; i++) {
//                         int r_local = i / tileW;
//                         int c_local = i % tileW;
//                         localOut[oc2][r_local][c_local] =
//                             output[oc_offset + oc2][(out_row0 + r_local) * out_w + (out_col0 + c_local)];
//                     }
//                 }

//                 // (D) Loop over input-channel tiles.
//                 for (int ict = 0; ict < in_ch_tiles; ict++) {
//                     int ic_offset = ict * BLOCK_IN_CH;
//                     int cur_in_ch = (ic_offset + BLOCK_IN_CH <= in_ch) ? BLOCK_IN_CH : (in_ch - ic_offset);

//                     // Build localW for the current slice.
//                     data_t localW[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE];

//                     for (int block_out_ch_j = 0; block_out_ch_j < BLOCK_OUT_CH; block_out_ch_j++)
//                     {
//                         for (int block_in_ch_j = 0; block_in_ch_j < BLOCK_IN_CH; block_in_ch_j++)
//                         {
//                             for (int k_i = 0; k_i < KSIZE; k_i++)
//                             {
//                                 for (int k_j = 0; k_j < KSIZE; k_j++)
//                                 {
//                                     localW[block_out_ch_j][block_in_ch_j][k_i][k_j] = (data_t) 0;
//                                 }
//                             }
//                         }
//                     }

//                     for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
//                         for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
//                             for (int kr = 0; kr < KSIZE; kr++) {
//                                 for (int kc = 0; kc < KSIZE; kc++) {
//                                     localW[oc2][ic2][kr][kc] =
//                                         weight[oc_offset + oc2][ic_offset + ic2][kr][kc];
//                                 }
//                             }
//                         }
//                     }

//                     // Build localIn for the current input-channel tile.
//                     data_t localIn[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE];
                    
//                     for (int block_in_ch_k = 0; block_in_ch_k < BLOCK_IN_CH; block_in_ch_k++)
//                     {
//                         for (int block_in_h_k = 0; block_in_h_k < MAX_LOCAL_SIZE; block_in_h_k++)
//                         {
//                             for (int block_in_w_k = 0; block_in_w_k < MAX_LOCAL_SIZE; block_in_w_k++)
//                             {
//                                 localIn[block_in_ch_k][block_in_h_k][block_in_w_k] = (data_t) 0;
//                             }
//                         }
//                     }
//                     for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
//                         for (int rr = 0; rr < localInH; rr++) {
//                             for (int cc = 0; cc < localInW; cc++) {
//                                 int global_r = out_row0 * stride - pad + rr;
//                                 int global_c = out_col0 * stride - pad + cc;
//                                 if (global_r >= 0 && global_r < H && global_c >= 0 && global_c < W) {
//                                     localIn[ic2][rr][cc] =
//                                         input[ic_offset + ic2][global_r * W + global_c];
//                                 }
//                             }
//                         }
//                     }

//                     // (E) Call conv_kernel for this tile.
//                     conv_kernel(localIn, localW, localOut, tileH, tileW, localInH, localInW, stride);
//                 } // end input-channel tile loop

//                 // (F) Write the updated localOut back to global output.
//                 for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
//                     for (int r = 0; r < tileH; r++) {
//                         for (int c = 0; c < tileW; c++) {
//                             output[oc_offset + oc2][(out_row0 + r) * out_w + (out_col0 + c)] =
//                                 localOut[oc2][r][c];
//                         }
//                     }
//                 }
//             } // end output-channel tile loop
//         } // end spatial tile loop (tc)
//     } // end spatial tile loop (tr)
// }


void batch_norm_kernel(
    data_t localInput[BLOCK_C][BLOCK_H][BLOCK_W],
    data_t localWeights[4][BLOCK_C],
    data_t localOutput[BLOCK_C][BLOCK_H][BLOCK_W],
    int tile_C, int tile_H, int tile_W
)
{
    #pragma HLS inline off
    for (int c = 0; c < tile_C; c++) {
        data_t denom = hls::sqrt(localWeights[3][c] + (data_t) EPSILON);
        for (int h = 0; h < tile_H; h++) {
            for (int w = 0; w < tile_W; w++) {
                float norm = (localInput[c][h][w] - localWeights[2][c]) / denom;
                localOutput[c][h][w] = localWeights[0][c] * norm + localWeights[1][c];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3) Tiled Batch Normalization
//
//    Divides the channels and spatial dimensions into tiles and calls the
//    batch_norm_kernel function for each tile.
// -----------------------------------------------------------------------------
void batch_norm_tiled(
    int C, int H, int W,
    data_t input[MAX_C][MAX_H][MAX_W],
    data_t weights[4][MAX_C],
    data_t output[MAX_C][MAX_H][MAX_W]
)
{
    int channel_tiles = ceil_div(C, BLOCK_C);
    int tile_rows     = ceil_div(H, BLOCK_H);
    int tile_cols     = ceil_div(W, BLOCK_W);

    // Loop over channel tiles.
    for (int ct = 0; ct < channel_tiles; ct++) {
        int c_offset = ct * BLOCK_C;
        int tile_C   = min(BLOCK_C, C - c_offset);

        // Loop over spatial tiles (rows and columns).
        for (int tr = 0; tr < tile_rows; tr++) {
            int h_offset = tr * BLOCK_H;
            int tile_H   = min(BLOCK_H, H - h_offset);
            for (int tc = 0; tc < tile_cols; tc++) {
                int w_offset = tc * BLOCK_W;
                int tile_W   = min(BLOCK_W, W - w_offset);

                // Local buffers for input and output tile.
                data_t localInput[BLOCK_C][BLOCK_H][BLOCK_W];
                data_t localOutput[BLOCK_C][BLOCK_H][BLOCK_W];
                // Local weights buffer for the current channel tile.
                data_t localWeights[4][BLOCK_C];

                // Copy global input tile into localInput.
                for (int c = 0; c < tile_C; c++) {
                    for (int h = 0; h < tile_H; h++) {
                        for (int w = 0; w < tile_W; w++) {
                            localInput[c][h][w] = input[c_offset + c][h_offset + h][w_offset + w];
                        }
                    }
                }
                // Copy corresponding weights into localWeights.
                for (int c = 0; c < tile_C; c++) {
                    for (int i = 0; i < 4; i++) {
                        localWeights[i][c] = weights[i][c_offset + c];
                    }
                }

                // Call the fixed-size kernel for this tile.
                batch_norm_kernel(localInput, localWeights, localOutput, tile_C, tile_H, tile_W);

                // Write the computed localOutput back to the global output.
                for (int c = 0; c < tile_C; c++) {
                    for (int h = 0; h < tile_H; h++) {
                        for (int w = 0; w < tile_W; w++) {
                            output[c_offset + c][h_offset + h][w_offset + w] = localOutput[c][h][w];
                        }
                    }
                }
            }
        }
    }
}


void relu_kernel(
    data_t localInput[BLOCK_C][BLOCK_H][BLOCK_W],
    data_t localOutput[BLOCK_C][BLOCK_H][BLOCK_W],
    int tile_C, int tile_H, int tile_W
)
{
    for (int c = 0; c < tile_C; c++) {
        for (int h = 0; h < tile_H; h++) {
            for (int w = 0; w < tile_W; w++) {
                localOutput[c][h][w] = (localInput[c][h][w] > (data_t) 0) ? localInput[c][h][w] : (data_t) 0;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3) Tiled ReLU
//
//    Divides the input into tiles and calls relu_kernel for each tile.
// -----------------------------------------------------------------------------
void relu_tiled(
    int C, int H, int W,
    data_t input[MAX_C][MAX_H][MAX_W],
    data_t output[MAX_C][MAX_H][MAX_W]
)
{
    int channel_tiles = ceil_div(C, BLOCK_C);
    int tile_rows     = ceil_div(H, BLOCK_H);
    int tile_cols     = ceil_div(W, BLOCK_W);

    // Loop over channel tiles.
    for (int ct = 0; ct < channel_tiles; ct++) {
        int c_offset = ct * BLOCK_C;
        int tile_C = min(BLOCK_C, C - c_offset);

        // Loop over spatial tiles (rows and columns).
        for (int tr = 0; tr < tile_rows; tr++) {
            int h_offset = tr * BLOCK_H;
            int tile_H = min(BLOCK_H, H - h_offset);
            for (int tc = 0; tc < tile_cols; tc++) {
                int w_offset = tc * BLOCK_W;
                int tile_W = min(BLOCK_W, W - w_offset);

                // Local buffers for input and output tile.
                data_t localInput[BLOCK_C][BLOCK_H][BLOCK_W];
                data_t localOutput[BLOCK_C][BLOCK_H][BLOCK_W];

                // Copy global input tile into localInput.
                for (int c = 0; c < tile_C; c++) {
                    for (int h = 0; h < tile_H; h++) {
                        for (int w = 0; w < tile_W; w++) {
                            localInput[c][h][w] = input[c_offset + c][h_offset + h][w_offset + w];
                        }
                    }
                }

                // Call the fixed-size kernel for this tile.
                relu_kernel(localInput, localOutput, tile_C, tile_H, tile_W);

                // Write the computed localOutput back to the global output.
                for (int c = 0; c < tile_C; c++) {
                    for (int h = 0; h < tile_H; h++) {
                        for (int w = 0; w < tile_W; w++) {
                            output[c_offset + c][h_offset + h][w_offset + w] = localOutput[c][h][w];
                        }
                    }
                }
            }
        }
    }
}


void maxpool_kernel(
    data_t localInput[BLOCK_C][MAX_LOCAL_IN_H][MAX_LOCAL_IN_W],
    data_t localOutput[BLOCK_C][BLOCK_H][BLOCK_W],
    int tile_C, int tile_H, int tile_W
)
{
    for (int c = 0; c < tile_C; c++) {
        for (int i = 0; i < tile_H; i++) {
            for (int j = 0; j < tile_W; j++) {
                // Compute pooling over the window.
                data_t max_val = localInput[c][i * STRIDE_H][j * STRIDE_W];
                for (int kh = 0; kh < K_H; kh++) {
                    for (int kw = 0; kw < K_W; kw++) {
                        int r = i * STRIDE_H + kh;
                        int col = j * STRIDE_W + kw;
                        if (localInput[c][r][col] > max_val) {
                            max_val = localInput[c][r][col];
                        }
                    }
                }
                localOutput[c][i][j] = max_val;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3) Tiled MaxPool
//
//    Divides the channels and output spatial dimensions into tiles and calls
//    maxpool_kernel for each tile.
// -----------------------------------------------------------------------------
void maxpool_tiled(
    int C, int H_IN, int W_IN,
    data_t input[MAX_C][MAX_H_IN][MAX_W_IN],
    int H_OUT, int W_OUT,
    data_t output[MAX_C][MAX_H_IN][MAX_W_IN]   // Output region [C][H_OUT][W_OUT]
)
{
    // Divide channels and spatial dimensions (of the output) into tiles.
    int channel_tiles = ceil_div(C, BLOCK_C);
    int tile_rows = ceil_div(H_OUT, BLOCK_H);
    int tile_cols = ceil_div(W_OUT, BLOCK_W);

    // Loop over channel tiles.
    for (int ct = 0; ct < channel_tiles; ct++) {
        int c_offset = ct * BLOCK_C;
        int tile_C = min(BLOCK_C, C - c_offset);

        // Loop over spatial tiles.
        for (int tr = 0; tr < tile_rows; tr++) {
            int out_row_offset = tr * BLOCK_H;
            int tile_H = min(BLOCK_H, H_OUT - out_row_offset);
            // Compute corresponding input tile height.
            int localInH = (tile_H - 1) * STRIDE_H + K_H;

            for (int tc = 0; tc < tile_cols; tc++) {
                int out_col_offset = tc * BLOCK_W;
                int tile_W = min(BLOCK_W, W_OUT - out_col_offset);
                // Compute corresponding input tile width.
                int localInW = (tile_W - 1) * STRIDE_W + K_W;

                // Local buffers.
                data_t localInput[BLOCK_C][MAX_LOCAL_IN_H][MAX_LOCAL_IN_W];
                data_t localOutput[BLOCK_C][BLOCK_H][BLOCK_W];

                // Copy the relevant region from global input to localInput.
                // For each channel in this tile.
                for (int c = 0; c < tile_C; c++) {
                    for (int r = 0; r < localInH; r++) {
                        for (int col = 0; col < localInW; col++) {
                            // Global input row = (out_row_offset * STRIDE_H) + r
                            // Global input col = (out_col_offset * STRIDE_W) + col
                            localInput[c][r][col] = input[c_offset + c][(out_row_offset * STRIDE_H) + r][(out_col_offset * STRIDE_W) + col];
                        }
                    }
                }

                // Call the maxpool kernel for this tile.
                maxpool_kernel(localInput, localOutput, tile_C, tile_H, tile_W);

                // Write the computed tile back to global output.
                for (int c = 0; c < tile_C; c++) {
                    for (int i = 0; i < tile_H; i++) {
                        for (int j = 0; j < tile_W; j++) {
                            output[c_offset + c][out_row_offset + i][out_col_offset + j] = localOutput[c][i][j];
                        }
                    }
                }
            }
        }
    }
}




void load_feature_map(data_t input_dram[MAX_C][MAX_H][MAX_W], data_t input_buffer[MAX_C][MAX_H][MAX_W], int C, int H, int W)
{
    for (int c = 0; c < C; c++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                input_buffer[c][h][w] = input_dram[c][h][w];
            }
        }
    }
}

void store_feature_map(data_t output_buffer[MAX_C][MAX_H][MAX_W], data_t output_dram[MAX_C][MAX_H][MAX_W], int C, int H, int W)
{
    for (int c = 0; c < C; c++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                output_dram[c][h][w] = output_buffer[c][h][w];
            }
        }
    }
}

void load_weights(data_t weight_dram[MAX_C][MAX_C][KSIZE][KSIZE], data_t weight_buffer[MAX_C][MAX_C][KSIZE][KSIZE], int C_OUT, int C_IN, int K)
{
    for (int cout = 0; cout < C_OUT; cout++)
    {
        for (int cin = 0; cin < C_IN; h++)
        {
            for (int k1 = 0; k1 < K; k1++)
            {
                for (int k2 = 0; k2 < K; k2++)
                {
                    weight_buffer[cout][cin][k1][k2] = weight_dram[cout][cin][k1][k2];
                }
                
            }
        }
    }
}

void load_bias(data_t bias_dram[MAX_C], data_t bias_buffer[MAX_C], int C)
{
    for (int c = 0; c < C; c++)
    {
        bias_buffer[c] = bias_dram[c];
    }
}

data_t FM_buffer_1[MAX_C][MAX_H][MAX_W];
data_t weight_buffer[MAX_C][MAX_C][KSIZE][KSIZE];
data_t bias_buffer[MAX_C]
data_t FM_buffer_2[MAX_C][MAX_H][MAX_W];


void top_A(data_t input [128][56][56], 
            data_t conv_weight_1[256][128][3][3], 
            data_t conv_bias_1[256],
            data_t conv_weight_2[256][256][3][3],
            data_t conv_bias_2[256],
            data_t conv_weight_3[256][256][3][3],
            data_t conv_bias_3[256],
            data_t conv_weight_4[256][256][3][3],
            data_t conv_bias_4[256],
            data_t output[256][28][28], 
            int in_ch, int out_ch, int H, int W, int stride, int pad
        )
{
    load_feature_map(input, FM_buffer_1, 128, 56, 56);
    load_weights(conv_weight_1, weight_buffer, 256, 128, 3);
    load_bias(conv_bias_1, bias_buffer, 256);
    conv_via_tiling_3x3(128, 256, 56, 56, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    
    load_weights(conv_weight_2, weight_buffer, 256, 256, 3);
    load_bias(conv_bias_2, bias_buffer, 256);
    conv_via_tiling_3x3(256, 256, 56, 56, FM_buffer_2, weight_buffer, bias_buffer, FM_buffer_1, 1, 1);
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

    int in_ch,
    int out_ch,
    int H, int W,  // input spatial dimensions
    data_t input[MAX_CH][MAX_H][MAX_W],
    data_t weight[MAX_CH][MAX_CH][KSIZE][KSIZE],
    data_t bias[MAX_CH],
    data_t output[MAX_CH][MAX_OUT_H][MAX_OUT_W],
    int stride,
    int pad

}






