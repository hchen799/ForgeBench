#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>

// Maximum spatial dimension (enough for up to 14x14 = 196 elements)
#define MAX_SPATIAL 256
// Maximum channels supported (e.g., 256)
#define MAX_CH 256

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------
static int out_dim(int size, int pad, int stride, int kernel = 3) {
    // Standard convolution output dimension formula
    return (size + 2 * pad - kernel) / stride + 1;
}
static int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

// -----------------------------------------------------------------------------
// 1) Block Function: conv3x3_block_64_7x7
//
//    This function computes a 3x3 convolution on a block with:
//      - exactly 64 input channels and 64 output channels,
//      - a fixed output tile size of 7x7.
//    It assumes that the caller has built a local input patch (of size 
//    localInH x localInW = (7-1)*stride + 3) with any needed zero-padding.
//    It performs pure accumulation (no bias added) over the 64 input channels.
// -----------------------------------------------------------------------------

// Maximum local input dimension (set large enough for typical stride values)
#define MAX_LOCAL_SIZE 15

static const int BLOCK_IN_CH  = 64;
static const int BLOCK_OUT_CH = 64;
static const int BLOCK_OUT_H  = 7;
static const int BLOCK_OUT_W  = 7;
static const int KSIZE        = 3;

void conv3x3_block_64_7x7(
    float inData[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE],
    float weight[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE],
    float outData[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W],
    int localInH,
    int localInW,
    int stride
)
{
    // For each of the 64 input channels, accumulate contributions
    for (int ic = 0; ic < BLOCK_IN_CH; ic++) {
        for (int oc = 0; oc < BLOCK_OUT_CH; oc++) {
            for (int oh = 0; oh < BLOCK_OUT_H; oh++) {
                for (int ow = 0; ow < BLOCK_OUT_W; ow++) {
                    float sum_val = 0.0f;
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

// -----------------------------------------------------------------------------
// 2) Reference Convolution: conv3x3_ref
//
//    This is a standard (non‑tiled) 3x3 convolution with given stride and pad.
//    The input and output are stored in flattened 1D arrays per channel.
// -----------------------------------------------------------------------------
void conv3x3_ref(
    int in_ch,
    int out_ch,
    int H,
    int W,
    float input[MAX_CH][MAX_SPATIAL],          // [in_ch][H*W]
    float weight[MAX_CH][MAX_CH][KSIZE][KSIZE],  // [out_ch][in_ch][3][3]
    float bias[MAX_CH],
    float output[MAX_CH][MAX_SPATIAL],           // [out_ch][out_h*out_w]
    int stride,
    int pad
)
{
    int out_h = out_dim(H, pad, stride, KSIZE);
    int out_w = out_dim(W, pad, stride, KSIZE);
    // Initialize output with bias
    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                output[oc][oh * out_w + ow] = bias[oc];
            }
        }
    }
    // Convolution accumulate
    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum_val = 0.0f;
                    int in_r0 = oh * stride - pad;
                    int in_c0 = ow * stride - pad;
                    for (int kh = 0; kh < KSIZE; kh++) {
                        for (int kw = 0; kw < KSIZE; kw++) {
                            int in_r = in_r0 + kh;
                            int in_c = in_c0 + kw;
                            if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W) {
                                sum_val += input[ic][in_r * W + in_c] * weight[oc][ic][kh][kw];
                            }
                        }
                    }
                    output[oc][oh * out_w + ow] += sum_val;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3) Block-Based Orchestrator: conv3x3_via_block_64_7x7
//
//    This function computes the full convolution for arbitrary input/output
//    channel counts and spatial dimensions (up to MAX_CH channels and 14x14, for example)
//    using the 64x7x7 block function. It supports user-specified stride and pad.
//
//    The strategy is to:
//      (A) Pre-initialize the full output with bias (once).
//      (B) Loop over spatial tiles (7x7 sub‑regions).
//      (C) For each spatial tile, loop over output-channel tiles,
//          and for each, loop over input-channel tiles.
//      (D) For each such sub‑block, copy the relevant sub‑region of the output
//          into a local buffer, update it using conv3x3_block_64_7x7, then write
//          it back.
// -----------------------------------------------------------------------------
void conv3x3_via_block_64_7x7(
    int in_ch,
    int out_ch,
    int H,
    int W,
    float input[MAX_CH][MAX_SPATIAL],           // [in_ch][H*W]
    float weight[MAX_CH][MAX_CH][KSIZE][KSIZE],   // [out_ch][in_ch][3][3]
    float bias[MAX_CH],
    float output[MAX_CH][MAX_SPATIAL],            // [out_ch][out_h*out_w]
    int stride,
    int pad
)
{
    int out_h = out_dim(H, pad, stride, KSIZE);
    int out_w = out_dim(W, pad, stride, KSIZE);

    // (A) Initialize the entire output with bias exactly once.
    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                output[oc][oh * out_w + ow] = bias[oc];
            }
        }
    }

    // Tiling parameters:
    int tile_rows = ceil_div(out_h, BLOCK_OUT_H);  // number of 7x7 tiles vertically
    int tile_cols = ceil_div(out_w, BLOCK_OUT_W);    // horizontally
    int out_ch_tiles = ceil_div(out_ch, BLOCK_OUT_CH);
    int in_ch_tiles  = ceil_div(in_ch, BLOCK_IN_CH);

    // local patch size for each 7x7 output tile:
    int localInH = (BLOCK_OUT_H - 1) * stride + KSIZE; // = 6*stride + 3
    int localInW = (BLOCK_OUT_W - 1) * stride + KSIZE;

    // Local arrays (fixed sizes)
    static float localIn[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE];   // [64][localInH][localInW]
    static float localOut[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W];         // [64][7][7]
    static float localW[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE];            // [64][64][3][3]

    // (B) Loop over spatial tiles first:
    for (int tr = 0; tr < tile_rows; tr++) {
        for (int tc = 0; tc < tile_cols; tc++) {
            int out_row0 = tr * BLOCK_OUT_H;
            int out_col0 = tc * BLOCK_OUT_W;
            int tileH = std::min(BLOCK_OUT_H, out_h - out_row0);
            int tileW = std::min(BLOCK_OUT_W, out_w - out_col0);

            // (C) For each output-channel tile:
            for (int oct = 0; oct < out_ch_tiles; oct++) {
                int oc_offset = oct * BLOCK_OUT_CH;
                int cur_out_ch = (oc_offset + BLOCK_OUT_CH <= out_ch) ? BLOCK_OUT_CH : (out_ch - oc_offset);

                // Copy the current partial sums for this spatial tile from global output into localOut.
                for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                    for (int i = 0; i < tileH * tileW; i++) {
                        int r_local = i / tileW;
                        int c_local = i % tileW;
                        localOut[oc2][r_local][c_local] = output[oc_offset + oc2][(out_row0 + r_local) * out_w + (out_col0 + c_local)];
                    }
                    // Clear the rest of the 7x7 block for safety.
                    for (int r = tileH; r < BLOCK_OUT_H; r++) {
                        for (int c = 0; c < BLOCK_OUT_W; c++) {
                            localOut[oc2][r][c] = 0.f;
                        }
                    }
                    for (int r = 0; r < tileH; r++) {
                        for (int c = tileW; c < BLOCK_OUT_W; c++) {
                            localOut[oc2][r][c] = 0.f;
                        }
                    }
                }
                // For any unused channels in localOut, clear them.
                for (int oc2 = cur_out_ch; oc2 < BLOCK_OUT_CH; oc2++) {
                    for (int r = 0; r < BLOCK_OUT_H; r++) {
                        for (int c = 0; c < BLOCK_OUT_W; c++) {
                            localOut[oc2][r][c] = 0.f;
                        }
                    }
                }

                // (D) Loop over input-channel tiles for this spatial and out-channel tile.
                for (int ict = 0; ict < in_ch_tiles; ict++) {
                    int ic_offset = ict * BLOCK_IN_CH;
                    int cur_in_ch = (ic_offset + BLOCK_IN_CH <= in_ch) ? BLOCK_IN_CH : (in_ch - ic_offset);

                    // Build localW for current out-channel and in-channel slice.
                    for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                        for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
                            for (int kr = 0; kr < KSIZE; kr++) {
                                for (int kc = 0; kc < KSIZE; kc++) {
                                    localW[oc2][ic2][kr][kc] =
                                        weight[oc_offset + oc2][ic_offset + ic2][kr][kc];
                                }
                            }
                        }
                        // For unused input channels, set weights to zero.
                        for (int ic2 = cur_in_ch; ic2 < BLOCK_IN_CH; ic2++) {
                            for (int kr = 0; kr < KSIZE; kr++) {
                                for (int kc = 0; kc < KSIZE; kc++) {
                                    localW[oc2][ic2][kr][kc] = 0.f;
                                }
                            }
                        }
                    }
                    // For unused output channels, clear localW.
                    for (int oc2 = cur_out_ch; oc2 < BLOCK_OUT_CH; oc2++) {
                        for (int ic2 = 0; ic2 < BLOCK_IN_CH; ic2++) {
                            for (int kr = 0; kr < KSIZE; kr++) {
                                for (int kc = 0; kc < KSIZE; kc++) {
                                    localW[oc2][ic2][kr][kc] = 0.f;
                                }
                            }
                        }
                    }

                    // Build localIn for the current input-channel tile and spatial tile.
                    // The top-left global input coordinate for this tile:
                    int in_row0 = out_row0 * stride - pad;
                    int in_col0 = out_col0 * stride - pad;
                    // Clear localIn first:
                    for (int ic2 = 0; ic2 < BLOCK_IN_CH; ic2++) {
                        for (int rr = 0; rr < localInH; rr++) {
                            for (int cc = 0; cc < localInW; cc++) {
                                localIn[ic2][rr][cc] = 0.f;
                            }
                        }
                    }
                    for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
                        for (int rr = 0; rr < localInH; rr++) {
                            for (int cc = 0; cc < localInW; cc++) {
                                int global_r = in_row0 + rr;
                                int global_c = in_col0 + cc;
                                if (global_r >= 0 && global_r < H && global_c >= 0 && global_c < W) {
                                    localIn[ic2][rr][cc] = input[ic_offset + ic2][global_r * W + global_c];
                                }
                            }
                        }
                    }
                    // For unused channels in localIn, they remain zero.

                    // (E) Call the block function to accumulate partial sums.
                    conv3x3_block_64_7x7(localIn, localW, localOut, localInH, localInW, stride);
                } // end input-channel tile loop

                // (F) Write the updated localOut back to the global output.
                for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                    for (int r = 0; r < tileH; r++) {
                        for (int c = 0; c < tileW; c++) {
                            output[oc_offset + oc2][(out_row0 + r) * out_w + (out_col0 + c)] = localOut[oc2][r][c];
                        }
                    }
                }
            } // end out-channel tile loop
        } // end spatial tile loop (tc)
    } // end spatial tile loop (tr)
}

// -----------------------------------------------------------------------------
// 4) Main Test Driver
// -----------------------------------------------------------------------------
int main()
{
    srand((unsigned)time(NULL));

    // Allocate arrays for up to MAX_CH channels and images with spatial dimension up to MAX_SPATIAL.
    static float input[MAX_CH][MAX_SPATIAL];          // [in_ch][H*W]
    static float weight[MAX_CH][MAX_CH][3][3];          // [out_ch][in_ch][3][3]
    static float bias[MAX_CH];
    static float out_ref[MAX_CH][MAX_SPATIAL];          // [out_ch][out_h*out_w]
    static float out_block[MAX_CH][MAX_SPATIAL];        // same

    auto runTest = [&](int inC, int outC, int H, int W, int stride, int pad)
    {
        std::cout << "\n=== Test inC=" << inC << ", outC=" << outC
                  << ", H=" << H << ", W=" << W
                  << ", stride=" << stride << ", pad=" << pad << " ===\n";
        int out_h = out_dim(H, pad, stride, 3);
        int out_w = out_dim(W, pad, stride, 3);
        if (out_h <= 0 || out_w <= 0) {
            std::cout << "  [Invalid output size => skipping]\n";
            return;
        }

        // Initialize input and weights randomly.
        for (int ic = 0; ic < inC; ic++) {
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    input[ic][r * W + c] = float(rand() % 201 - 100) / 25.f;
                }
            }
        }
        for (int oc = 0; oc < outC; oc++) {
            bias[oc] = float(rand() % 101 - 50) / 25.f;
            for (int ic = 0; ic < inC; ic++) {
                for (int kr = 0; kr < 3; kr++) {
                    for (int kc = 0; kc < 3; kc++) {
                        weight[oc][ic][kr][kc] = float(rand() % 101 - 50) / 25.f;
                    }
                }
            }
        }
        // Zero out unused portions.
        for (int ic = inC; ic < MAX_CH; ic++) {
            for (int i = 0; i < MAX_SPATIAL; i++) {
                input[ic][i] = 0.f;
            }
        }
        for (int oc = outC; oc < MAX_CH; oc++) {
            bias[oc] = 0.f;
            for (int ic = 0; ic < MAX_CH; ic++) {
                for (int kr = 0; kr < 3; kr++) {
                    for (int kc = 0; kc < 3; kc++) {
                        weight[oc][ic][kr][kc] = 0.f;
                    }
                }
            }
        }

        // Run reference convolution.
        conv3x3_ref(inC, outC, H, W, input, weight, bias, out_ref, stride, pad);
        // Run block-based convolution.
        conv3x3_via_block_64_7x7(inC, outC, H, W, input, weight, bias, out_block, stride, pad);

        // Compare results.
        double max_diff = 0.0;
        for (int oc = 0; oc < outC; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    double rVal = out_ref[oc][oh * out_w + ow];
                    double bVal = out_block[oc][oh * out_w + ow];
                    double diff = std::fabs(rVal - bVal);
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
    // (5) 128->128, input 28x28, output 14x14, stride=2, pad=1
    runTest(128, 128, 28, 28, 2, 1);
    // (6) 256->256, 14x14, stride=1, pad=1
    runTest(256, 256, 14, 14, 1, 1);

    return 0;
}




