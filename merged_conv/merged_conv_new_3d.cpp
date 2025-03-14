#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>

// -----------------------------------------------------------------------------
// Configuration and Constants
// -----------------------------------------------------------------------------
// Maximum dimensions for the 3-D input and output arrays.
// Adjust these as needed for your application.
#define MAX_H      224   // maximum input height
#define MAX_W      224   // maximum input width
#define MAX_OUT_H  224   // maximum output height
#define MAX_OUT_W  224   // maximum output width
// Maximum channels supported.
#define MAX_CH     256

// -----------------------------------------------------------------------------
// Convolution parameters
// -----------------------------------------------------------------------------
// Kernel size (e.g., 3x3)
static const int KSIZE = 3;

// Helper functions to compute output dimensions.
static int out_dim(int size, int pad, int stride, int kernel = KSIZE) {
    return (size + 2 * pad - kernel) / stride + 1;
}
static int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

// -----------------------------------------------------------------------------
// Tiling (block) parameters
// -----------------------------------------------------------------------------
// Define the maximum size of a local patch. This must be set such that
// stride*(BLOCK_OUT_H - 1) + KSIZE <= MAX_LOCAL_SIZE.
#define MAX_LOCAL_SIZE 23

// Tiled kernel block dimensions.
static const int BLOCK_IN_CH  = 64;
static const int BLOCK_OUT_CH = 64;
static const int BLOCK_OUT_H  = 7;  // maximum tile height (output)
static const int BLOCK_OUT_W  = 7;  // maximum tile width (output)

// -----------------------------------------------------------------------------
// Block Function: conv_kernel
//
// Computes a convolution on a tile (block) with fixed input and output channel
// counts over a valid region defined by outTileH x outTileW. The input patch
// is assumed to be prepared with proper zero-padding.
// -----------------------------------------------------------------------------
void conv_kernel(
    float inData[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE],
    float weight[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE],
    float outData[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W],
    int outTileH, int outTileW,
    int localInH, int localInW,
    int stride
)
{
    for (int ic = 0; ic < BLOCK_IN_CH; ic++) {
        for (int oc = 0; oc < BLOCK_OUT_CH; oc++) {
            for (int oh = 0; oh < outTileH; oh++) {
                for (int ow = 0; ow < outTileW; ow++) {
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
// Reference Convolution: conv_ref
//
// Standard (non‑tiled) 3‑D convolution with given stride and pad. The input and
// output are stored as 3‑D arrays: [channel][row][column].
// -----------------------------------------------------------------------------
void conv_ref(
    int in_ch,
    int out_ch,
    int H, int W,  // input spatial dimensions
    float input[MAX_CH][MAX_H][MAX_W],
    float weight[MAX_CH][MAX_CH][KSIZE][KSIZE],
    float bias[MAX_CH],
    float output[MAX_CH][MAX_OUT_H][MAX_OUT_W],
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
                output[oc][oh][ow] = bias[oc];
            }
        }
    }
    // Convolution accumulation.
    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int in_r0 = oh * stride - pad;
                    int in_c0 = ow * stride - pad;
                    float sum_val = 0.0f;
                    for (int kh = 0; kh < KSIZE; kh++) {
                        for (int kw = 0; kw < KSIZE; kw++) {
                            int in_r = in_r0 + kh;
                            int in_c = in_c0 + kw;
                            if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W) {
                                sum_val += input[ic][in_r][in_c] * weight[oc][ic][kh][kw];
                            }
                        }
                    }
                    output[oc][oh][ow] += sum_val;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Block-Based Orchestrator: conv_via_tiling
//
// Computes the full convolution using the block function (conv_kernel) over
// tiles. Input and output are 3‑D arrays. This function handles arbitrary
// input/output channel counts and spatial dimensions.
// -----------------------------------------------------------------------------
void conv_via_tiling(
    int in_ch,
    int out_ch,
    int H, int W,  // input spatial dimensions
    float input[MAX_CH][MAX_H][MAX_W],
    float weight[MAX_CH][MAX_CH][KSIZE][KSIZE],
    float bias[MAX_CH],
    float output[MAX_CH][MAX_OUT_H][MAX_OUT_W],
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
            int tileH = std::min(BLOCK_OUT_H, out_h - out_row0);
            int tileW = std::min(BLOCK_OUT_W, out_w - out_col0);

            // Compute local input patch size for this tile.
            int localInH = (tileH - 1) * stride + KSIZE;
            int localInW = (tileW - 1) * stride + KSIZE;

            // (C) Loop over output-channel tiles.
            for (int oct = 0; oct < out_ch_tiles; oct++) {
                int oc_offset = oct * BLOCK_OUT_CH;
                int cur_out_ch = (oc_offset + BLOCK_OUT_CH <= out_ch) ? BLOCK_OUT_CH : (out_ch - oc_offset);

                // Copy current partial sums from global output into a local buffer.
                float localOut[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W] = {0};
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
                    float localW[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE] = {{{{0}}}};
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
                    float localIn[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE] = {{{0}}};
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
                    conv_kernel(localIn, localW, localOut, tileH, tileW, localInH, localInW, stride);
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

// -----------------------------------------------------------------------------
// Main Test Driver
// -----------------------------------------------------------------------------
int main()
{
    srand((unsigned)time(NULL));

    // Global arrays for input, weights, biases, and outputs.
    static float input[MAX_CH][MAX_H][MAX_W];              // 3-D input: [channel][H][W]
    static float weight[MAX_CH][MAX_CH][KSIZE][KSIZE];       // weights: [out_ch][in_ch][KSIZE][KSIZE]
    static float bias[MAX_CH];                               // bias for each output channel
    static float out_ref[MAX_CH][MAX_OUT_H][MAX_OUT_W];      // output from conv_ref
    static float out_tile[MAX_CH][MAX_OUT_H][MAX_OUT_W];     // output from conv_via_tiling

    // Lambda to run a single test case.
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
        // Initialize input with random values.
        for (int ic = 0; ic < inC; ic++) {
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    input[ic][r][c] = float(rand() % 201 - 100) / 25.f;
                }
            }
        }
        // Initialize weights and bias.
        for (int oc = 0; oc < outC; oc++) {
            bias[oc] = float(rand() % 101 - 50) / 25.f;
            for (int ic = 0; ic < inC; ic++) {
                for (int kr = 0; kr < KSIZE; kr++) {
                    for (int kc = 0; kc < KSIZE; kc++) {
                        weight[oc][ic][kr][kc] = float(rand() % 101 - 50) / 25.f;
                    }
                }
            }
        }
        // Zero out unused portions.
        for (int ic = inC; ic < MAX_CH; ic++) {
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    input[ic][r][c] = 0.0f;
                }
            }
        }
        for (int oc = outC; oc < MAX_CH; oc++) {
            bias[oc] = 0.0f;
            for (int ic = 0; ic < MAX_CH; ic++) {
                for (int kr = 0; kr < KSIZE; kr++) {
                    for (int kc = 0; kc < KSIZE; kc++) {
                        weight[oc][ic][kr][kc] = 0.0f;
                    }
                }
            }
        }
        // Clear output arrays.
        for (int oc = 0; oc < MAX_CH; oc++) {
            for (int r = 0; r < MAX_OUT_H; r++) {
                for (int c = 0; c < MAX_OUT_W; c++) {
                    out_ref[oc][r][c] = 0.0f;
                    out_tile[oc][r][c] = 0.0f;
                }
            }
        }

        // Compute reference output.
        conv_ref(inC, outC, H, W, input, weight, bias, out_ref, stride, pad);
        // Compute tiled output.
        conv_via_tiling(inC, outC, H, W, input, weight, bias, out_tile, stride, pad);

        // Compare the outputs.
        double max_diff = 0.0;
        for (int oc = 0; oc < outC; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    double diff = std::fabs(out_ref[oc][oh][ow] - out_tile[oc][oh][ow]);
                    if (diff > max_diff)
                        max_diff = diff;
                }
            }
        }
        std::cout << "  Output shape: " << out_h << "x" << out_w
                  << ", max diff = " << max_diff << "\n";
        std::cout << "  Output ch0, top-left 4x4 => ref | tile\n";
        for (int r = 0; r < std::min(4, out_h); r++) {
            for (int c = 0; c < std::min(4, out_w); c++) {
                std::cout << out_ref[0][r][c] << " ";
            }
            std::cout << " | ";
            for (int c = 0; c < std::min(4, out_w); c++) {
                std::cout << out_tile[0][r][c] << " ";
            }
            std::cout << "\n";
        }
    };

    // -------------------------------------------------------------------------
    // Run test cases.
    // -------------------------------------------------------------------------
    runTest(64, 64, 14, 14, 1, 1);
    runTest(128, 128, 7, 7, 1, 1);
    runTest(128, 128, 14, 14, 1, 1);
    runTest(64, 128, 10, 10, 2, 2);
    runTest(128, 128, 28, 28, 2, 1);
    runTest(256, 256, 14, 14, 1, 1);
    runTest(128, 128, 28, 14, 3, 2);
    runTest(64, 128, 14, 14, 1, 0);
    runTest(128, 64, 14, 14, 1, 0);

    return 0;
}
