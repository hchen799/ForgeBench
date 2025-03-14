#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <vector>

// Maximum spatial dimension (enough for up to 14x14 = 196 elements)
#define MAX_SPATIAL 256
// Maximum channels supported (e.g., 256)
#define MAX_CH 256

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------
static int out_dim(int size, int pad, int stride, int kernel = 5) {
    // Standard convolution output dimension formula
    return (size + 2 * pad - kernel) / stride + 1;
}
static int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

// -----------------------------------------------------------------------------
// 1) Block Function: conv_kernel
//
//    Computes a 5x5 convolution on a block with:
//      - exactly BLOCK_IN_CH input channels and BLOCK_OUT_CH output channels,
//      - a fixed maximum output tile size of BLOCK_OUT_H x BLOCK_OUT_W,
//      - but only over a valid region (outTileH x outTileW).
//    Assumes that a local input patch of size (tileH-1)*stride+KSIZE has been built,
//    with any needed zero‑padding.
// -----------------------------------------------------------------------------

#define MAX_LOCAL_SIZE 30
static const int BLOCK_IN_CH  = 128;
static const int BLOCK_OUT_CH = 128;
static const int BLOCK_OUT_H  = 14; // maximum tile height
static const int BLOCK_OUT_W  = 14; // maximum tile width
static const int KSIZE        = 5;

void conv_kernel(
    float inData[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE],
    float weight[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE],
    float outData[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W],
    int outTileH, int outTileW,
    int localInH, int localInW,
    int stride
)
{
    // Loop over the valid output region.
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
// 2) Reference Convolution: conv_ref
//
//    Standard (non‑tiled) 5x5 convolution with given stride and pad.
//    Input and output are stored in flattened 1D arrays per channel.
// -----------------------------------------------------------------------------
void conv_ref(
    int in_ch,
    int out_ch,
    int H,
    int W,
    float input[MAX_CH][MAX_SPATIAL],
    float weight[MAX_CH][MAX_CH][KSIZE][KSIZE],
    float bias[MAX_CH],
    float output[MAX_CH][MAX_SPATIAL],
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
                    float sum_val = 0.0f;
                    int in_r0 = oh * stride - pad;
                    int in_c0 = ow * stride - pad;
                    for (int kh = 0; kh < KSIZE; kh++) {
                        for (int kw = 0; kw < KSIZE; kw++) {
                            int in_r = in_r0 + kh;
                            int in_c = in_c0 + kw;
                            if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W)
                                sum_val += input[ic][in_r * W + in_c] * weight[oc][ic][kh][kw];
                        }
                    }
                    output[oc][oh * out_w + ow] += sum_val;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3) Block-Based Orchestrator: conv_via_tiling
//
//    Computes the full convolution for arbitrary input/output channel counts and
//    spatial dimensions using the block function (conv_kernel). Supports a user‑specified
//    stride and pad.
// -----------------------------------------------------------------------------
void conv_via_tiling(
    int in_ch,
    int out_ch,
    int H,
    int W,
    float input[MAX_CH][MAX_SPATIAL],
    float weight[MAX_CH][MAX_CH][KSIZE][KSIZE],
    float bias[MAX_CH],
    float output[MAX_CH][MAX_SPATIAL],
    int stride,
    int pad
)
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

    // Loop over spatial tiles.
    for (int tr = 0; tr < tile_rows; tr++) {
        for (int tc = 0; tc < tile_cols; tc++) {
            int out_row0 = tr * BLOCK_OUT_H;
            int out_col0 = tc * BLOCK_OUT_W;
            int tileH = std::min(BLOCK_OUT_H, out_h - out_row0);
            int tileW = std::min(BLOCK_OUT_W, out_w - out_col0);
            // Compute the local input patch size for this tile.
            int localInH = (tileH - 1) * stride + KSIZE;
            int localInW = (tileW - 1) * stride + KSIZE;

            // Allocate local buffers on the heap for this tile.
            std::vector<float> localOutVec(BLOCK_OUT_CH * BLOCK_OUT_H * BLOCK_OUT_W, 0.f);
            std::vector<float> localWVec(BLOCK_OUT_CH * BLOCK_IN_CH * KSIZE * KSIZE, 0.f);
            std::vector<float> localInVec(BLOCK_IN_CH * MAX_LOCAL_SIZE * MAX_LOCAL_SIZE, 0.f);

            // Lambda functions to index the 3D local buffers.
            auto localOut = [&](int oc, int r, int c) -> float& {
                return localOutVec[oc * (BLOCK_OUT_H * BLOCK_OUT_W) + r * BLOCK_OUT_W + c];
            };
            auto localW = [&](int oc, int ic, int kr, int kc) -> float& {
                return localWVec[oc * (BLOCK_IN_CH * KSIZE * KSIZE) + ic * (KSIZE * KSIZE) + kr * KSIZE + kc];
            };
            auto localIn = [&](int ic, int r, int c) -> float& {
                return localInVec[ic * (MAX_LOCAL_SIZE * MAX_LOCAL_SIZE) + r * MAX_LOCAL_SIZE + c];
            };

            // Loop over output-channel tiles.
            for (int oct = 0; oct < out_ch_tiles; oct++) {
                int oc_offset = oct * BLOCK_OUT_CH;
                int cur_out_ch = (oc_offset + BLOCK_OUT_CH <= out_ch) ? BLOCK_OUT_CH : (out_ch - oc_offset);
                // Copy the current partial sums for this spatial tile into localOut.
                for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                    for (int i = 0; i < tileH * tileW; i++) {
                        int r_local = i / tileW;
                        int c_local = i % tileW;
                        localOut(oc2, r_local, c_local) = output[oc_offset + oc2][(out_row0 + r_local) * out_w + (out_col0 + c_local)];
                    }
                    // Clear remaining parts.
                    for (int r = tileH; r < BLOCK_OUT_H; r++)
                        for (int c = 0; c < BLOCK_OUT_W; c++)
                            localOut(oc2, r, c) = 0.f;
                    for (int r = 0; r < tileH; r++)
                        for (int c = tileW; c < BLOCK_OUT_W; c++)
                            localOut(oc2, r, c) = 0.f;
                }
                for (int oc2 = cur_out_ch; oc2 < BLOCK_OUT_CH; oc2++)
                    for (int r = 0; r < BLOCK_OUT_H; r++)
                        for (int c = 0; c < BLOCK_OUT_W; c++)
                            localOut(oc2, r, c) = 0.f;

                // Loop over input-channel tiles.
                for (int ict = 0; ict < in_ch_tiles; ict++) {
                    int ic_offset = ict * BLOCK_IN_CH;
                    int cur_in_ch = (ic_offset + BLOCK_IN_CH <= in_ch) ? BLOCK_IN_CH : (in_ch - ic_offset);
                    // Build localW for the current slice.
                    for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                        for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
                            for (int kr = 0; kr < KSIZE; kr++) {
                                for (int kc = 0; kc < KSIZE; kc++) {
                                    localW(oc2, ic2, kr, kc) = weight[oc_offset + oc2][ic_offset + ic2][kr][kc];
                                }
                            }
                        }
                        for (int ic2 = cur_in_ch; ic2 < BLOCK_IN_CH; ic2++)
                            for (int kr = 0; kr < KSIZE; kr++)
                                for (int kc = 0; kc < KSIZE; kc++)
                                    localW(oc2, ic2, kr, kc) = 0.f;
                    }
                    for (int oc2 = cur_out_ch; oc2 < BLOCK_OUT_CH; oc2++)
                        for (int ic2 = 0; ic2 < BLOCK_IN_CH; ic2++)
                            for (int kr = 0; kr < KSIZE; kr++)
                                for (int kc = 0; kc < KSIZE; kc++)
                                    localW(oc2, ic2, kr, kc) = 0.f;
                    
                    // Build localIn for the current input-channel slice.
                    for (int ic2 = 0; ic2 < BLOCK_IN_CH; ic2++) {
                        for (int rr = 0; rr < localInH; rr++)
                            for (int cc = 0; cc < localInW; cc++)
                                localIn(ic2, rr, cc) = 0.f;
                    }
                    for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
                        for (int rr = 0; rr < localInH; rr++) {
                            for (int cc = 0; cc < localInW; cc++) {
                                int global_r = out_row0 * stride - pad + rr;
                                int global_c = out_col0 * stride - pad + cc;
                                if (global_r >= 0 && global_r < H && global_c >= 0 && global_c < W)
                                    localIn(ic2, rr, cc) = input[ic_offset + ic2][global_r * W + global_c];
                            }
                        }
                    }
                    // Call the block kernel over the valid tile region.
                    conv_kernel(
                        reinterpret_cast<float (*)[MAX_LOCAL_SIZE][MAX_LOCAL_SIZE]>(localInVec.data()),
                        reinterpret_cast<float (*)[BLOCK_IN_CH][KSIZE][KSIZE]>(localWVec.data()),
                        reinterpret_cast<float (*)[BLOCK_OUT_H][BLOCK_OUT_W]>(localOutVec.data()),
                        tileH, tileW, localInH, localInW, stride
                    );
                } // end input-channel tile loop

                // Write the updated localOut back to the global output.
                for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                    for (int r = 0; r < tileH; r++) {
                        for (int c = 0; c < tileW; c++) {
                            output[oc_offset + oc2][(out_row0 + r) * out_w + (out_col0 + c)] = localOut(oc2, r, c);
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
int main() {
    srand((unsigned)time(NULL));

    static float input[MAX_CH][MAX_SPATIAL];
    static float weight[MAX_CH][MAX_CH][KSIZE][KSIZE];
    static float bias[MAX_CH];
    static float out_ref[MAX_CH][MAX_SPATIAL];
    static float out_block[MAX_CH][MAX_SPATIAL];

    auto runTest = [&](int inC, int outC, int H, int W, int stride, int pad) {
        std::cout << "\n=== Test inC=" << inC << ", outC=" << outC
                  << ", H=" << H << ", W=" << W
                  << ", stride=" << stride << ", pad=" << pad << " ===\n";
        int out_h = out_dim(H, pad, stride, KSIZE);
        int out_w = out_dim(W, pad, stride, KSIZE);
        if (out_h <= 0 || out_w <= 0) {
            std::cout << "  [Invalid output size => skipping]\n";
            return;
        }
        // Initialize input.
        for (int ic = 0; ic < inC; ic++) {
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    input[ic][r * W + c] = float(rand() % 201 - 100) / 25.f;
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
        for (int ic = inC; ic < MAX_CH; ic++) {
            for (int i = 0; i < MAX_SPATIAL; i++) {
                input[ic][i] = 0.f;
            }
        }
        for (int oc = outC; oc < MAX_CH; oc++) {
            bias[oc] = 0.f;
            for (int ic = 0; ic < MAX_CH; ic++) {
                for (int kr = 0; kr < KSIZE; kr++) {
                    for (int kc = 0; kc < KSIZE; kc++) {
                        weight[oc][ic][kr][kc] = 0.f;
                    }
                }
            }
        }

        conv_ref(inC, outC, H, W, input, weight, bias, out_ref, stride, pad);
        conv_via_tiling(inC, outC, H, W, input, weight, bias, out_block, stride, pad);

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

    // Run test cases.
    runTest(128, 128, 14, 14, 1, 1);
    runTest(128, 128, 28, 28, 3, 2); // expected output 10x10
    runTest(256, 256, 14, 14, 1, 1);

    return 0;
}
