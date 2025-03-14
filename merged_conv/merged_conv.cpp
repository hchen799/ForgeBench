#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

//========================================================
// 1) Base Block Function: conv3x3_block_64_7x7
//    - Exactly 64 in-ch, 64 out-ch, 7×7 output
//    - Because we do pad=1, stride=1, the local input
//      tile is 9×9 (7 + 2 for the padding).
//    - If "firstInputTile" is true, we initialize
//      output with bias. Otherwise we accumulate partial
//      sums onto existing output values.
//========================================================
static const int BLOCK_IN_CH  = 64;
static const int BLOCK_OUT_CH = 64;
static const int BLOCK_H      = 7;
static const int BLOCK_W      = 7;

void conv3x3_block_64_7x7(
    float inData[BLOCK_IN_CH][BLOCK_H+2][BLOCK_W+2],  // [64][9][9]
    float weight[BLOCK_OUT_CH][BLOCK_IN_CH][3][3],    // [64][64][3][3]
    float bias[BLOCK_OUT_CH],                         // [64]
    float outData[BLOCK_OUT_CH][BLOCK_H][BLOCK_W],    // [64][7][7]
    bool firstInputTile
)
{
    // If this is the first input‑channel tile, initialize output with bias;
    // otherwise, we keep partial sums from a previous pass.
    if (firstInputTile) {
        for (int oc = 0; oc < BLOCK_OUT_CH; oc++) {
            for (int oh = 0; oh < BLOCK_H; oh++) {
                for (int ow = 0; ow < BLOCK_W; ow++) {
                    outData[oc][oh][ow] = bias[oc];
                }
            }
        }
    }

    // Accumulate over these 64 input channels
    for (int ic = 0; ic < BLOCK_IN_CH; ic++) {
        for (int oc = 0; oc < BLOCK_OUT_CH; oc++) {
            for (int oh = 0; oh < BLOCK_H; oh++) {
                for (int ow = 0; ow < BLOCK_W; ow++) {
                    float sum_val = 0.0f;
                    // 3×3 kernel
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            // Because pad=1, the (oh,ow) in output maps to (oh+1, ow+1) in inData
                            int in_r = oh + kh;
                            int in_c = ow + kw;
                            sum_val += inData[ic][in_r][in_c] * weight[oc][ic][kh][kw];
                        }
                    }
                    outData[oc][oh][ow] += sum_val;
                }
            }
        }
    }
}

//========================================================
// 2) Reference Convolution (Direct), up to 128×14×14
//    - "same" convolution => pad=1, stride=1
//    - No tiling. Just straightforward loops.
//========================================================
void conv3x3_ref(
    const int in_ch,
    const int out_ch,
    const int H,
    const int W,
    float input[128][14][14],         // big enough for max
    float weight[128][128][3][3],     // big enough for max
    float bias[128],
    float output[128][14][14]
)
{
    // For "same" conv (pad=1, stride=1), output dims = H×W
    // Initialize output with bias
    for (int oc = 0; oc < out_ch; oc++) {
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                output[oc][r][c] = bias[oc];
            }
        }
    }

    // Accumulate
    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    float sum_val = 0.0f;
                    // Map output -> input top-left corner
                    int in_r0 = r - 1; // pad=1
                    int in_c0 = c - 1;
                    for (int kr = 0; kr < 3; kr++) {
                        for (int kc = 0; kc < 3; kc++) {
                            int in_r = in_r0 + kr;
                            int in_c = in_c0 + kc;
                            if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W) {
                                sum_val += input[ic][in_r][in_c] * weight[oc][ic][kr][kc];
                            }
                        }
                    }
                    output[oc][r][c] += sum_val;
                }
            }
        }
    }
}

//========================================================
// 3) Orchestrator: conv3x3_via_block_64_7x7
//    - Can handle in_ch, out_ch ∈ {64,128} and H,W ∈ {7,14}.
//    - Splits input channels in chunks of 64 (if needed).
//    - Splits output channels in chunks of 64 (if needed).
//    - Splits the spatial dimension into 7×7 tiles if H=14 or W=14.
//    - Calls conv3x3_block_64_7x7 to accumulate partial sums.
//
//    For each sub-tile, we gather a local [64][9][9] input region,
//    a local [64][64][3][3] weight, partial or fresh output, etc.
//========================================================
void conv3x3_via_block_64_7x7(
    const int in_ch,
    const int out_ch,
    const int H,
    const int W,
    float input[128][14][14],
    float weight[128][128][3][3],
    float bias[128],
    float output[128][14][14]
)
{
    // Clear the output array to 0 (we will add bias only in the block calls).
    for (int oc = 0; oc < out_ch; oc++) {
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                output[oc][r][c] = 0.0f;
            }
        }
    }

    // Number of input-channel tiles
    int in_ch_tiles = (in_ch + 63) / 64;   // 1 or 2
    // Number of output-channel tiles
    int out_ch_tiles = (out_ch + 63) / 64; // 1 or 2

    // Number of vertical tiles in the spatial dimension
    // (7 or 14 => 14/7=2 tiles, 7/7=1 tile)
    int tile_rows = (H + 6) / 7;  // if H=14 => 2, if H=7 => 1
    int tile_cols = (W + 6) / 7;

    // For each tile in output channels
    for (int oct = 0; oct < out_ch_tiles; oct++) {
        int oc_offset = oct * 64;  // 0 or 64

        // For each tile in input channels
        for (int ict = 0; ict < in_ch_tiles; ict++) {
            int ic_offset = ict * 64;  // 0 or 64

            // For each tile in the spatial dimension
            for (int tr = 0; tr < tile_rows; tr++) {
                for (int tc = 0; tc < tile_cols; tc++) {

                    // The actual tile height/width might be 7 or less
                    // but in this problem it's always exactly 7
                    // if within range, because H is 7 or 14
                    int tileH = 7; 
                    int tileW = 7;

                    int out_row0 = tr * 7;  // top-left row of this tile in the big output
                    int out_col0 = tc * 7;  // top-left col

                    // local arrays
                    static float localIn[64][9][9];
                    static float localW[64][64][3][3];
                    static float localOut[64][7][7];
                    static float localBias[64];

                    // 1) Copy the relevant weights:
                    //    64 out-ch from [oc_offset..oc_offset+63]
                    //    64 in-ch from [ic_offset..ic_offset+63]
                    //    (bounded by out_ch, in_ch of course)
                    int cur_out_ch = ((oc_offset + 64) > out_ch) ? (out_ch - oc_offset) : 64;
                    int cur_in_ch  = ((ic_offset + 64) > in_ch)  ? (in_ch - ic_offset) : 64;

                    // Initialize localW to 0 just in case some out-of-range channels
                    for (int o = 0; o < 64; o++) {
                        for (int i = 0; i < 64; i++) {
                            for (int kr = 0; kr < 3; kr++) {
                                for (int kc = 0; kc < 3; kc++) {
                                    localW[o][i][kr][kc] = 0.0f;
                                }
                            }
                        }
                    }
                    // Fill the valid portion
                    for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                        for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
                            for (int kr = 0; kr < 3; kr++) {
                                for (int kc = 0; kc < 3; kc++) {
                                    localW[oc2][ic2][kr][kc] =
                                        weight[oc_offset + oc2][ic_offset + ic2][kr][kc];
                                }
                            }
                        }
                    }

                    // 2) Copy the relevant input tile region into localIn
                    //    with size [cur_in_ch][9][9], plus zero for out-of-range
                    //    pad=1 => we copy 7 + 2 = 9 in each dimension
                    //    The top-left in the big input is (out_row0 - 1, out_col0 - 1).

                    for (int ic2 = 0; ic2 < 64; ic2++) {
                        // set default 0
                        for (int rr = 0; rr < 9; rr++) {
                            for (int cc = 0; cc < 9; cc++) {
                                localIn[ic2][rr][cc] = 0.0f;
                            }
                        }
                    }
                    for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
                        for (int rr = 0; rr < 9; rr++) {
                            for (int cc = 0; cc < 9; cc++) {
                                int in_r = (out_row0 - 1) + rr; 
                                int in_c = (out_col0 - 1) + cc;
                                if (in_r >= 0 && in_r < H &&
                                    in_c >= 0 && in_c < W) {
                                    localIn[ic2][rr][cc] = input[ic_offset + ic2][in_r][in_c];
                                }
                            }
                        }
                    }

                    // 3) Copy bias
                    //    If it's the first input-tile for this out-ch tile, we need the real bias;
                    //    otherwise, 0 (so we can accumulate partial sums).
                    bool firstInputTile = (ict == 0); // only if we're in the first input-ch tile
                    for (int oc2 = 0; oc2 < 64; oc2++) {
                        if (oc2 < cur_out_ch) {
                            localBias[oc2] = bias[oc_offset + oc2];
                        } else {
                            localBias[oc2] = 0.0f;
                        }
                    }

                    // 4) Prepare localOut: if not firstInputTile, we copy partial sums
                    //    from the big output array. Otherwise, we let the block function
                    //    do "out = bias + sums".
                    if (firstInputTile) {
                        // zero out localOut
                        for (int o = 0; o < 64; o++) {
                            for (int rr = 0; rr < 7; rr++) {
                                for (int cc = 0; cc < 7; cc++) {
                                    localOut[o][rr][cc] = 0.0f;
                                }
                            }
                        }
                    } else {
                        // copy partial sums from big output
                        for (int oc2 = 0; oc2 < 64; oc2++) {
                            for (int rr = 0; rr < tileH; rr++) {
                                for (int cc = 0; cc < tileW; cc++) {
                                    if (oc2 < cur_out_ch) {
                                        localOut[oc2][rr][cc] =
                                          output[oc_offset + oc2][out_row0 + rr][out_col0 + cc];
                                    } else {
                                        // out-of-range
                                        localOut[oc2][rr][cc] = 0.0f;
                                    }
                                }
                            }
                        }
                    }

                    // 5) Call the block
                    conv3x3_block_64_7x7(
                        localIn,
                        localW,
                        localBias,
                        localOut,
                        firstInputTile
                    );

                    // 6) Copy localOut back to the global output array
                    //    Only for the valid out channels (cur_out_ch) and valid tileH×tileW
                    for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                        for (int rr = 0; rr < tileH; rr++) {
                            for (int cc = 0; cc < tileW; cc++) {
                                output[oc_offset + oc2][out_row0 + rr][out_col0 + cc]
                                    = localOut[oc2][rr][cc];
                            }
                        }
                    }

                } // tile_cols
            } // tile_rows
        } // in_ch_tiles
    } // out_ch_tiles
}

//========================================================
// 4) Main: Validate 3 scenarios using the "block" approach
//    vs. the reference approach.
//
//    - (64, 64, 14×14)
//    - (128, 128, 7×7)
//    - (128, 128, 14×14)
//
//  (You can add more if needed.)
//========================================================
int main()
{
    srand((unsigned)time(NULL));

    // We'll hold data in arrays sized for up to 128×14×14:
    static float input[128][14][14];
    static float weight[128][128][3][3];
    static float bias[128];
    static float out_ref[128][14][14];
    static float out_block[128][14][14];

    //====================================================
    // Helper lambda to test a specific (in_ch, out_ch, H, W)
    //====================================================
    auto TestCase = [&](int inC, int outC, int H, int W)
    {
        // 1) Random initialization
        for (int ic = 0; ic < inC; ic++) {
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    input[ic][r][c] = (float)(rand() % 256 - 128) / 64.0f;
                }
            }
        }
        for (int oc = 0; oc < outC; oc++) {
            bias[oc] = (float)(rand() % 200 - 100) / 50.0f;
            for (int ic = 0; ic < inC; ic++) {
                for (int kr = 0; kr < 3; kr++) {
                    for (int kc = 0; kc < 3; kc++) {
                        weight[oc][ic][kr][kc] = (float)(rand() % 200 - 100) / 50.0f;
                    }
                }
            }
        }
        // If we have fewer than 128 channels, fill the remainder with 0
        // so we don't use uninitialized data
        for (int oc = outC; oc < 128; oc++) {
            bias[oc] = 0.0f;
            for (int ic = 0; ic < 128; ic++) {
                for (int kr = 0; kr < 3; kr++) {
                    for (int kc = 0; kc < 3; kc++) {
                        weight[oc][ic][kr][kc] = 0.0f;
                    }
                }
            }
        }
        for (int ic = inC; ic < 128; ic++) {
            for (int r = 0; r < 14; r++) {
                for (int c = 0; c < 14; c++) {
                    input[ic][r][c] = 0.0f;
                }
            }
        }

        // 2) Reference
        conv3x3_ref(inC, outC, H, W, input, weight, bias, out_ref);

        // 3) Block-based
        conv3x3_via_block_64_7x7(inC, outC, H, W, input, weight, bias, out_block);

        // 4) Compare
        double max_diff = 0.0;
        for (int oc = 0; oc < outC; oc++) {
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    double df = std::fabs(out_ref[oc][r][c] - out_block[oc][r][c]);
                    if (df > max_diff) {
                        max_diff = df;
                    }
                }
            }
        }

        // Print results
        std::cout << "Test (inC=" << inC << ", outC=" << outC
                  << ", H=" << H << ", W=" << W << "):\n";
        std::cout << "  Max difference = " << max_diff << std::endl;
        std::cout << "  Output[0][0..3,0..3] (ref vs. block):\n";
        for (int r = 0; r < 4 && r < H; r++) {
            for (int c = 0; c < 4 && c < W; c++) {
                std::cout << out_ref[0][r][c] << " ";
            }
            std::cout << "  |  ";
            for (int c = 0; c < 4 && c < W; c++) {
                std::cout << out_block[0][r][c] << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    };

    //====================================================
    // Now run the 3 requested testcases
    //====================================================
    // A) 64 in-ch, 64 out-ch, 14×14
    TestCase(64, 64, 14, 14);

    // B) 128 in-ch, 128 out-ch, 7×7
    TestCase(128, 128, 7, 7);

    // C) 128 in-ch, 128 out-ch, 14×14
    TestCase(128, 128, 14, 14);

    return 0;
}



