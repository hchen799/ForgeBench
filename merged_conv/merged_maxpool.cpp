#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>

// -----------------------------------------------------------------------------
// Definitions and Constants
// -----------------------------------------------------------------------------
#define MAX_C     256   // Maximum channels
#define MAX_H_IN  256   // Maximum input height
#define MAX_W_IN  256   // Maximum input width

// Pooling parameters.
#define K_H       2     // Kernel height
#define K_W       2     // Kernel width
#define STRIDE_H  2     // Stride in height
#define STRIDE_W  2     // Stride in width

// Tiling parameters.
#define BLOCK_C   64    // Channel tile size
#define BLOCK_H   8     // Output tile height
#define BLOCK_W   8     // Output tile width

// For the tiled maxpool kernel, we need a local input buffer large enough to
// cover a tile of outputs. Its dimensions are computed as:
#define MAX_LOCAL_IN_H ((BLOCK_H - 1) * STRIDE_H + K_H)
#define MAX_LOCAL_IN_W ((BLOCK_W - 1) * STRIDE_W + K_W)

// -----------------------------------------------------------------------------
// Helper Function: Ceiling Division
// -----------------------------------------------------------------------------
static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// -----------------------------------------------------------------------------
// 1) Reference MaxPool
//
//    Given input of dimensions [C][H_IN][W_IN], the output is computed as:
//      For each channel c and each output cell (i, j):
//         output[c][i][j] = max_{kh,kw in window} input[c][i*STRIDE_H + kh][j*STRIDE_W + kw]
//    where the window dimensions are: K_H x K_W.
//    The output dimensions are assumed to be:
//         H_OUT = (H_IN - K_H) / STRIDE_H + 1
//         W_OUT = (W_IN - K_W) / STRIDE_W + 1
// -----------------------------------------------------------------------------
void maxpool_ref(
    int C, int H_IN, int W_IN,
    float input[MAX_C][MAX_H_IN][MAX_W_IN],
    int H_OUT, int W_OUT,
    float output[MAX_C][MAX_H_IN][MAX_W_IN]   // Use first H_OUT x W_OUT region
)
{
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H_OUT; i++) {
            for (int j = 0; j < W_OUT; j++) {
                // Initialize max value from the top-left corner of the window.
                float max_val = input[c][i * STRIDE_H][j * STRIDE_W];
                for (int kh = 0; kh < K_H; kh++) {
                    for (int kw = 0; kw < K_W; kw++) {
                        int row = i * STRIDE_H + kh;
                        int col = j * STRIDE_W + kw;
                        if (input[c][row][col] > max_val) {
                            max_val = input[c][row][col];
                        }
                    }
                }
                output[c][i][j] = max_val;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2) MaxPool Kernel (Fixed-Size Tile)
//
//    Processes a tile of output with dimensions up to BLOCK_C x BLOCK_H x BLOCK_W.
//    The corresponding local input tile has dimensions:
//         localInH = (tile_H - 1)*STRIDE_H + K_H,
//         localInW = (tile_W - 1)*STRIDE_W + K_W.
// -----------------------------------------------------------------------------
void maxpool_kernel(
    float localInput[BLOCK_C][MAX_LOCAL_IN_H][MAX_LOCAL_IN_W],
    float localOutput[BLOCK_C][BLOCK_H][BLOCK_W],
    int tile_C, int tile_H, int tile_W
)
{
    for (int c = 0; c < tile_C; c++) {
        for (int i = 0; i < tile_H; i++) {
            for (int j = 0; j < tile_W; j++) {
                // Compute pooling over the window.
                float max_val = localInput[c][i * STRIDE_H][j * STRIDE_W];
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
    float input[MAX_C][MAX_H_IN][MAX_W_IN],
    int H_OUT, int W_OUT,
    float output[MAX_C][MAX_H_IN][MAX_W_IN]   // Output region [C][H_OUT][W_OUT]
)
{
    // Divide channels and spatial dimensions (of the output) into tiles.
    int channel_tiles = ceil_div(C, BLOCK_C);
    int tile_rows = ceil_div(H_OUT, BLOCK_H);
    int tile_cols = ceil_div(W_OUT, BLOCK_W);

    // Loop over channel tiles.
    for (int ct = 0; ct < channel_tiles; ct++) {
        int c_offset = ct * BLOCK_C;
        int tile_C = std::min(BLOCK_C, C - c_offset);

        // Loop over spatial tiles.
        for (int tr = 0; tr < tile_rows; tr++) {
            int out_row_offset = tr * BLOCK_H;
            int tile_H = std::min(BLOCK_H, H_OUT - out_row_offset);
            // Compute corresponding input tile height.
            int localInH = (tile_H - 1) * STRIDE_H + K_H;

            for (int tc = 0; tc < tile_cols; tc++) {
                int out_col_offset = tc * BLOCK_W;
                int tile_W = std::min(BLOCK_W, W_OUT - out_col_offset);
                // Compute corresponding input tile width.
                int localInW = (tile_W - 1) * STRIDE_W + K_W;

                // Local buffers.
                float localInput[BLOCK_C][MAX_LOCAL_IN_H][MAX_LOCAL_IN_W] = {{{0}}};
                float localOutput[BLOCK_C][BLOCK_H][BLOCK_W] = {{{0}}};

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

// -----------------------------------------------------------------------------
// 4) Test Driver: runRandomTest
//
//    Initializes the input with random values, computes the maxpool output using
//    both the reference and tiled implementations, and compares the results.
// -----------------------------------------------------------------------------
void runRandomTest(int C, int H_IN, int W_IN)
{
    // Compute output dimensions.
    int H_OUT = (H_IN - K_H) / STRIDE_H + 1;
    int W_OUT = (W_IN - K_W) / STRIDE_W + 1;

    static float input[MAX_C][MAX_H_IN][MAX_W_IN];
    static float output_ref[MAX_C][MAX_H_IN][MAX_W_IN];
    static float output_tiled[MAX_C][MAX_H_IN][MAX_W_IN];

    // Initialize input with random values in the range [-5.0, 5.0].
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < H_IN; r++) {
            for (int col = 0; col < W_IN; col++) {
                input[c][r][col] = (rand() % 1001 - 500) / 100.0f;
            }
        }
    }

    // Compute the reference maxpool output.
    maxpool_ref(C, H_IN, W_IN, input, H_OUT, W_OUT, output_ref);
    // Compute the tiled maxpool output.
    maxpool_tiled(C, H_IN, W_IN, input, H_OUT, W_OUT, output_tiled);

    // Validate: compute the maximum absolute difference.
    float max_diff = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H_OUT; i++) {
            for (int j = 0; j < W_OUT; j++) {
                float diff = std::fabs(output_ref[c][i][j] - output_tiled[c][i][j]);
                if (diff > max_diff)
                    max_diff = diff;
            }
        }
    }

    std::cout << "\n=== Random Input Tiled MaxPool Test ===\n";
    std::cout << "Input dims: " << H_IN << "x" << W_IN << ", Output dims: " << H_OUT << "x" << W_OUT << "\n";
    std::cout << "Max difference between reference and tiled maxpool: " << max_diff << "\n";

    // Print a 4x4 region from channel 0 (top-left corner) of the tiled output.
    std::cout << "Output channel 0 (first 4x4 region):\n";
    for (int i = 0; i < std::min(4, H_OUT); i++) {
        for (int j = 0; j < std::min(4, W_OUT); j++) {
            std::cout << output_tiled[0][i][j] << " ";
        }
        std::cout << "\n";
    }
}

// -----------------------------------------------------------------------------
// Main: Entry Point
// -----------------------------------------------------------------------------
int main()
{
    srand((unsigned)time(NULL));

    runRandomTest(64, 14, 14);
    runRandomTest(128, 28, 28);
    runRandomTest(256, 32, 32);

    return 0;
}
