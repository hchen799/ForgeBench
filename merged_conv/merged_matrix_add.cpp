#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>

// -----------------------------------------------------------------------------
// Global Definitions and Constants
// -----------------------------------------------------------------------------
#define MAX_C 256   // Maximum channels
#define MAX_H 256   // Maximum height
#define MAX_W 256   // Maximum width

// Tiling parameters for matrix add.
#define BLOCK_C 64  // Channel tile size
#define BLOCK_H 8   // Tile height
#define BLOCK_W 8   // Tile width

// -----------------------------------------------------------------------------
// Helper Function: Ceiling Division
// -----------------------------------------------------------------------------
static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// -----------------------------------------------------------------------------
// 1) Reference Matrix Add
//
//    Computes elementwise addition of in1 and in2 and writes the result to out.
// -----------------------------------------------------------------------------
void matrix_add_ref(
    int C, int H, int W,
    float in1[MAX_C][MAX_H][MAX_W],
    float in2[MAX_C][MAX_H][MAX_W],
    float out[MAX_C][MAX_H][MAX_W]
)
{
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                out[c][i][j] = in1[c][i][j] + in2[c][i][j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2) Matrix Add Kernel (Fixed-Size Tile)
//
//    Processes a tile of size up to BLOCK_C x BLOCK_H x BLOCK_W.
//    Only the valid region given by tile_C, tile_H, tile_W is processed.
// -----------------------------------------------------------------------------
void matrix_add_kernel(
    float localIn1[BLOCK_C][BLOCK_H][BLOCK_W],
    float localIn2[BLOCK_C][BLOCK_H][BLOCK_W],
    float localOut[BLOCK_C][BLOCK_H][BLOCK_W],
    int tile_C, int tile_H, int tile_W
)
{
    for (int c = 0; c < tile_C; c++) {
        for (int i = 0; i < tile_H; i++) {
            for (int j = 0; j < tile_W; j++) {
                localOut[c][i][j] = localIn1[c][i][j] + localIn2[c][i][j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3) Tiled Matrix Add
//
//    Divides the input tensors into tiles and calls matrix_add_kernel for each tile.
// -----------------------------------------------------------------------------
void matrix_add_tiled(
    int C, int H, int W,
    float in1[MAX_C][MAX_H][MAX_W],
    float in2[MAX_C][MAX_H][MAX_W],
    float out[MAX_C][MAX_H][MAX_W]
)
{
    int channel_tiles = ceil_div(C, BLOCK_C);
    int tile_rows = ceil_div(H, BLOCK_H);
    int tile_cols = ceil_div(W, BLOCK_W);

    // Loop over channel tiles.
    for (int ct = 0; ct < channel_tiles; ct++) {
        int c_offset = ct * BLOCK_C;
        int tile_C = std::min(BLOCK_C, C - c_offset);

        // Loop over spatial tiles.
        for (int tr = 0; tr < tile_rows; tr++) {
            int h_offset = tr * BLOCK_H;
            int tile_H = std::min(BLOCK_H, H - h_offset);
            for (int tc = 0; tc < tile_cols; tc++) {
                int w_offset = tc * BLOCK_W;
                int tile_W = std::min(BLOCK_W, W - w_offset);

                // Local buffers for the current tile.
                float localIn1[BLOCK_C][BLOCK_H][BLOCK_W];
                float localIn2[BLOCK_C][BLOCK_H][BLOCK_W];
                float localOut[BLOCK_C][BLOCK_H][BLOCK_W];

                // Copy global data into local buffers.
                for (int c = 0; c < tile_C; c++) {
                    for (int i = 0; i < tile_H; i++) {
                        for (int j = 0; j < tile_W; j++) {
                            localIn1[c][i][j] = in1[c_offset + c][h_offset + i][w_offset + j];
                            localIn2[c][i][j] = in2[c_offset + c][h_offset + i][w_offset + j];
                        }
                    }
                }

                // Call the fixed-size kernel for this tile.
                matrix_add_kernel(localIn1, localIn2, localOut, tile_C, tile_H, tile_W);

                // Write the computed local output back to the global output.
                for (int c = 0; c < tile_C; c++) {
                    for (int i = 0; i < tile_H; i++) {
                        for (int j = 0; j < tile_W; j++) {
                            out[c_offset + c][h_offset + i][w_offset + j] = localOut[c][i][j];
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
//    Initializes two input matrices with random values, computes matrix addition
//    using both the reference and tiled versions, and compares the outputs.
// -----------------------------------------------------------------------------
void runRandomTest(int C, int H, int W)
{
    static float in1[MAX_C][MAX_H][MAX_W];
    static float in2[MAX_C][MAX_H][MAX_W];
    static float out_ref[MAX_C][MAX_H][MAX_W];
    static float out_tiled[MAX_C][MAX_H][MAX_W];

    // Initialize in1 and in2 with random values in range [-10, 10].
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                in1[c][i][j] = (rand() % 2001 - 1000) / 100.0f;  // [-10.0, 10.0]
                in2[c][i][j] = (rand() % 2001 - 1000) / 100.0f;  // [-10.0, 10.0]
            }
        }
    }

    // Compute the reference matrix addition.
    matrix_add_ref(C, H, W, in1, in2, out_ref);
    // Compute the tiled matrix addition.
    matrix_add_tiled(C, H, W, in1, in2, out_tiled);

    // Validate: compute the maximum absolute difference between reference and tiled outputs.
    float max_diff = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                float diff = std::fabs(out_ref[c][i][j] - out_tiled[c][i][j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }
    }
    std::cout << "\n=== Random Input Tiled Matrix Add Test ===\n";
    std::cout << "Max difference between reference and tiled: " << max_diff << "\n";

    // Print a 4x4 region from channel 0 (top-left) for inspection.
    std::cout << "Output channel 0 (first 4x4 region):\n";
    for (int i = 0; i < std::min(4, H); i++) {
        for (int j = 0; j < std::min(4, W); j++) {
            std::cout << out_tiled[0][i][j] << " ";
        }
        std::cout << "\n";
    }
}

// -----------------------------------------------------------------------------
// Main: Entry Point
// -----------------------------------------------------------------------------
int main()
{
    srand(static_cast<unsigned int>(time(NULL)));

    runRandomTest(64, 14, 14);
    runRandomTest(128, 28, 28);
    runRandomTest(256, 32, 32);

    return 0;
}
