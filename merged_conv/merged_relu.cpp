#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>

// -----------------------------------------------------------------------------
// Definitions and Constants
// -----------------------------------------------------------------------------
#define MAX_C 256   // Maximum channels
#define MAX_H 256   // Maximum height
#define MAX_W 256   // Maximum width

#define BLOCK_C 64  // Channel tile size
#define BLOCK_H 8   // Spatial tile height
#define BLOCK_W 8   // Spatial tile width

// -----------------------------------------------------------------------------
// Helper Function: Ceiling Division
// -----------------------------------------------------------------------------
static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// -----------------------------------------------------------------------------
// 1) Reference ReLU
//
//    Applies ReLU on the input for each channel and spatial location:
//      output = (input > 0) ? input : 0
// -----------------------------------------------------------------------------
void relu_ref(
    int C, int H, int W,
    float input[MAX_C][MAX_H][MAX_W],
    float output[MAX_C][MAX_H][MAX_W]
)
{
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                output[c][i][j] = (input[c][i][j] > 0) ? input[c][i][j] : 0.0f;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2) ReLU Kernel (Fixed-Size Tile)
//
//    Processes a tile of dimensions up to BLOCK_C x BLOCK_H x BLOCK_W.
//    Only the valid portion (tile_C, tile_H, tile_W) is processed.
// -----------------------------------------------------------------------------
void relu_kernel(
    float localInput[BLOCK_C][BLOCK_H][BLOCK_W],
    float localOutput[BLOCK_C][BLOCK_H][BLOCK_W],
    int tile_C, int tile_H, int tile_W
)
{
    for (int c = 0; c < tile_C; c++) {
        for (int h = 0; h < tile_H; h++) {
            for (int w = 0; w < tile_W; w++) {
                localOutput[c][h][w] = (localInput[c][h][w] > 0) ? localInput[c][h][w] : 0.0f;
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
    float input[MAX_C][MAX_H][MAX_W],
    float output[MAX_C][MAX_H][MAX_W]
)
{
    int channel_tiles = ceil_div(C, BLOCK_C);
    int tile_rows     = ceil_div(H, BLOCK_H);
    int tile_cols     = ceil_div(W, BLOCK_W);

    // Loop over channel tiles.
    for (int ct = 0; ct < channel_tiles; ct++) {
        int c_offset = ct * BLOCK_C;
        int tile_C = std::min(BLOCK_C, C - c_offset);

        // Loop over spatial tiles (rows and columns).
        for (int tr = 0; tr < tile_rows; tr++) {
            int h_offset = tr * BLOCK_H;
            int tile_H = std::min(BLOCK_H, H - h_offset);
            for (int tc = 0; tc < tile_cols; tc++) {
                int w_offset = tc * BLOCK_W;
                int tile_W = std::min(BLOCK_W, W - w_offset);

                // Local buffers for input and output tile.
                float localInput[BLOCK_C][BLOCK_H][BLOCK_W] = {{{0}}};
                float localOutput[BLOCK_C][BLOCK_H][BLOCK_W] = {{{0}}};

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

// -----------------------------------------------------------------------------
// 4) Test Driver: runRandomTest
//
//    Initializes the input with random values, computes the ReLU using both
//    the reference and tiled versions, and compares the outputs.
// -----------------------------------------------------------------------------
void runRandomTest(int C, int H, int W)
{
    static float input[MAX_C][MAX_H][MAX_W];
    static float output_ref[MAX_C][MAX_H][MAX_W];
    static float output_tiled[MAX_C][MAX_H][MAX_W];

    // Initialize input with random values in range [-5.0, 5.0].
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input[c][h][w] = (rand() % 1001 - 500) / 100.0f;
            }
        }
    }

    // Compute the reference ReLU.
    relu_ref(C, H, W, input, output_ref);
    // Compute the tiled ReLU.
    relu_tiled(C, H, W, input, output_tiled);

    // Validate: compute maximum absolute difference between the two outputs.
    float max_diff = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float diff = std::fabs(output_ref[c][h][w] - output_tiled[c][h][w]);
                if (diff > max_diff)
                    max_diff = diff;
            }
        }
    }

    std::cout << "\n=== Random Input Tiled ReLU Test ===\n";
    std::cout << "Max difference between reference and tiled ReLU: " << max_diff << "\n";

    // Print a 4x4 region from channel 0 (top-left corner) for inspection.
    std::cout << "Output channel 0 (first 4x4 region):\n";
    for (int h = 0; h < std::min(4, H); h++) {
        for (int w = 0; w < std::min(4, W); w++) {
            std::cout << output_tiled[0][h][w] << " ";
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
