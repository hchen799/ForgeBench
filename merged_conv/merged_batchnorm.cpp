#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>

// -----------------------------------------------------------------------------
// Definitions and Constants
// -----------------------------------------------------------------------------
#define MAX_C 256   // Maximum channels
#define MAX_H 256   // Maximum height
#define MAX_W 256   // Maximum width

// Tiling parameters for batch norm.
#define BLOCK_C 64  // Channel tile size
#define BLOCK_H 8   // Spatial tile height
#define BLOCK_W 8   // Spatial tile width

#define EPSILON 1e-5f

// Helper function: Ceiling division.
static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// -----------------------------------------------------------------------------
// 1) Reference Batch Normalization (non‑tiled)
//
//    Computes batch normalization for each element:
//      norm = (input - mean) / sqrt(variance + EPSILON)
//      output = gamma * norm + beta
//
//    We assume input and output are 3D arrays with dimensions [C][H][W].
//    weights[0] = gamma, weights[1] = beta, weights[2] = mean, weights[3] = variance.
// -----------------------------------------------------------------------------
void batch_norm_ref(
    int C, int H, int W,
    float input[MAX_C][MAX_H][MAX_W],
    float weights[4][MAX_C],
    float output[MAX_C][MAX_H][MAX_W]
) {
    for (int c = 0; c < C; c++) {
        // Precompute the denominator for the channel.
        float denom = std::sqrt(weights[3][c] + EPSILON);
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float norm = (input[c][h][w] - weights[2][c]) / denom;
                output[c][h][w] = weights[0][c] * norm + weights[1][c];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2) Tiled Batch Normalization
//
//    The tiled version splits the channels and spatial dimensions into blocks,
//    processing one tile at a time. This approach mimics the tiling strategy used
//    in the convolution code.
// -----------------------------------------------------------------------------
void batch_norm_tiled(
    int C, int H, int W,
    float input[MAX_C][MAX_H][MAX_W],
    float weights[4][MAX_C],
    float output[MAX_C][MAX_H][MAX_W]
) {
    // Calculate the number of tiles needed.
    int tile_rows = ceil_div(H, BLOCK_H);
    int tile_cols = ceil_div(W, BLOCK_W);
    int channel_tiles = ceil_div(C, BLOCK_C);

    // Loop over channel tiles.
    for (int ct = 0; ct < channel_tiles; ct++) {
        int c_offset = ct * BLOCK_C;
        int cur_C = std::min(BLOCK_C, C - c_offset);

        // Loop over spatial tiles.
        for (int tr = 0; tr < tile_rows; tr++) {
            int h_offset = tr * BLOCK_H;
            int cur_H = std::min(BLOCK_H, H - h_offset);
            for (int tc = 0; tc < tile_cols; tc++) {
                int w_offset = tc * BLOCK_W;
                int cur_W = std::min(BLOCK_W, W - w_offset);

                // Process the current tile for each channel in the block.
                for (int c = 0; c < cur_C; c++) {
                    int global_c = c_offset + c;
                    float denom = std::sqrt(weights[3][global_c] + EPSILON);
                    for (int i = 0; i < cur_H; i++) {
                        int global_h = h_offset + i;
                        for (int j = 0; j < cur_W; j++) {
                            int global_w = w_offset + j;
                            float norm = (input[global_c][global_h][global_w] - weights[2][global_c]) / denom;
                            output[global_c][global_h][global_w] = weights[0][global_c] * norm + weights[1][global_c];
                        }
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3) Test Driver: runTest
//
//    Initializes random input and batch-norm weights, runs both the reference
//    and tiled versions, and prints out the maximum difference along with a
//    sample tile of the output from channel 0.
// -----------------------------------------------------------------------------
void runTest(int C, int H, int W)
{
    // Declare static arrays for input, weights, and outputs.
    static float input[MAX_C][MAX_H][MAX_W];
    static float weights[4][MAX_C];      // weights[0]=gamma, [1]=beta, [2]=mean, [3]=variance
    static float output_ref[MAX_C][MAX_H][MAX_W];
    static float output_tile[MAX_C][MAX_H][MAX_W];

    std::cout << "\n=== Test BatchNorm with C=" << C << ", H=" << H << ", W=" << W << " ===\n";

    // Initialize input with random values in a typical range.
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input[c][h][w] = float(rand() % 201 - 100) / 25.f;
            }
        }
    }
    // Initialize batch-norm weights.
    for (int c = 0; c < C; c++) {
        weights[0][c] = float(rand() % 101 - 50) / 25.f;     // gamma
        weights[1][c] = float(rand() % 101 - 50) / 25.f;     // beta
        weights[2][c] = float(rand() % 201 - 100) / 25.f;      // mean
        // Ensure variance is positive; add a small constant.
        weights[3][c] = (float(rand() % 101) / 25.f) + 0.1f;  
    }

    // Run the reference batch normalization.
    batch_norm_ref(C, H, W, input, weights, output_ref);
    // Run the tiled batch normalization.
    batch_norm_tiled(C, H, W, input, weights, output_tile);

    // Validate: compute maximum absolute difference between outputs.
    float max_diff = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float diff = std::fabs(output_ref[c][h][w] - output_tile[c][h][w]);
                if (diff > max_diff)
                    max_diff = diff;
            }
        }
    }
    std::cout << "Max difference = " << max_diff << "\n";

    // Print a 4x4 region from channel 0 (top-left corner) for both outputs.
    std::cout << "Output channel 0, top-left 4x4 => ref | tiled\n";
    for (int h = 0; h < std::min(4, H); h++) {
        for (int w = 0; w < std::min(4, W); w++) {
            std::cout << output_ref[0][h][w] << " ";
        }
        std::cout << " | ";
        for (int w = 0; w < std::min(4, W); w++) {
            std::cout << output_tile[0][h][w] << " ";
        }
        std::cout << "\n";
    }
}

// -----------------------------------------------------------------------------
// Main: Entry point for testing the batch normalization functions.
// -----------------------------------------------------------------------------
int main()
{
    srand((unsigned)time(NULL));

    // You can add multiple test cases.
    runTest(64, 14, 14);
    runTest(128, 28, 28);
    runTest(256, 32, 32);

    return 0;
}
