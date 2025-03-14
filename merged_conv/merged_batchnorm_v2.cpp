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

#define BLOCK_C 64  // Channel tile size
#define BLOCK_H 8   // Spatial tile height
#define BLOCK_W 8   // Spatial tile width

#define EPSILON 1e-5f

// -----------------------------------------------------------------------------
// Helper Function: Ceiling Division
// -----------------------------------------------------------------------------
static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// -----------------------------------------------------------------------------
// 1) Reference Batch Normalization (non‑tiled)
//
//    For each channel and spatial location, the computation is:
//      norm = (input - mean) / sqrt(variance + EPSILON)
//      output = gamma * norm + beta
//
//    We use weights as follows:
//      weights[0] = gamma, weights[1] = beta,
//      weights[2] = mean,  weights[3] = variance.
// -----------------------------------------------------------------------------
void batch_norm_ref(
    int C, int H, int W,
    float input[MAX_C][MAX_H][MAX_W],
    float weights[4][MAX_C],
    float output[MAX_C][MAX_H][MAX_W]
)
{
    for (int c = 0; c < C; c++) {
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
// 2) Batch Norm Kernel (Fixed-Size Tile)
//
//    Processes a tile of dimensions up to BLOCK_C x BLOCK_H x BLOCK_W.
//    Only the valid dimensions (tile_C, tile_H, tile_W) are processed.
// -----------------------------------------------------------------------------
void batch_norm_kernel(
    float localInput[BLOCK_C][BLOCK_H][BLOCK_W],
    float localWeights[4][BLOCK_C],
    float localOutput[BLOCK_C][BLOCK_H][BLOCK_W],
    int tile_C, int tile_H, int tile_W
)
{
    for (int c = 0; c < tile_C; c++) {
        float denom = std::sqrt(localWeights[3][c] + EPSILON);
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
    float input[MAX_C][MAX_H][MAX_W],
    float weights[4][MAX_C],
    float output[MAX_C][MAX_H][MAX_W]
)
{
    int channel_tiles = ceil_div(C, BLOCK_C);
    int tile_rows     = ceil_div(H, BLOCK_H);
    int tile_cols     = ceil_div(W, BLOCK_W);

    // Loop over channel tiles.
    for (int ct = 0; ct < channel_tiles; ct++) {
        int c_offset = ct * BLOCK_C;
        int tile_C   = std::min(BLOCK_C, C - c_offset);

        // Loop over spatial tiles (rows and columns).
        for (int tr = 0; tr < tile_rows; tr++) {
            int h_offset = tr * BLOCK_H;
            int tile_H   = std::min(BLOCK_H, H - h_offset);
            for (int tc = 0; tc < tile_cols; tc++) {
                int w_offset = tc * BLOCK_W;
                int tile_W   = std::min(BLOCK_W, W - w_offset);

                // Local buffers for input and output tile.
                float localInput[BLOCK_C][BLOCK_H][BLOCK_W];
                float localOutput[BLOCK_C][BLOCK_H][BLOCK_W];
                // Local weights buffer for the current channel tile.
                float localWeights[4][BLOCK_C];

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

// -----------------------------------------------------------------------------
// 4) Test Driver: runRandomTest
//
//    Initializes input and weights with random values, computes the reference
//    batch normalization and then the tiled version (calling the kernel multiple times).
//    The outputs are compared to verify the correctness of the tiled implementation.
// -----------------------------------------------------------------------------
void runRandomTest(int C, int H, int W, int iterations)
{
    static float input[MAX_C][MAX_H][MAX_W];
    static float weights[4][MAX_C];      // weights[0]=gamma, [1]=beta, [2]=mean, [3]=variance
    static float output_ref[MAX_C][MAX_H][MAX_W];
    static float output_tiled[MAX_C][MAX_H][MAX_W];

    // Initialize input with random values in a given range.
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                // Random float in range [-2.0, 2.0]
                input[c][h][w] = (rand() % 400 - 200) / 100.0f;
            }
        }
    }

    // Initialize weights with random values.
    // For variance, ensure a positive value by adding a constant.
    for (int c = 0; c < C; c++) {
        weights[0][c] = (rand() % 200 - 100) / 50.0f;  // gamma
        weights[1][c] = (rand() % 200 - 100) / 50.0f;  // beta
        weights[2][c] = (rand() % 400 - 200) / 100.0f; // mean
        weights[3][c] = ((rand() % 100) / 50.0f) + 0.1f; // variance (ensuring > 0)
    }

    // Compute reference output using non‑tiled batch norm.
    batch_norm_ref(C, H, W, input, weights, output_ref);

    // Call tiled batch normalization multiple times.
    for (int iter = 0; iter < iterations; iter++) {
        batch_norm_tiled(C, H, W, input, weights, output_tiled);
    }

    // Validate: compute maximum absolute difference between reference and tiled outputs.
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

    std::cout << "\n=== Random Input/Output Tiled BatchNorm Test ===\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Max difference between reference and tiled: " << max_diff << "\n";

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


    runRandomTest(64, 14, 14, 1);
    runRandomTest(128, 28, 28, 1);
    runRandomTest(256, 32, 32, 1);


    return 0;
}
