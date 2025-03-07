#include <cmath>

#include <iostream>

using namespace std;

typedef float data_t;

data_t BRAM_1[32][64][64];
data_t BRAM_2[32][64][64];
data_t BRAM_3[32][32][3][3];
data_t BRAM_4[32][64][64];
data_t BRAM_5[32][32][32];

void load_32_64_64_float(data_t input[32][64][64], data_t output[32][64][64])
{
    for (int idx0 = 0; idx0 < 32; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            for (int idx2 = 0; idx2 < 64; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}


/*
 * Auto-generated Batch Normalization HLS Code
 *
 * Dimensions:
 *   Input/Output: [32][64][64]
 * Data type: float
 * Epsilon: 1e-05
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type

void batch_norm_32_64_64_float(
    data_t input[32][64][64],
    data_t gamma[32],
    data_t beta[32],
    data_t mean[32],
    data_t variance[32],
    data_t output[32][64][64]
)
{
    for (int c = 0; c < 32; c++) {
        for (int h = 0; h < 64; h++) {
            for (int w = 0; w < 64; w++) {
                data_t norm = (input[c][h][w] - mean[c]) / sqrt(variance[c] + 1e-05);
                output[c][h][w] = gamma[c] * norm + beta[c];
            }
        }
    }
}


/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [32][64][64]
 *   Kernel : [32][32][3][3]
 *   Output : [32][64][64]
 *
 * Data type: float
 *
 * If bias is included, a 1D bias array of size [32] is expected.
 * For group convolution, we assume:
 *   32 % groups == 0
 *   32 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (Standard)
//////////////////////////////////////////

void conv2d_32_32_64_64_3_float(
    data_t input[32][64][64],
    data_t kernel[32][32][3][3],
    data_t output[32][64][64]
)
{
    // Initialize output to ((data_t)0)
    for (int co = 0; co < 32; co++) {
        for (int row = 0; row < 64; row++) {
            for (int col = 0; col < 64; col++) {
                #pragma HLS PIPELINE II=1
                output[co][row][col] = ((data_t)0);
            }
        }
    }

    // 6 nested loops: co, ci, row, col, kh, kw
    for (int co = 0; co < 32; co++) {
        for (int ci = 0; ci < 32; ci++) {
            for (int row = 0; row < 64; row++) {
                for (int col = 0; col < 64; col++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            output[co][row][col] +=
                                input[ci][row + kh][col + kw] * kernel[co][ci][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: float
 * Tensor dimensions: [32][64][64]
 *
 * Available functions:
 *   - relu
 *   - leaky_relu
 *   - prelu
 *   - rrelu
 *   - thresholded_relu
 *   - relu6
 *   - sigmoid
 *   - tanh_act
 *   - elu
 *   - selu
 *   - gelu
 *   - swish
 *   - softmax
 */

void relu_32_64_64_float(
    data_t input[32][64][64],
    data_t output[32][64][64]
)
{
    for (int c = 0; c < 32; c++) {
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                output[c][i][j] = (input[c][i][j] > 0) ? input[c][i][j] : (data_t)0;
            }
        }
    }
}


/*
 * Auto-generated Max Pooling HLS Code
 *
 * Dimensions:
 *   Input  : [32][64][64]
 *   Output : [32][32][32]
 *   Pooling Window: [2][2]
 *   Stride: [2][2]
 *
 * Data type: float
 */

void maxpool_32_64_64_32_32_2_2_2_2_float(
    data_t input[32][64][64],
    data_t output[32][32][32]
)
{
    for (int c = 0; c < 32; c++) {
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                // Initialize max value from the top-left corner of the window
                data_t max_val = input[c][i * 2][j * 2];
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int row = i * 2 + kh;
                        int col = j * 2 + kw;
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


void store_32_32_32_float(data_t input[32][32][32], data_t output[32][32][32])
{
    for (int idx0 = 0; idx0 < 32; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            for (int idx2 = 0; idx2 < 32; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}

void top(data_t DRAM_1[32][64][64], data_t DRAM_2[32][64][64], data_t DRAM_3[32][32][3][3], data_t DRAM_4[32][32][32])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    load_32_64_64_float(DRAM_1, BRAM_1);
    batch_norm_32_64_64_float(BRAM_1, gamma, beta, mean, variance, BRAM_2);
    conv2d_32_32_64_64_3_float(BRAM_2, BRAM_3, BRAM_4);
    relu_32_64_64_float(BRAM_4, BRAM_4);
    maxpool_32_64_64_32_32_2_2_2_2_float(BRAM_4, BRAM_5);
    store_32_32_32_float(BRAM_5, DRAM_4);
    store_32_32_32_float(BRAM_3, DRAM_3);
}