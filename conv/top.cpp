
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>
using namespace std;

typedef ap_fixed<16, 5> data_t;

data_t BRAM_1[2][4][4];
data_t BRAM_2[2][2][3][3];
data_t BRAM_3[2];
data_t BRAM_4[4][2];
data_t BRAM_5[2][4][4];
data_t BRAM_6[2][2][2];

void load_2_4_4_ap_fixed_16_5_(data_t input[2][4][4], data_t output[2][4][4])
{
    for (int idx0 = 0; idx0 < 2; idx0++) {
        for (int idx1 = 0; idx1 < 4; idx1++) {
            for (int idx2 = 0; idx2 < 4; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}

void load_2_2_3_3_ap_fixed_16_5_(data_t input[2][2][3][3], data_t output[2][2][3][3])
{
    for (int idx0 = 0; idx0 < 2; idx0++) {
        for (int idx1 = 0; idx1 < 2; idx1++) {
            for (int idx2 = 0; idx2 < 3; idx2++) {
                for (int idx3 = 0; idx3 < 3; idx3++) {
                    output[idx0][idx1][idx2][idx3] = input[idx0][idx1][idx2][idx3];
                }
            }
        }
    }
}

void load_2_ap_fixed_16_5_(data_t input[2], data_t output[2])
{
    for (int idx0 = 0; idx0 < 2; idx0++) {
        output[idx0] = input[idx0];
    }
}

void load_4_2_ap_fixed_16_5_(data_t input[4][2], data_t output[4][2])
{
    for (int idx0 = 0; idx0 < 4; idx0++) {
        for (int idx1 = 0; idx1 < 2; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}


/*
 * Auto-generated Batch Normalization HLS Code
 *
 * Dimensions:
 *   Input/Output: [2][4][4]
 * Data type: ap_fixed<16, 5>
 * Epsilon: 1e-05
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_2_4_4_ap_fixed_16_5_(
    data_t input[2][4][4],
    data_t weights[4][2],
    data_t output[2][4][4]
)
{
    for (int c = 0; c < 2; c++) {
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                data_t norm = (input[c][h][w] - weights[2][c]) / hls::sqrt(weights[3][c] + (data_t)1e-05);
                output[c][h][w] = weights[0][c] * norm + weights[1][c];
            }
        }
    }
}


/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [2][4][4]
 *   Kernel : [2][2][3][3]
 *   Output : [2][4][4]
 *
 * Data type: ap_fixed<16, 5>
 *
 * If bias is included, a 1D bias array of size [2] is expected.
 * For group convolution, we assume:
 *   2 % groups == 0
 *   2 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (Standard)
//////////////////////////////////////////

void group_conv2d_2_2_4_4_3_ap_fixed_16_5__bias(
    data_t input[2][4][4],
    data_t kernel[2][2][3][3],
    data_t bias[2],
    data_t output[2][4][4],
    int groups
)
{
    int c_in_per_group  = 2 / groups;
    int c_out_per_group = 2 / groups;

    // Initialize output to bias[co]
    for (int co = 0; co < 2; co++) {
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                #pragma HLS PIPELINE II=1
                output[co][row][col] = bias[co];
            }
        }
    }

    // Group convolution: partition input/output channels
    for (int g = 0; g < groups; g++) {
        int co_start = g * c_out_per_group;
        int co_end   = co_start + c_out_per_group;

        int ci_start = g * c_in_per_group;
        int ci_end   = ci_start + c_in_per_group;

        for (int co = co_start; co < co_end; co++) {
            for (int ci = ci_start; ci < ci_end; ci++) {
                for (int row = 0; row < 4; row++) {
                    for (int col = 0; col < 4; col++) {
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
}

/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: ap_fixed<16, 5>
 * Tensor dimensions: [2][4][4]
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

void relu_2_4_4_ap_fixed_16_5_(
    data_t input[2][4][4],
    data_t output[2][4][4]
)
{
    for (int c = 0; c < 2; c++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                output[c][i][j] = (input[c][i][j] > 0) ? input[c][i][j] : (data_t)0;
            }
        }
    }
}


/*
 * Auto-generated Max Pooling HLS Code
 *
 * Dimensions:
 *   Input  : [2][4][4]
 *   Output : [2][2][2]
 *   Pooling Window: [2][2]
 *   Stride: [2][2]
 *
 * Data type: ap_fixed<16, 5>
 */

void maxpool_2_4_4_2_2_2_2_2_2_ap_fixed_16_5_(
    data_t input[2][4][4],
    data_t output[2][2][2]
)
{
    for (int c = 0; c < 2; c++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
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


void store_2_2_2_ap_fixed_16_5_(data_t input[2][2][2], data_t output[2][2][2])
{
    for (int idx0 = 0; idx0 < 2; idx0++) {
        for (int idx1 = 0; idx1 < 2; idx1++) {
            for (int idx2 = 0; idx2 < 2; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}


void matrix_add_2_2_2_ap_fixed_16_5_(
    data_t in1[2][2][2],
    data_t in2[2][2][2],
    data_t out[2][2][2]
)
{
    for (int c = 0; c < 2; c++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                out[c][i][j] = in1[c][i][j] + in2[c][i][j];
            }
        }
    }
}







void top(data_t DRAM_1[2][4][4], data_t DRAM_2[2][2][3][3], data_t DRAM_3[2], data_t DRAM_4[4][2], data_t DRAM_5[2][2][2])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_5 offset=slave bundle=mem2

    load_2_4_4_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_2_2_3_3_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_2_ap_fixed_16_5_(DRAM_3, BRAM_3);
    load_4_2_ap_fixed_16_5_(DRAM_4, BRAM_4);
    batch_norm_2_4_4_ap_fixed_16_5_(BRAM_1, BRAM_4, BRAM_5);
    group_conv2d_2_2_4_4_3_ap_fixed_16_5__bias(BRAM_5, BRAM_2, BRAM_3, BRAM_1, 2);
    relu_2_4_4_ap_fixed_16_5_(BRAM_1, BRAM_5);
    maxpool_2_4_4_2_2_2_2_2_2_ap_fixed_16_5_(BRAM_5, BRAM_6);
    store_2_2_2_ap_fixed_16_5_(BRAM_6, DRAM_5);
    matrix_add_2_2_2_ap_fixed_16_5_(BRAM_4, BRAM_5, BRAM_6);
}