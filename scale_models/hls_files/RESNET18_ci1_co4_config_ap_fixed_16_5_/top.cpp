
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

typedef ap_fixed<16,5> data_t;
typedef ap_fixed<32,10> acc_t;

data_t BRAM_feat_input[3][224][224];
data_t BRAM_feat_stem[64][112][112];
data_t BRAM_feat_pool[64][56][56];
data_t BRAM_w_stem[64][3][7][7];
data_t BRAM_bn_stem[4][64];
data_t BRAM_feat_s1_tmp1[64][56][56];
data_t BRAM_feat_s1_tmp2[64][56][56];
data_t BRAM_feat_s1_b0[64][56][56];
data_t BRAM_w_s1_b0_1[64][64][3][3];
data_t BRAM_bn_s1_b0_1[4][64];
data_t BRAM_w_s1_b0_2[64][64][3][3];
data_t BRAM_bn_s1_b0_2[4][64];
data_t BRAM_feat_s1_b1[64][56][56];
data_t BRAM_w_s1_b1_1[64][64][3][3];
data_t BRAM_bn_s1_b1_1[4][64];
data_t BRAM_w_s1_b1_2[64][64][3][3];
data_t BRAM_bn_s1_b1_2[4][64];
data_t BRAM_feat_s2_tmp1[128][28][28];
data_t BRAM_feat_s2_tmp2[128][28][28];
data_t BRAM_feat_s2_skip[128][28][28];
data_t BRAM_feat_s2_b0[128][28][28];
data_t BRAM_w_s2_b0_1[128][64][3][3];
data_t BRAM_bn_s2_b0_1[4][128];
data_t BRAM_w_s2_b0_2[128][128][3][3];
data_t BRAM_bn_s2_b0_2[4][128];
data_t BRAM_w_s2_b0_down[128][64][1][1];
data_t BRAM_feat_s2_b1[128][28][28];
data_t BRAM_w_s2_b1_1[128][128][3][3];
data_t BRAM_bn_s2_b1_1[4][128];
data_t BRAM_w_s2_b1_2[128][128][3][3];
data_t BRAM_bn_s2_b1_2[4][128];
data_t BRAM_feat_s3_tmp1[256][14][14];
data_t BRAM_feat_s3_tmp2[256][14][14];
data_t BRAM_feat_s3_skip[256][14][14];
data_t BRAM_feat_s3_b0[256][14][14];
data_t BRAM_w_s3_b0_1[256][128][3][3];
data_t BRAM_bn_s3_b0_1[4][256];
data_t BRAM_w_s3_b0_2[256][256][3][3];
data_t BRAM_bn_s3_b0_2[4][256];
data_t BRAM_w_s3_b0_down[256][128][1][1];
data_t BRAM_feat_s3_b1[256][14][14];
data_t BRAM_w_s3_b1_1[256][256][3][3];
data_t BRAM_bn_s3_b1_1[4][256];
data_t BRAM_w_s3_b1_2[256][256][3][3];
data_t BRAM_bn_s3_b1_2[4][256];
data_t BRAM_feat_s4_tmp1[512][7][7];
data_t BRAM_feat_s4_tmp2[512][7][7];
data_t BRAM_feat_s4_skip[512][7][7];
data_t BRAM_feat_s4_b0[512][7][7];
data_t BRAM_w_s4_b0_1[512][256][3][3];
data_t BRAM_bn_s4_b0_1[4][512];
data_t BRAM_w_s4_b0_2[512][512][3][3];
data_t BRAM_bn_s4_b0_2[4][512];
data_t BRAM_w_s4_b0_down[512][256][1][1];
data_t BRAM_feat_s4_b1[512][7][7];
data_t BRAM_w_s4_b1_1[512][512][3][3];
data_t BRAM_bn_s4_b1_1[4][512];
data_t BRAM_w_s4_b1_2[512][512][3][3];
data_t BRAM_bn_s4_b1_2[4][512];
data_t BRAM_pool[512][1][1];
data_t BRAM_out[1000][1][1];
data_t BRAM_fc[1000][512][1][1];

void load_3_224_224_ap_fixed_16_5_(data_t input[3][224][224], data_t output[3][224][224])
{
    for (int idx0 = 0; idx0 < 3; idx0++) {
        for (int idx1 = 0; idx1 < 224; idx1++) {
            for (int idx2 = 0; idx2 < 224; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}

/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [3][224][224]
 *   Kernel : [64][3][7][7]
 *   Output : [64][112][112]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [64] is expected.
 * For group convolution, we assume:
 *   3 % groups == 0
 *   64 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_3_64_224_224_112_112_7_3_2_ap_fixed_16_5_(
    data_t input[3][224][224],
    data_t kernel[64][3][7][7],
    data_t output[64][112][112]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int co = 0; co < 64; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int kh = 0; kh < 7; kh++) {
                for (int kw = 0; kw < 7; kw++) {
                    int in_row = i * 2 - 3 + kh;
                    int in_col = j * 2 - 3 + kw;
                    if (in_row >= 0 && in_row < 224 && in_col >= 0 && in_col < 224) {
                        for (int ci = 0; ci < 3; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 64; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}


/*
 * Auto-generated Batch Normalization HLS Code
 *
 * Dimensions:
 *   Input/Output: [64][112][112]
 * Data type: ap_fixed<16,5>
 * Epsilon: 0
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_64_112_112_ap_fixed_16_5_(
    data_t input[64][112][112],
    data_t weights[4][64],
    data_t output[64][112][112]
)
{
    for (int c = 0; c < 64; c++) {
        for (int h = 0; h < 112; h++) {
            for (int w = 0; w < 112; w++) {
                data_t norm = (input[c][h][w] - weights[2][c]) / hls::sqrt(weights[3][c] + (data_t)0);
                output[c][h][w] = weights[0][c] * norm + weights[1][c];
            }
        }
    }
}


/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: ap_fixed<16,5>
 * Tensor dimensions: [64][112][112]
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

void relu_64_112_112_ap_fixed_16_5_(
    data_t input[64][112][112],
    data_t output[64][112][112]
)
{
    for (int c = 0; c < 64; c++) {
        for (int i = 0; i < 112; i++) {
            for (int j = 0; j < 112; j++) {
                output[c][i][j] = (input[c][i][j] > 0) ? input[c][i][j] : (data_t)0;
            }
        }
    }
}

/*
 * Auto-generated Max Pooling HLS Code with explicit padding support.
 *
 * Dimensions:
 *   Input  : [64][112][112]
 *   Output : [64][56][56]
 *   Pooling Window: [3][3]
 *   Stride: [2][2]
 *   Padding: [1][1]
 *
 * Data type: ap_fixed<16,5>
 */

void maxpool_pad_64_112_112_56_56_3_3_2_2_1_1_ap_fixed_16_5_(
    data_t input[64][112][112],
    data_t output[64][56][56]
)
{
    for (int c = 0; c < 64; c++) {
        for (int i = 0; i < 56; i++) {
            for (int j = 0; j < 56; j++) {
                bool initialized = false;
                data_t max_val = (data_t)0;
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int row = i * 2 + kh - 1;
                        int col = j * 2 + kw - 1;
                        if (row >= 0 && row < 112 && col >= 0 && col < 112) {
                            if (!initialized || input[c][row][col] > max_val) {
                                max_val = input[c][row][col];
                                initialized = true;
                            }
                        }
                    }
                }
                output[c][i][j] = initialized ? max_val : (data_t)0;
            }
        }
    }
}


/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [64][56][56]
 *   Kernel : [64][64][3][3]
 *   Output : [64][56][56]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [64] is expected.
 * For group convolution, we assume:
 *   64 % groups == 0
 *   64 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_64_64_56_56_56_56_3_1_1_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t kernel[64][64][3][3],
    data_t output[64][56][56]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int co = 0; co < 64; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int in_row = i * 1 - 1 + kh;
                    int in_col = j * 1 - 1 + kw;
                    if (in_row >= 0 && in_row < 56 && in_col >= 0 && in_col < 56) {
                        for (int ci = 0; ci < 64; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 64; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}


/*
 * Auto-generated Batch Normalization HLS Code
 *
 * Dimensions:
 *   Input/Output: [64][56][56]
 * Data type: ap_fixed<16,5>
 * Epsilon: 0
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_64_56_56_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t weights[4][64],
    data_t output[64][56][56]
)
{
    for (int c = 0; c < 64; c++) {
        for (int h = 0; h < 56; h++) {
            for (int w = 0; w < 56; w++) {
                data_t norm = (input[c][h][w] - weights[2][c]) / hls::sqrt(weights[3][c] + (data_t)0);
                output[c][h][w] = weights[0][c] * norm + weights[1][c];
            }
        }
    }
}


/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: ap_fixed<16,5>
 * Tensor dimensions: [64][56][56]
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

void relu_64_56_56_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t output[64][56][56]
)
{
    for (int c = 0; c < 64; c++) {
        for (int i = 0; i < 56; i++) {
            for (int j = 0; j < 56; j++) {
                output[c][i][j] = (input[c][i][j] > 0) ? input[c][i][j] : (data_t)0;
            }
        }
    }
}


void matrix_add_64_56_56_ap_fixed_16_5_(
    data_t in1[64][56][56],
    data_t in2[64][56][56],
    data_t out[64][56][56]
)
{
    for (int c = 0; c < 64; c++) {
        for (int i = 0; i < 56; i++) {
            for (int j = 0; j < 56; j++) {
                out[c][i][j] = in1[c][i][j] + in2[c][i][j];
            }
        }
    }
}







/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [64][56][56]
 *   Kernel : [128][64][3][3]
 *   Output : [128][28][28]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [128] is expected.
 * For group convolution, we assume:
 *   64 % groups == 0
 *   128 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_64_128_56_56_28_28_3_1_2_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t kernel[128][64][3][3],
    data_t output[128][28][28]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int co = 0; co < 128; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int in_row = i * 2 - 1 + kh;
                    int in_col = j * 2 - 1 + kw;
                    if (in_row >= 0 && in_row < 56 && in_col >= 0 && in_col < 56) {
                        for (int ci = 0; ci < 64; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 128; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}


/*
 * Auto-generated Batch Normalization HLS Code
 *
 * Dimensions:
 *   Input/Output: [128][28][28]
 * Data type: ap_fixed<16,5>
 * Epsilon: 0
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_128_28_28_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t weights[4][128],
    data_t output[128][28][28]
)
{
    for (int c = 0; c < 128; c++) {
        for (int h = 0; h < 28; h++) {
            for (int w = 0; w < 28; w++) {
                data_t norm = (input[c][h][w] - weights[2][c]) / hls::sqrt(weights[3][c] + (data_t)0);
                output[c][h][w] = weights[0][c] * norm + weights[1][c];
            }
        }
    }
}


/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: ap_fixed<16,5>
 * Tensor dimensions: [128][28][28]
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

void relu_128_28_28_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t output[128][28][28]
)
{
    for (int c = 0; c < 128; c++) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                output[c][i][j] = (input[c][i][j] > 0) ? input[c][i][j] : (data_t)0;
            }
        }
    }
}

/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [128][28][28]
 *   Kernel : [128][128][3][3]
 *   Output : [128][28][28]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [128] is expected.
 * For group convolution, we assume:
 *   128 % groups == 0
 *   128 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_128_128_28_28_28_28_3_1_1_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t kernel[128][128][3][3],
    data_t output[128][28][28]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int co = 0; co < 128; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int in_row = i * 1 - 1 + kh;
                    int in_col = j * 1 - 1 + kw;
                    if (in_row >= 0 && in_row < 28 && in_col >= 0 && in_col < 28) {
                        for (int ci = 0; ci < 128; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 128; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [64][56][56]
 *   Kernel : [128][64][1][1]
 *   Output : [128][28][28]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [128] is expected.
 * For group convolution, we assume:
 *   64 % groups == 0
 *   128 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_64_128_56_56_28_28_1_0_2_ap_fixed_16_5_(
    data_t input[64][56][56],
    data_t kernel[128][64][1][1],
    data_t output[128][28][28]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int co = 0; co < 128; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    int in_row = i * 2 - 0 + kh;
                    int in_col = j * 2 - 0 + kw;
                    if (in_row >= 0 && in_row < 56 && in_col >= 0 && in_col < 56) {
                        for (int ci = 0; ci < 64; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 128; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}


void matrix_add_128_28_28_ap_fixed_16_5_(
    data_t in1[128][28][28],
    data_t in2[128][28][28],
    data_t out[128][28][28]
)
{
    for (int c = 0; c < 128; c++) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                out[c][i][j] = in1[c][i][j] + in2[c][i][j];
            }
        }
    }
}







/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [128][28][28]
 *   Kernel : [256][128][3][3]
 *   Output : [256][14][14]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [256] is expected.
 * For group convolution, we assume:
 *   128 % groups == 0
 *   256 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_128_256_28_28_14_14_3_1_2_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t kernel[256][128][3][3],
    data_t output[256][14][14]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int co = 0; co < 256; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int in_row = i * 2 - 1 + kh;
                    int in_col = j * 2 - 1 + kw;
                    if (in_row >= 0 && in_row < 28 && in_col >= 0 && in_col < 28) {
                        for (int ci = 0; ci < 128; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 256; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}


/*
 * Auto-generated Batch Normalization HLS Code
 *
 * Dimensions:
 *   Input/Output: [256][14][14]
 * Data type: ap_fixed<16,5>
 * Epsilon: 0
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_256_14_14_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t weights[4][256],
    data_t output[256][14][14]
)
{
    for (int c = 0; c < 256; c++) {
        for (int h = 0; h < 14; h++) {
            for (int w = 0; w < 14; w++) {
                data_t norm = (input[c][h][w] - weights[2][c]) / hls::sqrt(weights[3][c] + (data_t)0);
                output[c][h][w] = weights[0][c] * norm + weights[1][c];
            }
        }
    }
}


/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: ap_fixed<16,5>
 * Tensor dimensions: [256][14][14]
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

void relu_256_14_14_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t output[256][14][14]
)
{
    for (int c = 0; c < 256; c++) {
        for (int i = 0; i < 14; i++) {
            for (int j = 0; j < 14; j++) {
                output[c][i][j] = (input[c][i][j] > 0) ? input[c][i][j] : (data_t)0;
            }
        }
    }
}

/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [256][14][14]
 *   Kernel : [256][256][3][3]
 *   Output : [256][14][14]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [256] is expected.
 * For group convolution, we assume:
 *   256 % groups == 0
 *   256 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_256_256_14_14_14_14_3_1_1_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t kernel[256][256][3][3],
    data_t output[256][14][14]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int co = 0; co < 256; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int in_row = i * 1 - 1 + kh;
                    int in_col = j * 1 - 1 + kw;
                    if (in_row >= 0 && in_row < 14 && in_col >= 0 && in_col < 14) {
                        for (int ci = 0; ci < 256; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 256; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [128][28][28]
 *   Kernel : [256][128][1][1]
 *   Output : [256][14][14]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [256] is expected.
 * For group convolution, we assume:
 *   128 % groups == 0
 *   256 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_128_256_28_28_14_14_1_0_2_ap_fixed_16_5_(
    data_t input[128][28][28],
    data_t kernel[256][128][1][1],
    data_t output[256][14][14]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int co = 0; co < 256; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    int in_row = i * 2 - 0 + kh;
                    int in_col = j * 2 - 0 + kw;
                    if (in_row >= 0 && in_row < 28 && in_col >= 0 && in_col < 28) {
                        for (int ci = 0; ci < 128; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 256; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}


void matrix_add_256_14_14_ap_fixed_16_5_(
    data_t in1[256][14][14],
    data_t in2[256][14][14],
    data_t out[256][14][14]
)
{
    for (int c = 0; c < 256; c++) {
        for (int i = 0; i < 14; i++) {
            for (int j = 0; j < 14; j++) {
                out[c][i][j] = in1[c][i][j] + in2[c][i][j];
            }
        }
    }
}







/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [256][14][14]
 *   Kernel : [512][256][3][3]
 *   Output : [512][7][7]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [512] is expected.
 * For group convolution, we assume:
 *   256 % groups == 0
 *   512 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_256_512_14_14_7_7_3_1_2_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t kernel[512][256][3][3],
    data_t output[512][7][7]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int co = 0; co < 512; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int in_row = i * 2 - 1 + kh;
                    int in_col = j * 2 - 1 + kw;
                    if (in_row >= 0 && in_row < 14 && in_col >= 0 && in_col < 14) {
                        for (int ci = 0; ci < 256; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 512; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}


/*
 * Auto-generated Batch Normalization HLS Code
 *
 * Dimensions:
 *   Input/Output: [512][7][7]
 * Data type: ap_fixed<16,5>
 * Epsilon: 0
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_512_7_7_ap_fixed_16_5_(
    data_t input[512][7][7],
    data_t weights[4][512],
    data_t output[512][7][7]
)
{
    for (int c = 0; c < 512; c++) {
        for (int h = 0; h < 7; h++) {
            for (int w = 0; w < 7; w++) {
                data_t norm = (input[c][h][w] - weights[2][c]) / hls::sqrt(weights[3][c] + (data_t)0);
                output[c][h][w] = weights[0][c] * norm + weights[1][c];
            }
        }
    }
}


/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: ap_fixed<16,5>
 * Tensor dimensions: [512][7][7]
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

void relu_512_7_7_ap_fixed_16_5_(
    data_t input[512][7][7],
    data_t output[512][7][7]
)
{
    for (int c = 0; c < 512; c++) {
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                output[c][i][j] = (input[c][i][j] > 0) ? input[c][i][j] : (data_t)0;
            }
        }
    }
}

/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [512][7][7]
 *   Kernel : [512][512][3][3]
 *   Output : [512][7][7]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [512] is expected.
 * For group convolution, we assume:
 *   512 % groups == 0
 *   512 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_512_512_7_7_7_7_3_1_1_ap_fixed_16_5_(
    data_t input[512][7][7],
    data_t kernel[512][512][3][3],
    data_t output[512][7][7]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int co = 0; co < 512; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int in_row = i * 1 - 1 + kh;
                    int in_col = j * 1 - 1 + kw;
                    if (in_row >= 0 && in_row < 7 && in_col >= 0 && in_col < 7) {
                        for (int ci = 0; ci < 512; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 512; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [256][14][14]
 *   Kernel : [512][256][1][1]
 *   Output : [512][7][7]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [512] is expected.
 * For group convolution, we assume:
 *   256 % groups == 0
 *   512 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_256_512_14_14_7_7_1_0_2_ap_fixed_16_5_(
    data_t input[256][14][14],
    data_t kernel[512][256][1][1],
    data_t output[512][7][7]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int co = 0; co < 512; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    int in_row = i * 2 - 0 + kh;
                    int in_col = j * 2 - 0 + kw;
                    if (in_row >= 0 && in_row < 14 && in_col >= 0 && in_col < 14) {
                        for (int ci = 0; ci < 256; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 512; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}


void matrix_add_512_7_7_ap_fixed_16_5_(
    data_t in1[512][7][7],
    data_t in2[512][7][7],
    data_t out[512][7][7]
)
{
    for (int c = 0; c < 512; c++) {
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                out[c][i][j] = in1[c][i][j] + in2[c][i][j];
            }
        }
    }
}







void adaptive_avgpool_512_7_7_1_1_ap_fixed_16_5_(
    data_t input[512][7][7],
    data_t output[512][1][1]
)
{
    for (int c = 0; c < 512; c++) {
        for (int oh = 0; oh < 1; oh++) {
            int h_start = (int)floor(( (float)oh * 7 ) / 1);
            int h_end   = (int)ceil(( (float)(oh + 1) * 7 ) / 1);
            if (h_end > 7) {
                h_end = 7;
            }
            for (int ow = 0; ow < 1; ow++) {
                int w_start = (int)floor(( (float)ow * 7 ) / 1);
                int w_end   = (int)ceil(( (float)(ow + 1) * 7 ) / 1);
                if (w_end > 7) {
                    w_end = 7;
                }

                data_t sum = 0;
                int count = 0;
                for (int h = h_start; h < h_end; h++) {
                    for (int w = w_start; w < w_end; w++) {
                        sum += input[c][h][w];
                        count++;
                    }
                }
                if (count > 0)
                    output[c][oh][ow] = sum / (data_t)count;
                else
                    output[c][oh][ow] = 0;
            }
        }
    }
}






/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [512][1][1]
 *   Kernel : [1000][512][1][1]
 *   Output : [1000][1][1]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [1000] is expected.
 * For group convolution, we assume:
 *   512 % groups == 0
 *   1000 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_512_1000_1_1_1_1_1_0_1_ap_fixed_16_5_(
    data_t input[512][1][1],
    data_t kernel[1000][512][1][1],
    data_t output[1000][1][1]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++) {
            for (int co = 0; co < 1000; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    int in_row = i * 1 - 0 + kh;
                    int in_col = j * 1 - 0 + kw;
                    if (in_row >= 0 && in_row < 1 && in_col >= 0 && in_col < 1) {
                        for (int ci = 0; ci < 512; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 1000; co++) {
                                #pragma HLS unroll factor=4
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}

void store_1000_1_1_ap_fixed_16_5_(data_t input[1000][1][1], data_t output[1000][1][1])
{
    for (int idx0 = 0; idx0 < 1000; idx0++) {
        for (int idx1 = 0; idx1 < 1; idx1++) {
            for (int idx2 = 0; idx2 < 1; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}

void top(data_t DRAM_input[3][224][224], data_t DRAM_w_stem[64][3][7][7], data_t DRAM_bn_stem[4][64], data_t DRAM_w_s1_b0_1[64][64][3][3], data_t DRAM_bn_s1_b0_1[4][64], data_t DRAM_w_s1_b0_2[64][64][3][3], data_t DRAM_bn_s1_b0_2[4][64], data_t DRAM_w_s1_b1_1[64][64][3][3], data_t DRAM_bn_s1_b1_1[4][64], data_t DRAM_w_s1_b1_2[64][64][3][3], data_t DRAM_bn_s1_b1_2[4][64], data_t DRAM_w_s2_b0_1[128][64][3][3], data_t DRAM_bn_s2_b0_1[4][128], data_t DRAM_w_s2_b0_2[128][128][3][3], data_t DRAM_bn_s2_b0_2[4][128], data_t DRAM_w_s2_b0_down[128][64][1][1], data_t DRAM_w_s2_b1_1[128][128][3][3], data_t DRAM_bn_s2_b1_1[4][128], data_t DRAM_w_s2_b1_2[128][128][3][3], data_t DRAM_bn_s2_b1_2[4][128], data_t DRAM_w_s3_b0_1[256][128][3][3], data_t DRAM_bn_s3_b0_1[4][256], data_t DRAM_w_s3_b0_2[256][256][3][3], data_t DRAM_bn_s3_b0_2[4][256], data_t DRAM_w_s3_b0_down[256][128][1][1], data_t DRAM_w_s3_b1_1[256][256][3][3], data_t DRAM_bn_s3_b1_1[4][256], data_t DRAM_w_s3_b1_2[256][256][3][3], data_t DRAM_bn_s3_b1_2[4][256], data_t DRAM_w_s4_b0_1[512][256][3][3], data_t DRAM_bn_s4_b0_1[4][512], data_t DRAM_w_s4_b0_2[512][512][3][3], data_t DRAM_bn_s4_b0_2[4][512], data_t DRAM_w_s4_b0_down[512][256][1][1], data_t DRAM_w_s4_b1_1[512][512][3][3], data_t DRAM_bn_s4_b1_1[4][512], data_t DRAM_w_s4_b1_2[512][512][3][3], data_t DRAM_bn_s4_b1_2[4][512], data_t DRAM_out[1000][1][1], data_t DRAM_fc[1000][512][1][1])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem_input
    #pragma HLS interface m_axi port=DRAM_w_stem offset=slave bundle=mem_w_stem
    #pragma HLS interface m_axi port=DRAM_bn_stem offset=slave bundle=mem_bn_stem
    #pragma HLS interface m_axi port=DRAM_w_s1_b0_1 offset=slave bundle=mem_w_s1_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s1_b0_1 offset=slave bundle=mem_bn_s1_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s1_b0_2 offset=slave bundle=mem_w_s1_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s1_b0_2 offset=slave bundle=mem_bn_s1_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s1_b1_1 offset=slave bundle=mem_w_s1_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s1_b1_1 offset=slave bundle=mem_bn_s1_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s1_b1_2 offset=slave bundle=mem_w_s1_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s1_b1_2 offset=slave bundle=mem_bn_s1_b1_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_1 offset=slave bundle=mem_w_s2_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b0_1 offset=slave bundle=mem_bn_s2_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_2 offset=slave bundle=mem_w_s2_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b0_2 offset=slave bundle=mem_bn_s2_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s2_b0_down offset=slave bundle=mem_w_s2_b0_down
    #pragma HLS interface m_axi port=DRAM_w_s2_b1_1 offset=slave bundle=mem_w_s2_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s2_b1_1 offset=slave bundle=mem_bn_s2_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s2_b1_2 offset=slave bundle=mem_w_s2_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s2_b1_2 offset=slave bundle=mem_bn_s2_b1_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_1 offset=slave bundle=mem_w_s3_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b0_1 offset=slave bundle=mem_bn_s3_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_2 offset=slave bundle=mem_w_s3_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b0_2 offset=slave bundle=mem_bn_s3_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s3_b0_down offset=slave bundle=mem_w_s3_b0_down
    #pragma HLS interface m_axi port=DRAM_w_s3_b1_1 offset=slave bundle=mem_w_s3_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s3_b1_1 offset=slave bundle=mem_bn_s3_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s3_b1_2 offset=slave bundle=mem_w_s3_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s3_b1_2 offset=slave bundle=mem_bn_s3_b1_2
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_1 offset=slave bundle=mem_w_s4_b0_1
    #pragma HLS interface m_axi port=DRAM_bn_s4_b0_1 offset=slave bundle=mem_bn_s4_b0_1
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_2 offset=slave bundle=mem_w_s4_b0_2
    #pragma HLS interface m_axi port=DRAM_bn_s4_b0_2 offset=slave bundle=mem_bn_s4_b0_2
    #pragma HLS interface m_axi port=DRAM_w_s4_b0_down offset=slave bundle=mem_w_s4_b0_down
    #pragma HLS interface m_axi port=DRAM_w_s4_b1_1 offset=slave bundle=mem_w_s4_b1_1
    #pragma HLS interface m_axi port=DRAM_bn_s4_b1_1 offset=slave bundle=mem_bn_s4_b1_1
    #pragma HLS interface m_axi port=DRAM_w_s4_b1_2 offset=slave bundle=mem_w_s4_b1_2
    #pragma HLS interface m_axi port=DRAM_bn_s4_b1_2 offset=slave bundle=mem_bn_s4_b1_2
    #pragma HLS interface m_axi port=DRAM_out offset=slave bundle=mem_out
    #pragma HLS interface m_axi port=DRAM_fc offset=slave bundle=mem_fc

    load_3_224_224_ap_fixed_16_5_(DRAM_input, BRAM_feat_input);
    conv2d_3_64_224_224_112_112_7_3_2_ap_fixed_16_5_(BRAM_feat_input, BRAM_w_stem, BRAM_feat_stem);
    batch_norm_64_112_112_ap_fixed_16_5_(BRAM_feat_stem, BRAM_bn_stem, BRAM_feat_stem);
    relu_64_112_112_ap_fixed_16_5_(BRAM_feat_stem, BRAM_feat_stem);
    maxpool_pad_64_112_112_56_56_3_3_2_2_1_1_ap_fixed_16_5_(BRAM_feat_stem, BRAM_feat_pool);
    conv2d_64_64_56_56_56_56_3_1_1_ap_fixed_16_5_(BRAM_feat_pool, BRAM_w_s1_b0_1, BRAM_feat_s1_tmp1);
    batch_norm_64_56_56_ap_fixed_16_5_(BRAM_feat_s1_tmp1, BRAM_bn_s1_b0_1, BRAM_feat_s1_tmp1);
    relu_64_56_56_ap_fixed_16_5_(BRAM_feat_s1_tmp1, BRAM_feat_s1_tmp1);
    conv2d_64_64_56_56_56_56_3_1_1_ap_fixed_16_5_(BRAM_feat_s1_tmp1, BRAM_w_s1_b0_2, BRAM_feat_s1_tmp2);
    batch_norm_64_56_56_ap_fixed_16_5_(BRAM_feat_s1_tmp2, BRAM_bn_s1_b0_2, BRAM_feat_s1_tmp2);
    matrix_add_64_56_56_ap_fixed_16_5_(BRAM_feat_s1_tmp2, BRAM_feat_pool, BRAM_feat_s1_b0);
    relu_64_56_56_ap_fixed_16_5_(BRAM_feat_s1_b0, BRAM_feat_s1_b0);
    conv2d_64_64_56_56_56_56_3_1_1_ap_fixed_16_5_(BRAM_feat_s1_b0, BRAM_w_s1_b1_1, BRAM_feat_s1_tmp1);
    batch_norm_64_56_56_ap_fixed_16_5_(BRAM_feat_s1_tmp1, BRAM_bn_s1_b1_1, BRAM_feat_s1_tmp1);
    relu_64_56_56_ap_fixed_16_5_(BRAM_feat_s1_tmp1, BRAM_feat_s1_tmp1);
    conv2d_64_64_56_56_56_56_3_1_1_ap_fixed_16_5_(BRAM_feat_s1_tmp1, BRAM_w_s1_b1_2, BRAM_feat_s1_tmp2);
    batch_norm_64_56_56_ap_fixed_16_5_(BRAM_feat_s1_tmp2, BRAM_bn_s1_b1_2, BRAM_feat_s1_tmp2);
    matrix_add_64_56_56_ap_fixed_16_5_(BRAM_feat_s1_tmp2, BRAM_feat_s1_b0, BRAM_feat_s1_b1);
    relu_64_56_56_ap_fixed_16_5_(BRAM_feat_s1_b1, BRAM_feat_s1_b1);
    conv2d_64_128_56_56_28_28_3_1_2_ap_fixed_16_5_(BRAM_feat_s1_b1, BRAM_w_s2_b0_1, BRAM_feat_s2_tmp1);
    batch_norm_128_28_28_ap_fixed_16_5_(BRAM_feat_s2_tmp1, BRAM_bn_s2_b0_1, BRAM_feat_s2_tmp1);
    relu_128_28_28_ap_fixed_16_5_(BRAM_feat_s2_tmp1, BRAM_feat_s2_tmp1);
    conv2d_128_128_28_28_28_28_3_1_1_ap_fixed_16_5_(BRAM_feat_s2_tmp1, BRAM_w_s2_b0_2, BRAM_feat_s2_tmp2);
    batch_norm_128_28_28_ap_fixed_16_5_(BRAM_feat_s2_tmp2, BRAM_bn_s2_b0_2, BRAM_feat_s2_tmp2);
    conv2d_64_128_56_56_28_28_1_0_2_ap_fixed_16_5_(BRAM_feat_s1_b1, BRAM_w_s2_b0_down, BRAM_feat_s2_skip);
    matrix_add_128_28_28_ap_fixed_16_5_(BRAM_feat_s2_tmp2, BRAM_feat_s2_skip, BRAM_feat_s2_b0);
    relu_128_28_28_ap_fixed_16_5_(BRAM_feat_s2_b0, BRAM_feat_s2_b0);
    conv2d_128_128_28_28_28_28_3_1_1_ap_fixed_16_5_(BRAM_feat_s2_b0, BRAM_w_s2_b1_1, BRAM_feat_s2_tmp1);
    batch_norm_128_28_28_ap_fixed_16_5_(BRAM_feat_s2_tmp1, BRAM_bn_s2_b1_1, BRAM_feat_s2_tmp1);
    relu_128_28_28_ap_fixed_16_5_(BRAM_feat_s2_tmp1, BRAM_feat_s2_tmp1);
    conv2d_128_128_28_28_28_28_3_1_1_ap_fixed_16_5_(BRAM_feat_s2_tmp1, BRAM_w_s2_b1_2, BRAM_feat_s2_tmp2);
    batch_norm_128_28_28_ap_fixed_16_5_(BRAM_feat_s2_tmp2, BRAM_bn_s2_b1_2, BRAM_feat_s2_tmp2);
    matrix_add_128_28_28_ap_fixed_16_5_(BRAM_feat_s2_tmp2, BRAM_feat_s2_b0, BRAM_feat_s2_b1);
    relu_128_28_28_ap_fixed_16_5_(BRAM_feat_s2_b1, BRAM_feat_s2_b1);
    conv2d_128_256_28_28_14_14_3_1_2_ap_fixed_16_5_(BRAM_feat_s2_b1, BRAM_w_s3_b0_1, BRAM_feat_s3_tmp1);
    batch_norm_256_14_14_ap_fixed_16_5_(BRAM_feat_s3_tmp1, BRAM_bn_s3_b0_1, BRAM_feat_s3_tmp1);
    relu_256_14_14_ap_fixed_16_5_(BRAM_feat_s3_tmp1, BRAM_feat_s3_tmp1);
    conv2d_256_256_14_14_14_14_3_1_1_ap_fixed_16_5_(BRAM_feat_s3_tmp1, BRAM_w_s3_b0_2, BRAM_feat_s3_tmp2);
    batch_norm_256_14_14_ap_fixed_16_5_(BRAM_feat_s3_tmp2, BRAM_bn_s3_b0_2, BRAM_feat_s3_tmp2);
    conv2d_128_256_28_28_14_14_1_0_2_ap_fixed_16_5_(BRAM_feat_s2_b1, BRAM_w_s3_b0_down, BRAM_feat_s3_skip);
    matrix_add_256_14_14_ap_fixed_16_5_(BRAM_feat_s3_tmp2, BRAM_feat_s3_skip, BRAM_feat_s3_b0);
    relu_256_14_14_ap_fixed_16_5_(BRAM_feat_s3_b0, BRAM_feat_s3_b0);
    conv2d_256_256_14_14_14_14_3_1_1_ap_fixed_16_5_(BRAM_feat_s3_b0, BRAM_w_s3_b1_1, BRAM_feat_s3_tmp1);
    batch_norm_256_14_14_ap_fixed_16_5_(BRAM_feat_s3_tmp1, BRAM_bn_s3_b1_1, BRAM_feat_s3_tmp1);
    relu_256_14_14_ap_fixed_16_5_(BRAM_feat_s3_tmp1, BRAM_feat_s3_tmp1);
    conv2d_256_256_14_14_14_14_3_1_1_ap_fixed_16_5_(BRAM_feat_s3_tmp1, BRAM_w_s3_b1_2, BRAM_feat_s3_tmp2);
    batch_norm_256_14_14_ap_fixed_16_5_(BRAM_feat_s3_tmp2, BRAM_bn_s3_b1_2, BRAM_feat_s3_tmp2);
    matrix_add_256_14_14_ap_fixed_16_5_(BRAM_feat_s3_tmp2, BRAM_feat_s3_b0, BRAM_feat_s3_b1);
    relu_256_14_14_ap_fixed_16_5_(BRAM_feat_s3_b1, BRAM_feat_s3_b1);
    conv2d_256_512_14_14_7_7_3_1_2_ap_fixed_16_5_(BRAM_feat_s3_b1, BRAM_w_s4_b0_1, BRAM_feat_s4_tmp1);
    batch_norm_512_7_7_ap_fixed_16_5_(BRAM_feat_s4_tmp1, BRAM_bn_s4_b0_1, BRAM_feat_s4_tmp1);
    relu_512_7_7_ap_fixed_16_5_(BRAM_feat_s4_tmp1, BRAM_feat_s4_tmp1);
    conv2d_512_512_7_7_7_7_3_1_1_ap_fixed_16_5_(BRAM_feat_s4_tmp1, BRAM_w_s4_b0_2, BRAM_feat_s4_tmp2);
    batch_norm_512_7_7_ap_fixed_16_5_(BRAM_feat_s4_tmp2, BRAM_bn_s4_b0_2, BRAM_feat_s4_tmp2);
    conv2d_256_512_14_14_7_7_1_0_2_ap_fixed_16_5_(BRAM_feat_s3_b1, BRAM_w_s4_b0_down, BRAM_feat_s4_skip);
    matrix_add_512_7_7_ap_fixed_16_5_(BRAM_feat_s4_tmp2, BRAM_feat_s4_skip, BRAM_feat_s4_b0);
    relu_512_7_7_ap_fixed_16_5_(BRAM_feat_s4_b0, BRAM_feat_s4_b0);
    conv2d_512_512_7_7_7_7_3_1_1_ap_fixed_16_5_(BRAM_feat_s4_b0, BRAM_w_s4_b1_1, BRAM_feat_s4_tmp1);
    batch_norm_512_7_7_ap_fixed_16_5_(BRAM_feat_s4_tmp1, BRAM_bn_s4_b1_1, BRAM_feat_s4_tmp1);
    relu_512_7_7_ap_fixed_16_5_(BRAM_feat_s4_tmp1, BRAM_feat_s4_tmp1);
    conv2d_512_512_7_7_7_7_3_1_1_ap_fixed_16_5_(BRAM_feat_s4_tmp1, BRAM_w_s4_b1_2, BRAM_feat_s4_tmp2);
    batch_norm_512_7_7_ap_fixed_16_5_(BRAM_feat_s4_tmp2, BRAM_bn_s4_b1_2, BRAM_feat_s4_tmp2);
    matrix_add_512_7_7_ap_fixed_16_5_(BRAM_feat_s4_tmp2, BRAM_feat_s4_b0, BRAM_feat_s4_b1);
    relu_512_7_7_ap_fixed_16_5_(BRAM_feat_s4_b1, BRAM_feat_s4_b1);
    adaptive_avgpool_512_7_7_1_1_ap_fixed_16_5_(BRAM_feat_s4_b1, BRAM_pool);
    conv2d_512_1000_1_1_1_1_1_0_1_ap_fixed_16_5_(BRAM_pool, BRAM_fc, BRAM_out);
    store_1000_1_1_ap_fixed_16_5_(BRAM_out, DRAM_out);
}