
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

data_t BRAM_feat_0[3][224][224];
data_t BRAM_feat_1[3][112][112];
data_t BRAM_feat_2[3][112][112];
data_t BRAM_w_stem[32][3][3][3];
data_t BRAM_bn_stem[4][32];
data_t BRAM_w_dw_0[32][32][3][3];
data_t BRAM_bn_dw_0[4][32];
data_t BRAM_w_proj_0[16][32][1][1];
data_t BRAM_bn_proj_0[4][16];
data_t BRAM_w_exp_1[96][16][1][1];
data_t BRAM_bn_exp_1[4][96];
data_t BRAM_w_dw_1[96][96][3][3];
data_t BRAM_bn_dw_1[4][96];
data_t BRAM_w_proj_1[24][96][1][1];
data_t BRAM_bn_proj_1[4][24];
data_t BRAM_w_exp_2[144][24][1][1];
data_t BRAM_bn_exp_2[4][144];
data_t BRAM_w_dw_2[144][144][3][3];
data_t BRAM_bn_dw_2[4][144];
data_t BRAM_w_proj_2[24][144][1][1];
data_t BRAM_bn_proj_2[4][24];
data_t BRAM_w_exp_3[144][24][1][1];
data_t BRAM_bn_exp_3[4][144];
data_t BRAM_w_dw_3[144][144][3][3];
data_t BRAM_bn_dw_3[4][144];
data_t BRAM_w_proj_3[32][144][1][1];
data_t BRAM_bn_proj_3[4][32];
data_t BRAM_w_exp_4[192][32][1][1];
data_t BRAM_bn_exp_4[4][192];
data_t BRAM_w_dw_4[192][192][3][3];
data_t BRAM_bn_dw_4[4][192];
data_t BRAM_w_proj_4[32][192][1][1];
data_t BRAM_bn_proj_4[4][32];
data_t BRAM_w_exp_5[192][32][1][1];
data_t BRAM_bn_exp_5[4][192];
data_t BRAM_w_dw_5[192][192][3][3];
data_t BRAM_bn_dw_5[4][192];
data_t BRAM_w_proj_5[32][192][1][1];
data_t BRAM_bn_proj_5[4][32];
data_t BRAM_w_exp_6[192][32][1][1];
data_t BRAM_bn_exp_6[4][192];
data_t BRAM_w_dw_6[192][192][3][3];
data_t BRAM_bn_dw_6[4][192];
data_t BRAM_w_proj_6[64][192][1][1];
data_t BRAM_bn_proj_6[4][64];
data_t BRAM_w_exp_7[384][64][1][1];
data_t BRAM_bn_exp_7[4][384];
data_t BRAM_w_dw_7[384][384][3][3];
data_t BRAM_bn_dw_7[4][384];
data_t BRAM_w_proj_7[64][384][1][1];
data_t BRAM_bn_proj_7[4][64];
data_t BRAM_w_exp_8[384][64][1][1];
data_t BRAM_bn_exp_8[4][384];
data_t BRAM_w_dw_8[384][384][3][3];
data_t BRAM_bn_dw_8[4][384];
data_t BRAM_w_proj_8[64][384][1][1];
data_t BRAM_bn_proj_8[4][64];
data_t BRAM_w_exp_9[384][64][1][1];
data_t BRAM_bn_exp_9[4][384];
data_t BRAM_w_dw_9[384][384][3][3];
data_t BRAM_bn_dw_9[4][384];
data_t BRAM_w_proj_9[64][384][1][1];
data_t BRAM_bn_proj_9[4][64];
data_t BRAM_w_exp_10[384][64][1][1];
data_t BRAM_bn_exp_10[4][384];
data_t BRAM_w_dw_10[384][384][3][3];
data_t BRAM_bn_dw_10[4][384];
data_t BRAM_w_proj_10[96][384][1][1];
data_t BRAM_bn_proj_10[4][96];
data_t BRAM_w_exp_11[576][96][1][1];
data_t BRAM_bn_exp_11[4][576];
data_t BRAM_w_dw_11[576][576][3][3];
data_t BRAM_bn_dw_11[4][576];
data_t BRAM_w_proj_11[96][576][1][1];
data_t BRAM_bn_proj_11[4][96];
data_t BRAM_w_exp_12[576][96][1][1];
data_t BRAM_bn_exp_12[4][576];
data_t BRAM_w_dw_12[576][576][3][3];
data_t BRAM_bn_dw_12[4][576];
data_t BRAM_w_proj_12[96][576][1][1];
data_t BRAM_bn_proj_12[4][96];
data_t BRAM_w_exp_13[576][96][1][1];
data_t BRAM_bn_exp_13[4][576];
data_t BRAM_w_dw_13[576][576][3][3];
data_t BRAM_bn_dw_13[4][576];
data_t BRAM_w_proj_13[160][576][1][1];
data_t BRAM_bn_proj_13[4][160];
data_t BRAM_w_exp_14[960][160][1][1];
data_t BRAM_bn_exp_14[4][960];
data_t BRAM_w_dw_14[960][960][3][3];
data_t BRAM_bn_dw_14[4][960];
data_t BRAM_w_proj_14[160][960][1][1];
data_t BRAM_bn_proj_14[4][160];
data_t BRAM_w_exp_15[960][160][1][1];
data_t BRAM_bn_exp_15[4][960];
data_t BRAM_w_dw_15[960][960][3][3];
data_t BRAM_bn_dw_15[4][960];
data_t BRAM_w_proj_15[160][960][1][1];
data_t BRAM_bn_proj_15[4][160];
data_t BRAM_w_exp_16[960][160][1][1];
data_t BRAM_bn_exp_16[4][960];
data_t BRAM_w_dw_16[960][960][3][3];
data_t BRAM_bn_dw_16[4][960];
data_t BRAM_w_proj_16[320][960][1][1];
data_t BRAM_bn_proj_16[4][320];
data_t BRAM_w_final[1280][320][1][1];
data_t BRAM_bn_final[4][1280];
data_t BRAM_pool[1280][1][1];
data_t BRAM_out[1000][1][1];
data_t BRAM_fc[1000][1280][1][1];

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
 *   Kernel : [32][3][3][3]
 *   Output : [32][112][112]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [32] is expected.
 * For group convolution, we assume:
 *   3 % groups == 0
 *   32 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_3_32_224_224_112_112_3_1_2_ap_fixed_16_5_(
    data_t input[3][224][224],
    data_t kernel[32][3][3][3],
    data_t output[32][112][112]
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
            for (int co = 0; co < 32; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int in_row = i * 2 - 1 + kh;
                    int in_col = j * 2 - 1 + kw;
                    if (in_row >= 0 && in_row < 224 && in_col >= 0 && in_col < 224) {
                        for (int ci = 0; ci < 3; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 32; co++) {
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
 *   Input/Output: [32][112][112]
 * Data type: ap_fixed<16,5>
 * Epsilon: 0
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_32_112_112_ap_fixed_16_5_(
    data_t input[32][112][112],
    data_t weights[4][32],
    data_t output[32][112][112]
)
{
    for (int c = 0; c < 32; c++) {
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
 * Tensor dimensions: [32][112][112]
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

void relu6_32_112_112_ap_fixed_16_5_(
    data_t input[32][112][112],
    data_t output[32][112][112],
    data_t cap  // cap value, typically 6
)
{
    for (int c = 0; c < 32; c++) {
        for (int i = 0; i < 112; i++) {
            for (int j = 0; j < 112; j++) {
                data_t temp = (input[c][i][j] > 0) ? input[c][i][j] : (data_t)0;
                output[c][i][j] = (temp < cap) ? temp : cap;
            }
        }
    }
}

/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [960][112][112]
 *   Kernel : [960][960][3][3]
 *   Output : [960][112][112]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [960] is expected.
 * For group convolution, we assume:
 *   960 % groups == 0
 *   960 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void group_conv2d_960_960_112_112_112_112_3_1_1_ap_fixed_16_5_(
    data_t input[960][112][112],
    data_t kernel[960][960][3][3],
    data_t output[960][112][112],
    int groups
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=1  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=4 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=1 dim=2
    // #pragma HLS array_partition variable=bias   type=cyclic factor=4   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=4  dim=1
    int c_in_per_group  = 960 / groups;
    int c_out_per_group = 960 / groups;

    // Initialize output to ((data_t)0), with co in the innermost loop
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int co = 0; co < 960; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Group convolution (co in the innermost loop)
    for (int g = 0; g < groups; g++) {
        int co_start = g * c_out_per_group;
        int co_end   = co_start + c_out_per_group;

        int ci_start = g * c_in_per_group;
        int ci_end   = ci_start + c_in_per_group;

        for (int i = 0; i < 112; i++) {
            for (int j = 0; j < 112; j++) {
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int in_row = i * 1 - 1 + kh;
                        int in_col = j * 1 - 1 + kw;
                        if (in_row >= 0 && in_row < 112 && in_col >= 0 && in_col < 112) {
                            for (int ci = ci_start; ci < ci_end; ci++) {
                            #pragma HLS unroll factor=1
                                for (int co = co_start; co < co_end; co++) {
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
}


/*
 * Auto-generated Batch Normalization HLS Code
 *
 * Dimensions:
 *   Input/Output: [960][112][112]
 * Data type: ap_fixed<16,5>
 * Epsilon: 0
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_960_112_112_ap_fixed_16_5_(
    data_t input[960][112][112],
    data_t weights[4][960],
    data_t output[960][112][112]
)
{
    for (int c = 0; c < 960; c++) {
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
 * Tensor dimensions: [960][112][112]
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

void relu6_960_112_112_ap_fixed_16_5_(
    data_t input[960][112][112],
    data_t output[960][112][112],
    data_t cap  // cap value, typically 6
)
{
    for (int c = 0; c < 960; c++) {
        for (int i = 0; i < 112; i++) {
            for (int j = 0; j < 112; j++) {
                data_t temp = (input[c][i][j] > 0) ? input[c][i][j] : (data_t)0;
                output[c][i][j] = (temp < cap) ? temp : cap;
            }
        }
    }
}

/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [960][112][112]
 *   Kernel : [320][960][1][1]
 *   Output : [320][112][112]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [320] is expected.
 * For group convolution, we assume:
 *   960 % groups == 0
 *   320 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_960_320_112_112_112_112_1_0_1_ap_fixed_16_5_(
    data_t input[960][112][112],
    data_t kernel[320][960][1][1],
    data_t output[320][112][112]
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
            for (int co = 0; co < 320; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    int in_row = i * 1 - 0 + kh;
                    int in_col = j * 1 - 0 + kw;
                    if (in_row >= 0 && in_row < 112 && in_col >= 0 && in_col < 112) {
                        for (int ci = 0; ci < 960; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 320; co++) {
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
 *   Input/Output: [320][112][112]
 * Data type: ap_fixed<16,5>
 * Epsilon: 0
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_320_112_112_ap_fixed_16_5_(
    data_t input[320][112][112],
    data_t weights[4][320],
    data_t output[320][112][112]
)
{
    for (int c = 0; c < 320; c++) {
        for (int h = 0; h < 112; h++) {
            for (int w = 0; w < 112; w++) {
                data_t norm = (input[c][h][w] - weights[2][c]) / hls::sqrt(weights[3][c] + (data_t)0);
                output[c][h][w] = weights[0][c] * norm + weights[1][c];
            }
        }
    }
}


/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [160][112][112]
 *   Kernel : [960][160][1][1]
 *   Output : [960][112][112]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [960] is expected.
 * For group convolution, we assume:
 *   160 % groups == 0
 *   960 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_160_960_112_112_112_112_1_0_1_ap_fixed_16_5_(
    data_t input[160][112][112],
    data_t kernel[960][160][1][1],
    data_t output[960][112][112]
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
            for (int co = 0; co < 960; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    int in_row = i * 1 - 0 + kh;
                    int in_col = j * 1 - 0 + kw;
                    if (in_row >= 0 && in_row < 112 && in_col >= 0 && in_col < 112) {
                        for (int ci = 0; ci < 160; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 960; co++) {
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


void matrix_add_160_112_112_ap_fixed_16_5_(
    data_t in1[160][112][112],
    data_t in2[160][112][112],
    data_t out[160][112][112]
)
{
    for (int c = 0; c < 160; c++) {
        for (int i = 0; i < 112; i++) {
            for (int j = 0; j < 112; j++) {
                out[c][i][j] = in1[c][i][j] + in2[c][i][j];
            }
        }
    }
}







/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [320][112][112]
 *   Kernel : [1280][320][1][1]
 *   Output : [1280][112][112]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [1280] is expected.
 * For group convolution, we assume:
 *   320 % groups == 0
 *   1280 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_320_1280_112_112_112_112_1_0_1_ap_fixed_16_5_(
    data_t input[320][112][112],
    data_t kernel[1280][320][1][1],
    data_t output[1280][112][112]
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
            for (int co = 0; co < 1280; co++) {
            #pragma HLS unroll factor=4
                output[co][i][j] = ((data_t)0);
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            for (int kh = 0; kh < 1; kh++) {
                for (int kw = 0; kw < 1; kw++) {
                    int in_row = i * 1 - 0 + kh;
                    int in_col = j * 1 - 0 + kw;
                    if (in_row >= 0 && in_row < 112 && in_col >= 0 && in_col < 112) {
                        for (int ci = 0; ci < 320; ci++) {
                        #pragma HLS unroll factor=1
                            for (int co = 0; co < 1280; co++) {
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
 *   Input/Output: [1280][112][112]
 * Data type: ap_fixed<16,5>
 * Epsilon: 0
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_1280_112_112_ap_fixed_16_5_(
    data_t input[1280][112][112],
    data_t weights[4][1280],
    data_t output[1280][112][112]
)
{
    for (int c = 0; c < 1280; c++) {
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
 * Tensor dimensions: [1280][112][112]
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

void relu6_1280_112_112_ap_fixed_16_5_(
    data_t input[1280][112][112],
    data_t output[1280][112][112],
    data_t cap  // cap value, typically 6
)
{
    for (int c = 0; c < 1280; c++) {
        for (int i = 0; i < 112; i++) {
            for (int j = 0; j < 112; j++) {
                data_t temp = (input[c][i][j] > 0) ? input[c][i][j] : (data_t)0;
                output[c][i][j] = (temp < cap) ? temp : cap;
            }
        }
    }
}

void adaptive_avgpool_1280_112_112_1_1_ap_fixed_16_5_(
    data_t input[1280][112][112],
    data_t output[1280][1][1]
)
{
    for (int c = 0; c < 1280; c++) {
        for (int oh = 0; oh < 1; oh++) {
            int h_start = (int)floor(( (float)oh * 112 ) / 1);
            int h_end   = (int)ceil(( (float)(oh + 1) * 112 ) / 1);
            if (h_end > 112) {
                h_end = 112;
            }
            for (int ow = 0; ow < 1; ow++) {
                int w_start = (int)floor(( (float)ow * 112 ) / 1);
                int w_end   = (int)ceil(( (float)(ow + 1) * 112 ) / 1);
                if (w_end > 112) {
                    w_end = 112;
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
 *   Input  : [1280][1][1]
 *   Kernel : [1000][1280][1][1]
 *   Output : [1000][1][1]
 *
 * Data type: ap_fixed<16,5>
 *
 * If bias is included, a 1D bias array of size [1000] is expected.
 * For group convolution, we assume:
 *   1280 % groups == 0
 *   1000 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_1280_1000_1_1_1_1_1_0_1_ap_fixed_16_5_(
    data_t input[1280][1][1],
    data_t kernel[1000][1280][1][1],
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
                        for (int ci = 0; ci < 1280; ci++) {
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

void top(data_t DRAM_input[3][224][224], data_t DRAM_w_stem[32][3][3][3], data_t DRAM_bn_stem[4][32], data_t DRAM_w_dw_0[32][32][3][3], data_t DRAM_bn_dw_0[4][32], data_t DRAM_w_proj_0[16][32][1][1], data_t DRAM_bn_proj_0[4][16], data_t DRAM_w_exp_1[96][16][1][1], data_t DRAM_bn_exp_1[4][96], data_t DRAM_w_dw_1[96][96][3][3], data_t DRAM_bn_dw_1[4][96], data_t DRAM_w_proj_1[24][96][1][1], data_t DRAM_bn_proj_1[4][24], data_t DRAM_w_exp_2[144][24][1][1], data_t DRAM_bn_exp_2[4][144], data_t DRAM_w_dw_2[144][144][3][3], data_t DRAM_bn_dw_2[4][144], data_t DRAM_w_proj_2[24][144][1][1], data_t DRAM_bn_proj_2[4][24], data_t DRAM_w_exp_3[144][24][1][1], data_t DRAM_bn_exp_3[4][144], data_t DRAM_w_dw_3[144][144][3][3], data_t DRAM_bn_dw_3[4][144], data_t DRAM_w_proj_3[32][144][1][1], data_t DRAM_bn_proj_3[4][32], data_t DRAM_w_exp_4[192][32][1][1], data_t DRAM_bn_exp_4[4][192], data_t DRAM_w_dw_4[192][192][3][3], data_t DRAM_bn_dw_4[4][192], data_t DRAM_w_proj_4[32][192][1][1], data_t DRAM_bn_proj_4[4][32], data_t DRAM_w_exp_5[192][32][1][1], data_t DRAM_bn_exp_5[4][192], data_t DRAM_w_dw_5[192][192][3][3], data_t DRAM_bn_dw_5[4][192], data_t DRAM_w_proj_5[32][192][1][1], data_t DRAM_bn_proj_5[4][32], data_t DRAM_w_exp_6[192][32][1][1], data_t DRAM_bn_exp_6[4][192], data_t DRAM_w_dw_6[192][192][3][3], data_t DRAM_bn_dw_6[4][192], data_t DRAM_w_proj_6[64][192][1][1], data_t DRAM_bn_proj_6[4][64], data_t DRAM_w_exp_7[384][64][1][1], data_t DRAM_bn_exp_7[4][384], data_t DRAM_w_dw_7[384][384][3][3], data_t DRAM_bn_dw_7[4][384], data_t DRAM_w_proj_7[64][384][1][1], data_t DRAM_bn_proj_7[4][64], data_t DRAM_w_exp_8[384][64][1][1], data_t DRAM_bn_exp_8[4][384], data_t DRAM_w_dw_8[384][384][3][3], data_t DRAM_bn_dw_8[4][384], data_t DRAM_w_proj_8[64][384][1][1], data_t DRAM_bn_proj_8[4][64], data_t DRAM_w_exp_9[384][64][1][1], data_t DRAM_bn_exp_9[4][384], data_t DRAM_w_dw_9[384][384][3][3], data_t DRAM_bn_dw_9[4][384], data_t DRAM_w_proj_9[64][384][1][1], data_t DRAM_bn_proj_9[4][64], data_t DRAM_w_exp_10[384][64][1][1], data_t DRAM_bn_exp_10[4][384], data_t DRAM_w_dw_10[384][384][3][3], data_t DRAM_bn_dw_10[4][384], data_t DRAM_w_proj_10[96][384][1][1], data_t DRAM_bn_proj_10[4][96], data_t DRAM_w_exp_11[576][96][1][1], data_t DRAM_bn_exp_11[4][576], data_t DRAM_w_dw_11[576][576][3][3], data_t DRAM_bn_dw_11[4][576], data_t DRAM_w_proj_11[96][576][1][1], data_t DRAM_bn_proj_11[4][96], data_t DRAM_w_exp_12[576][96][1][1], data_t DRAM_bn_exp_12[4][576], data_t DRAM_w_dw_12[576][576][3][3], data_t DRAM_bn_dw_12[4][576], data_t DRAM_w_proj_12[96][576][1][1], data_t DRAM_bn_proj_12[4][96], data_t DRAM_w_exp_13[576][96][1][1], data_t DRAM_bn_exp_13[4][576], data_t DRAM_w_dw_13[576][576][3][3], data_t DRAM_bn_dw_13[4][576], data_t DRAM_w_proj_13[160][576][1][1], data_t DRAM_bn_proj_13[4][160], data_t DRAM_w_exp_14[960][160][1][1], data_t DRAM_bn_exp_14[4][960], data_t DRAM_w_dw_14[960][960][3][3], data_t DRAM_bn_dw_14[4][960], data_t DRAM_w_proj_14[160][960][1][1], data_t DRAM_bn_proj_14[4][160], data_t DRAM_w_exp_15[960][160][1][1], data_t DRAM_bn_exp_15[4][960], data_t DRAM_w_dw_15[960][960][3][3], data_t DRAM_bn_dw_15[4][960], data_t DRAM_w_proj_15[160][960][1][1], data_t DRAM_bn_proj_15[4][160], data_t DRAM_w_exp_16[960][160][1][1], data_t DRAM_bn_exp_16[4][960], data_t DRAM_w_dw_16[960][960][3][3], data_t DRAM_bn_dw_16[4][960], data_t DRAM_w_proj_16[320][960][1][1], data_t DRAM_bn_proj_16[4][320], data_t DRAM_w_final[1280][320][1][1], data_t DRAM_bn_final[4][1280], data_t DRAM_out[1000][1][1], data_t DRAM_fc[1000][1280][1][1])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem_input
    #pragma HLS interface m_axi port=DRAM_w_stem offset=slave bundle=mem_w_stem
    #pragma HLS interface m_axi port=DRAM_bn_stem offset=slave bundle=mem_bn_stem
    #pragma HLS interface m_axi port=DRAM_w_dw_0 offset=slave bundle=mem_w_dw_0
    #pragma HLS interface m_axi port=DRAM_bn_dw_0 offset=slave bundle=mem_bn_dw_0
    #pragma HLS interface m_axi port=DRAM_w_proj_0 offset=slave bundle=mem_w_proj_0
    #pragma HLS interface m_axi port=DRAM_bn_proj_0 offset=slave bundle=mem_bn_proj_0
    #pragma HLS interface m_axi port=DRAM_w_exp_1 offset=slave bundle=mem_w_exp_1
    #pragma HLS interface m_axi port=DRAM_bn_exp_1 offset=slave bundle=mem_bn_exp_1
    #pragma HLS interface m_axi port=DRAM_w_dw_1 offset=slave bundle=mem_w_dw_1
    #pragma HLS interface m_axi port=DRAM_bn_dw_1 offset=slave bundle=mem_bn_dw_1
    #pragma HLS interface m_axi port=DRAM_w_proj_1 offset=slave bundle=mem_w_proj_1
    #pragma HLS interface m_axi port=DRAM_bn_proj_1 offset=slave bundle=mem_bn_proj_1
    #pragma HLS interface m_axi port=DRAM_w_exp_2 offset=slave bundle=mem_w_exp_2
    #pragma HLS interface m_axi port=DRAM_bn_exp_2 offset=slave bundle=mem_bn_exp_2
    #pragma HLS interface m_axi port=DRAM_w_dw_2 offset=slave bundle=mem_w_dw_2
    #pragma HLS interface m_axi port=DRAM_bn_dw_2 offset=slave bundle=mem_bn_dw_2
    #pragma HLS interface m_axi port=DRAM_w_proj_2 offset=slave bundle=mem_w_proj_2
    #pragma HLS interface m_axi port=DRAM_bn_proj_2 offset=slave bundle=mem_bn_proj_2
    #pragma HLS interface m_axi port=DRAM_w_exp_3 offset=slave bundle=mem_w_exp_3
    #pragma HLS interface m_axi port=DRAM_bn_exp_3 offset=slave bundle=mem_bn_exp_3
    #pragma HLS interface m_axi port=DRAM_w_dw_3 offset=slave bundle=mem_w_dw_3
    #pragma HLS interface m_axi port=DRAM_bn_dw_3 offset=slave bundle=mem_bn_dw_3
    #pragma HLS interface m_axi port=DRAM_w_proj_3 offset=slave bundle=mem_w_proj_3
    #pragma HLS interface m_axi port=DRAM_bn_proj_3 offset=slave bundle=mem_bn_proj_3
    #pragma HLS interface m_axi port=DRAM_w_exp_4 offset=slave bundle=mem_w_exp_4
    #pragma HLS interface m_axi port=DRAM_bn_exp_4 offset=slave bundle=mem_bn_exp_4
    #pragma HLS interface m_axi port=DRAM_w_dw_4 offset=slave bundle=mem_w_dw_4
    #pragma HLS interface m_axi port=DRAM_bn_dw_4 offset=slave bundle=mem_bn_dw_4
    #pragma HLS interface m_axi port=DRAM_w_proj_4 offset=slave bundle=mem_w_proj_4
    #pragma HLS interface m_axi port=DRAM_bn_proj_4 offset=slave bundle=mem_bn_proj_4
    #pragma HLS interface m_axi port=DRAM_w_exp_5 offset=slave bundle=mem_w_exp_5
    #pragma HLS interface m_axi port=DRAM_bn_exp_5 offset=slave bundle=mem_bn_exp_5
    #pragma HLS interface m_axi port=DRAM_w_dw_5 offset=slave bundle=mem_w_dw_5
    #pragma HLS interface m_axi port=DRAM_bn_dw_5 offset=slave bundle=mem_bn_dw_5
    #pragma HLS interface m_axi port=DRAM_w_proj_5 offset=slave bundle=mem_w_proj_5
    #pragma HLS interface m_axi port=DRAM_bn_proj_5 offset=slave bundle=mem_bn_proj_5
    #pragma HLS interface m_axi port=DRAM_w_exp_6 offset=slave bundle=mem_w_exp_6
    #pragma HLS interface m_axi port=DRAM_bn_exp_6 offset=slave bundle=mem_bn_exp_6
    #pragma HLS interface m_axi port=DRAM_w_dw_6 offset=slave bundle=mem_w_dw_6
    #pragma HLS interface m_axi port=DRAM_bn_dw_6 offset=slave bundle=mem_bn_dw_6
    #pragma HLS interface m_axi port=DRAM_w_proj_6 offset=slave bundle=mem_w_proj_6
    #pragma HLS interface m_axi port=DRAM_bn_proj_6 offset=slave bundle=mem_bn_proj_6
    #pragma HLS interface m_axi port=DRAM_w_exp_7 offset=slave bundle=mem_w_exp_7
    #pragma HLS interface m_axi port=DRAM_bn_exp_7 offset=slave bundle=mem_bn_exp_7
    #pragma HLS interface m_axi port=DRAM_w_dw_7 offset=slave bundle=mem_w_dw_7
    #pragma HLS interface m_axi port=DRAM_bn_dw_7 offset=slave bundle=mem_bn_dw_7
    #pragma HLS interface m_axi port=DRAM_w_proj_7 offset=slave bundle=mem_w_proj_7
    #pragma HLS interface m_axi port=DRAM_bn_proj_7 offset=slave bundle=mem_bn_proj_7
    #pragma HLS interface m_axi port=DRAM_w_exp_8 offset=slave bundle=mem_w_exp_8
    #pragma HLS interface m_axi port=DRAM_bn_exp_8 offset=slave bundle=mem_bn_exp_8
    #pragma HLS interface m_axi port=DRAM_w_dw_8 offset=slave bundle=mem_w_dw_8
    #pragma HLS interface m_axi port=DRAM_bn_dw_8 offset=slave bundle=mem_bn_dw_8
    #pragma HLS interface m_axi port=DRAM_w_proj_8 offset=slave bundle=mem_w_proj_8
    #pragma HLS interface m_axi port=DRAM_bn_proj_8 offset=slave bundle=mem_bn_proj_8
    #pragma HLS interface m_axi port=DRAM_w_exp_9 offset=slave bundle=mem_w_exp_9
    #pragma HLS interface m_axi port=DRAM_bn_exp_9 offset=slave bundle=mem_bn_exp_9
    #pragma HLS interface m_axi port=DRAM_w_dw_9 offset=slave bundle=mem_w_dw_9
    #pragma HLS interface m_axi port=DRAM_bn_dw_9 offset=slave bundle=mem_bn_dw_9
    #pragma HLS interface m_axi port=DRAM_w_proj_9 offset=slave bundle=mem_w_proj_9
    #pragma HLS interface m_axi port=DRAM_bn_proj_9 offset=slave bundle=mem_bn_proj_9
    #pragma HLS interface m_axi port=DRAM_w_exp_10 offset=slave bundle=mem_w_exp_10
    #pragma HLS interface m_axi port=DRAM_bn_exp_10 offset=slave bundle=mem_bn_exp_10
    #pragma HLS interface m_axi port=DRAM_w_dw_10 offset=slave bundle=mem_w_dw_10
    #pragma HLS interface m_axi port=DRAM_bn_dw_10 offset=slave bundle=mem_bn_dw_10
    #pragma HLS interface m_axi port=DRAM_w_proj_10 offset=slave bundle=mem_w_proj_10
    #pragma HLS interface m_axi port=DRAM_bn_proj_10 offset=slave bundle=mem_bn_proj_10
    #pragma HLS interface m_axi port=DRAM_w_exp_11 offset=slave bundle=mem_w_exp_11
    #pragma HLS interface m_axi port=DRAM_bn_exp_11 offset=slave bundle=mem_bn_exp_11
    #pragma HLS interface m_axi port=DRAM_w_dw_11 offset=slave bundle=mem_w_dw_11
    #pragma HLS interface m_axi port=DRAM_bn_dw_11 offset=slave bundle=mem_bn_dw_11
    #pragma HLS interface m_axi port=DRAM_w_proj_11 offset=slave bundle=mem_w_proj_11
    #pragma HLS interface m_axi port=DRAM_bn_proj_11 offset=slave bundle=mem_bn_proj_11
    #pragma HLS interface m_axi port=DRAM_w_exp_12 offset=slave bundle=mem_w_exp_12
    #pragma HLS interface m_axi port=DRAM_bn_exp_12 offset=slave bundle=mem_bn_exp_12
    #pragma HLS interface m_axi port=DRAM_w_dw_12 offset=slave bundle=mem_w_dw_12
    #pragma HLS interface m_axi port=DRAM_bn_dw_12 offset=slave bundle=mem_bn_dw_12
    #pragma HLS interface m_axi port=DRAM_w_proj_12 offset=slave bundle=mem_w_proj_12
    #pragma HLS interface m_axi port=DRAM_bn_proj_12 offset=slave bundle=mem_bn_proj_12
    #pragma HLS interface m_axi port=DRAM_w_exp_13 offset=slave bundle=mem_w_exp_13
    #pragma HLS interface m_axi port=DRAM_bn_exp_13 offset=slave bundle=mem_bn_exp_13
    #pragma HLS interface m_axi port=DRAM_w_dw_13 offset=slave bundle=mem_w_dw_13
    #pragma HLS interface m_axi port=DRAM_bn_dw_13 offset=slave bundle=mem_bn_dw_13
    #pragma HLS interface m_axi port=DRAM_w_proj_13 offset=slave bundle=mem_w_proj_13
    #pragma HLS interface m_axi port=DRAM_bn_proj_13 offset=slave bundle=mem_bn_proj_13
    #pragma HLS interface m_axi port=DRAM_w_exp_14 offset=slave bundle=mem_w_exp_14
    #pragma HLS interface m_axi port=DRAM_bn_exp_14 offset=slave bundle=mem_bn_exp_14
    #pragma HLS interface m_axi port=DRAM_w_dw_14 offset=slave bundle=mem_w_dw_14
    #pragma HLS interface m_axi port=DRAM_bn_dw_14 offset=slave bundle=mem_bn_dw_14
    #pragma HLS interface m_axi port=DRAM_w_proj_14 offset=slave bundle=mem_w_proj_14
    #pragma HLS interface m_axi port=DRAM_bn_proj_14 offset=slave bundle=mem_bn_proj_14
    #pragma HLS interface m_axi port=DRAM_w_exp_15 offset=slave bundle=mem_w_exp_15
    #pragma HLS interface m_axi port=DRAM_bn_exp_15 offset=slave bundle=mem_bn_exp_15
    #pragma HLS interface m_axi port=DRAM_w_dw_15 offset=slave bundle=mem_w_dw_15
    #pragma HLS interface m_axi port=DRAM_bn_dw_15 offset=slave bundle=mem_bn_dw_15
    #pragma HLS interface m_axi port=DRAM_w_proj_15 offset=slave bundle=mem_w_proj_15
    #pragma HLS interface m_axi port=DRAM_bn_proj_15 offset=slave bundle=mem_bn_proj_15
    #pragma HLS interface m_axi port=DRAM_w_exp_16 offset=slave bundle=mem_w_exp_16
    #pragma HLS interface m_axi port=DRAM_bn_exp_16 offset=slave bundle=mem_bn_exp_16
    #pragma HLS interface m_axi port=DRAM_w_dw_16 offset=slave bundle=mem_w_dw_16
    #pragma HLS interface m_axi port=DRAM_bn_dw_16 offset=slave bundle=mem_bn_dw_16
    #pragma HLS interface m_axi port=DRAM_w_proj_16 offset=slave bundle=mem_w_proj_16
    #pragma HLS interface m_axi port=DRAM_bn_proj_16 offset=slave bundle=mem_bn_proj_16
    #pragma HLS interface m_axi port=DRAM_w_final offset=slave bundle=mem_w_final
    #pragma HLS interface m_axi port=DRAM_bn_final offset=slave bundle=mem_bn_final
    #pragma HLS interface m_axi port=DRAM_out offset=slave bundle=mem_out
    #pragma HLS interface m_axi port=DRAM_fc offset=slave bundle=mem_fc

    load_3_224_224_ap_fixed_16_5_(DRAM_input, BRAM_feat_0);
    conv2d_3_32_224_224_112_112_3_1_2_ap_fixed_16_5_(BRAM_feat_0, BRAM_w_stem, BRAM_feat_1);
    batch_norm_32_112_112_ap_fixed_16_5_(BRAM_feat_1, BRAM_bn_stem, BRAM_feat_1);
    relu6_32_112_112_ap_fixed_16_5_(BRAM_feat_1, BRAM_feat_1, 6);
    group_conv2d_960_960_112_112_112_112_3_1_1_ap_fixed_16_5_(BRAM_feat_2, BRAM_w_dw_16, BRAM_feat_2, 960);
    batch_norm_960_112_112_ap_fixed_16_5_(BRAM_feat_2, BRAM_bn_dw_16, BRAM_feat_2);
    relu6_960_112_112_ap_fixed_16_5_(BRAM_feat_2, BRAM_feat_2, 6);
    conv2d_960_320_112_112_112_112_1_0_1_ap_fixed_16_5_(BRAM_feat_2, BRAM_w_proj_16, BRAM_feat_1);
    batch_norm_320_112_112_ap_fixed_16_5_(BRAM_feat_1, BRAM_bn_proj_16, BRAM_feat_1);
    conv2d_160_960_112_112_112_112_1_0_1_ap_fixed_16_5_(BRAM_feat_1, BRAM_w_exp_16, BRAM_feat_2);
    batch_norm_960_112_112_ap_fixed_16_5_(BRAM_feat_2, BRAM_bn_exp_16, BRAM_feat_2);
    relu6_960_112_112_ap_fixed_16_5_(BRAM_feat_2, BRAM_feat_2, 6);
    matrix_add_160_112_112_ap_fixed_16_5_(BRAM_feat_2, BRAM_feat_2, BRAM_feat_2);
    conv2d_320_1280_112_112_112_112_1_0_1_ap_fixed_16_5_(BRAM_feat_2, BRAM_w_final, BRAM_feat_1);
    batch_norm_1280_112_112_ap_fixed_16_5_(BRAM_feat_1, BRAM_bn_final, BRAM_feat_1);
    relu6_1280_112_112_ap_fixed_16_5_(BRAM_feat_1, BRAM_feat_1, 6);
    adaptive_avgpool_1280_112_112_1_1_ap_fixed_16_5_(BRAM_feat_1, BRAM_pool);
    conv2d_1280_1000_1_1_1_1_1_0_1_ap_fixed_16_5_(BRAM_pool, BRAM_fc, BRAM_out);
    store_1000_1_1_ap_fixed_16_5_(BRAM_out, DRAM_out);
}