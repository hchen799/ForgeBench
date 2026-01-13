
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

data_t BRAM_input_0[2048][768];
data_t BRAM_input_1[2048][768];
data_t BRAM_tmp_0[2048][768];
data_t BRAM_ln1_0[768];
data_t BRAM_ln2_0[768];
data_t BRAM_wq_0[768][768];
data_t BRAM_wk_0[768][768];
data_t BRAM_wv_0[768][768];
data_t BRAM_wo_0[768][768];
data_t BRAM_mlp_w1_0[768][3072];
data_t BRAM_mlp_w2_0[3072][768];
data_t BRAM_input_2[2048][768];
data_t BRAM_tmp_1[2048][768];
data_t BRAM_ln1_1[768];
data_t BRAM_ln2_1[768];
data_t BRAM_wq_1[768][768];
data_t BRAM_wk_1[768][768];
data_t BRAM_wv_1[768][768];
data_t BRAM_wo_1[768][768];
data_t BRAM_mlp_w1_1[768][3072];
data_t BRAM_mlp_w2_1[3072][768];
data_t BRAM_input_3[2048][768];
data_t BRAM_tmp_2[2048][768];
data_t BRAM_ln1_2[768];
data_t BRAM_ln2_2[768];
data_t BRAM_wq_2[768][768];
data_t BRAM_wk_2[768][768];
data_t BRAM_wv_2[768][768];
data_t BRAM_wo_2[768][768];
data_t BRAM_mlp_w1_2[768][3072];
data_t BRAM_mlp_w2_2[3072][768];
data_t BRAM_input_4[2048][768];
data_t BRAM_tmp_3[2048][768];
data_t BRAM_ln1_3[768];
data_t BRAM_ln2_3[768];
data_t BRAM_wq_3[768][768];
data_t BRAM_wk_3[768][768];
data_t BRAM_wv_3[768][768];
data_t BRAM_wo_3[768][768];
data_t BRAM_mlp_w1_3[768][3072];
data_t BRAM_mlp_w2_3[3072][768];
data_t BRAM_input_5[2048][768];
data_t BRAM_tmp_4[2048][768];
data_t BRAM_ln1_4[768];
data_t BRAM_ln2_4[768];
data_t BRAM_wq_4[768][768];
data_t BRAM_wk_4[768][768];
data_t BRAM_wv_4[768][768];
data_t BRAM_wo_4[768][768];
data_t BRAM_mlp_w1_4[768][3072];
data_t BRAM_mlp_w2_4[3072][768];
data_t BRAM_input_6[2048][768];
data_t BRAM_tmp_5[2048][768];
data_t BRAM_ln1_5[768];
data_t BRAM_ln2_5[768];
data_t BRAM_wq_5[768][768];
data_t BRAM_wk_5[768][768];
data_t BRAM_wv_5[768][768];
data_t BRAM_wo_5[768][768];
data_t BRAM_mlp_w1_5[768][3072];
data_t BRAM_mlp_w2_5[3072][768];
data_t BRAM_input_7[2048][768];
data_t BRAM_tmp_6[2048][768];
data_t BRAM_ln1_6[768];
data_t BRAM_ln2_6[768];
data_t BRAM_wq_6[768][768];
data_t BRAM_wk_6[768][768];
data_t BRAM_wv_6[768][768];
data_t BRAM_wo_6[768][768];
data_t BRAM_mlp_w1_6[768][3072];
data_t BRAM_mlp_w2_6[3072][768];
data_t BRAM_input_8[2048][768];
data_t BRAM_tmp_7[2048][768];
data_t BRAM_ln1_7[768];
data_t BRAM_ln2_7[768];
data_t BRAM_wq_7[768][768];
data_t BRAM_wk_7[768][768];
data_t BRAM_wv_7[768][768];
data_t BRAM_wo_7[768][768];
data_t BRAM_mlp_w1_7[768][3072];
data_t BRAM_mlp_w2_7[3072][768];
data_t BRAM_input_9[2048][768];
data_t BRAM_tmp_8[2048][768];
data_t BRAM_ln1_8[768];
data_t BRAM_ln2_8[768];
data_t BRAM_wq_8[768][768];
data_t BRAM_wk_8[768][768];
data_t BRAM_wv_8[768][768];
data_t BRAM_wo_8[768][768];
data_t BRAM_mlp_w1_8[768][3072];
data_t BRAM_mlp_w2_8[3072][768];
data_t BRAM_input_10[2048][768];
data_t BRAM_tmp_9[2048][768];
data_t BRAM_ln1_9[768];
data_t BRAM_ln2_9[768];
data_t BRAM_wq_9[768][768];
data_t BRAM_wk_9[768][768];
data_t BRAM_wv_9[768][768];
data_t BRAM_wo_9[768][768];
data_t BRAM_mlp_w1_9[768][3072];
data_t BRAM_mlp_w2_9[3072][768];
data_t BRAM_input_11[2048][768];
data_t BRAM_tmp_10[2048][768];
data_t BRAM_ln1_10[768];
data_t BRAM_ln2_10[768];
data_t BRAM_wq_10[768][768];
data_t BRAM_wk_10[768][768];
data_t BRAM_wv_10[768][768];
data_t BRAM_wo_10[768][768];
data_t BRAM_mlp_w1_10[768][3072];
data_t BRAM_mlp_w2_10[3072][768];
data_t BRAM_input_12[2048][768];
data_t BRAM_tmp_11[2048][768];
data_t BRAM_ln1_11[768];
data_t BRAM_ln2_11[768];
data_t BRAM_wq_11[768][768];
data_t BRAM_wk_11[768][768];
data_t BRAM_wv_11[768][768];
data_t BRAM_wo_11[768][768];
data_t BRAM_mlp_w1_11[768][3072];
data_t BRAM_mlp_w2_11[3072][768];

void load_2048_768_ap_fixed_16_5_(data_t input[2048][768], data_t output[2048][768])
{
    for (int idx0 = 0; idx0 < 2048; idx0++) {
        for (int idx1 = 0; idx1 < 768; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_768_ap_fixed_16_5_(data_t input[768], data_t output[768])
{
    for (int idx0 = 0; idx0 < 768; idx0++) {
        output[idx0] = input[idx0];
    }
}

void load_768_768_ap_fixed_16_5_(data_t input[768][768], data_t output[768][768])
{
    for (int idx0 = 0; idx0 < 768; idx0++) {
        for (int idx1 = 0; idx1 < 768; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_768_3072_ap_fixed_16_5_(data_t input[768][3072], data_t output[768][3072])
{
    for (int idx0 = 0; idx0 < 768; idx0++) {
        for (int idx1 = 0; idx1 < 3072; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_3072_768_ap_fixed_16_5_(data_t input[3072][768], data_t output[3072][768])
{
    for (int idx0 = 0; idx0 < 3072; idx0++) {
        for (int idx1 = 0; idx1 < 768; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}



void layer_norm_2048_768_ap_fixed_16_5_(
    data_t input[2048][768],
    data_t gamma[768],
    data_t beta[768],
    data_t output[2048][768]
)
{
    for (int i = 0; i < 2048; i++) {
        // Compute mean for the i-th sequence element
        data_t sum = (data_t)0;
        for (int j = 0; j < 768; j++) {
            sum += input[i][j];
        }
        data_t mean = sum / 768;

        // Compute variance for the i-th sequence element
        data_t var_sum = (data_t)0;
        for (int j = 0; j < 768; j++) {
            data_t diff = input[i][j] - mean;
            var_sum += diff * diff;
        }
        data_t variance = var_sum / 768;

        // Normalize and scale: output = gamma * (x - mean) / sqrt(variance + epsilon) + beta
        for (int j = 0; j < 768; j++) {
            output[i][j] = gamma[j] * ((input[i][j] - mean) / hls::sqrt(variance + (data_t)1e-06)) + beta[j];
        }
    }
}



void sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(
    data_t input[2048][768],
    data_t W_q[768][768],
    data_t W_k[768][768],
    data_t W_v[768][768],
    data_t output[2048][768],
    int window_size
)
{
    const int num_heads = 12;   // total number of heads (must equal DIM_OUT / HEAD_DIM)
    const int head_dim = 64;       // dimension per head
    const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);

    data_t Q[2048][768];
    data_t K[2048][768];
    data_t V[2048][768];

    // Compute Q, K, V
    for (int i = 0; i < 2048; i++) {
        for (int d = 0; d < 768; d++) {
            Q[i][d] = 0;
            K[i][d] = 0;
            V[i][d] = 0;
            for (int j = 0; j < 768; j++) {
                Q[i][d] += input[i][j] * W_q[d][j];
                K[i][d] += input[i][j] * W_k[d][j];
                V[i][d] += input[i][j] * W_v[d][j];
            }
        }
    }

    /*==== BEGIN OPTIONAL ROPE LOGIC ====*/
    
    /*==== END OPTIONAL ROPE LOGIC ====*/

    // Sliding window attention: for each head, each query attends only to a local window.
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < 2048; i++) {
            // Determine window bounds
            int start = (i - window_size < 0) ? 0 : i - window_size;
            int end = (i + window_size >= 2048) ? 2048 - 1 : i + window_size;
            data_t scores[2048]; // Allocate full length for simplicity

            // Compute scaled dot-product scores for indices within the window.
            for (int j = start; j <= end; j++) {
                data_t sum = 0;
                for (int d = 0; d < head_dim; d++) {
                    int idx = h * head_dim + d;
                    sum += Q[i][idx] * K[j][idx];
                }
                scores[j] = sum * scale;
            }

            // Apply softmax over the window.
            data_t max_val = scores[start];
            for (int j = start + 1; j <= end; j++) {
                if (scores[j] > max_val)
                    max_val = scores[j];
            }
            data_t sum_exp = 0;
            for (int j = start; j <= end; j++) {
                scores[j] = hls::exp(scores[j] - max_val);
                sum_exp += scores[j];
            }
            for (int j = start; j <= end; j++) {
                scores[j] /= sum_exp;
            }

            // Compute context vector for the current head.
            for (int d = 0; d < head_dim; d++) {
                data_t context = 0;
                for (int j = start; j <= end; j++) {
                    int idx = h * head_dim + d;
                    context += scores[j] * V[j][idx];
                }
                output[i][h * head_dim + d] = context;
            }
        }
    }
}



void matrix_add_2048_768_ap_fixed_16_5_(
    data_t in1[2048][768],
    data_t in2[2048][768],
    data_t out[2048][768]
)
{
    
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 768; j++) {
            out[i][j] = in1[i][j] + in2[i][j];
        }
    }
    
}









void matmul_2048_768_3072_bias_ap_fixed_16_5_(
    data_t input[2048][768],
    data_t weights[3072][768],
    data_t bias[3072],
    data_t output[2048][3072]
)
{
    // Initialize output to bias[j]
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 3072; j++) {
            output[i][j] = bias[j];
        }
    }

    // Matrix multiplication
    for (int i = 0; i < 2048; i++) {
        for (int k = 0; k < 768; k++) {
            for (int j = 0; j < 3072; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}




void gelu_2048_3072_ap_fixed_16_5_(
    data_t input[2048][3072],
    data_t output[2048][3072]
)
{
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const data_t sqrt_2_over_pi = hls::sqrt((data_t)2/(data_t)3.141592653589793);
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 3072; j++) {
            data_t x = input[i][j];
            data_t x_cube = x * x * x;
            data_t tanh_arg = sqrt_2_over_pi * (x + (data_t)0.044715 * x_cube);
            output[i][j] = (data_t)0.5 * x * (1 + hls::tanh(tanh_arg));
        }
    }
}



void matmul_2048_3072_768_bias_ap_fixed_16_5_(
    data_t input[2048][3072],
    data_t weights[768][3072],
    data_t bias[768],
    data_t output[2048][768]
)
{
    // Initialize output to bias[j]
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 768; j++) {
            output[i][j] = bias[j];
        }
    }

    // Matrix multiplication
    for (int i = 0; i < 2048; i++) {
        for (int k = 0; k < 3072; k++) {
            for (int j = 0; j < 768; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}


void store_2048_768_ap_fixed_16_5_(data_t input[2048][768], data_t output[2048][768])
{
    for (int idx0 = 0; idx0 < 2048; idx0++) {
        for (int idx1 = 0; idx1 < 768; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_input[2048][768], data_t DRAM_ln1_0[768], data_t DRAM_ln2_0[768], data_t DRAM_wq_0[768][768], data_t DRAM_wk_0[768][768], data_t DRAM_wv_0[768][768], data_t DRAM_wo_0[768][768], data_t DRAM_mlp_w1_0[768][3072], data_t DRAM_mlp_w2_0[3072][768], data_t DRAM_ln1_1[768], data_t DRAM_ln2_1[768], data_t DRAM_wq_1[768][768], data_t DRAM_wk_1[768][768], data_t DRAM_wv_1[768][768], data_t DRAM_wo_1[768][768], data_t DRAM_mlp_w1_1[768][3072], data_t DRAM_mlp_w2_1[3072][768], data_t DRAM_ln1_2[768], data_t DRAM_ln2_2[768], data_t DRAM_wq_2[768][768], data_t DRAM_wk_2[768][768], data_t DRAM_wv_2[768][768], data_t DRAM_wo_2[768][768], data_t DRAM_mlp_w1_2[768][3072], data_t DRAM_mlp_w2_2[3072][768], data_t DRAM_ln1_3[768], data_t DRAM_ln2_3[768], data_t DRAM_wq_3[768][768], data_t DRAM_wk_3[768][768], data_t DRAM_wv_3[768][768], data_t DRAM_wo_3[768][768], data_t DRAM_mlp_w1_3[768][3072], data_t DRAM_mlp_w2_3[3072][768], data_t DRAM_ln1_4[768], data_t DRAM_ln2_4[768], data_t DRAM_wq_4[768][768], data_t DRAM_wk_4[768][768], data_t DRAM_wv_4[768][768], data_t DRAM_wo_4[768][768], data_t DRAM_mlp_w1_4[768][3072], data_t DRAM_mlp_w2_4[3072][768], data_t DRAM_ln1_5[768], data_t DRAM_ln2_5[768], data_t DRAM_wq_5[768][768], data_t DRAM_wk_5[768][768], data_t DRAM_wv_5[768][768], data_t DRAM_wo_5[768][768], data_t DRAM_mlp_w1_5[768][3072], data_t DRAM_mlp_w2_5[3072][768], data_t DRAM_ln1_6[768], data_t DRAM_ln2_6[768], data_t DRAM_wq_6[768][768], data_t DRAM_wk_6[768][768], data_t DRAM_wv_6[768][768], data_t DRAM_wo_6[768][768], data_t DRAM_mlp_w1_6[768][3072], data_t DRAM_mlp_w2_6[3072][768], data_t DRAM_ln1_7[768], data_t DRAM_ln2_7[768], data_t DRAM_wq_7[768][768], data_t DRAM_wk_7[768][768], data_t DRAM_wv_7[768][768], data_t DRAM_wo_7[768][768], data_t DRAM_mlp_w1_7[768][3072], data_t DRAM_mlp_w2_7[3072][768], data_t DRAM_ln1_8[768], data_t DRAM_ln2_8[768], data_t DRAM_wq_8[768][768], data_t DRAM_wk_8[768][768], data_t DRAM_wv_8[768][768], data_t DRAM_wo_8[768][768], data_t DRAM_mlp_w1_8[768][3072], data_t DRAM_mlp_w2_8[3072][768], data_t DRAM_ln1_9[768], data_t DRAM_ln2_9[768], data_t DRAM_wq_9[768][768], data_t DRAM_wk_9[768][768], data_t DRAM_wv_9[768][768], data_t DRAM_wo_9[768][768], data_t DRAM_mlp_w1_9[768][3072], data_t DRAM_mlp_w2_9[3072][768], data_t DRAM_ln1_10[768], data_t DRAM_ln2_10[768], data_t DRAM_wq_10[768][768], data_t DRAM_wk_10[768][768], data_t DRAM_wv_10[768][768], data_t DRAM_wo_10[768][768], data_t DRAM_mlp_w1_10[768][3072], data_t DRAM_mlp_w2_10[3072][768], data_t DRAM_ln1_11[768], data_t DRAM_ln2_11[768], data_t DRAM_wq_11[768][768], data_t DRAM_wk_11[768][768], data_t DRAM_wv_11[768][768], data_t DRAM_wo_11[768][768], data_t DRAM_mlp_w1_11[768][3072], data_t DRAM_mlp_w2_11[3072][768], data_t DRAM_output[2048][768])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem_input
    #pragma HLS interface m_axi port=DRAM_ln1_0 offset=slave bundle=mem_ln1_0
    #pragma HLS interface m_axi port=DRAM_ln2_0 offset=slave bundle=mem_ln2_0
    #pragma HLS interface m_axi port=DRAM_wq_0 offset=slave bundle=mem_wq_0
    #pragma HLS interface m_axi port=DRAM_wk_0 offset=slave bundle=mem_wk_0
    #pragma HLS interface m_axi port=DRAM_wv_0 offset=slave bundle=mem_wv_0
    #pragma HLS interface m_axi port=DRAM_wo_0 offset=slave bundle=mem_wo_0
    #pragma HLS interface m_axi port=DRAM_mlp_w1_0 offset=slave bundle=mem_mlp1_0
    #pragma HLS interface m_axi port=DRAM_mlp_w2_0 offset=slave bundle=mem_mlp2_0
    #pragma HLS interface m_axi port=DRAM_ln1_1 offset=slave bundle=mem_ln1_1
    #pragma HLS interface m_axi port=DRAM_ln2_1 offset=slave bundle=mem_ln2_1
    #pragma HLS interface m_axi port=DRAM_wq_1 offset=slave bundle=mem_wq_1
    #pragma HLS interface m_axi port=DRAM_wk_1 offset=slave bundle=mem_wk_1
    #pragma HLS interface m_axi port=DRAM_wv_1 offset=slave bundle=mem_wv_1
    #pragma HLS interface m_axi port=DRAM_wo_1 offset=slave bundle=mem_wo_1
    #pragma HLS interface m_axi port=DRAM_mlp_w1_1 offset=slave bundle=mem_mlp1_1
    #pragma HLS interface m_axi port=DRAM_mlp_w2_1 offset=slave bundle=mem_mlp2_1
    #pragma HLS interface m_axi port=DRAM_ln1_2 offset=slave bundle=mem_ln1_2
    #pragma HLS interface m_axi port=DRAM_ln2_2 offset=slave bundle=mem_ln2_2
    #pragma HLS interface m_axi port=DRAM_wq_2 offset=slave bundle=mem_wq_2
    #pragma HLS interface m_axi port=DRAM_wk_2 offset=slave bundle=mem_wk_2
    #pragma HLS interface m_axi port=DRAM_wv_2 offset=slave bundle=mem_wv_2
    #pragma HLS interface m_axi port=DRAM_wo_2 offset=slave bundle=mem_wo_2
    #pragma HLS interface m_axi port=DRAM_mlp_w1_2 offset=slave bundle=mem_mlp1_2
    #pragma HLS interface m_axi port=DRAM_mlp_w2_2 offset=slave bundle=mem_mlp2_2
    #pragma HLS interface m_axi port=DRAM_ln1_3 offset=slave bundle=mem_ln1_3
    #pragma HLS interface m_axi port=DRAM_ln2_3 offset=slave bundle=mem_ln2_3
    #pragma HLS interface m_axi port=DRAM_wq_3 offset=slave bundle=mem_wq_3
    #pragma HLS interface m_axi port=DRAM_wk_3 offset=slave bundle=mem_wk_3
    #pragma HLS interface m_axi port=DRAM_wv_3 offset=slave bundle=mem_wv_3
    #pragma HLS interface m_axi port=DRAM_wo_3 offset=slave bundle=mem_wo_3
    #pragma HLS interface m_axi port=DRAM_mlp_w1_3 offset=slave bundle=mem_mlp1_3
    #pragma HLS interface m_axi port=DRAM_mlp_w2_3 offset=slave bundle=mem_mlp2_3
    #pragma HLS interface m_axi port=DRAM_ln1_4 offset=slave bundle=mem_ln1_4
    #pragma HLS interface m_axi port=DRAM_ln2_4 offset=slave bundle=mem_ln2_4
    #pragma HLS interface m_axi port=DRAM_wq_4 offset=slave bundle=mem_wq_4
    #pragma HLS interface m_axi port=DRAM_wk_4 offset=slave bundle=mem_wk_4
    #pragma HLS interface m_axi port=DRAM_wv_4 offset=slave bundle=mem_wv_4
    #pragma HLS interface m_axi port=DRAM_wo_4 offset=slave bundle=mem_wo_4
    #pragma HLS interface m_axi port=DRAM_mlp_w1_4 offset=slave bundle=mem_mlp1_4
    #pragma HLS interface m_axi port=DRAM_mlp_w2_4 offset=slave bundle=mem_mlp2_4
    #pragma HLS interface m_axi port=DRAM_ln1_5 offset=slave bundle=mem_ln1_5
    #pragma HLS interface m_axi port=DRAM_ln2_5 offset=slave bundle=mem_ln2_5
    #pragma HLS interface m_axi port=DRAM_wq_5 offset=slave bundle=mem_wq_5
    #pragma HLS interface m_axi port=DRAM_wk_5 offset=slave bundle=mem_wk_5
    #pragma HLS interface m_axi port=DRAM_wv_5 offset=slave bundle=mem_wv_5
    #pragma HLS interface m_axi port=DRAM_wo_5 offset=slave bundle=mem_wo_5
    #pragma HLS interface m_axi port=DRAM_mlp_w1_5 offset=slave bundle=mem_mlp1_5
    #pragma HLS interface m_axi port=DRAM_mlp_w2_5 offset=slave bundle=mem_mlp2_5
    #pragma HLS interface m_axi port=DRAM_ln1_6 offset=slave bundle=mem_ln1_6
    #pragma HLS interface m_axi port=DRAM_ln2_6 offset=slave bundle=mem_ln2_6
    #pragma HLS interface m_axi port=DRAM_wq_6 offset=slave bundle=mem_wq_6
    #pragma HLS interface m_axi port=DRAM_wk_6 offset=slave bundle=mem_wk_6
    #pragma HLS interface m_axi port=DRAM_wv_6 offset=slave bundle=mem_wv_6
    #pragma HLS interface m_axi port=DRAM_wo_6 offset=slave bundle=mem_wo_6
    #pragma HLS interface m_axi port=DRAM_mlp_w1_6 offset=slave bundle=mem_mlp1_6
    #pragma HLS interface m_axi port=DRAM_mlp_w2_6 offset=slave bundle=mem_mlp2_6
    #pragma HLS interface m_axi port=DRAM_ln1_7 offset=slave bundle=mem_ln1_7
    #pragma HLS interface m_axi port=DRAM_ln2_7 offset=slave bundle=mem_ln2_7
    #pragma HLS interface m_axi port=DRAM_wq_7 offset=slave bundle=mem_wq_7
    #pragma HLS interface m_axi port=DRAM_wk_7 offset=slave bundle=mem_wk_7
    #pragma HLS interface m_axi port=DRAM_wv_7 offset=slave bundle=mem_wv_7
    #pragma HLS interface m_axi port=DRAM_wo_7 offset=slave bundle=mem_wo_7
    #pragma HLS interface m_axi port=DRAM_mlp_w1_7 offset=slave bundle=mem_mlp1_7
    #pragma HLS interface m_axi port=DRAM_mlp_w2_7 offset=slave bundle=mem_mlp2_7
    #pragma HLS interface m_axi port=DRAM_ln1_8 offset=slave bundle=mem_ln1_8
    #pragma HLS interface m_axi port=DRAM_ln2_8 offset=slave bundle=mem_ln2_8
    #pragma HLS interface m_axi port=DRAM_wq_8 offset=slave bundle=mem_wq_8
    #pragma HLS interface m_axi port=DRAM_wk_8 offset=slave bundle=mem_wk_8
    #pragma HLS interface m_axi port=DRAM_wv_8 offset=slave bundle=mem_wv_8
    #pragma HLS interface m_axi port=DRAM_wo_8 offset=slave bundle=mem_wo_8
    #pragma HLS interface m_axi port=DRAM_mlp_w1_8 offset=slave bundle=mem_mlp1_8
    #pragma HLS interface m_axi port=DRAM_mlp_w2_8 offset=slave bundle=mem_mlp2_8
    #pragma HLS interface m_axi port=DRAM_ln1_9 offset=slave bundle=mem_ln1_9
    #pragma HLS interface m_axi port=DRAM_ln2_9 offset=slave bundle=mem_ln2_9
    #pragma HLS interface m_axi port=DRAM_wq_9 offset=slave bundle=mem_wq_9
    #pragma HLS interface m_axi port=DRAM_wk_9 offset=slave bundle=mem_wk_9
    #pragma HLS interface m_axi port=DRAM_wv_9 offset=slave bundle=mem_wv_9
    #pragma HLS interface m_axi port=DRAM_wo_9 offset=slave bundle=mem_wo_9
    #pragma HLS interface m_axi port=DRAM_mlp_w1_9 offset=slave bundle=mem_mlp1_9
    #pragma HLS interface m_axi port=DRAM_mlp_w2_9 offset=slave bundle=mem_mlp2_9
    #pragma HLS interface m_axi port=DRAM_ln1_10 offset=slave bundle=mem_ln1_10
    #pragma HLS interface m_axi port=DRAM_ln2_10 offset=slave bundle=mem_ln2_10
    #pragma HLS interface m_axi port=DRAM_wq_10 offset=slave bundle=mem_wq_10
    #pragma HLS interface m_axi port=DRAM_wk_10 offset=slave bundle=mem_wk_10
    #pragma HLS interface m_axi port=DRAM_wv_10 offset=slave bundle=mem_wv_10
    #pragma HLS interface m_axi port=DRAM_wo_10 offset=slave bundle=mem_wo_10
    #pragma HLS interface m_axi port=DRAM_mlp_w1_10 offset=slave bundle=mem_mlp1_10
    #pragma HLS interface m_axi port=DRAM_mlp_w2_10 offset=slave bundle=mem_mlp2_10
    #pragma HLS interface m_axi port=DRAM_ln1_11 offset=slave bundle=mem_ln1_11
    #pragma HLS interface m_axi port=DRAM_ln2_11 offset=slave bundle=mem_ln2_11
    #pragma HLS interface m_axi port=DRAM_wq_11 offset=slave bundle=mem_wq_11
    #pragma HLS interface m_axi port=DRAM_wk_11 offset=slave bundle=mem_wk_11
    #pragma HLS interface m_axi port=DRAM_wv_11 offset=slave bundle=mem_wv_11
    #pragma HLS interface m_axi port=DRAM_wo_11 offset=slave bundle=mem_wo_11
    #pragma HLS interface m_axi port=DRAM_mlp_w1_11 offset=slave bundle=mem_mlp1_11
    #pragma HLS interface m_axi port=DRAM_mlp_w2_11 offset=slave bundle=mem_mlp2_11
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem_output

    load_2048_768_ap_fixed_16_5_(DRAM_input, BRAM_input_0);
    load_768_ap_fixed_16_5_(DRAM_ln1_0, BRAM_ln1_0);
    load_768_ap_fixed_16_5_(DRAM_ln2_0, BRAM_ln2_0);
    load_768_768_ap_fixed_16_5_(DRAM_wq_0, BRAM_wq_0);
    load_768_768_ap_fixed_16_5_(DRAM_wk_0, BRAM_wk_0);
    load_768_768_ap_fixed_16_5_(DRAM_wv_0, BRAM_wv_0);
    load_768_768_ap_fixed_16_5_(DRAM_wo_0, BRAM_wo_0);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_0, BRAM_mlp_w1_0);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_0, BRAM_mlp_w2_0);
    load_768_ap_fixed_16_5_(DRAM_ln1_1, BRAM_ln1_1);
    load_768_ap_fixed_16_5_(DRAM_ln2_1, BRAM_ln2_1);
    load_768_768_ap_fixed_16_5_(DRAM_wq_1, BRAM_wq_1);
    load_768_768_ap_fixed_16_5_(DRAM_wk_1, BRAM_wk_1);
    load_768_768_ap_fixed_16_5_(DRAM_wv_1, BRAM_wv_1);
    load_768_768_ap_fixed_16_5_(DRAM_wo_1, BRAM_wo_1);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_1, BRAM_mlp_w1_1);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_1, BRAM_mlp_w2_1);
    load_768_ap_fixed_16_5_(DRAM_ln1_2, BRAM_ln1_2);
    load_768_ap_fixed_16_5_(DRAM_ln2_2, BRAM_ln2_2);
    load_768_768_ap_fixed_16_5_(DRAM_wq_2, BRAM_wq_2);
    load_768_768_ap_fixed_16_5_(DRAM_wk_2, BRAM_wk_2);
    load_768_768_ap_fixed_16_5_(DRAM_wv_2, BRAM_wv_2);
    load_768_768_ap_fixed_16_5_(DRAM_wo_2, BRAM_wo_2);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_2, BRAM_mlp_w1_2);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_2, BRAM_mlp_w2_2);
    load_768_ap_fixed_16_5_(DRAM_ln1_3, BRAM_ln1_3);
    load_768_ap_fixed_16_5_(DRAM_ln2_3, BRAM_ln2_3);
    load_768_768_ap_fixed_16_5_(DRAM_wq_3, BRAM_wq_3);
    load_768_768_ap_fixed_16_5_(DRAM_wk_3, BRAM_wk_3);
    load_768_768_ap_fixed_16_5_(DRAM_wv_3, BRAM_wv_3);
    load_768_768_ap_fixed_16_5_(DRAM_wo_3, BRAM_wo_3);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_3, BRAM_mlp_w1_3);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_3, BRAM_mlp_w2_3);
    load_768_ap_fixed_16_5_(DRAM_ln1_4, BRAM_ln1_4);
    load_768_ap_fixed_16_5_(DRAM_ln2_4, BRAM_ln2_4);
    load_768_768_ap_fixed_16_5_(DRAM_wq_4, BRAM_wq_4);
    load_768_768_ap_fixed_16_5_(DRAM_wk_4, BRAM_wk_4);
    load_768_768_ap_fixed_16_5_(DRAM_wv_4, BRAM_wv_4);
    load_768_768_ap_fixed_16_5_(DRAM_wo_4, BRAM_wo_4);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_4, BRAM_mlp_w1_4);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_4, BRAM_mlp_w2_4);
    load_768_ap_fixed_16_5_(DRAM_ln1_5, BRAM_ln1_5);
    load_768_ap_fixed_16_5_(DRAM_ln2_5, BRAM_ln2_5);
    load_768_768_ap_fixed_16_5_(DRAM_wq_5, BRAM_wq_5);
    load_768_768_ap_fixed_16_5_(DRAM_wk_5, BRAM_wk_5);
    load_768_768_ap_fixed_16_5_(DRAM_wv_5, BRAM_wv_5);
    load_768_768_ap_fixed_16_5_(DRAM_wo_5, BRAM_wo_5);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_5, BRAM_mlp_w1_5);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_5, BRAM_mlp_w2_5);
    load_768_ap_fixed_16_5_(DRAM_ln1_6, BRAM_ln1_6);
    load_768_ap_fixed_16_5_(DRAM_ln2_6, BRAM_ln2_6);
    load_768_768_ap_fixed_16_5_(DRAM_wq_6, BRAM_wq_6);
    load_768_768_ap_fixed_16_5_(DRAM_wk_6, BRAM_wk_6);
    load_768_768_ap_fixed_16_5_(DRAM_wv_6, BRAM_wv_6);
    load_768_768_ap_fixed_16_5_(DRAM_wo_6, BRAM_wo_6);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_6, BRAM_mlp_w1_6);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_6, BRAM_mlp_w2_6);
    load_768_ap_fixed_16_5_(DRAM_ln1_7, BRAM_ln1_7);
    load_768_ap_fixed_16_5_(DRAM_ln2_7, BRAM_ln2_7);
    load_768_768_ap_fixed_16_5_(DRAM_wq_7, BRAM_wq_7);
    load_768_768_ap_fixed_16_5_(DRAM_wk_7, BRAM_wk_7);
    load_768_768_ap_fixed_16_5_(DRAM_wv_7, BRAM_wv_7);
    load_768_768_ap_fixed_16_5_(DRAM_wo_7, BRAM_wo_7);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_7, BRAM_mlp_w1_7);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_7, BRAM_mlp_w2_7);
    load_768_ap_fixed_16_5_(DRAM_ln1_8, BRAM_ln1_8);
    load_768_ap_fixed_16_5_(DRAM_ln2_8, BRAM_ln2_8);
    load_768_768_ap_fixed_16_5_(DRAM_wq_8, BRAM_wq_8);
    load_768_768_ap_fixed_16_5_(DRAM_wk_8, BRAM_wk_8);
    load_768_768_ap_fixed_16_5_(DRAM_wv_8, BRAM_wv_8);
    load_768_768_ap_fixed_16_5_(DRAM_wo_8, BRAM_wo_8);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_8, BRAM_mlp_w1_8);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_8, BRAM_mlp_w2_8);
    load_768_ap_fixed_16_5_(DRAM_ln1_9, BRAM_ln1_9);
    load_768_ap_fixed_16_5_(DRAM_ln2_9, BRAM_ln2_9);
    load_768_768_ap_fixed_16_5_(DRAM_wq_9, BRAM_wq_9);
    load_768_768_ap_fixed_16_5_(DRAM_wk_9, BRAM_wk_9);
    load_768_768_ap_fixed_16_5_(DRAM_wv_9, BRAM_wv_9);
    load_768_768_ap_fixed_16_5_(DRAM_wo_9, BRAM_wo_9);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_9, BRAM_mlp_w1_9);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_9, BRAM_mlp_w2_9);
    load_768_ap_fixed_16_5_(DRAM_ln1_10, BRAM_ln1_10);
    load_768_ap_fixed_16_5_(DRAM_ln2_10, BRAM_ln2_10);
    load_768_768_ap_fixed_16_5_(DRAM_wq_10, BRAM_wq_10);
    load_768_768_ap_fixed_16_5_(DRAM_wk_10, BRAM_wk_10);
    load_768_768_ap_fixed_16_5_(DRAM_wv_10, BRAM_wv_10);
    load_768_768_ap_fixed_16_5_(DRAM_wo_10, BRAM_wo_10);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_10, BRAM_mlp_w1_10);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_10, BRAM_mlp_w2_10);
    load_768_ap_fixed_16_5_(DRAM_ln1_11, BRAM_ln1_11);
    load_768_ap_fixed_16_5_(DRAM_ln2_11, BRAM_ln2_11);
    load_768_768_ap_fixed_16_5_(DRAM_wq_11, BRAM_wq_11);
    load_768_768_ap_fixed_16_5_(DRAM_wk_11, BRAM_wk_11);
    load_768_768_ap_fixed_16_5_(DRAM_wv_11, BRAM_wv_11);
    load_768_768_ap_fixed_16_5_(DRAM_wo_11, BRAM_wo_11);
    load_768_3072_ap_fixed_16_5_(DRAM_mlp_w1_11, BRAM_mlp_w1_11);
    load_3072_768_ap_fixed_16_5_(DRAM_mlp_w2_11, BRAM_mlp_w2_11);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_0, BRAM_ln1_0, BRAM_input_0);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_0, BRAM_wq_0, BRAM_wk_0, BRAM_wv_0, BRAM_wo_0, BRAM_tmp_0, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_0, BRAM_input_0, BRAM_input_0);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_0, BRAM_ln2_0, BRAM_input_0);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_0, BRAM_mlp_w1_0, BRAM_tmp_0);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_0, BRAM_tmp_0);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_0, BRAM_mlp_w2_0, BRAM_tmp_0);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_0, BRAM_input_0, BRAM_input_1);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_1, BRAM_ln1_1, BRAM_input_1);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_1, BRAM_wq_1, BRAM_wk_1, BRAM_wv_1, BRAM_wo_1, BRAM_tmp_1, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_1, BRAM_input_1);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_1, BRAM_ln2_1, BRAM_input_1);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_1, BRAM_mlp_w1_1, BRAM_tmp_1);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_1, BRAM_tmp_1);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_1, BRAM_mlp_w2_1, BRAM_tmp_1);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_1, BRAM_input_2);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_2, BRAM_ln1_2, BRAM_input_2);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_2, BRAM_wq_2, BRAM_wk_2, BRAM_wv_2, BRAM_wo_2, BRAM_tmp_2, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_2, BRAM_input_2, BRAM_input_2);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_2, BRAM_ln2_2, BRAM_input_2);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_2, BRAM_mlp_w1_2, BRAM_tmp_2);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_2, BRAM_mlp_w2_2, BRAM_tmp_2);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_2, BRAM_input_2, BRAM_input_3);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_3, BRAM_ln1_3, BRAM_input_3);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_3, BRAM_wq_3, BRAM_wk_3, BRAM_wv_3, BRAM_wo_3, BRAM_tmp_3, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_3, BRAM_input_3, BRAM_input_3);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_3, BRAM_ln2_3, BRAM_input_3);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_3, BRAM_mlp_w1_3, BRAM_tmp_3);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mlp_w2_3, BRAM_tmp_3);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_3, BRAM_input_3, BRAM_input_4);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_4, BRAM_ln1_4, BRAM_input_4);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_4, BRAM_wq_4, BRAM_wk_4, BRAM_wv_4, BRAM_wo_4, BRAM_tmp_4, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_4, BRAM_input_4, BRAM_input_4);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_4, BRAM_ln2_4, BRAM_input_4);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_4, BRAM_mlp_w1_4, BRAM_tmp_4);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_4, BRAM_tmp_4);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_4, BRAM_mlp_w2_4, BRAM_tmp_4);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_4, BRAM_input_4, BRAM_input_5);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_5, BRAM_ln1_5, BRAM_input_5);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_5, BRAM_wq_5, BRAM_wk_5, BRAM_wv_5, BRAM_wo_5, BRAM_tmp_5, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_5, BRAM_input_5, BRAM_input_5);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_5, BRAM_ln2_5, BRAM_input_5);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_5, BRAM_mlp_w1_5, BRAM_tmp_5);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_5, BRAM_tmp_5);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_5, BRAM_mlp_w2_5, BRAM_tmp_5);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_5, BRAM_input_5, BRAM_input_6);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_6, BRAM_ln1_6, BRAM_input_6);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_6, BRAM_wq_6, BRAM_wk_6, BRAM_wv_6, BRAM_wo_6, BRAM_tmp_6, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_6, BRAM_input_6, BRAM_input_6);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_6, BRAM_ln2_6, BRAM_input_6);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_6, BRAM_mlp_w1_6, BRAM_tmp_6);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_6, BRAM_tmp_6);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_6, BRAM_mlp_w2_6, BRAM_tmp_6);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_6, BRAM_input_6, BRAM_input_7);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_7, BRAM_ln1_7, BRAM_input_7);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_7, BRAM_wq_7, BRAM_wk_7, BRAM_wv_7, BRAM_wo_7, BRAM_tmp_7, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_7, BRAM_input_7, BRAM_input_7);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_7, BRAM_ln2_7, BRAM_input_7);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_7, BRAM_mlp_w1_7, BRAM_tmp_7);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_7, BRAM_tmp_7);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_7, BRAM_mlp_w2_7, BRAM_tmp_7);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_7, BRAM_input_7, BRAM_input_8);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_8, BRAM_ln1_8, BRAM_input_8);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_8, BRAM_wq_8, BRAM_wk_8, BRAM_wv_8, BRAM_wo_8, BRAM_tmp_8, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_8, BRAM_input_8, BRAM_input_8);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_8, BRAM_ln2_8, BRAM_input_8);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_8, BRAM_mlp_w1_8, BRAM_tmp_8);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_8, BRAM_tmp_8);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_8, BRAM_mlp_w2_8, BRAM_tmp_8);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_8, BRAM_input_8, BRAM_input_9);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_9, BRAM_ln1_9, BRAM_input_9);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_9, BRAM_wq_9, BRAM_wk_9, BRAM_wv_9, BRAM_wo_9, BRAM_tmp_9, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_9, BRAM_input_9, BRAM_input_9);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_9, BRAM_ln2_9, BRAM_input_9);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_9, BRAM_mlp_w1_9, BRAM_tmp_9);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_9, BRAM_tmp_9);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_9, BRAM_mlp_w2_9, BRAM_tmp_9);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_9, BRAM_input_9, BRAM_input_10);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_10, BRAM_ln1_10, BRAM_input_10);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_10, BRAM_wq_10, BRAM_wk_10, BRAM_wv_10, BRAM_wo_10, BRAM_tmp_10, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_10, BRAM_input_10, BRAM_input_10);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_10, BRAM_ln2_10, BRAM_input_10);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_10, BRAM_mlp_w1_10, BRAM_tmp_10);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_10, BRAM_tmp_10);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_10, BRAM_mlp_w2_10, BRAM_tmp_10);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_10, BRAM_input_10, BRAM_input_11);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_11, BRAM_ln1_11, BRAM_input_11);
    sliding_window_attention_2048_768_12_64_ap_fixed_16_5_(BRAM_input_11, BRAM_wq_11, BRAM_wk_11, BRAM_wv_11, BRAM_wo_11, BRAM_tmp_11, 512);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_11, BRAM_input_11, BRAM_input_11);
    layer_norm_2048_768_ap_fixed_16_5_(BRAM_input_11, BRAM_ln2_11, BRAM_input_11);
    matmul_2048_768_3072_bias_ap_fixed_16_5_(BRAM_input_11, BRAM_mlp_w1_11, BRAM_tmp_11);
    gelu_2048_3072_ap_fixed_16_5_(BRAM_tmp_11, BRAM_tmp_11);
    matmul_2048_3072_768_bias_ap_fixed_16_5_(BRAM_tmp_11, BRAM_mlp_w2_11, BRAM_tmp_11);
    matrix_add_2048_768_ap_fixed_16_5_(BRAM_tmp_11, BRAM_input_11, BRAM_input_12);
    store_2048_768_ap_fixed_16_5_(BRAM_input_12, DRAM_output);
}