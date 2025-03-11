
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

data_t BRAM_attn_input[128][512];
data_t BRAM_weights_q[512][512];
data_t BRAM_weights_k[512][512];
data_t BRAM_weights_v[512][512];
data_t BRAM_1[128][512];
data_t BRAM_2[128][512];
data_t BRAM_layer_norm_weights_1[2][512];
data_t FF_weights_1[2048][512];
data_t FF_weights_2[512][2048];
data_t BRAM_layer_norm_weights_2[2][512];

void load_128_512_ap_fixed_16_5_(data_t input[128][512], data_t output[128][512])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 512; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_512_512_ap_fixed_16_5_(data_t input[512][512], data_t output[512][512])
{
    for (int idx0 = 0; idx0 < 512; idx0++) {
        for (int idx1 = 0; idx1 < 512; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_2_512_ap_fixed_16_5_(data_t input[2][512], data_t output[2][512])
{
    for (int idx0 = 0; idx0 < 2; idx0++) {
        for (int idx1 = 0; idx1 < 512; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_2048_512_ap_fixed_16_5_(data_t input[2048][512], data_t output[2048][512])
{
    for (int idx0 = 0; idx0 < 2048; idx0++) {
        for (int idx1 = 0; idx1 < 512; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_512_2048_ap_fixed_16_5_(data_t input[512][2048], data_t output[512][2048])
{
    for (int idx0 = 0; idx0 < 512; idx0++) {
        for (int idx1 = 0; idx1 < 2048; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

#include <math.h>
#include <stdio.h>

typedef ap_fixed<16, 5> data_t;

/*
 * Auto-generated Grouped Multi-head Attention (with optional inline RoPE)
 *
 * Input     : [128][512]
 * W_q/k/v   : [512][512], DIM_OUT = NUM_HEADS * HEAD_DIM
 * Output    : [128][512]
 *
 * Data type : ap_fixed<16, 5>
 * Num Heads : 8
 * Head Dim  : 64
 */

void grouped_multihead_attention(
    data_t input[128][512],
    data_t W_q[512][512],
    data_t W_k[512][512],
    data_t W_v[512][512],
    data_t output[128][512],
    int groups
)
{
    const int num_heads = 8;
    const int head_dim = 64;
    const data_t scale = 1.0 / sqrt((data_t)head_dim);

    data_t Q[128][512];
    data_t K[128][512];
    data_t V[128][512];

    // Compute Q, K, V
    for (int seq = 0; seq < 128; seq++) {
        for (int dout = 0; dout < 512; dout++) {
            Q[seq][dout] = 0;
            K[seq][dout] = 0;
            V[seq][dout] = 0;
            for (int din = 0; din < 512; din++) {
                Q[seq][dout] += input[seq][din] * W_q[dout][din];
                K[seq][dout] += input[seq][din] * W_k[dout][din];
                V[seq][dout] += input[seq][din] * W_v[dout][din];
            }
        }
    }

    /*==== BEGIN OPTIONAL ROPE LOGIC ====*/
    
    /*==== END OPTIONAL ROPE LOGIC ====*/

    // Compute Attention per head
    for (int h = 0; h < num_heads; h++) {
        data_t scores[128][128];

        // Scaled Dot-product: Q x K^T
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < 128; j++) {
                scores[i][j] = 0;
                for (int d = 0; d < head_dim; d++) {
                    int idx = h * head_dim + d;
                    scores[i][j] += Q[i][idx] * K[j][idx];
                }
                scores[i][j] *= scale;
            }
        }

        // Softmax
        for (int i = 0; i < 128; i++) {
            data_t sum_exp = 0;
            data_t max_score = scores[i][0];
            for (int j = 1; j < 128; j++)
                if (scores[i][j] > max_score)
                    max_score = scores[i][j];

            for (int j = 0; j < 128; j++) {
                scores[i][j] = exp(scores[i][j] - max_score);
                sum_exp += scores[i][j];
            }
            for (int j = 0; j < 128; j++)
                scores[i][j] /= sum_exp;
        }

        // Compute context: scores x V
        for (int i = 0; i < 128; i++) {
            for (int d = 0; d < head_dim; d++) {
                data_t context = 0;
                for (int j = 0; j < 128; j++)
                    context += scores[i][j] * V[j][h * head_dim + d];
                output[i][h * head_dim + d] = context;
            }
        }
    }
}



#include <math.h>
#include <stdio.h>

typedef ap_fixed<16, 5> data_t;

// Simple LCG for pseudo-random numbers.
unsigned int lcg_rand(unsigned int *seed) {
    *seed = (1103515245 * (*seed) + 12345) & 0x7fffffff;
    return *seed;
}

void dropout(
    data_t input[128][512],
    data_t output[128][512],
    data_t dropout_prob,
    unsigned int seed
)
{
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 512; j++) {
            // Generate a random value between 0 and 1.
            unsigned int r = lcg_rand(&seed);
            data_t rand_val = (data_t)r / (data_t)2147483647;
            // Apply dropout: if random value is below dropout_prob, output is 0;
            // otherwise, scale the input.
            if (rand_val < dropout_prob) {
                output[i][j] = (data_t)0;
            } else {
                output[i][j] = input[i][j] / (1 - dropout_prob);
            }
        }
    }
}


#include <math.h>
#include <stdio.h>

typedef ap_fixed<16, 5> data_t;

void layer_norm(
    data_t input[128][512],
    data_t gamma[512],
    data_t beta[512],
    data_t output[128][512]
)
{
    for (int i = 0; i < 128; i++) {
        // Compute mean for the i-th sequence element
        data_t sum = (data_t)0;
        for (int j = 0; j < 512; j++) {
            sum += input[i][j];
        }
        data_t mean = sum / 512;

        // Compute variance for the i-th sequence element
        data_t var_sum = (data_t)0;
        for (int j = 0; j < 512; j++) {
            data_t diff = input[i][j] - mean;
            var_sum += diff * diff;
        }
        data_t variance = var_sum / 512;

        // Normalize and scale: output = gamma * (x - mean) / sqrt(variance + epsilon) + beta
        for (int j = 0; j < 512; j++) {
            output[i][j] = gamma[j] * ((input[i][j] - mean) / sqrt(variance + 1e-05)) + beta[j];
        }
    }
}


#include <stdio.h>
#include <math.h>

typedef ap_fixed<16, 5> data_t;

void matmul(
    data_t input[128][512],
    data_t weights[2048][512],
    data_t output[128][2048]
)
{
    // Initialize output to ((data_t)0)
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 2048; j++) {
            output[i][j] = ((data_t)0);
        }
    }

    // Matrix multiplication
    for (int i = 0; i < 128; i++) {
        for (int k = 0; k < 512; k++) {
            for (int j = 0; j < 2048; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}


#include <math.h>
#include <stdio.h>

typedef ap_fixed<16, 5> data_t;

/*
 * Auto-generated Activation Functions for 2D Tensors
 * Tensor dimensions: [128][512]
 * Data type: ap_fixed<16, 5>
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
 *   - softmax  // Softmax is computed along the hidden_dim axis.
 */

void gelu_128_512_ap_fixed_16_5_(
    data_t input[128][512],
    data_t output[128][512]
)
{
    // Exact GELU using the error function:
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 512; j++) {
            data_t x = input[i][j];
            output[i][j] = 0.5 * x * (1 + erf(x / sqrt((data_t)2)));
        }
    }
}

#include <stdio.h>
#include <math.h>

typedef ap_fixed<16, 5> data_t;

void matmul(
    data_t input[128][2048],
    data_t weights[512][2048],
    data_t output[128][512]
)
{
    // Initialize output to ((data_t)0)
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 512; j++) {
            output[i][j] = ((data_t)0);
        }
    }

    // Matrix multiplication
    for (int i = 0; i < 128; i++) {
        for (int k = 0; k < 2048; k++) {
            for (int j = 0; j < 512; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}


void store_128_512_ap_fixed_16_5_(data_t input[128][512], data_t output[128][512])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 512; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_attn_input[128][512], data_t DRAM_weights_q[512][512], data_t DRAM_weights_k[512][512], data_t DRAM_weights_v[512][512], data_t DRAM_layer_norm_weights_1[2][512], data_t DRAM_FF_weights_1[2048][512], data_t DRAM_FF_weights_2[512][2048], data_t DRAM_layer_norm_weights_2[2][512], data_t DRAM_output[128][512])
{
    #pragma HLS interface m_axi port=DRAM_attn_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_q offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_k offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_v offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_layer_norm_weights_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_FF_weights_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_FF_weights_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_layer_norm_weights_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_128_512_ap_fixed_16_5_(DRAM_attn_input, BRAM_attn_input);
    load_512_512_ap_fixed_16_5_(DRAM_weights_q, BRAM_weights_q);
    load_512_512_ap_fixed_16_5_(DRAM_weights_k, BRAM_weights_k);
    load_512_512_ap_fixed_16_5_(DRAM_weights_v, BRAM_weights_v);
    load_2_512_ap_fixed_16_5_(DRAM_layer_norm_weights_1, BRAM_layer_norm_weights_1);
    load_2048_512_ap_fixed_16_5_(DRAM_FF_weights_1, FF_weights_1);
    load_512_2048_ap_fixed_16_5_(DRAM_FF_weights_2, FF_weights_2);
    load_2_512_ap_fixed_16_5_(DRAM_layer_norm_weights_2, BRAM_layer_norm_weights_2);
    grouped_multihead_attention_128_512_8_64_ap_fixed_16_5_(BRAM_attn_input, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1, 8);
    dropout_128_512_ap_fixed_16_5_(BRAM_1, BRAM_2, 0.5, 47);
    layer_norm_128_512_ap_fixed_16_5_(BRAM_2, BRAM_layer_norm_weights_1[0], BRAM_layer_norm_weights_1[1], BRAM_1);
    matmul_128_512_2048_ap_fixed_16_5_(BRAM_1, FF_weights_1, BRAM_2);
    gelu_128_512_ap_fixed_16_5_(BRAM_2, BRAM_1);
    matmul_128_2048_512_ap_fixed_16_5_(BRAM_1, FF_weights_2, BRAM_2);
    dropout_128_512_ap_fixed_16_5_(BRAM_2, BRAM_1, 0.5, 47);
    layer_norm_128_512_ap_fixed_16_5_(BRAM_1, BRAM_layer_norm_weights_2[0], BRAM_layer_norm_weights_2[1], BRAM_2);
    store_128_512_ap_fixed_16_5_(BRAM_2, DRAM_output);
}