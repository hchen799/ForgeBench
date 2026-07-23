
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

typedef float data_t;

data_t BRAM_attn_input[8][32];
data_t BRAM_weights_q[32][32];
data_t BRAM_weights_k[32][32];
data_t BRAM_weights_v[32][32];
data_t BRAM_1[8][32];

void load_8_32_float(data_t input[8][32], data_t output[8][32])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_32_32_float(data_t input[32][32], data_t output[32][32])
{
    for (int idx0 = 0; idx0 < 32; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}


void sliding_window_attention_8_32_2_16_float(
    data_t input[8][32],
    data_t W_q[32][32],
    data_t W_k[32][32],
    data_t W_v[32][32],
    data_t output[8][32],
    int window_size
)
{
    const int num_heads = 2;   // total number of heads (must equal DIM_OUT / HEAD_DIM)
    const int head_dim = 16;       // dimension per head
    const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);

    data_t Q[8][32];
    data_t K[8][32];
    data_t V[8][32];

    // Compute Q, K, V
    for (int i = 0; i < 8; i++) {
        for (int d = 0; d < 32; d++) {
            Q[i][d] = 0;
            K[i][d] = 0;
            V[i][d] = 0;
            for (int j = 0; j < 32; j++) {
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
        for (int i = 0; i < 8; i++) {
            // Determine window bounds
            int start = (i - window_size < 0) ? 0 : i - window_size;
            int end = (i + window_size >= 8) ? 8 - 1 : i + window_size;
            data_t scores[8]; // Allocate full length for simplicity

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


void store_8_32_float(data_t input[8][32], data_t output[8][32])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_attn_input[8][32], data_t DRAM_weights_q[32][32], data_t DRAM_weights_k[32][32], data_t DRAM_weights_v[32][32], data_t DRAM_output[8][32])
{
    #pragma HLS interface m_axi port=DRAM_attn_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_q offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_k offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_v offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_32_float(DRAM_attn_input, BRAM_attn_input);
    load_32_32_float(DRAM_weights_q, BRAM_weights_q);
    load_32_32_float(DRAM_weights_k, BRAM_weights_k);
    load_32_32_float(DRAM_weights_v, BRAM_weights_v);
    sliding_window_attention_8_32_2_16_float(BRAM_attn_input, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1, 4);
    store_8_32_float(BRAM_1, DRAM_output);
}