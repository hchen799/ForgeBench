
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

data_t BRAM_buffer_1[8][16][16];
data_t BRAM_batch_norm_weights[4][8];
data_t BRAM_buffer_2[8][16][16];

void load_8_16_16_float(data_t input[8][16][16], data_t output[8][16][16])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 16; idx1++) {
            for (int idx2 = 0; idx2 < 16; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}

void load_4_8_float(data_t input[4][8], data_t output[4][8])
{
    for (int idx0 = 0; idx0 < 4; idx0++) {
        for (int idx1 = 0; idx1 < 8; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}


/*
 * Auto-generated Batch Normalization HLS Code
 *
 * Dimensions:
 *   Input/Output: [8][16][16]
 * Data type: float
 * Epsilon: 1e-05
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm_8_16_16_float(
    data_t input[8][16][16],
    data_t weights[4][8],
    data_t output[8][16][16]
)
{
    for (int c = 0; c < 8; c++) {
        for (int h = 0; h < 16; h++) {
            for (int w = 0; w < 16; w++) {
                data_t norm = (input[c][h][w] - weights[2][c]) / hls::sqrt(weights[3][c] + (data_t)1e-05);
                output[c][h][w] = weights[0][c] * norm + weights[1][c];
            }
        }
    }
}


void store_8_16_16_float(data_t input[8][16][16], data_t output[8][16][16])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 16; idx1++) {
            for (int idx2 = 0; idx2 < 16; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}

void top(data_t DRAM_input[8][16][16], data_t DRAM_weights[4][8], data_t DRAM_output[8][16][16])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_16_16_float(DRAM_input, BRAM_buffer_1);
    load_4_8_float(DRAM_weights, BRAM_batch_norm_weights);
    batch_norm_8_16_16_float(BRAM_buffer_1, BRAM_batch_norm_weights, BRAM_buffer_2);
    store_8_16_16_float(BRAM_buffer_2, DRAM_output);
}