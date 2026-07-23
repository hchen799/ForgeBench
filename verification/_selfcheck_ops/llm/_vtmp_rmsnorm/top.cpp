
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
data_t BRAM_rms_norm_weights_1[2][32];
data_t BRAM_2[8][32];

void load_8_32_float(data_t input[8][32], data_t output[8][32])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_2_32_float(data_t input[2][32], data_t output[2][32])
{
    for (int idx0 = 0; idx0 < 2; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}


void rms_norm_8_32_float(
    data_t input[8][32],
    data_t gamma[32],
    data_t output[8][32]
)
{
    for (int i = 0; i < 8; i++) {
        data_t sum_sq = (data_t)0;
        for (int j = 0; j < 32; j++) {
            sum_sq += input[i][j] * input[i][j];
        }
        data_t rms = hls::sqrt(sum_sq / (data_t)32 + (data_t)0.01);
        for (int j = 0; j < 32; j++) {
            output[i][j] = gamma[j] * input[i][j] / rms;
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

void top(data_t DRAM_input[8][32], data_t DRAM_weights[2][32], data_t DRAM_output[8][32])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_32_float(DRAM_input, BRAM_attn_input);
    load_2_32_float(DRAM_weights, BRAM_rms_norm_weights_1);
    rms_norm_8_32_float(BRAM_attn_input, BRAM_rms_norm_weights_1[0], BRAM_2);
    store_8_32_float(BRAM_2, DRAM_output);
}