
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

data_t BRAM_1[8][32];
data_t FF_weights_1[128][32];
data_t BRAM_MLP_1[8][128];

void load_8_32_float(data_t input[8][32], data_t output[8][32])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_128_32_float(data_t input[128][32], data_t output[128][32])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}



void matmul_8_32_128_float(
    data_t input[8][32],
    data_t weights[128][32],
    data_t output[8][128]
)
{
    // Initialize output to ((data_t)0)
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 128; j++) {
            output[i][j] = ((data_t)0);
        }
    }

    // Matrix multiplication
    for (int i = 0; i < 8; i++) {
        for (int k = 0; k < 32; k++) {
            for (int j = 0; j < 128; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}


void store_8_128_float(data_t input[8][128], data_t output[8][128])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_input[8][32], data_t DRAM_weights[128][32], data_t DRAM_output[8][128])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_32_float(DRAM_input, BRAM_1);
    load_128_32_float(DRAM_weights, FF_weights_1);
    matmul_8_32_128_float(BRAM_1, FF_weights_1, BRAM_MLP_1);
    store_8_128_float(BRAM_MLP_1, DRAM_output);
}