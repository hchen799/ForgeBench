
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
data_t BRAM_2[8][32];
data_t BRAM_1[8][32];

void load_8_32_float(data_t input[8][32], data_t output[8][32])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}


void matrix_add_8_32_float(
    data_t in1[8][32],
    data_t in2[8][32],
    data_t out[8][32]
)
{
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 32; j++) {
            out[i][j] = in1[i][j] + in2[i][j];
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

void top(data_t DRAM_input_1[8][32], data_t DRAM_input_2[8][32], data_t DRAM_output[8][32])
{
    #pragma HLS interface m_axi port=DRAM_input_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_input_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_32_float(DRAM_input_1, BRAM_attn_input);
    load_8_32_float(DRAM_input_2, BRAM_2);
    matrix_add_8_32_float(BRAM_attn_input, BRAM_2, BRAM_1);
    store_8_32_float(BRAM_1, DRAM_output);
}