
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
data_t BRAM_image_input[8][16][16];
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


void matrix_add_8_16_16_float(
    data_t in1[8][16][16],
    data_t in2[8][16][16],
    data_t out[8][16][16]
)
{
    for (int c = 0; c < 8; c++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                out[c][i][j] = in1[c][i][j] + in2[c][i][j];
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

void top(data_t DRAM_input_1[8][16][16], data_t DRAM_input_2[8][16][16], data_t DRAM_output[8][16][16])
{
    #pragma HLS interface m_axi port=DRAM_input_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_input_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_16_16_float(DRAM_input_1, BRAM_buffer_1);
    load_8_16_16_float(DRAM_input_2, BRAM_image_input);
    matrix_add_8_16_16_float(BRAM_buffer_1, BRAM_image_input, BRAM_buffer_2);
    store_8_16_16_float(BRAM_buffer_2, DRAM_output);
}