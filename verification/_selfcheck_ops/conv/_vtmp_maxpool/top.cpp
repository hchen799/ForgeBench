
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
data_t BRAM_buffer_2[8][8][8];

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


/*
 * Auto-generated Max Pooling HLS Code
 *
 * Dimensions:
 *   Input  : [8][16][16]
 *   Output : [8][8][8]
 *   Pooling Window: [2][2]
 *   Stride: [2][2]
 *
 * Data type: float
 */

void maxpool_8_16_16_8_8_2_2_2_2_float(
    data_t input[8][16][16],
    data_t output[8][8][8]
)
{
    for (int c = 0; c < 8; c++) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                // Initialize max value from the top-left corner of the window
                data_t max_val = input[c][i * 2][j * 2];
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int row = i * 2 + kh;
                        int col = j * 2 + kw;
                        if (input[c][row][col] > max_val) {
                            max_val = input[c][row][col];
                        }
                    }
                }
                output[c][i][j] = max_val;
            }
        }
    }
}


void store_8_8_8_float(data_t input[8][8][8], data_t output[8][8][8])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 8; idx1++) {
            for (int idx2 = 0; idx2 < 8; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}

void top(data_t DRAM_input[8][16][16], data_t DRAM_output[8][8][8])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_16_16_float(DRAM_input, BRAM_buffer_1);
    maxpool_8_16_16_8_8_2_2_2_2_float(BRAM_buffer_1, BRAM_buffer_2);
    store_8_8_8_float(BRAM_buffer_2, DRAM_output);
}