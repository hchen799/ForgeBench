
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
data_t BRAM_buffer_2[8][4][4];

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

void adaptive_avgpool_8_16_16_4_4_float(
    data_t input[8][16][16],
    data_t output[8][4][4]
)
{
    for (int c = 0; c < 8; c++) {
        for (int oh = 0; oh < 4; oh++) {
            int h_start = (int)floor(( (float)oh * 16 ) / 4);
            int h_end   = (int)ceil(( (float)(oh + 1) * 16 ) / 4);
            if (h_end > 16) {
                h_end = 16;
            }
            for (int ow = 0; ow < 4; ow++) {
                int w_start = (int)floor(( (float)ow * 16 ) / 4);
                int w_end   = (int)ceil(( (float)(ow + 1) * 16 ) / 4);
                if (w_end > 16) {
                    w_end = 16;
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






void store_8_4_4_float(data_t input[8][4][4], data_t output[8][4][4])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 4; idx1++) {
            for (int idx2 = 0; idx2 < 4; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}

void top(data_t DRAM_input[8][16][16], data_t DRAM_output[8][4][4])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_16_16_float(DRAM_input, BRAM_buffer_1);
    adaptive_avgpool_8_16_16_4_4_float(BRAM_buffer_1, BRAM_buffer_2);
    store_8_4_4_float(BRAM_buffer_2, DRAM_output);
}