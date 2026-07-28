
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

data_t BRAM_MLP_1[8][128];
data_t BRAM_MLP_2[8][128];

void load_8_128_float(data_t input[8][128], data_t output[8][128])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}



void relu6_8_128_float(
    data_t input[8][128],
    data_t output[8][128],
    data_t cap
)
{
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 128; j++) {
            data_t temp = (input[i][j] > 0) ? input[i][j] : (data_t)0;
            output[i][j] = (temp < cap) ? temp : cap;
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

void top(data_t DRAM_input[8][128], data_t DRAM_output[8][128])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_128_float(DRAM_input, BRAM_MLP_1);
    relu6_8_128_float(BRAM_MLP_1, BRAM_MLP_2, (data_t)6.0);
    store_8_128_float(BRAM_MLP_2, DRAM_output);
}