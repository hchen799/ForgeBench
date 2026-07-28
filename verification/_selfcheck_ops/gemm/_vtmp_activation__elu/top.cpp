
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

data_t BRAM_1[16][32];
data_t BRAM_2[16][32];

void load_16_32_float(data_t input[16][32], data_t output[16][32])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: float
 * Tensor dimensions: [16][32]
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
 *   - softmax
 */

void elu_16_32_float(
    data_t input[16][32],
    data_t output[16][32],
    data_t alpha  // ELU parameter
)
{
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            if (input[i][j] >= 0) {
                output[i][j] = input[i][j];
            } else {
                output[i][j] = alpha * (hls::exp(input[i][j]) - 1);
            }
        }
    }
}

void store_16_32_float(data_t input[16][32], data_t output[16][32])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_1[16][32], data_t DRAM_2[16][32])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem2

    load_16_32_float(DRAM_1, BRAM_1);
    elu_16_32_float(BRAM_1, BRAM_2, (data_t)1.0);
    store_16_32_float(BRAM_2, DRAM_2);
}