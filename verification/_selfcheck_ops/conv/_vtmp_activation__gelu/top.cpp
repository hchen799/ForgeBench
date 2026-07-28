
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

/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: float
 * Tensor dimensions: [8][16][16]
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

void gelu_8_16_16_float(
    data_t input[8][16][16],
    data_t output[8][16][16]
)
{
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    const data_t sqrt_2_over_pi = hls::sqrt((data_t)2/(data_t)3.141592653589793);
    for (int c = 0; c < 8; c++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                data_t x = input[c][i][j];
                data_t x_cube = x * x * x;
                data_t tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cube);
                output[c][i][j] = 0.5 * x * (1 + hls::tanh(tanh_arg));
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

void top(data_t DRAM_input[8][16][16], data_t DRAM_output[8][16][16])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_16_16_float(DRAM_input, BRAM_buffer_1);
    gelu_8_16_16_float(BRAM_buffer_1, BRAM_buffer_2);
    store_8_16_16_float(BRAM_buffer_2, DRAM_output);
}