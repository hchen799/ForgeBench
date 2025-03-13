
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

typedef ap_fixed<16, 5> data_t;

data_t BRAM_1[16][8];
data_t BRAM_2[8][32];
data_t BRAM_3[16][32];
data_t BRAM_4[16][32];
data_t BRAM_5[32];
data_t BRAM_6[16];
data_t BRAM_7[16];
data_t BRAM_8[32];
data_t BRAM_9[32];
data_t BRAM_10[32];
data_t BRAM_11[1];
data_t BRAM_12[1];

void load_16_8_ap_fixed_16_5_(data_t input[16][8], data_t output[16][8])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 8; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_8_32_ap_fixed_16_5_(data_t input[8][32], data_t output[8][32])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_16_32_ap_fixed_16_5_(data_t input[16][32], data_t output[16][32])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

//////////////////////////////////////////
// Begin: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////
/*==== GEMM_IJK_BIAS FUNCTION START ====*/
void gemm_ijk_bias(
    ap_fixed<16, 5> input_A[16][8],
    ap_fixed<16, 5> input_B[8][32],
    ap_fixed<16, 5> bias[16][32],
    ap_fixed<16, 5> output[16][32]
)

{
for (int i = 0; i < 16; i++) {
for (int j = 0; j < 8; j++) {
for (int k = 0; k < 32; k++) {
    output[i][k] = bias[i][k];
}
}
}


for (int i = 0; i < 16; i++) {
for (int j = 0; j < 8; j++) {
for (int k = 0; k < 32; k++) {
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}
/*==== GEMM_IJK_BIAS FUNCTION END ====*/
//////////////////////////////////////////
// END: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////


void store_16_32_ap_fixed_16_5_(data_t input[16][32], data_t output[16][32])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

//////////////////////////////////////////
// Begin: DOT_PRODUCT_BIAS FUNCTION with BIAS
//////////////////////////////////////////
/*==== DOT_PRODUCT_BIAS FUNCTION START ====*/
void dot_product_bias(
    ap_fixed<16, 5> input_A[16],
    ap_fixed<16, 5> input_B[16],
    ap_fixed<16, 5> bias[1],
    ap_fixed<16, 5> output[1]
)
{
for (int i = 0; i < 16; i++) {
    output[0] = bias[0];
}


for (int i = 0; i < 16; i++) {
    output[0] += input_A[i] * input_B[i];
}
}
/*==== DOT_PRODUCT_BIAS FUNCTION END ====*/
//////////////////////////////////////////
// END: DOT_PRODUCT_BIAS FUNCTION with BIAS
//////////////////////////////////////////


void store_1_ap_fixed_16_5_(data_t input[1], data_t output[1])
{
    for (int idx0 = 0; idx0 < 1; idx0++) {
        output[idx0] = input[idx0];
    }
}

void top(data_t DRAM_1[16][8], data_t DRAM_2[8][32], data_t DRAM_3[16][32], data_t DRAM_4[16][32], data_t DRAM_5[32], data_t DRAM_10[32], data_t DRAM_11[1], data_t DRAM_12[1])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2
    #pragma HLS interface m_axi port=DRAM_5 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_10 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_11 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_12 offset=slave bundle=mem2

    load_16_8_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_8_32_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_16_32_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
// Begin: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////
for (int i = 0; i < 16; i++) {
for (int j = 0; j < 8; j++) {
for (int k = 0; k < 32; k++) {
    BRAM_4[i][k] = BRAM_3[i][k];
}
}
}


for (int i = 0; i < 16; i++) {
for (int j = 0; j < 8; j++) {
for (int k = 0; k < 32; k++) {
    BRAM_4[i][k] += BRAM_1[i][j] * BRAM_2[j][k];
}
}
}
//////////////////////////////////////////
// End: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////

    store_16_32_ap_fixed_16_5_(BRAM_4, DRAM_4);
    //////////////////////////////////////////
// Begin: Inline implementation of DOT_PRODUCT_BIAS
//////////////////////////////////////////
for (int i = 0; i < 16; i++) {
    BRAM_12[0] = BRAM_11[0];
}


for (int i = 0; i < 16; i++) {
    BRAM_12[0] += BRAM_9[i] * BRAM_10[i];
}
//////////////////////////////////////////
// End: Inline implementation of DOT_PRODUCT_BIAS
//////////////////////////////////////////

    store_1_ap_fixed_16_5_(BRAM_12, DRAM_12);
}