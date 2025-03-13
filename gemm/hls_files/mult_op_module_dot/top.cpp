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

// LOADS A
void load_16_64_ap_fixed_16_5_(data_t input[16][64], data_t output[16][64])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_64_32_ap_fixed_16_5_(data_t input[64][32], data_t output[64][64])
{
    for (int idx0 = 0; idx0 < 64; idx0++) {
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

void load_64_ap_fixed_16_5_(data_t input[64], data_t output[64])
{
    for (int idx0 = 0; idx0 < 64; idx0++) {
        output[idx0] = input[idx0];
    }
}

void load_16_ap_fixed_16_5_(data_t input[16], data_t output[16])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        output[idx0] = input[idx0];
    }
}

void load_1_ap_fixed_16_5_(data_t input[1], data_t output[1])
{
    for (int idx0 = 0; idx0 < 1; idx0++) {
        output[idx0] = input[idx0];
    }
}

void gemm_ijk(
    ap_fixed<16, 5> input_A[16][64],
    ap_fixed<16, 5> input_B[64][32],
    ap_fixed<16, 5> output[16][32]
)

{
for (int i = 0; i < 16; i++) {
for (int j = 0; j < 64; j++) {
for (int k = 0; k < 32; k++) {
    output[i][k] = 0;
}
}
}


for (int i = 0; i < 16; i++) {
for (int j = 0; j < 64; j++) {
for (int k = 0; k < 32; k++) {
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}

void mmv_ij(
    ap_fixed<16, 5> input_A[16][64],
    ap_fixed<16, 5> input_B[64],
    ap_fixed<16, 5> output[16]
)
{
for (int i = 0; i < 16; i++) {
for (int j = 0; j < 64; j++) {
    output[i] = 0;
}
}


for (int i = 0; i < 16; i++) {
for (int j = 0; j < 64; j++) {
    output[i] += input_A[i][j] * input_B[j];
}
}
}

void dot_product(
    ap_fixed<16, 5> input_A[64],
    ap_fixed<16, 5> input_B[64],
    ap_fixed<16, 5> output[1]
)
{
for (int i = 0; i < 64; i++) {
    output[0] = 0;
}


for (int i = 0; i < 64; i++) {
    output[0] += input_A[i] * input_B[i];
}
}

void transpose(
    ap_fixed<16, 5> input[64][64],
)
{
for (int i = 0; i < 64; i++) {
for (int j = 0; j < 64; j++) {
    input[i][j] = input[j][i];
}
}
}


void gemm_ijk_with_dot(
    ap_fixed<16, 5> input_A[16][64],
    ap_fixed<16, 5> input_B[64][64],
    ap_fixed<16, 5> output[16][32],
)

{
    data_t dot_out[1];
    transpose(input_B);
    
    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 32; k++) {
            dot_prouct(input_A[i], input_B[k], dot_out);
            output[i][k] = dot_out[0];
        }
    }
}

void mmv_ij_with_dot(
    ap_fixed<16, 5> input_A[16][64],
    ap_fixed<16, 5> input_B[64],
    ap_fixed<16, 5> output[16]
)
{
    data_t dot_out[1];
    for (int i = 0; i < 16; i++) {
        dot_prouct(input_A[i], input_B, dot_out);
        output[i] = dot_out[0];
    }
}


// STORES A
void store_16_32_ap_fixed_16_5_(data_t input[16][32], data_t output[16][32])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

// STORES B
void store_16_ap_fixed_16_5_(data_t input[16], data_t output[16])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        output[idx0] = input[idx0];
    }
}

// STORES C
void store_1_ap_fixed_16_5_(data_t input[1], data_t output[1])
{
    for (int idx0 = 0; idx0 gemm_ijk_with_mmv< 1; idx0++) {
        output[idx0] = input[idx0];
    }
}


void top_A(data_t DRAM_1[16][64], data_t DRAM_2[64][32], data_t DRAM_3[16][32], data_t DRAM_4[16][32])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    data_t BRAM_1[16][64];
    data_t BRAM_2[64][64];
    data_t BRAM_3[16][32];
    data_t BRAM_4[16][32];

    load_16_64_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_64_32_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_16_32_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
// Begin: Call to GEMM_IJK
//////////////////////////////////////////
gemm_ijk_with_dot(BRAM_1, BRAM_2, BRAM_4);
//////////////////////////////////////////
// End: Call to GEMM_IJK
//////////////////////////////////////////

    store_16_32_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top_B(data_t DRAM_1[16][64], data_t DRAM_2[64], data_t DRAM_3[16], data_t DRAM_4[16])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    data_t BRAM_1[16][64];
    data_t BRAM_2[64];
    data_t BRAM_3[16];
    data_t BRAM_4[16];

    load_16_64_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_64_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_16_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
    // Begin: Call to MMV_IJ
//////////////////////////////////////////
mmv_ij_with_dot(BRAM_1, BRAM_2, BRAM_4);
//////////////////////////////////////////
// End: Call to MMV_IJ
//////////////////////////////////////////

    store_16_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top_C(data_t DRAM_1[64], data_t DRAM_2[64], data_t DRAM_3[1], data_t DRAM_4[1])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    data_t BRAM_1[64];
    data_t BRAM_2[64];
    data_t BRAM_3[1];
    data_t BRAM_4[1];

    load_64_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_64_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_1_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
    // Begin: Call to DOT_PRODUCT
//////////////////////////////////////////
dot_product(BRAM_1, BRAM_2, BRAM_4);
//////////////////////////////////////////
// End: Call to DOT_PRODUCT
//////////////////////////////////////////

    store_1_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top(
    data_t DRAM_A1[16][64], data_t DRAM_A2[64][32], data_t DRAM_A3[16][32], data_t DRAM_A4[16][32],
    data_t DRAM_B1[16][64], data_t DRAM_B2[64], data_t DRAM_B3[16], data_t DRAM_B4[16],
    data_t DRAM_C1[64], data_t DRAM_C2[64], data_t DRAM_C3[1], data_t DRAM_C4[1]
){
    top_A(DRAM_A1, DRAM_A2, DRAM_A3, DRAM_A4);
    top_B(DRAM_B1, DRAM_B2, DRAM_B3, DRAM_B4);
    top_C(DRAM_C1, DRAM_C2, DRAM_C3, DRAM_C4);
}