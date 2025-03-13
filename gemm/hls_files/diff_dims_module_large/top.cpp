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
void load_96_512_ap_fixed_16_5_(data_t input[96][512], data_t output[256][512])
{
    for (int idx0 = 0; idx0 < 96; idx0++) {
        for (int idx1 = 0; idx1 < 512; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_512_128_ap_fixed_16_5_(data_t input[512][128], data_t output[512][192])
{
    for (int idx0 = 0; idx0 < 512; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_96_128_ap_fixed_16_5_(data_t input[96][128], data_t output[256][192])
{
    for (int idx0 = 0; idx0 < 96; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

// LOADS B
void load_128_256_ap_fixed_16_5_(data_t input[128][256], data_t output[256][512])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 256; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_256_64_ap_fixed_16_5_(data_t input[256][64], data_t output[512][192])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_128_64_ap_fixed_16_5_(data_t input[128][64], data_t output[256][192])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

// LOADS C
void load_256_128_ap_fixed_16_5_(data_t input[256][128], data_t output[256][512])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_128_192_ap_fixed_16_5_(data_t input[128][192], data_t output[512][192])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 192; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_256_192_ap_fixed_16_5_(data_t input[256][192], data_t output[256][192])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 192; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}


//////////////////////////////////////////
// Begin: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////
/*==== GEMM_IJK_BIAS FUNCTION START ====*/
void gemm_ijk_bias(
    ap_fixed<16, 5> input_A[256][512],
    ap_fixed<16, 5> input_B[512][192],
    ap_fixed<16, 5> bias[256][192],
    ap_fixed<16, 5> output[256][192]
)

{
for (int i = 0; i < 96; i++) {
for (int j = 0; j < 512; j++) {
for (int k = 0; k < 128; k++) {
    output[i][k] = bias[i][k];
}
}
}


for (int i = 0; i < 96; i++) {
for (int j = 0; j < 512; j++) {
for (int k = 0; k < 128; k++) {
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}
/*==== GEMM_IJK_BIAS FUNCTION END ====*/
//////////////////////////////////////////
// END: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////

void tiled_gemm(
    ap_fixed<16, 5> input_A[256][512],
    ap_fixed<16, 5> input_B[512][192],
    ap_fixed<16, 5> bias[256][192],
    ap_fixed<16, 5> output[256][192]
)

{
    for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 512; j++) {
    for (int k = 0; k < 192; k++) {
        output[i][k] = bias[i][k];
    }
    }
    }
    
    
    for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 512; j++) {
    for (int k = 0; k < 192; k++) {
        output[i][k] += input_A[i][j] * input_B[j][k_offset];
    }
    }
    }
}

void store_96_128_ap_fixed_16_5_(data_t input[256][192], data_t output[96][128])
{
    for (int idx0 = 0; idx0 < 96; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void store_128_64_ap_fixed_16_5_(data_t input[256][192], data_t output[128][64])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void store_256_192_ap_fixed_16_5_(data_t input[256][192], data_t output[256][192])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 192; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}


void top_A(data_t DRAM_1[96][512], data_t DRAM_2[512][128], data_t DRAM_3[96][128], data_t DRAM_4[96][128])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    data_t BRAM_1[256][512];
    data_t BRAM_2[512][192];
    data_t BRAM_3[256][192];
    data_t BRAM_4[256][192];

    load_96_512_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_512_128_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_96_128_ap_fixed_16_5_(DRAM_3, BRAM_3);


//////////////////////////////////////////
// Begin: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////
    tiled_gemm(BRAM_1, BRAM_2, BRAM_3, BRAM_4); 
//////////////////////////////////////////
// End: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////
    store_96_128_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top_B(data_t DRAM_1[128][256], data_t DRAM_2[256][64], data_t DRAM_3[128][64], data_t DRAM_4[128][64])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    data_t BRAM_1[256][512];
    data_t BRAM_2[512][192];
    data_t BRAM_3[256][192];
    data_t BRAM_4[256][192];

    load_128_256_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_256_64_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_128_64_ap_fixed_16_5_(DRAM_3, BRAM_3);


//////////////////////////////////////////
// Begin: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////

    tiled_gemm(BRAM_1, BRAM_2, BRAM_3, BRAM_4); 

//////////////////////////////////////////
// End: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////

    store_128_64_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top_C(data_t DRAM_1[256][128], data_t DRAM_2[128][192], data_t DRAM_3[256][192], data_t DRAM_4[256][192])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    data_t BRAM_1[256][512];
    data_t BRAM_2[512][192];
    data_t BRAM_3[256][192];
    data_t BRAM_4[256][192];

    load_256_128_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_128_192_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_256_192_ap_fixed_16_5_(DRAM_3, BRAM_3);


//////////////////////////////////////////
// Begin: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////

    tiled_gemm(BRAM_1, BRAM_2, BRAM_3, BRAM_4); 

//////////////////////////////////////////
// End: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////

    store_256_192_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top(
    data_t DRAM_A1[96][512], data_t DRAM_A2[512][128], data_t DRAM_A3[96][128], data_t DRAM_A4[96][128],
    data_t DRAM_B1[128][256], data_t DRAM_B2[256][64], data_t DRAM_B3[128][64], data_t DRAM_B4[128][64],
    data_t DRAM_C1[256][128], data_t DRAM_C2[128][192], data_t DRAM_C3[256][192], data_t DRAM_C4[256][192]
){
    top_A(DRAM_A1, DRAM_A2, DRAM_A3, DRAM_A4);
    top_B(DRAM_B1, DRAM_B2, DRAM_B3, DRAM_B4);
    top_C(DRAM_C1, DRAM_C2, DRAM_C3, DRAM_C4);
}