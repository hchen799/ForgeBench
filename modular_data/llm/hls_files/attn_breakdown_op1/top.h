#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>

#ifndef TOP_H
#define TOP_H


typedef ap_fixed<32, 10> data_t;
typedef ap_fixed<32, 10> acc_t;    // accumulators and normalization arithmetic
typedef ap_fixed<32, 10> math_t;   // operand type of hls::sqrt / hls::exp (needs default AP_TRN/AP_WRAP modes)

#define SEQ_LENGTH 8
#define DIM 32
#define NUM_HEAD 8
#define HEAD_DIM 4

void top(data_t input_dram_A[SEQ_LENGTH][DIM], data_t Q_weight_dram_A[DIM][DIM], data_t K_weight_dram_A[DIM][DIM], data_t V_weight_dram_A[DIM][DIM], data_t output_dram_A[SEQ_LENGTH][DIM],
        data_t input_dram_B[SEQ_LENGTH][DIM], data_t Q_weight_dram_B[DIM][DIM], data_t K_weight_dram_B[DIM][DIM], data_t V_weight_dram_B[DIM][DIM], data_t output_dram_B[SEQ_LENGTH][DIM]);

#endif // TOP_H
