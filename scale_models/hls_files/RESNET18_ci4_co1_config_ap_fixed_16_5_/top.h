#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16,5> data_t;

void top(data_t DRAM_input[3][224][224], data_t DRAM_w_stem[64][3][7][7], data_t DRAM_bn_stem[4][64], data_t DRAM_w_s1_b0_1[64][64][3][3], data_t DRAM_bn_s1_b0_1[4][64], data_t DRAM_w_s1_b0_2[64][64][3][3], data_t DRAM_bn_s1_b0_2[4][64], data_t DRAM_w_s1_b1_1[64][64][3][3], data_t DRAM_bn_s1_b1_1[4][64], data_t DRAM_w_s1_b1_2[64][64][3][3], data_t DRAM_bn_s1_b1_2[4][64], data_t DRAM_w_s2_b0_1[128][64][3][3], data_t DRAM_bn_s2_b0_1[4][128], data_t DRAM_w_s2_b0_2[128][128][3][3], data_t DRAM_bn_s2_b0_2[4][128], data_t DRAM_w_s2_b0_down[128][64][1][1], data_t DRAM_w_s2_b1_1[128][128][3][3], data_t DRAM_bn_s2_b1_1[4][128], data_t DRAM_w_s2_b1_2[128][128][3][3], data_t DRAM_bn_s2_b1_2[4][128], data_t DRAM_w_s3_b0_1[256][128][3][3], data_t DRAM_bn_s3_b0_1[4][256], data_t DRAM_w_s3_b0_2[256][256][3][3], data_t DRAM_bn_s3_b0_2[4][256], data_t DRAM_w_s3_b0_down[256][128][1][1], data_t DRAM_w_s3_b1_1[256][256][3][3], data_t DRAM_bn_s3_b1_1[4][256], data_t DRAM_w_s3_b1_2[256][256][3][3], data_t DRAM_bn_s3_b1_2[4][256], data_t DRAM_w_s4_b0_1[512][256][3][3], data_t DRAM_bn_s4_b0_1[4][512], data_t DRAM_w_s4_b0_2[512][512][3][3], data_t DRAM_bn_s4_b0_2[4][512], data_t DRAM_w_s4_b0_down[512][256][1][1], data_t DRAM_w_s4_b1_1[512][512][3][3], data_t DRAM_bn_s4_b1_1[4][512], data_t DRAM_w_s4_b1_2[512][512][3][3], data_t DRAM_bn_s4_b1_2[4][512], data_t DRAM_out[1000][1][1], data_t DRAM_fc[1000][512][1][1]);

#endif // TOP_H