#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16,5> data_t;

void top(data_t DRAM_input[2048][4096], data_t DRAM_rms_norm_w1_0[4096], data_t DRAM_rms_norm_w2_0[4096], data_t DRAM_wq_0[4096][4096], data_t DRAM_wk_0[4096][4096], data_t DRAM_wv_0[4096][4096], data_t DRAM_mm_w1_0[14336][4096], data_t DRAM_mm_w2_0[14336][4096], data_t DRAM_mm_w3_0[4096][14336], data_t DRAM_rms_norm_w1_1[4096], data_t DRAM_rms_norm_w2_1[4096], data_t DRAM_wq_1[4096][4096], data_t DRAM_wk_1[4096][4096], data_t DRAM_wv_1[4096][4096], data_t DRAM_mm_w1_1[14336][4096], data_t DRAM_mm_w2_1[14336][4096], data_t DRAM_mm_w3_1[4096][14336], data_t DRAM_rms_norm_w1_2[4096], data_t DRAM_rms_norm_w2_2[4096], data_t DRAM_wq_2[4096][4096], data_t DRAM_wk_2[4096][4096], data_t DRAM_wv_2[4096][4096], data_t DRAM_mm_w1_2[14336][4096], data_t DRAM_mm_w2_2[14336][4096], data_t DRAM_mm_w3_2[4096][14336], data_t DRAM_rms_norm_w1_3[4096], data_t DRAM_rms_norm_w2_3[4096], data_t DRAM_wq_3[4096][4096], data_t DRAM_wk_3[4096][4096], data_t DRAM_wv_3[4096][4096], data_t DRAM_mm_w1_3[14336][4096], data_t DRAM_mm_w2_3[14336][4096], data_t DRAM_mm_w3_3[4096][14336], data_t DRAM_output[2048][4096]);

#endif // TOP_H