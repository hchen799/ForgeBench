#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_1[64][32], data_t DRAM_2[32][128], data_t DRAM_3[128], data_t DRAM_4[64][128], data_t DRAM_5[128], data_t DRAM_10[128], data_t DRAM_11[1], data_t DRAM_12[1]);

#endif // TOP_H