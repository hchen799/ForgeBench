#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_1[2][4][4], data_t DRAM_2[2][2][3][3], data_t DRAM_3[2], data_t DRAM_4[4][2], data_t DRAM_5[2][2][2]);

#endif // TOP_H