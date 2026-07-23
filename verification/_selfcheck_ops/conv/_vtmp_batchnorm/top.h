#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef float data_t;

void top(data_t DRAM_input[8][16][16], data_t DRAM_weights[4][8], data_t DRAM_output[8][16][16]);

#endif // TOP_H