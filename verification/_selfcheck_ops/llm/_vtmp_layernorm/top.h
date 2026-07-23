#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef float data_t;

void top(data_t DRAM_input[8][32], data_t DRAM_weights[2][32], data_t DRAM_output[8][32]);

#endif // TOP_H