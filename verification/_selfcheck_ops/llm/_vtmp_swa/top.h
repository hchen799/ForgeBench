#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef float data_t;

void top(data_t DRAM_attn_input[8][32], data_t DRAM_weights_q[32][32], data_t DRAM_weights_k[32][32], data_t DRAM_weights_v[32][32], data_t DRAM_output[8][32]);

#endif // TOP_H