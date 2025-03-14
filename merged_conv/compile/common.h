#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>
// Maximum spatial dimension (enough for up to 14x14 = 196 elements)
#define MAX_SPATIAL 256
// Maximum channels supported
#define MAX_CH 256
#define MAX_LOCAL_SIZE 23

static const int BLOCK_IN_CH  = 64;
static const int BLOCK_OUT_CH = 64;
static const int BLOCK_OUT_H  = 7;  // maximum tile height (output)
static const int BLOCK_OUT_W  = 7;  // maximum tile width (output)
static const int KSIZE        = 3;

typedef float data_t;

// Inline functions or function definitions in a header should be marked inline
inline int out_dim(int size, int pad, int stride, int kernel) {
    return (size + 2 * pad - kernel) / stride + 1;
}

inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

inline int min_value (int x, int y) {
    return (x < y) ? x : y;
}

#endif // COMMON_H
