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
#define MAX_SPATIAL 3136
// Maximum channels supported
#define MAX_CH 256
#define MAX_LOCAL_SIZE 9 //stride*(BLOCK_OUT_H - 1) + KSIZE


#define BLOCK_IN_CH 64
#define BLOCK_OUT_CH 64
#define BLOCK_OUT_H 7  // maximum tile height (output)
#define BLOCK_OUT_W 7  // maximum tile width (output)
#define KSIZE 3


#define MAX_C 256   // Maximum channels
#define MAX_H 56   // Maximum height
#define MAX_W 56   // Maximum width
#define MAX_OUT_H 56   // Maximum height
#define MAX_OUT_W 56   // Maximum width

#define BLOCK_C 64  // Channel tile size
#define BLOCK_H 7   // Spatial tile height
#define BLOCK_W 7   // Spatial tile width

#define EPSILON 1e-5

#define MAX_H_IN  256   // Maximum input height
#define MAX_W_IN  256   // Maximum input width

// Pooling parameters.
#define K_H       2     // Kernel height
#define K_W       2     // Kernel width
#define STRIDE_H  2     // Stride in height
#define STRIDE_W  2     // Stride in width

// For the tiled maxpool kernel, we need a local input buffer large enough to
// cover a tile of outputs. Its dimensions are computed as:
#define MAX_LOCAL_IN_H ((BLOCK_H - 1) * STRIDE_H + K_H)
#define MAX_LOCAL_IN_W ((BLOCK_W - 1) * STRIDE_W + K_W)



typedef ap_fixed<16, 5> data_t;

// Inline functions or function definitions in a header should be marked inline
int out_dim(int size, int pad, int stride, int kernel) {
    return (size + 2 * pad - kernel) / stride + 1;
}

int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

int min_value (int x, int y) {
    return (x < y) ? x : y;
}

#endif // COMMON_H
