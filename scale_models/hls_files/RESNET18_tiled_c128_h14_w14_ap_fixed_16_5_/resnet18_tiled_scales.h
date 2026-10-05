#pragma once
#include <ap_fixed.h>

// Shared-exponent output shifts for tiled reductions.
// Defaults are conservative placeholders and can be tuned per layer later.
// 0: SHIFT_STEM
// 1: SHIFT_S1_B0_1
// 2: SHIFT_S1_B0_2
// 3: SHIFT_S1_B1_1
// 4: SHIFT_S1_B1_2
// 5: SHIFT_S2_B0_1
// 6: SHIFT_S2_B0_2
// 7: SHIFT_S2_B0_DOWN
// 8: SHIFT_S2_B1_1
// 9: SHIFT_S2_B1_2
// 10: SHIFT_S3_B0_1
// 11: SHIFT_S3_B0_2
// 12: SHIFT_S3_B0_DOWN
// 13: SHIFT_S3_B1_1
// 14: SHIFT_S3_B1_2
// 15: SHIFT_S4_B0_1
// 16: SHIFT_S4_B0_2
// 17: SHIFT_S4_B0_DOWN
// 18: SHIFT_S4_B1_1
// 19: SHIFT_S4_B1_2
// 20: SHIFT_FC
enum ConvShiftIndex {
    SHIFT_STEM = 0,
    SHIFT_S1_B0_1 = 1,
    SHIFT_S1_B0_2 = 2,
    SHIFT_S1_B1_1 = 3,
    SHIFT_S1_B1_2 = 4,
    SHIFT_S2_B0_1 = 5,
    SHIFT_S2_B0_2 = 6,
    SHIFT_S2_B0_DOWN = 7,
    SHIFT_S2_B1_1 = 8,
    SHIFT_S2_B1_2 = 9,
    SHIFT_S3_B0_1 = 10,
    SHIFT_S3_B0_2 = 11,
    SHIFT_S3_B0_DOWN = 12,
    SHIFT_S3_B1_1 = 13,
    SHIFT_S3_B1_2 = 14,
    SHIFT_S4_B0_1 = 15,
    SHIFT_S4_B0_2 = 16,
    SHIFT_S4_B0_DOWN = 17,
    SHIFT_S4_B1_1 = 18,
    SHIFT_S4_B1_2 = 19,
    SHIFT_FC = 20
};

static const int kConvOutputShift[21] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static const int kRenormGuard = 256;
