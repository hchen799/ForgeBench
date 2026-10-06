
/*
 * Auto-generated Batch Normalization HLS Code
 *
 * Dimensions:
 *   Input/Output: [{C_OUT}][{H}][{W}]
 * Data type: {DATA_TYPE}
 * Epsilon: {EPSILON}
 *
 * The batch normalization is computed as:
 *   output[c][h][w] = gamma[c]*(input[c][h][w] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];
 */

// Use a typedef for the data type
// weights[4][C_OUT], 0: gamma, 1: beta, 2: mean, 3: variance
void batch_norm(
    data_t input[{C_OUT}][{H}][{W}],
    data_t weights[4][{C_OUT}],
    data_t output[{C_OUT}][{H}][{W}]
)
{{
    for (int c = 0; c < {C_OUT}; c++) {{
        for (int h = 0; h < {H}; h++) {{
            for (int w = 0; w < {W}; w++) {{
                acc_t norm = fb_div(acc_t(input[c][h][w]) - acc_t(weights[2][c]), fb_sqrt(acc_t(weights[3][c]) + FB_EPS));
                output[c][h][w] = acc_t(weights[0][c]) * norm + acc_t(weights[1][c]);
            }}
        }}
    }}
}}
