
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

void batch_norm(
    data_t input[{C_OUT}][{H}][{W}],
    data_t gamma[{C_OUT}],
    data_t beta[{C_OUT}],
    data_t mean[{C_OUT}],
    data_t variance[{C_OUT}],
    data_t output[{C_OUT}][{H}][{W}]
)
{{
    for (int c = 0; c < {C_OUT}; c++) {{
        for (int h = 0; h < {H}; h++) {{
            for (int w = 0; w < {W}; w++) {{
                data_t norm = (input[c][h][w] - mean[c]) / sqrt(variance[c] + {EPSILON});
                output[c][h][w] = gamma[c] * norm + beta[c];
            }}
        }}
    }}
}}
