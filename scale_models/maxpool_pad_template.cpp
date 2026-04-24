/*
 * Auto-generated Max Pooling HLS Code with explicit padding support.
 *
 * Dimensions:
 *   Input  : [{C}][{H_IN}][{W_IN}]
 *   Output : [{C}][{H_OUT}][{W_OUT}]
 *   Pooling Window: [{K_H}][{K_W}]
 *   Stride: [{STRIDE_H}][{STRIDE_W}]
 *   Padding: [{PAD_H}][{PAD_W}]
 *
 * Data type: {DATA_TYPE}
 */

void maxpool_pad(
    data_t input[{C}][{H_IN}][{W_IN}],
    data_t output[{C}][{H_OUT}][{W_OUT}]
)
{{
    for (int c = 0; c < {C}; c++) {{
        for (int i = 0; i < {H_OUT}; i++) {{
            for (int j = 0; j < {W_OUT}; j++) {{
                bool initialized = false;
                data_t max_val = (data_t)0;
                for (int kh = 0; kh < {K_H}; kh++) {{
                    for (int kw = 0; kw < {K_W}; kw++) {{
                        int row = i * {STRIDE_H} + kh - {PAD_H};
                        int col = j * {STRIDE_W} + kw - {PAD_W};
                        if (row >= 0 && row < {H_IN} && col >= 0 && col < {W_IN}) {{
                            if (!initialized || input[c][row][col] > max_val) {{
                                max_val = input[c][row][col];
                                initialized = true;
                            }}
                        }}
                    }}
                }}
                output[c][i][j] = initialized ? max_val : (data_t)0;
            }}
        }}
    }}
}}
