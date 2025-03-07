
/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [{C_IN}][{H}][{W}]
 *   Kernel : [{C_OUT}][{C_IN}][{K}][{K}]
 *   Output : [{C_OUT}][{H}][{W}]
 *
 * Data type: {DATA_TYPE}
 *
 * If bias is included, a 1D bias array of size [{C_OUT}] is expected.
 * For group convolution, we assume:
 *   {C_IN} % groups == 0
 *   {C_OUT} % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (Standard)
//////////////////////////////////////////
/*==== CONV2D FUNCTION START ====*/
void conv2d(
    data_t input[{C_IN}][{H}][{W}],
    data_t kernel[{C_OUT}][{C_IN}][{K}][{K}],
{CONV_BIAS_ARG}    data_t output[{C_OUT}][{H}][{W}]
)
{{
    // Initialize output to {BIAS_INIT_EXPR}
    for (int co = 0; co < {C_OUT}; co++) {{
        for (int row = 0; row < {H}; row++) {{
            for (int col = 0; col < {W}; col++) {{
                #pragma HLS PIPELINE II=1
                output[co][row][col] = {BIAS_INIT_EXPR};
            }}
        }}
    }}

    // 6 nested loops: co, ci, row, col, kh, kw
    for (int co = 0; co < {C_OUT}; co++) {{
        for (int ci = 0; ci < {C_IN}; ci++) {{
            for (int row = 0; row < {H}; row++) {{
                for (int col = 0; col < {W}; col++) {{
                    for (int kh = 0; kh < {K}; kh++) {{
                        for (int kw = 0; kw < {K}; kw++) {{
                            output[co][row][col] +=
                                input[ci][row + kh][col + kw] * kernel[co][ci][kh][kw];
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
/*==== CONV2D FUNCTION END ====*/
//////////////////////////////////////////
// End: CONV2D FUNCTION (Standard)
//////////////////////////////////////////

//////////////////////////////////////////
// Begin: GROUP_CONV2D FUNCTION
//////////////////////////////////////////
/*==== GROUP_CONV2D FUNCTION START ====*/
void group_conv2d(
    data_t input[{C_IN}][{H}][{W}],
    data_t kernel[{C_OUT}][{C_IN}][{K}][{K}],
{GROUP_BIAS_ARG}    data_t output[{C_OUT}][{H}][{W}],
    int groups
)
{{
    int c_in_per_group  = {C_IN} / groups;
    int c_out_per_group = {C_OUT} / groups;

    // Initialize output to {GROUP_BIAS_INIT_EXPR}
    for (int co = 0; co < {C_OUT}; co++) {{
        for (int row = 0; row < {H}; row++) {{
            for (int col = 0; col < {W}; col++) {{
                #pragma HLS PIPELINE II=1
                output[co][row][col] = {GROUP_BIAS_INIT_EXPR};
            }}
        }}
    }}

    // Group convolution: partition input/output channels
    for (int g = 0; g < groups; g++) {{
        int co_start = g * c_out_per_group;
        int co_end   = co_start + c_out_per_group;

        int ci_start = g * c_in_per_group;
        int ci_end   = ci_start + c_in_per_group;

        for (int co = co_start; co < co_end; co++) {{
            for (int ci = ci_start; ci < ci_end; ci++) {{
                for (int row = 0; row < {H}; row++) {{
                    for (int col = 0; col < {W}; col++) {{
                        for (int kh = 0; kh < {K}; kh++) {{
                            for (int kw = 0; kw < {K}; kw++) {{
                                output[co][row][col] +=
                                    input[ci][row + kh][col + kw] * kernel[co][ci][kh][kw];
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
/*==== GROUP_CONV2D FUNCTION END ====*/
//////////////////////////////////////////
// End: GROUP_CONV2D FUNCTION
//////////////////////////////////////////

