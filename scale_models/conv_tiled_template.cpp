/*
 * Auto-generated tiled Convolution HLS Code.
 *
 * Dimensions:
 *   Input  : [{C_IN}][{H_IN}][{W_IN}]
 *   Kernel : [{C_OUT}][{C_IN}][{K}][{K}]
 *   Output : [{C_OUT}][{H_OUT}][{W_OUT}]
 *
 * Tile parameters:
 *   TILE_C       : {TILE_C}
 *   TILE_H       : {TILE_H}
 *   TILE_W       : {TILE_W}
 *   IC_CHUNK     : {IC_CHUNK}
 *   OC_CHUNK     : {OC_CHUNK}
 *   RENORM_GUARD : {RENORM_GUARD}
 *
 * Data type: {DATA_TYPE}
 */

/*==== CONV2D_TILED FUNCTION START ====*/
void conv2d_tiled(
    data_t input[{C_IN}][{H_IN}][{W_IN}],
    data_t kernel[{C_OUT}][{C_IN}][{K}][{K}],
    data_t output[{C_OUT}][{H_OUT}][{W_OUT}]
)
{{
    data_t input_tile[{IC_CHUNK}][{PATCH_H}][{PATCH_W}];
    data_t weight_tile[{OC_CHUNK}][{IC_CHUNK}][{K}][{K}];
    data_t output_tile[{OC_CHUNK}][{TILE_H}][{TILE_W}];

    #pragma HLS array_partition variable=input_tile type=cyclic factor=8 dim=1
    #pragma HLS array_partition variable=weight_tile type=cyclic factor=8 dim=1
    #pragma HLS array_partition variable=weight_tile type=cyclic factor=8 dim=2
    #pragma HLS array_partition variable=output_tile type=cyclic factor=8 dim=1

    for (int co_base = 0; co_base < {C_OUT}; co_base += {OC_CHUNK}) {{
        int valid_oc = ((co_base + {OC_CHUNK}) <= {C_OUT}) ? {OC_CHUNK} : ({C_OUT} - co_base);
        for (int oh_base = 0; oh_base < {H_OUT}; oh_base += {TILE_H}) {{
            int valid_h = ((oh_base + {TILE_H}) <= {H_OUT}) ? {TILE_H} : ({H_OUT} - oh_base);
            for (int ow_base = 0; ow_base < {W_OUT}; ow_base += {TILE_W}) {{
                int valid_w = ((ow_base + {TILE_W}) <= {W_OUT}) ? {TILE_W} : ({W_OUT} - ow_base);

                for (int oc = 0; oc < valid_oc; ++oc) {{
                    for (int th = 0; th < valid_h; ++th) {{
                        for (int tw = 0; tw < valid_w; ++tw) {{
                            output_tile[oc][th][tw] = (data_t)0;
                        }}
                    }}
                }}

                for (int ci_base = 0; ci_base < {C_IN}; ci_base += {IC_CHUNK}) {{
                    int valid_ci = ((ci_base + {IC_CHUNK}) <= {C_IN}) ? {IC_CHUNK} : ({C_IN} - ci_base);

                    for (int ic = 0; ic < valid_ci; ++ic) {{
                        for (int ph = 0; ph < {PATCH_H}; ++ph) {{
                            for (int pw = 0; pw < {PATCH_W}; ++pw) {{
                                int in_row = oh_base * {STRIDE} - {PAD} + ph;
                                int in_col = ow_base * {STRIDE} - {PAD} + pw;
                                if (in_row >= 0 && in_row < {H_IN} && in_col >= 0 && in_col < {W_IN}) {{
                                    input_tile[ic][ph][pw] = input[ci_base + ic][in_row][in_col];
                                }} else {{
                                    input_tile[ic][ph][pw] = (data_t)0;
                                }}
                            }}
                        }}
                    }}

                    for (int oc = 0; oc < valid_oc; ++oc) {{
                        for (int ic = 0; ic < valid_ci; ++ic) {{
                            for (int kh = 0; kh < {K}; ++kh) {{
                                for (int kw = 0; kw < {K}; ++kw) {{
                                    weight_tile[oc][ic][kh][kw] = kernel[co_base + oc][ci_base + ic][kh][kw];
                                }}
                            }}
                        }}
                    }}

                    for (int oc = 0; oc < valid_oc; ++oc) {{
                        for (int th = 0; th < valid_h; ++th) {{
                            for (int tw = 0; tw < valid_w; ++tw) {{
                                data_t sum = output_tile[oc][th][tw];
                                for (int ic = 0; ic < valid_ci; ++ic) {{
                                    for (int kh = 0; kh < {K}; ++kh) {{
                                        for (int kw = 0; kw < {K}; ++kw) {{
                                            sum += input_tile[ic][th * {STRIDE} + kh][tw * {STRIDE} + kw] *
                                                   weight_tile[oc][ic][kh][kw];
                                        }}
                                    }}
                                }}
                                output_tile[oc][th][tw] = sum;
                            }}
                        }}
                    }}

                    for (int oc = 0; oc < valid_oc; ++oc) {{
                        for (int th = 0; th < valid_h; ++th) {{
                            for (int tw = 0; tw < valid_w; ++tw) {{
                            }}
                        }}
                    }}
                }}

                for (int oc = 0; oc < valid_oc; ++oc) {{
                    for (int th = 0; th < valid_h; ++th) {{
                        for (int tw = 0; tw < valid_w; ++tw) {{
                            output[co_base + oc][oh_base + th][ow_base + tw] = output_tile[oc][th][tw];
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
/*==== CONV2D_TILED FUNCTION END ====*/
