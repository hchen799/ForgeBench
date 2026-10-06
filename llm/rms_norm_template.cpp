
void rms_norm(
    data_t input[{SEQ_LENGTH}][{DIM}],
    data_t gamma[{DIM}],
    data_t output[{SEQ_LENGTH}][{DIM}]
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        {ACC} sum_sq = ({ACC})0;
        for (int j = 0; j < {DIM}; j++) {{
            sum_sq += input[i][j] * input[i][j];
        }}
        {ACC} rms = hls::sqrt({SQRT_ARG});
        for (int j = 0; j < {DIM}; j++) {{
            output[i][j] = gamma[j] * input[i][j] / rms;
        }}
    }}
}}
