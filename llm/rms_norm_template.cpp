#include <math.h>
#include <stdio.h>

typedef {DATA_TYPE} data_t;

void rms_norm(
    data_t input[{SEQ_LENGTH}][{DIM}],
    data_t gamma[{DIM}],
    data_t output[{SEQ_LENGTH}][{DIM}]
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        data_t sum_sq = (data_t)0;
        for (int j = 0; j < {DIM}; j++) {{
            sum_sq += input[i][j] * input[i][j];
        }}
        data_t rms = sqrt(sum_sq / {DIM} + {EPSILON});
        for (int j = 0; j < {DIM}; j++) {{
            output[i][j] = gamma[j] * input[i][j] / rms;
        }}
    }}
}}
