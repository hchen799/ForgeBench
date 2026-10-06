

void matmul(
    data_t input[{SEQ_LENGTH}][{DIM_IN}],
    data_t weights[{DIM_OUT}][{DIM_IN}],
{BIAS_ARG}    data_t output[{SEQ_LENGTH}][{DIM_OUT}]
)
{{
{BODY}}}
