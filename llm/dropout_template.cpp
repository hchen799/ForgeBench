
#include <math.h>
#include <stdio.h>

typedef {DATA_TYPE} data_t;

// Simple LCG for pseudo-random numbers.
unsigned int lcg_rand(unsigned int *seed) {{
    *seed = (1103515245 * (*seed) + 12345) & 0x7fffffff;
    return *seed;
}}

void dropout(
    data_t input[{SEQ_LENGTH}][{DIM}],
    data_t output[{SEQ_LENGTH}][{DIM}],
    data_t dropout_prob,
    unsigned int seed
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {DIM}; j++) {{
            // Generate a random value between 0 and 1.
            unsigned int r = lcg_rand(&seed);
            data_t rand_val = (data_t)r / (data_t)2147483647;
            // Apply dropout: if random value is below dropout_prob, output is 0;
            // otherwise, scale the input.
            if (rand_val < dropout_prob) {{
                output[i][j] = (data_t)0;
            }} else {{
                output[i][j] = input[i][j] / (1 - dropout_prob);
            }}
        }}
    }}
}}
