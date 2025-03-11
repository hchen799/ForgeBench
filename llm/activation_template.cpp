
#include <math.h>
#include <stdio.h>

typedef {DATA_TYPE} data_t;

/*
 * Auto-generated Activation Functions for 2D Tensors
 * Tensor dimensions: [{SEQ_LENGTH}][{HIDDEN_DIM}]
 * Data type: {DATA_TYPE}
 *
 * Available functions:
 *   - relu
 *   - leaky_relu
 *   - prelu
 *   - rrelu
 *   - thresholded_relu
 *   - relu6
 *   - sigmoid
 *   - tanh_act
 *   - elu
 *   - selu
 *   - gelu
 *   - swish
 *   - softmax  // Softmax is computed along the hidden_dim axis.
 */

/*==== RELU FUNCTION START ====*/
void relu(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}]
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            output[i][j] = (input[i][j] > 0) ? input[i][j] : (data_t)0;
        }}
    }}
}}
/*==== RELU FUNCTION END ====*/

/*==== LEAKY_RELU FUNCTION START ====*/
void leaky_relu(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t alpha
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            output[i][j] = (input[i][j] >= 0) ? input[i][j] : alpha * input[i][j];
        }}
    }}
}}
/*==== LEAKY_RELU FUNCTION END ====*/

/*==== PRELU FUNCTION START ====*/
void prelu(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t alpha
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            output[i][j] = (input[i][j] >= 0) ? input[i][j] : alpha * input[i][j];
        }}
    }}
}}
/*==== PRELU FUNCTION END ====*/

/*==== RRELU FUNCTION START ====*/
void rrelu(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t lower,
    data_t upper
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            if (input[i][j] >= 0) {{
                output[i][j] = input[i][j];
            }} else {{
                data_t ralpha = (lower + upper) / 2; // fixed value for demonstration
                output[i][j] = ralpha * input[i][j];
            }}
        }}
    }}
}}
/*==== RRELU FUNCTION END ====*/

/*==== THRESHOLDED_RELU FUNCTION START ====*/
void thresholded_relu(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t theta
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            output[i][j] = (input[i][j] > theta) ? input[i][j] : (data_t)0;
        }}
    }}
}}
/*==== THRESHOLDED_RELU FUNCTION END ====*/

/*==== RELU6 FUNCTION START ====*/
void relu6(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t cap
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            data_t temp = (input[i][j] > 0) ? input[i][j] : (data_t)0;
            output[i][j] = (temp < cap) ? temp : cap;
        }}
    }}
}}
/*==== RELU6 FUNCTION END ====*/

/*==== SIGMOID FUNCTION START ====*/
void sigmoid(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}]
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            output[i][j] = (data_t)1 / ((data_t)1 + exp(-input[i][j]));
        }}
    }}
}}
/*==== SIGMOID FUNCTION END ====*/

/*==== TANH FUNCTION START ====*/
void tanh_act(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}]
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            output[i][j] = tanh(input[i][j]);
        }}
    }}
}}
/*==== TANH FUNCTION END ====*/

/*==== ELU FUNCTION START ====*/
void elu(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t alpha
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            if (input[i][j] >= 0) {{
                output[i][j] = input[i][j];
            }} else {{
                output[i][j] = alpha * (exp(input[i][j]) - 1);
            }}
        }}
    }}
}}
/*==== ELU FUNCTION END ====*/

/*==== SELU FUNCTION START ====*/
void selu(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t alpha,
    data_t lambda
)
{{
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            if (input[i][j] >= 0) {{
                output[i][j] = lambda * input[i][j];
            }} else {{
                output[i][j] = lambda * alpha * (exp(input[i][j]) - 1);
            }}
        }}
    }}
}}
/*==== SELU FUNCTION END ====*/

/*==== GELU FUNCTION START ====*/
void gelu(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}]
)
{{
    // Exact GELU using the error function:
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            data_t x = input[i][j];
            output[i][j] = 0.5 * x * (1 + erf(x / sqrt((data_t)2)));
        }}
    }}
}}
/*==== GELU FUNCTION END ====*/

/*==== SWISH FUNCTION START ====*/
void swish(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}]
)
{{
    // Swish: x * sigmoid(x)
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            data_t sig = (data_t)1 / ((data_t)1 + exp(-input[i][j]));
            output[i][j] = input[i][j] * sig;
        }}
    }}
}}
/*==== SWISH FUNCTION END ====*/

/*==== SOFTMAX FUNCTION START ====*/
void softmax(
    data_t input[{SEQ_LENGTH}][{HIDDEN_DIM}],
    data_t output[{SEQ_LENGTH}][{HIDDEN_DIM}]
)
{{
    // Compute softmax along the hidden dimension for each row.
    for (int i = 0; i < {SEQ_LENGTH}; i++) {{
        data_t sum = 0;
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            output[i][j] = exp(input[i][j]);
            sum += output[i][j];
        }}
        for (int j = 0; j < {HIDDEN_DIM}; j++) {{
            output[i][j] /= sum;
        }}
    }}
}}
/*==== SOFTMAX FUNCTION END ====*/
