
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>
using namespace std;

typedef ap_fixed<16,5> data_t;

data_t BRAM_input_0[2048][4096];
data_t BRAM_tmp_1[2048][4096];
data_t BRAM_tmp_2[2048][14336];
data_t BRAM_tmp_3[2048][14336];
data_t BRAM_input_1[2048][4096];
data_t BRAM_rms_norm_w1_0[4096];
data_t BRAM_rms_norm_w2_0[4096];
data_t BRAM_wq_0[4096][4096];
data_t BRAM_wk_0[4096][4096];
data_t BRAM_wv_0[4096][4096];
data_t BRAM_mm_w1_0[14336][4096];
data_t BRAM_mm_w2_0[14336][4096];
data_t BRAM_mm_w3_0[4096][14336];
data_t BRAM_input_2[2048][4096];
data_t BRAM_rms_norm_w1_1[4096];
data_t BRAM_rms_norm_w2_1[4096];
data_t BRAM_wq_1[4096][4096];
data_t BRAM_wk_1[4096][4096];
data_t BRAM_wv_1[4096][4096];
data_t BRAM_mm_w1_1[14336][4096];
data_t BRAM_mm_w2_1[14336][4096];
data_t BRAM_mm_w3_1[4096][14336];
data_t BRAM_input_3[2048][4096];
data_t BRAM_rms_norm_w1_2[4096];
data_t BRAM_rms_norm_w2_2[4096];
data_t BRAM_wq_2[4096][4096];
data_t BRAM_wk_2[4096][4096];
data_t BRAM_wv_2[4096][4096];
data_t BRAM_mm_w1_2[14336][4096];
data_t BRAM_mm_w2_2[14336][4096];
data_t BRAM_mm_w3_2[4096][14336];
data_t BRAM_input_4[2048][4096];
data_t BRAM_rms_norm_w1_3[4096];
data_t BRAM_rms_norm_w2_3[4096];
data_t BRAM_wq_3[4096][4096];
data_t BRAM_wk_3[4096][4096];
data_t BRAM_wv_3[4096][4096];
data_t BRAM_mm_w1_3[14336][4096];
data_t BRAM_mm_w2_3[14336][4096];
data_t BRAM_mm_w3_3[4096][14336];

void load_2048_4096_ap_fixed_16_5_(data_t input[2048][4096], data_t output[2048][4096])
{
    for (int idx0 = 0; idx0 < 2048; idx0++) {
        for (int idx1 = 0; idx1 < 4096; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_4096_ap_fixed_16_5_(data_t input[4096], data_t output[4096])
{
    for (int idx0 = 0; idx0 < 4096; idx0++) {
        output[idx0] = input[idx0];
    }
}

void load_4096_4096_ap_fixed_16_5_(data_t input[4096][4096], data_t output[4096][4096])
{
    for (int idx0 = 0; idx0 < 4096; idx0++) {
        for (int idx1 = 0; idx1 < 4096; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_14336_4096_ap_fixed_16_5_(data_t input[14336][4096], data_t output[14336][4096])
{
    for (int idx0 = 0; idx0 < 14336; idx0++) {
        for (int idx1 = 0; idx1 < 4096; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_4096_14336_ap_fixed_16_5_(data_t input[4096][14336], data_t output[4096][14336])
{
    for (int idx0 = 0; idx0 < 4096; idx0++) {
        for (int idx1 = 0; idx1 < 14336; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}


void rms_norm_2048_4096_ap_fixed_16_5_(
    data_t input[2048][4096],
    data_t gamma[4096],
    data_t output[2048][4096]
)
{
    for (int i = 0; i < 2048; i++) {
        data_t sum_sq = (data_t)0;
        for (int j = 0; j < 4096; j++) {
            sum_sq += input[i][j] * input[i][j];
        }
        data_t rms = hls::sqrt(sum_sq / (data_t)4096 + (data_t)1e-06);
        for (int j = 0; j < 4096; j++) {
            output[i][j] = gamma[j] * input[i][j] / rms;
        }
    }
}


/*
 * Auto-generated Grouped Multi-head Attention (with optional inline RoPE)
 *
 * Input     : [2048][4096]
 * W_q/k/v   : [4096][4096], DIM_OUT = NUM_HEADS * HEAD_DIM
 * Output    : [2048][4096]
 *
 * Data type : ap_fixed<16,5>
 * Num Heads : 32
 * Head Dim  : 128
 */

 void grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(
    data_t input[2048][4096],
    data_t W_q[4096][4096],
    data_t W_k[4096][4096],
    data_t W_v[4096][4096],
    data_t output[2048][4096],
    int groups
)
{
    const int num_heads = 32;   // total number of heads (must equal DIM_OUT / HEAD_DIM)
    const int head_dim = 128;       // dimension per head
    const int heads_per_group = num_heads / groups;
    const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);

    data_t Q[2048][4096];
    data_t K[2048][4096];
    data_t V[2048][4096];

    // Compute Q, K, V
    for (int seq = 0; seq < 2048; seq++) {
        for (int dout = 0; dout < 4096; dout++) {
            Q[seq][dout] = 0;
            K[seq][dout] = 0;
            V[seq][dout] = 0;
            for (int din = 0; din < 4096; din++) {
                Q[seq][dout] += input[seq][din] * W_q[dout][din];
                K[seq][dout] += input[seq][din] * W_k[dout][din];
                V[seq][dout] += input[seq][din] * W_v[dout][din];
            }
        }
    }

    /*==== BEGIN OPTIONAL ROPE LOGIC ====*/
    
    // Inline RoPE logic
    for (int seq = 0; seq < 2048; seq++) {
        for (int h = 0; h < 32; h++) {
            for (int d = 0; d < 128; d += 2) {
                int idx = h * 128 + d;
                data_t theta = (data_t)hls::powf(10000.0f, -((float)d) / (float)128);
                data_t angle = seq * theta;
                data_t cos_val = hls::cos(angle);
                data_t sin_val = hls::sin(angle);

                // Apply RoPE to Q
                data_t q0 = Q[seq][idx];
                data_t q1 = Q[seq][idx + 1];
                Q[seq][idx]     = q0 * cos_val - q1 * sin_val;
                Q[seq][idx + 1] = q0 * sin_val + q1 * cos_val;

                // Apply RoPE to K
                data_t k0 = K[seq][idx];
                data_t k1 = K[seq][idx + 1];
                K[seq][idx]     = k0 * cos_val - k1 * sin_val;
                K[seq][idx + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
    /*==== END OPTIONAL ROPE LOGIC ====*/

    // Compute Attention per head in groups.
    for (int g = 0; g < groups; g++) {
        for (int h = 0; h < heads_per_group; h++) {
            int head_index = g * heads_per_group + h;
            data_t scores[2048][2048];

            // Scaled Dot-product: Q x K^T for head head_index
            for (int i = 0; i < 2048; i++) {
                for (int j = 0; j < 2048; j++) {
                    scores[i][j] = 0;
                    for (int d = 0; d < head_dim; d++) {
                        int idx = head_index * head_dim + d;
                        scores[i][j] += Q[i][idx] * K[j][idx];
                    }
                    scores[i][j] *= scale;
                }
            }

            // Softmax for this head.
            for (int i = 0; i < 2048; i++) {
                data_t sum_exp = 0;
                data_t max_score = scores[i][0];
                for (int j = 1; j < 2048; j++) {
                    if (scores[i][j] > max_score)
                        max_score = scores[i][j];
                }
                for (int j = 0; j < 2048; j++) {
                    scores[i][j] = hls::exp(scores[i][j] - max_score);
                    sum_exp += scores[i][j];
                }
                for (int j = 0; j < 2048; j++) {
                    scores[i][j] /= sum_exp;
                }
            }

            // Compute context: scores x V for head head_index.
            for (int i = 0; i < 2048; i++) {
                for (int d = 0; d < head_dim; d++) {
                    data_t context = 0;
                    for (int j = 0; j < 2048; j++) {
                        context += scores[i][j] * V[j][head_index * head_dim + d];
                    }
                    output[i][head_index * head_dim + d] = context;
                }
            }
        }
    }
}


void matrix_add_2048_4096_ap_fixed_16_5_(
    data_t in1[2048][4096],
    data_t in2[2048][4096],
    data_t out[2048][4096]
)
{
    
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 4096; j++) {
            out[i][j] = in1[i][j] + in2[i][j];
        }
    }
    
}









void matmul_2048_4096_14336_ap_fixed_16_5_(
    data_t input[2048][4096],
    data_t weights[14336][4096],
    data_t output[2048][14336]
)
{
    // Initialize output to ((data_t)0)
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 14336; j++) {
            output[i][j] = ((data_t)0);
        }
    }

    // Matrix multiplication
    for (int i = 0; i < 2048; i++) {
        for (int k = 0; k < 4096; k++) {
            for (int j = 0; j < 14336; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}




void swish_2048_14336_ap_fixed_16_5_(
    data_t input[2048][14336],
    data_t output[2048][14336]
)
{
    // Swish: x * sigmoid(x)
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 14336; j++) {
            data_t sig = (data_t)1 / ((data_t)1 + hls::exp(-input[i][j]));
            output[i][j] = input[i][j] * sig;
        }
    }
}


void elementwise_mult_2048_14336_ap_fixed_16_5_(
    data_t in1[2048][14336],
    data_t in2[2048][14336],
    data_t out[2048][14336]
)
{
    
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 14336; j++) {
            out[i][j] = in1[i][j] * in2[i][j];
        }
    }
    
}









void matmul_2048_14336_4096_ap_fixed_16_5_(
    data_t input[2048][14336],
    data_t weights[4096][14336],
    data_t output[2048][4096]
)
{
    // Initialize output to ((data_t)0)
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 4096; j++) {
            output[i][j] = ((data_t)0);
        }
    }

    // Matrix multiplication
    for (int i = 0; i < 2048; i++) {
        for (int k = 0; k < 14336; k++) {
            for (int j = 0; j < 4096; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}


void store_2048_4096_ap_fixed_16_5_(data_t input[2048][4096], data_t output[2048][4096])
{
    for (int idx0 = 0; idx0 < 2048; idx0++) {
        for (int idx1 = 0; idx1 < 4096; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_input[2048][4096], data_t DRAM_rms_norm_w1_0[4096], data_t DRAM_rms_norm_w2_0[4096], data_t DRAM_wq_0[4096][4096], data_t DRAM_wk_0[4096][4096], data_t DRAM_wv_0[4096][4096], data_t DRAM_mm_w1_0[14336][4096], data_t DRAM_mm_w2_0[14336][4096], data_t DRAM_mm_w3_0[4096][14336], data_t DRAM_rms_norm_w1_1[4096], data_t DRAM_rms_norm_w2_1[4096], data_t DRAM_wq_1[4096][4096], data_t DRAM_wk_1[4096][4096], data_t DRAM_wv_1[4096][4096], data_t DRAM_mm_w1_1[14336][4096], data_t DRAM_mm_w2_1[14336][4096], data_t DRAM_mm_w3_1[4096][14336], data_t DRAM_rms_norm_w1_2[4096], data_t DRAM_rms_norm_w2_2[4096], data_t DRAM_wq_2[4096][4096], data_t DRAM_wk_2[4096][4096], data_t DRAM_wv_2[4096][4096], data_t DRAM_mm_w1_2[14336][4096], data_t DRAM_mm_w2_2[14336][4096], data_t DRAM_mm_w3_2[4096][14336], data_t DRAM_rms_norm_w1_3[4096], data_t DRAM_rms_norm_w2_3[4096], data_t DRAM_wq_3[4096][4096], data_t DRAM_wk_3[4096][4096], data_t DRAM_wv_3[4096][4096], data_t DRAM_mm_w1_3[14336][4096], data_t DRAM_mm_w2_3[14336][4096], data_t DRAM_mm_w3_3[4096][14336], data_t DRAM_output[2048][4096])
{
    #pragma HLS interface m_axi port=DRAM_input offset=slave bundle=mem_input
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_0 offset=slave bundle=mem_rms_norm_w1_0
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_0 offset=slave bundle=mem_rms_norm_w2_0
    #pragma HLS interface m_axi port=DRAM_wq_0 offset=slave bundle=mem_wq_0
    #pragma HLS interface m_axi port=DRAM_wk_0 offset=slave bundle=mem_wk_0
    #pragma HLS interface m_axi port=DRAM_wv_0 offset=slave bundle=mem_wv_0
    #pragma HLS interface m_axi port=DRAM_mm_w1_0 offset=slave bundle=mem_mm_w1_0
    #pragma HLS interface m_axi port=DRAM_mm_w2_0 offset=slave bundle=mem_mm_w2_0
    #pragma HLS interface m_axi port=DRAM_mm_w3_0 offset=slave bundle=mem_mm_w3_0
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_1 offset=slave bundle=mem_rms_norm_w1_1
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_1 offset=slave bundle=mem_rms_norm_w2_1
    #pragma HLS interface m_axi port=DRAM_wq_1 offset=slave bundle=mem_wq_1
    #pragma HLS interface m_axi port=DRAM_wk_1 offset=slave bundle=mem_wk_1
    #pragma HLS interface m_axi port=DRAM_wv_1 offset=slave bundle=mem_wv_1
    #pragma HLS interface m_axi port=DRAM_mm_w1_1 offset=slave bundle=mem_mm_w1_1
    #pragma HLS interface m_axi port=DRAM_mm_w2_1 offset=slave bundle=mem_mm_w2_1
    #pragma HLS interface m_axi port=DRAM_mm_w3_1 offset=slave bundle=mem_mm_w3_1
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_2 offset=slave bundle=mem_rms_norm_w1_2
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_2 offset=slave bundle=mem_rms_norm_w2_2
    #pragma HLS interface m_axi port=DRAM_wq_2 offset=slave bundle=mem_wq_2
    #pragma HLS interface m_axi port=DRAM_wk_2 offset=slave bundle=mem_wk_2
    #pragma HLS interface m_axi port=DRAM_wv_2 offset=slave bundle=mem_wv_2
    #pragma HLS interface m_axi port=DRAM_mm_w1_2 offset=slave bundle=mem_mm_w1_2
    #pragma HLS interface m_axi port=DRAM_mm_w2_2 offset=slave bundle=mem_mm_w2_2
    #pragma HLS interface m_axi port=DRAM_mm_w3_2 offset=slave bundle=mem_mm_w3_2
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_3 offset=slave bundle=mem_rms_norm_w1_3
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_3 offset=slave bundle=mem_rms_norm_w2_3
    #pragma HLS interface m_axi port=DRAM_wq_3 offset=slave bundle=mem_wq_3
    #pragma HLS interface m_axi port=DRAM_wk_3 offset=slave bundle=mem_wk_3
    #pragma HLS interface m_axi port=DRAM_wv_3 offset=slave bundle=mem_wv_3
    #pragma HLS interface m_axi port=DRAM_mm_w1_3 offset=slave bundle=mem_mm_w1_3
    #pragma HLS interface m_axi port=DRAM_mm_w2_3 offset=slave bundle=mem_mm_w2_3
    #pragma HLS interface m_axi port=DRAM_mm_w3_3 offset=slave bundle=mem_mm_w3_3
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem_output

    load_2048_4096_ap_fixed_16_5_(DRAM_input, BRAM_input_0);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_0, BRAM_rms_norm_w1_0);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_0, BRAM_rms_norm_w2_0);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_0, BRAM_wq_0);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_0, BRAM_wk_0);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_0, BRAM_wv_0);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_0, BRAM_mm_w1_0);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_0, BRAM_mm_w2_0);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_0, BRAM_mm_w3_0);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_1, BRAM_rms_norm_w1_1);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_1, BRAM_rms_norm_w2_1);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_1, BRAM_wq_1);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_1, BRAM_wk_1);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_1, BRAM_wv_1);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_1, BRAM_mm_w1_1);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_1, BRAM_mm_w2_1);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_1, BRAM_mm_w3_1);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_2, BRAM_rms_norm_w1_2);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_2, BRAM_rms_norm_w2_2);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_2, BRAM_wq_2);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_2, BRAM_wk_2);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_2, BRAM_wv_2);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_2, BRAM_mm_w1_2);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_2, BRAM_mm_w2_2);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_2, BRAM_mm_w3_2);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_3, BRAM_rms_norm_w1_3);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_3, BRAM_rms_norm_w2_3);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_3, BRAM_wq_3);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_3, BRAM_wk_3);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_3, BRAM_wv_3);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_3, BRAM_mm_w1_3);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_3, BRAM_mm_w2_3);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_3, BRAM_mm_w3_3);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_0, BRAM_rms_norm_w1_0, BRAM_input_0);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_0, BRAM_wq_0, BRAM_wk_0, BRAM_wv_0, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_0, BRAM_input_0);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_0, BRAM_rms_norm_w2_0, BRAM_input_0);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_0, BRAM_mm_w1_0, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_0, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_0, BRAM_input_1);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_1, BRAM_rms_norm_w1_1, BRAM_input_1);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_1, BRAM_wq_1, BRAM_wk_1, BRAM_wv_1, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_1, BRAM_input_1);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_1, BRAM_rms_norm_w2_1, BRAM_input_1);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_1, BRAM_mm_w1_1, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_1, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_1, BRAM_input_2);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_2, BRAM_rms_norm_w1_2, BRAM_input_2);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_2, BRAM_wq_2, BRAM_wk_2, BRAM_wv_2, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_2, BRAM_input_2);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_2, BRAM_rms_norm_w2_2, BRAM_input_2);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_2, BRAM_mm_w1_2, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_2, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_2, BRAM_input_3);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_3, BRAM_rms_norm_w1_3, BRAM_input_3);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_3, BRAM_wq_3, BRAM_wk_3, BRAM_wv_3, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_3, BRAM_input_3);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_3, BRAM_rms_norm_w2_3, BRAM_input_3);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_3, BRAM_mm_w1_3, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_3, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_3, BRAM_input_4);
    store_2048_4096_ap_fixed_16_5_(BRAM_input_32, DRAM_output);
}