
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
typedef ap_fixed<32,10> acc_t;

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
data_t BRAM_input_5[2048][4096];
data_t BRAM_rms_norm_w1_4[4096];
data_t BRAM_rms_norm_w2_4[4096];
data_t BRAM_wq_4[4096][4096];
data_t BRAM_wk_4[4096][4096];
data_t BRAM_wv_4[4096][4096];
data_t BRAM_mm_w1_4[14336][4096];
data_t BRAM_mm_w2_4[14336][4096];
data_t BRAM_mm_w3_4[4096][14336];
data_t BRAM_input_6[2048][4096];
data_t BRAM_rms_norm_w1_5[4096];
data_t BRAM_rms_norm_w2_5[4096];
data_t BRAM_wq_5[4096][4096];
data_t BRAM_wk_5[4096][4096];
data_t BRAM_wv_5[4096][4096];
data_t BRAM_mm_w1_5[14336][4096];
data_t BRAM_mm_w2_5[14336][4096];
data_t BRAM_mm_w3_5[4096][14336];
data_t BRAM_input_7[2048][4096];
data_t BRAM_rms_norm_w1_6[4096];
data_t BRAM_rms_norm_w2_6[4096];
data_t BRAM_wq_6[4096][4096];
data_t BRAM_wk_6[4096][4096];
data_t BRAM_wv_6[4096][4096];
data_t BRAM_mm_w1_6[14336][4096];
data_t BRAM_mm_w2_6[14336][4096];
data_t BRAM_mm_w3_6[4096][14336];
data_t BRAM_input_8[2048][4096];
data_t BRAM_rms_norm_w1_7[4096];
data_t BRAM_rms_norm_w2_7[4096];
data_t BRAM_wq_7[4096][4096];
data_t BRAM_wk_7[4096][4096];
data_t BRAM_wv_7[4096][4096];
data_t BRAM_mm_w1_7[14336][4096];
data_t BRAM_mm_w2_7[14336][4096];
data_t BRAM_mm_w3_7[4096][14336];
data_t BRAM_input_9[2048][4096];
data_t BRAM_rms_norm_w1_8[4096];
data_t BRAM_rms_norm_w2_8[4096];
data_t BRAM_wq_8[4096][4096];
data_t BRAM_wk_8[4096][4096];
data_t BRAM_wv_8[4096][4096];
data_t BRAM_mm_w1_8[14336][4096];
data_t BRAM_mm_w2_8[14336][4096];
data_t BRAM_mm_w3_8[4096][14336];
data_t BRAM_input_10[2048][4096];
data_t BRAM_rms_norm_w1_9[4096];
data_t BRAM_rms_norm_w2_9[4096];
data_t BRAM_wq_9[4096][4096];
data_t BRAM_wk_9[4096][4096];
data_t BRAM_wv_9[4096][4096];
data_t BRAM_mm_w1_9[14336][4096];
data_t BRAM_mm_w2_9[14336][4096];
data_t BRAM_mm_w3_9[4096][14336];
data_t BRAM_input_11[2048][4096];
data_t BRAM_rms_norm_w1_10[4096];
data_t BRAM_rms_norm_w2_10[4096];
data_t BRAM_wq_10[4096][4096];
data_t BRAM_wk_10[4096][4096];
data_t BRAM_wv_10[4096][4096];
data_t BRAM_mm_w1_10[14336][4096];
data_t BRAM_mm_w2_10[14336][4096];
data_t BRAM_mm_w3_10[4096][14336];
data_t BRAM_input_12[2048][4096];
data_t BRAM_rms_norm_w1_11[4096];
data_t BRAM_rms_norm_w2_11[4096];
data_t BRAM_wq_11[4096][4096];
data_t BRAM_wk_11[4096][4096];
data_t BRAM_wv_11[4096][4096];
data_t BRAM_mm_w1_11[14336][4096];
data_t BRAM_mm_w2_11[14336][4096];
data_t BRAM_mm_w3_11[4096][14336];
data_t BRAM_input_13[2048][4096];
data_t BRAM_rms_norm_w1_12[4096];
data_t BRAM_rms_norm_w2_12[4096];
data_t BRAM_wq_12[4096][4096];
data_t BRAM_wk_12[4096][4096];
data_t BRAM_wv_12[4096][4096];
data_t BRAM_mm_w1_12[14336][4096];
data_t BRAM_mm_w2_12[14336][4096];
data_t BRAM_mm_w3_12[4096][14336];
data_t BRAM_input_14[2048][4096];
data_t BRAM_rms_norm_w1_13[4096];
data_t BRAM_rms_norm_w2_13[4096];
data_t BRAM_wq_13[4096][4096];
data_t BRAM_wk_13[4096][4096];
data_t BRAM_wv_13[4096][4096];
data_t BRAM_mm_w1_13[14336][4096];
data_t BRAM_mm_w2_13[14336][4096];
data_t BRAM_mm_w3_13[4096][14336];
data_t BRAM_input_15[2048][4096];
data_t BRAM_rms_norm_w1_14[4096];
data_t BRAM_rms_norm_w2_14[4096];
data_t BRAM_wq_14[4096][4096];
data_t BRAM_wk_14[4096][4096];
data_t BRAM_wv_14[4096][4096];
data_t BRAM_mm_w1_14[14336][4096];
data_t BRAM_mm_w2_14[14336][4096];
data_t BRAM_mm_w3_14[4096][14336];
data_t BRAM_input_16[2048][4096];
data_t BRAM_rms_norm_w1_15[4096];
data_t BRAM_rms_norm_w2_15[4096];
data_t BRAM_wq_15[4096][4096];
data_t BRAM_wk_15[4096][4096];
data_t BRAM_wv_15[4096][4096];
data_t BRAM_mm_w1_15[14336][4096];
data_t BRAM_mm_w2_15[14336][4096];
data_t BRAM_mm_w3_15[4096][14336];
data_t BRAM_input_17[2048][4096];
data_t BRAM_rms_norm_w1_16[4096];
data_t BRAM_rms_norm_w2_16[4096];
data_t BRAM_wq_16[4096][4096];
data_t BRAM_wk_16[4096][4096];
data_t BRAM_wv_16[4096][4096];
data_t BRAM_mm_w1_16[14336][4096];
data_t BRAM_mm_w2_16[14336][4096];
data_t BRAM_mm_w3_16[4096][14336];
data_t BRAM_input_18[2048][4096];
data_t BRAM_rms_norm_w1_17[4096];
data_t BRAM_rms_norm_w2_17[4096];
data_t BRAM_wq_17[4096][4096];
data_t BRAM_wk_17[4096][4096];
data_t BRAM_wv_17[4096][4096];
data_t BRAM_mm_w1_17[14336][4096];
data_t BRAM_mm_w2_17[14336][4096];
data_t BRAM_mm_w3_17[4096][14336];
data_t BRAM_input_19[2048][4096];
data_t BRAM_rms_norm_w1_18[4096];
data_t BRAM_rms_norm_w2_18[4096];
data_t BRAM_wq_18[4096][4096];
data_t BRAM_wk_18[4096][4096];
data_t BRAM_wv_18[4096][4096];
data_t BRAM_mm_w1_18[14336][4096];
data_t BRAM_mm_w2_18[14336][4096];
data_t BRAM_mm_w3_18[4096][14336];
data_t BRAM_input_20[2048][4096];
data_t BRAM_rms_norm_w1_19[4096];
data_t BRAM_rms_norm_w2_19[4096];
data_t BRAM_wq_19[4096][4096];
data_t BRAM_wk_19[4096][4096];
data_t BRAM_wv_19[4096][4096];
data_t BRAM_mm_w1_19[14336][4096];
data_t BRAM_mm_w2_19[14336][4096];
data_t BRAM_mm_w3_19[4096][14336];
data_t BRAM_input_21[2048][4096];
data_t BRAM_rms_norm_w1_20[4096];
data_t BRAM_rms_norm_w2_20[4096];
data_t BRAM_wq_20[4096][4096];
data_t BRAM_wk_20[4096][4096];
data_t BRAM_wv_20[4096][4096];
data_t BRAM_mm_w1_20[14336][4096];
data_t BRAM_mm_w2_20[14336][4096];
data_t BRAM_mm_w3_20[4096][14336];
data_t BRAM_input_22[2048][4096];
data_t BRAM_rms_norm_w1_21[4096];
data_t BRAM_rms_norm_w2_21[4096];
data_t BRAM_wq_21[4096][4096];
data_t BRAM_wk_21[4096][4096];
data_t BRAM_wv_21[4096][4096];
data_t BRAM_mm_w1_21[14336][4096];
data_t BRAM_mm_w2_21[14336][4096];
data_t BRAM_mm_w3_21[4096][14336];
data_t BRAM_input_23[2048][4096];
data_t BRAM_rms_norm_w1_22[4096];
data_t BRAM_rms_norm_w2_22[4096];
data_t BRAM_wq_22[4096][4096];
data_t BRAM_wk_22[4096][4096];
data_t BRAM_wv_22[4096][4096];
data_t BRAM_mm_w1_22[14336][4096];
data_t BRAM_mm_w2_22[14336][4096];
data_t BRAM_mm_w3_22[4096][14336];
data_t BRAM_input_24[2048][4096];
data_t BRAM_rms_norm_w1_23[4096];
data_t BRAM_rms_norm_w2_23[4096];
data_t BRAM_wq_23[4096][4096];
data_t BRAM_wk_23[4096][4096];
data_t BRAM_wv_23[4096][4096];
data_t BRAM_mm_w1_23[14336][4096];
data_t BRAM_mm_w2_23[14336][4096];
data_t BRAM_mm_w3_23[4096][14336];
data_t BRAM_input_25[2048][4096];
data_t BRAM_rms_norm_w1_24[4096];
data_t BRAM_rms_norm_w2_24[4096];
data_t BRAM_wq_24[4096][4096];
data_t BRAM_wk_24[4096][4096];
data_t BRAM_wv_24[4096][4096];
data_t BRAM_mm_w1_24[14336][4096];
data_t BRAM_mm_w2_24[14336][4096];
data_t BRAM_mm_w3_24[4096][14336];
data_t BRAM_input_26[2048][4096];
data_t BRAM_rms_norm_w1_25[4096];
data_t BRAM_rms_norm_w2_25[4096];
data_t BRAM_wq_25[4096][4096];
data_t BRAM_wk_25[4096][4096];
data_t BRAM_wv_25[4096][4096];
data_t BRAM_mm_w1_25[14336][4096];
data_t BRAM_mm_w2_25[14336][4096];
data_t BRAM_mm_w3_25[4096][14336];
data_t BRAM_input_27[2048][4096];
data_t BRAM_rms_norm_w1_26[4096];
data_t BRAM_rms_norm_w2_26[4096];
data_t BRAM_wq_26[4096][4096];
data_t BRAM_wk_26[4096][4096];
data_t BRAM_wv_26[4096][4096];
data_t BRAM_mm_w1_26[14336][4096];
data_t BRAM_mm_w2_26[14336][4096];
data_t BRAM_mm_w3_26[4096][14336];
data_t BRAM_input_28[2048][4096];
data_t BRAM_rms_norm_w1_27[4096];
data_t BRAM_rms_norm_w2_27[4096];
data_t BRAM_wq_27[4096][4096];
data_t BRAM_wk_27[4096][4096];
data_t BRAM_wv_27[4096][4096];
data_t BRAM_mm_w1_27[14336][4096];
data_t BRAM_mm_w2_27[14336][4096];
data_t BRAM_mm_w3_27[4096][14336];
data_t BRAM_input_29[2048][4096];
data_t BRAM_rms_norm_w1_28[4096];
data_t BRAM_rms_norm_w2_28[4096];
data_t BRAM_wq_28[4096][4096];
data_t BRAM_wk_28[4096][4096];
data_t BRAM_wv_28[4096][4096];
data_t BRAM_mm_w1_28[14336][4096];
data_t BRAM_mm_w2_28[14336][4096];
data_t BRAM_mm_w3_28[4096][14336];
data_t BRAM_input_30[2048][4096];
data_t BRAM_rms_norm_w1_29[4096];
data_t BRAM_rms_norm_w2_29[4096];
data_t BRAM_wq_29[4096][4096];
data_t BRAM_wk_29[4096][4096];
data_t BRAM_wv_29[4096][4096];
data_t BRAM_mm_w1_29[14336][4096];
data_t BRAM_mm_w2_29[14336][4096];
data_t BRAM_mm_w3_29[4096][14336];
data_t BRAM_input_31[2048][4096];
data_t BRAM_rms_norm_w1_30[4096];
data_t BRAM_rms_norm_w2_30[4096];
data_t BRAM_wq_30[4096][4096];
data_t BRAM_wk_30[4096][4096];
data_t BRAM_wv_30[4096][4096];
data_t BRAM_mm_w1_30[14336][4096];
data_t BRAM_mm_w2_30[14336][4096];
data_t BRAM_mm_w3_30[4096][14336];
data_t BRAM_input_32[2048][4096];
data_t BRAM_rms_norm_w1_31[4096];
data_t BRAM_rms_norm_w2_31[4096];
data_t BRAM_wq_31[4096][4096];
data_t BRAM_wk_31[4096][4096];
data_t BRAM_wv_31[4096][4096];
data_t BRAM_mm_w1_31[14336][4096];
data_t BRAM_mm_w2_31[14336][4096];
data_t BRAM_mm_w3_31[4096][14336];

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

void top(data_t DRAM_input[2048][4096], data_t DRAM_rms_norm_w1_0[4096], data_t DRAM_rms_norm_w2_0[4096], data_t DRAM_wq_0[4096][4096], data_t DRAM_wk_0[4096][4096], data_t DRAM_wv_0[4096][4096], data_t DRAM_mm_w1_0[14336][4096], data_t DRAM_mm_w2_0[14336][4096], data_t DRAM_mm_w3_0[4096][14336], data_t DRAM_rms_norm_w1_1[4096], data_t DRAM_rms_norm_w2_1[4096], data_t DRAM_wq_1[4096][4096], data_t DRAM_wk_1[4096][4096], data_t DRAM_wv_1[4096][4096], data_t DRAM_mm_w1_1[14336][4096], data_t DRAM_mm_w2_1[14336][4096], data_t DRAM_mm_w3_1[4096][14336], data_t DRAM_rms_norm_w1_2[4096], data_t DRAM_rms_norm_w2_2[4096], data_t DRAM_wq_2[4096][4096], data_t DRAM_wk_2[4096][4096], data_t DRAM_wv_2[4096][4096], data_t DRAM_mm_w1_2[14336][4096], data_t DRAM_mm_w2_2[14336][4096], data_t DRAM_mm_w3_2[4096][14336], data_t DRAM_rms_norm_w1_3[4096], data_t DRAM_rms_norm_w2_3[4096], data_t DRAM_wq_3[4096][4096], data_t DRAM_wk_3[4096][4096], data_t DRAM_wv_3[4096][4096], data_t DRAM_mm_w1_3[14336][4096], data_t DRAM_mm_w2_3[14336][4096], data_t DRAM_mm_w3_3[4096][14336], data_t DRAM_rms_norm_w1_4[4096], data_t DRAM_rms_norm_w2_4[4096], data_t DRAM_wq_4[4096][4096], data_t DRAM_wk_4[4096][4096], data_t DRAM_wv_4[4096][4096], data_t DRAM_mm_w1_4[14336][4096], data_t DRAM_mm_w2_4[14336][4096], data_t DRAM_mm_w3_4[4096][14336], data_t DRAM_rms_norm_w1_5[4096], data_t DRAM_rms_norm_w2_5[4096], data_t DRAM_wq_5[4096][4096], data_t DRAM_wk_5[4096][4096], data_t DRAM_wv_5[4096][4096], data_t DRAM_mm_w1_5[14336][4096], data_t DRAM_mm_w2_5[14336][4096], data_t DRAM_mm_w3_5[4096][14336], data_t DRAM_rms_norm_w1_6[4096], data_t DRAM_rms_norm_w2_6[4096], data_t DRAM_wq_6[4096][4096], data_t DRAM_wk_6[4096][4096], data_t DRAM_wv_6[4096][4096], data_t DRAM_mm_w1_6[14336][4096], data_t DRAM_mm_w2_6[14336][4096], data_t DRAM_mm_w3_6[4096][14336], data_t DRAM_rms_norm_w1_7[4096], data_t DRAM_rms_norm_w2_7[4096], data_t DRAM_wq_7[4096][4096], data_t DRAM_wk_7[4096][4096], data_t DRAM_wv_7[4096][4096], data_t DRAM_mm_w1_7[14336][4096], data_t DRAM_mm_w2_7[14336][4096], data_t DRAM_mm_w3_7[4096][14336], data_t DRAM_rms_norm_w1_8[4096], data_t DRAM_rms_norm_w2_8[4096], data_t DRAM_wq_8[4096][4096], data_t DRAM_wk_8[4096][4096], data_t DRAM_wv_8[4096][4096], data_t DRAM_mm_w1_8[14336][4096], data_t DRAM_mm_w2_8[14336][4096], data_t DRAM_mm_w3_8[4096][14336], data_t DRAM_rms_norm_w1_9[4096], data_t DRAM_rms_norm_w2_9[4096], data_t DRAM_wq_9[4096][4096], data_t DRAM_wk_9[4096][4096], data_t DRAM_wv_9[4096][4096], data_t DRAM_mm_w1_9[14336][4096], data_t DRAM_mm_w2_9[14336][4096], data_t DRAM_mm_w3_9[4096][14336], data_t DRAM_rms_norm_w1_10[4096], data_t DRAM_rms_norm_w2_10[4096], data_t DRAM_wq_10[4096][4096], data_t DRAM_wk_10[4096][4096], data_t DRAM_wv_10[4096][4096], data_t DRAM_mm_w1_10[14336][4096], data_t DRAM_mm_w2_10[14336][4096], data_t DRAM_mm_w3_10[4096][14336], data_t DRAM_rms_norm_w1_11[4096], data_t DRAM_rms_norm_w2_11[4096], data_t DRAM_wq_11[4096][4096], data_t DRAM_wk_11[4096][4096], data_t DRAM_wv_11[4096][4096], data_t DRAM_mm_w1_11[14336][4096], data_t DRAM_mm_w2_11[14336][4096], data_t DRAM_mm_w3_11[4096][14336], data_t DRAM_rms_norm_w1_12[4096], data_t DRAM_rms_norm_w2_12[4096], data_t DRAM_wq_12[4096][4096], data_t DRAM_wk_12[4096][4096], data_t DRAM_wv_12[4096][4096], data_t DRAM_mm_w1_12[14336][4096], data_t DRAM_mm_w2_12[14336][4096], data_t DRAM_mm_w3_12[4096][14336], data_t DRAM_rms_norm_w1_13[4096], data_t DRAM_rms_norm_w2_13[4096], data_t DRAM_wq_13[4096][4096], data_t DRAM_wk_13[4096][4096], data_t DRAM_wv_13[4096][4096], data_t DRAM_mm_w1_13[14336][4096], data_t DRAM_mm_w2_13[14336][4096], data_t DRAM_mm_w3_13[4096][14336], data_t DRAM_rms_norm_w1_14[4096], data_t DRAM_rms_norm_w2_14[4096], data_t DRAM_wq_14[4096][4096], data_t DRAM_wk_14[4096][4096], data_t DRAM_wv_14[4096][4096], data_t DRAM_mm_w1_14[14336][4096], data_t DRAM_mm_w2_14[14336][4096], data_t DRAM_mm_w3_14[4096][14336], data_t DRAM_rms_norm_w1_15[4096], data_t DRAM_rms_norm_w2_15[4096], data_t DRAM_wq_15[4096][4096], data_t DRAM_wk_15[4096][4096], data_t DRAM_wv_15[4096][4096], data_t DRAM_mm_w1_15[14336][4096], data_t DRAM_mm_w2_15[14336][4096], data_t DRAM_mm_w3_15[4096][14336], data_t DRAM_rms_norm_w1_16[4096], data_t DRAM_rms_norm_w2_16[4096], data_t DRAM_wq_16[4096][4096], data_t DRAM_wk_16[4096][4096], data_t DRAM_wv_16[4096][4096], data_t DRAM_mm_w1_16[14336][4096], data_t DRAM_mm_w2_16[14336][4096], data_t DRAM_mm_w3_16[4096][14336], data_t DRAM_rms_norm_w1_17[4096], data_t DRAM_rms_norm_w2_17[4096], data_t DRAM_wq_17[4096][4096], data_t DRAM_wk_17[4096][4096], data_t DRAM_wv_17[4096][4096], data_t DRAM_mm_w1_17[14336][4096], data_t DRAM_mm_w2_17[14336][4096], data_t DRAM_mm_w3_17[4096][14336], data_t DRAM_rms_norm_w1_18[4096], data_t DRAM_rms_norm_w2_18[4096], data_t DRAM_wq_18[4096][4096], data_t DRAM_wk_18[4096][4096], data_t DRAM_wv_18[4096][4096], data_t DRAM_mm_w1_18[14336][4096], data_t DRAM_mm_w2_18[14336][4096], data_t DRAM_mm_w3_18[4096][14336], data_t DRAM_rms_norm_w1_19[4096], data_t DRAM_rms_norm_w2_19[4096], data_t DRAM_wq_19[4096][4096], data_t DRAM_wk_19[4096][4096], data_t DRAM_wv_19[4096][4096], data_t DRAM_mm_w1_19[14336][4096], data_t DRAM_mm_w2_19[14336][4096], data_t DRAM_mm_w3_19[4096][14336], data_t DRAM_rms_norm_w1_20[4096], data_t DRAM_rms_norm_w2_20[4096], data_t DRAM_wq_20[4096][4096], data_t DRAM_wk_20[4096][4096], data_t DRAM_wv_20[4096][4096], data_t DRAM_mm_w1_20[14336][4096], data_t DRAM_mm_w2_20[14336][4096], data_t DRAM_mm_w3_20[4096][14336], data_t DRAM_rms_norm_w1_21[4096], data_t DRAM_rms_norm_w2_21[4096], data_t DRAM_wq_21[4096][4096], data_t DRAM_wk_21[4096][4096], data_t DRAM_wv_21[4096][4096], data_t DRAM_mm_w1_21[14336][4096], data_t DRAM_mm_w2_21[14336][4096], data_t DRAM_mm_w3_21[4096][14336], data_t DRAM_rms_norm_w1_22[4096], data_t DRAM_rms_norm_w2_22[4096], data_t DRAM_wq_22[4096][4096], data_t DRAM_wk_22[4096][4096], data_t DRAM_wv_22[4096][4096], data_t DRAM_mm_w1_22[14336][4096], data_t DRAM_mm_w2_22[14336][4096], data_t DRAM_mm_w3_22[4096][14336], data_t DRAM_rms_norm_w1_23[4096], data_t DRAM_rms_norm_w2_23[4096], data_t DRAM_wq_23[4096][4096], data_t DRAM_wk_23[4096][4096], data_t DRAM_wv_23[4096][4096], data_t DRAM_mm_w1_23[14336][4096], data_t DRAM_mm_w2_23[14336][4096], data_t DRAM_mm_w3_23[4096][14336], data_t DRAM_rms_norm_w1_24[4096], data_t DRAM_rms_norm_w2_24[4096], data_t DRAM_wq_24[4096][4096], data_t DRAM_wk_24[4096][4096], data_t DRAM_wv_24[4096][4096], data_t DRAM_mm_w1_24[14336][4096], data_t DRAM_mm_w2_24[14336][4096], data_t DRAM_mm_w3_24[4096][14336], data_t DRAM_rms_norm_w1_25[4096], data_t DRAM_rms_norm_w2_25[4096], data_t DRAM_wq_25[4096][4096], data_t DRAM_wk_25[4096][4096], data_t DRAM_wv_25[4096][4096], data_t DRAM_mm_w1_25[14336][4096], data_t DRAM_mm_w2_25[14336][4096], data_t DRAM_mm_w3_25[4096][14336], data_t DRAM_rms_norm_w1_26[4096], data_t DRAM_rms_norm_w2_26[4096], data_t DRAM_wq_26[4096][4096], data_t DRAM_wk_26[4096][4096], data_t DRAM_wv_26[4096][4096], data_t DRAM_mm_w1_26[14336][4096], data_t DRAM_mm_w2_26[14336][4096], data_t DRAM_mm_w3_26[4096][14336], data_t DRAM_rms_norm_w1_27[4096], data_t DRAM_rms_norm_w2_27[4096], data_t DRAM_wq_27[4096][4096], data_t DRAM_wk_27[4096][4096], data_t DRAM_wv_27[4096][4096], data_t DRAM_mm_w1_27[14336][4096], data_t DRAM_mm_w2_27[14336][4096], data_t DRAM_mm_w3_27[4096][14336], data_t DRAM_rms_norm_w1_28[4096], data_t DRAM_rms_norm_w2_28[4096], data_t DRAM_wq_28[4096][4096], data_t DRAM_wk_28[4096][4096], data_t DRAM_wv_28[4096][4096], data_t DRAM_mm_w1_28[14336][4096], data_t DRAM_mm_w2_28[14336][4096], data_t DRAM_mm_w3_28[4096][14336], data_t DRAM_rms_norm_w1_29[4096], data_t DRAM_rms_norm_w2_29[4096], data_t DRAM_wq_29[4096][4096], data_t DRAM_wk_29[4096][4096], data_t DRAM_wv_29[4096][4096], data_t DRAM_mm_w1_29[14336][4096], data_t DRAM_mm_w2_29[14336][4096], data_t DRAM_mm_w3_29[4096][14336], data_t DRAM_rms_norm_w1_30[4096], data_t DRAM_rms_norm_w2_30[4096], data_t DRAM_wq_30[4096][4096], data_t DRAM_wk_30[4096][4096], data_t DRAM_wv_30[4096][4096], data_t DRAM_mm_w1_30[14336][4096], data_t DRAM_mm_w2_30[14336][4096], data_t DRAM_mm_w3_30[4096][14336], data_t DRAM_rms_norm_w1_31[4096], data_t DRAM_rms_norm_w2_31[4096], data_t DRAM_wq_31[4096][4096], data_t DRAM_wk_31[4096][4096], data_t DRAM_wv_31[4096][4096], data_t DRAM_mm_w1_31[14336][4096], data_t DRAM_mm_w2_31[14336][4096], data_t DRAM_mm_w3_31[4096][14336], data_t DRAM_output[2048][4096])
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
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_4 offset=slave bundle=mem_rms_norm_w1_4
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_4 offset=slave bundle=mem_rms_norm_w2_4
    #pragma HLS interface m_axi port=DRAM_wq_4 offset=slave bundle=mem_wq_4
    #pragma HLS interface m_axi port=DRAM_wk_4 offset=slave bundle=mem_wk_4
    #pragma HLS interface m_axi port=DRAM_wv_4 offset=slave bundle=mem_wv_4
    #pragma HLS interface m_axi port=DRAM_mm_w1_4 offset=slave bundle=mem_mm_w1_4
    #pragma HLS interface m_axi port=DRAM_mm_w2_4 offset=slave bundle=mem_mm_w2_4
    #pragma HLS interface m_axi port=DRAM_mm_w3_4 offset=slave bundle=mem_mm_w3_4
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_5 offset=slave bundle=mem_rms_norm_w1_5
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_5 offset=slave bundle=mem_rms_norm_w2_5
    #pragma HLS interface m_axi port=DRAM_wq_5 offset=slave bundle=mem_wq_5
    #pragma HLS interface m_axi port=DRAM_wk_5 offset=slave bundle=mem_wk_5
    #pragma HLS interface m_axi port=DRAM_wv_5 offset=slave bundle=mem_wv_5
    #pragma HLS interface m_axi port=DRAM_mm_w1_5 offset=slave bundle=mem_mm_w1_5
    #pragma HLS interface m_axi port=DRAM_mm_w2_5 offset=slave bundle=mem_mm_w2_5
    #pragma HLS interface m_axi port=DRAM_mm_w3_5 offset=slave bundle=mem_mm_w3_5
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_6 offset=slave bundle=mem_rms_norm_w1_6
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_6 offset=slave bundle=mem_rms_norm_w2_6
    #pragma HLS interface m_axi port=DRAM_wq_6 offset=slave bundle=mem_wq_6
    #pragma HLS interface m_axi port=DRAM_wk_6 offset=slave bundle=mem_wk_6
    #pragma HLS interface m_axi port=DRAM_wv_6 offset=slave bundle=mem_wv_6
    #pragma HLS interface m_axi port=DRAM_mm_w1_6 offset=slave bundle=mem_mm_w1_6
    #pragma HLS interface m_axi port=DRAM_mm_w2_6 offset=slave bundle=mem_mm_w2_6
    #pragma HLS interface m_axi port=DRAM_mm_w3_6 offset=slave bundle=mem_mm_w3_6
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_7 offset=slave bundle=mem_rms_norm_w1_7
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_7 offset=slave bundle=mem_rms_norm_w2_7
    #pragma HLS interface m_axi port=DRAM_wq_7 offset=slave bundle=mem_wq_7
    #pragma HLS interface m_axi port=DRAM_wk_7 offset=slave bundle=mem_wk_7
    #pragma HLS interface m_axi port=DRAM_wv_7 offset=slave bundle=mem_wv_7
    #pragma HLS interface m_axi port=DRAM_mm_w1_7 offset=slave bundle=mem_mm_w1_7
    #pragma HLS interface m_axi port=DRAM_mm_w2_7 offset=slave bundle=mem_mm_w2_7
    #pragma HLS interface m_axi port=DRAM_mm_w3_7 offset=slave bundle=mem_mm_w3_7
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_8 offset=slave bundle=mem_rms_norm_w1_8
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_8 offset=slave bundle=mem_rms_norm_w2_8
    #pragma HLS interface m_axi port=DRAM_wq_8 offset=slave bundle=mem_wq_8
    #pragma HLS interface m_axi port=DRAM_wk_8 offset=slave bundle=mem_wk_8
    #pragma HLS interface m_axi port=DRAM_wv_8 offset=slave bundle=mem_wv_8
    #pragma HLS interface m_axi port=DRAM_mm_w1_8 offset=slave bundle=mem_mm_w1_8
    #pragma HLS interface m_axi port=DRAM_mm_w2_8 offset=slave bundle=mem_mm_w2_8
    #pragma HLS interface m_axi port=DRAM_mm_w3_8 offset=slave bundle=mem_mm_w3_8
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_9 offset=slave bundle=mem_rms_norm_w1_9
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_9 offset=slave bundle=mem_rms_norm_w2_9
    #pragma HLS interface m_axi port=DRAM_wq_9 offset=slave bundle=mem_wq_9
    #pragma HLS interface m_axi port=DRAM_wk_9 offset=slave bundle=mem_wk_9
    #pragma HLS interface m_axi port=DRAM_wv_9 offset=slave bundle=mem_wv_9
    #pragma HLS interface m_axi port=DRAM_mm_w1_9 offset=slave bundle=mem_mm_w1_9
    #pragma HLS interface m_axi port=DRAM_mm_w2_9 offset=slave bundle=mem_mm_w2_9
    #pragma HLS interface m_axi port=DRAM_mm_w3_9 offset=slave bundle=mem_mm_w3_9
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_10 offset=slave bundle=mem_rms_norm_w1_10
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_10 offset=slave bundle=mem_rms_norm_w2_10
    #pragma HLS interface m_axi port=DRAM_wq_10 offset=slave bundle=mem_wq_10
    #pragma HLS interface m_axi port=DRAM_wk_10 offset=slave bundle=mem_wk_10
    #pragma HLS interface m_axi port=DRAM_wv_10 offset=slave bundle=mem_wv_10
    #pragma HLS interface m_axi port=DRAM_mm_w1_10 offset=slave bundle=mem_mm_w1_10
    #pragma HLS interface m_axi port=DRAM_mm_w2_10 offset=slave bundle=mem_mm_w2_10
    #pragma HLS interface m_axi port=DRAM_mm_w3_10 offset=slave bundle=mem_mm_w3_10
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_11 offset=slave bundle=mem_rms_norm_w1_11
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_11 offset=slave bundle=mem_rms_norm_w2_11
    #pragma HLS interface m_axi port=DRAM_wq_11 offset=slave bundle=mem_wq_11
    #pragma HLS interface m_axi port=DRAM_wk_11 offset=slave bundle=mem_wk_11
    #pragma HLS interface m_axi port=DRAM_wv_11 offset=slave bundle=mem_wv_11
    #pragma HLS interface m_axi port=DRAM_mm_w1_11 offset=slave bundle=mem_mm_w1_11
    #pragma HLS interface m_axi port=DRAM_mm_w2_11 offset=slave bundle=mem_mm_w2_11
    #pragma HLS interface m_axi port=DRAM_mm_w3_11 offset=slave bundle=mem_mm_w3_11
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_12 offset=slave bundle=mem_rms_norm_w1_12
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_12 offset=slave bundle=mem_rms_norm_w2_12
    #pragma HLS interface m_axi port=DRAM_wq_12 offset=slave bundle=mem_wq_12
    #pragma HLS interface m_axi port=DRAM_wk_12 offset=slave bundle=mem_wk_12
    #pragma HLS interface m_axi port=DRAM_wv_12 offset=slave bundle=mem_wv_12
    #pragma HLS interface m_axi port=DRAM_mm_w1_12 offset=slave bundle=mem_mm_w1_12
    #pragma HLS interface m_axi port=DRAM_mm_w2_12 offset=slave bundle=mem_mm_w2_12
    #pragma HLS interface m_axi port=DRAM_mm_w3_12 offset=slave bundle=mem_mm_w3_12
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_13 offset=slave bundle=mem_rms_norm_w1_13
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_13 offset=slave bundle=mem_rms_norm_w2_13
    #pragma HLS interface m_axi port=DRAM_wq_13 offset=slave bundle=mem_wq_13
    #pragma HLS interface m_axi port=DRAM_wk_13 offset=slave bundle=mem_wk_13
    #pragma HLS interface m_axi port=DRAM_wv_13 offset=slave bundle=mem_wv_13
    #pragma HLS interface m_axi port=DRAM_mm_w1_13 offset=slave bundle=mem_mm_w1_13
    #pragma HLS interface m_axi port=DRAM_mm_w2_13 offset=slave bundle=mem_mm_w2_13
    #pragma HLS interface m_axi port=DRAM_mm_w3_13 offset=slave bundle=mem_mm_w3_13
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_14 offset=slave bundle=mem_rms_norm_w1_14
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_14 offset=slave bundle=mem_rms_norm_w2_14
    #pragma HLS interface m_axi port=DRAM_wq_14 offset=slave bundle=mem_wq_14
    #pragma HLS interface m_axi port=DRAM_wk_14 offset=slave bundle=mem_wk_14
    #pragma HLS interface m_axi port=DRAM_wv_14 offset=slave bundle=mem_wv_14
    #pragma HLS interface m_axi port=DRAM_mm_w1_14 offset=slave bundle=mem_mm_w1_14
    #pragma HLS interface m_axi port=DRAM_mm_w2_14 offset=slave bundle=mem_mm_w2_14
    #pragma HLS interface m_axi port=DRAM_mm_w3_14 offset=slave bundle=mem_mm_w3_14
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_15 offset=slave bundle=mem_rms_norm_w1_15
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_15 offset=slave bundle=mem_rms_norm_w2_15
    #pragma HLS interface m_axi port=DRAM_wq_15 offset=slave bundle=mem_wq_15
    #pragma HLS interface m_axi port=DRAM_wk_15 offset=slave bundle=mem_wk_15
    #pragma HLS interface m_axi port=DRAM_wv_15 offset=slave bundle=mem_wv_15
    #pragma HLS interface m_axi port=DRAM_mm_w1_15 offset=slave bundle=mem_mm_w1_15
    #pragma HLS interface m_axi port=DRAM_mm_w2_15 offset=slave bundle=mem_mm_w2_15
    #pragma HLS interface m_axi port=DRAM_mm_w3_15 offset=slave bundle=mem_mm_w3_15
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_16 offset=slave bundle=mem_rms_norm_w1_16
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_16 offset=slave bundle=mem_rms_norm_w2_16
    #pragma HLS interface m_axi port=DRAM_wq_16 offset=slave bundle=mem_wq_16
    #pragma HLS interface m_axi port=DRAM_wk_16 offset=slave bundle=mem_wk_16
    #pragma HLS interface m_axi port=DRAM_wv_16 offset=slave bundle=mem_wv_16
    #pragma HLS interface m_axi port=DRAM_mm_w1_16 offset=slave bundle=mem_mm_w1_16
    #pragma HLS interface m_axi port=DRAM_mm_w2_16 offset=slave bundle=mem_mm_w2_16
    #pragma HLS interface m_axi port=DRAM_mm_w3_16 offset=slave bundle=mem_mm_w3_16
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_17 offset=slave bundle=mem_rms_norm_w1_17
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_17 offset=slave bundle=mem_rms_norm_w2_17
    #pragma HLS interface m_axi port=DRAM_wq_17 offset=slave bundle=mem_wq_17
    #pragma HLS interface m_axi port=DRAM_wk_17 offset=slave bundle=mem_wk_17
    #pragma HLS interface m_axi port=DRAM_wv_17 offset=slave bundle=mem_wv_17
    #pragma HLS interface m_axi port=DRAM_mm_w1_17 offset=slave bundle=mem_mm_w1_17
    #pragma HLS interface m_axi port=DRAM_mm_w2_17 offset=slave bundle=mem_mm_w2_17
    #pragma HLS interface m_axi port=DRAM_mm_w3_17 offset=slave bundle=mem_mm_w3_17
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_18 offset=slave bundle=mem_rms_norm_w1_18
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_18 offset=slave bundle=mem_rms_norm_w2_18
    #pragma HLS interface m_axi port=DRAM_wq_18 offset=slave bundle=mem_wq_18
    #pragma HLS interface m_axi port=DRAM_wk_18 offset=slave bundle=mem_wk_18
    #pragma HLS interface m_axi port=DRAM_wv_18 offset=slave bundle=mem_wv_18
    #pragma HLS interface m_axi port=DRAM_mm_w1_18 offset=slave bundle=mem_mm_w1_18
    #pragma HLS interface m_axi port=DRAM_mm_w2_18 offset=slave bundle=mem_mm_w2_18
    #pragma HLS interface m_axi port=DRAM_mm_w3_18 offset=slave bundle=mem_mm_w3_18
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_19 offset=slave bundle=mem_rms_norm_w1_19
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_19 offset=slave bundle=mem_rms_norm_w2_19
    #pragma HLS interface m_axi port=DRAM_wq_19 offset=slave bundle=mem_wq_19
    #pragma HLS interface m_axi port=DRAM_wk_19 offset=slave bundle=mem_wk_19
    #pragma HLS interface m_axi port=DRAM_wv_19 offset=slave bundle=mem_wv_19
    #pragma HLS interface m_axi port=DRAM_mm_w1_19 offset=slave bundle=mem_mm_w1_19
    #pragma HLS interface m_axi port=DRAM_mm_w2_19 offset=slave bundle=mem_mm_w2_19
    #pragma HLS interface m_axi port=DRAM_mm_w3_19 offset=slave bundle=mem_mm_w3_19
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_20 offset=slave bundle=mem_rms_norm_w1_20
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_20 offset=slave bundle=mem_rms_norm_w2_20
    #pragma HLS interface m_axi port=DRAM_wq_20 offset=slave bundle=mem_wq_20
    #pragma HLS interface m_axi port=DRAM_wk_20 offset=slave bundle=mem_wk_20
    #pragma HLS interface m_axi port=DRAM_wv_20 offset=slave bundle=mem_wv_20
    #pragma HLS interface m_axi port=DRAM_mm_w1_20 offset=slave bundle=mem_mm_w1_20
    #pragma HLS interface m_axi port=DRAM_mm_w2_20 offset=slave bundle=mem_mm_w2_20
    #pragma HLS interface m_axi port=DRAM_mm_w3_20 offset=slave bundle=mem_mm_w3_20
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_21 offset=slave bundle=mem_rms_norm_w1_21
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_21 offset=slave bundle=mem_rms_norm_w2_21
    #pragma HLS interface m_axi port=DRAM_wq_21 offset=slave bundle=mem_wq_21
    #pragma HLS interface m_axi port=DRAM_wk_21 offset=slave bundle=mem_wk_21
    #pragma HLS interface m_axi port=DRAM_wv_21 offset=slave bundle=mem_wv_21
    #pragma HLS interface m_axi port=DRAM_mm_w1_21 offset=slave bundle=mem_mm_w1_21
    #pragma HLS interface m_axi port=DRAM_mm_w2_21 offset=slave bundle=mem_mm_w2_21
    #pragma HLS interface m_axi port=DRAM_mm_w3_21 offset=slave bundle=mem_mm_w3_21
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_22 offset=slave bundle=mem_rms_norm_w1_22
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_22 offset=slave bundle=mem_rms_norm_w2_22
    #pragma HLS interface m_axi port=DRAM_wq_22 offset=slave bundle=mem_wq_22
    #pragma HLS interface m_axi port=DRAM_wk_22 offset=slave bundle=mem_wk_22
    #pragma HLS interface m_axi port=DRAM_wv_22 offset=slave bundle=mem_wv_22
    #pragma HLS interface m_axi port=DRAM_mm_w1_22 offset=slave bundle=mem_mm_w1_22
    #pragma HLS interface m_axi port=DRAM_mm_w2_22 offset=slave bundle=mem_mm_w2_22
    #pragma HLS interface m_axi port=DRAM_mm_w3_22 offset=slave bundle=mem_mm_w3_22
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_23 offset=slave bundle=mem_rms_norm_w1_23
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_23 offset=slave bundle=mem_rms_norm_w2_23
    #pragma HLS interface m_axi port=DRAM_wq_23 offset=slave bundle=mem_wq_23
    #pragma HLS interface m_axi port=DRAM_wk_23 offset=slave bundle=mem_wk_23
    #pragma HLS interface m_axi port=DRAM_wv_23 offset=slave bundle=mem_wv_23
    #pragma HLS interface m_axi port=DRAM_mm_w1_23 offset=slave bundle=mem_mm_w1_23
    #pragma HLS interface m_axi port=DRAM_mm_w2_23 offset=slave bundle=mem_mm_w2_23
    #pragma HLS interface m_axi port=DRAM_mm_w3_23 offset=slave bundle=mem_mm_w3_23
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_24 offset=slave bundle=mem_rms_norm_w1_24
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_24 offset=slave bundle=mem_rms_norm_w2_24
    #pragma HLS interface m_axi port=DRAM_wq_24 offset=slave bundle=mem_wq_24
    #pragma HLS interface m_axi port=DRAM_wk_24 offset=slave bundle=mem_wk_24
    #pragma HLS interface m_axi port=DRAM_wv_24 offset=slave bundle=mem_wv_24
    #pragma HLS interface m_axi port=DRAM_mm_w1_24 offset=slave bundle=mem_mm_w1_24
    #pragma HLS interface m_axi port=DRAM_mm_w2_24 offset=slave bundle=mem_mm_w2_24
    #pragma HLS interface m_axi port=DRAM_mm_w3_24 offset=slave bundle=mem_mm_w3_24
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_25 offset=slave bundle=mem_rms_norm_w1_25
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_25 offset=slave bundle=mem_rms_norm_w2_25
    #pragma HLS interface m_axi port=DRAM_wq_25 offset=slave bundle=mem_wq_25
    #pragma HLS interface m_axi port=DRAM_wk_25 offset=slave bundle=mem_wk_25
    #pragma HLS interface m_axi port=DRAM_wv_25 offset=slave bundle=mem_wv_25
    #pragma HLS interface m_axi port=DRAM_mm_w1_25 offset=slave bundle=mem_mm_w1_25
    #pragma HLS interface m_axi port=DRAM_mm_w2_25 offset=slave bundle=mem_mm_w2_25
    #pragma HLS interface m_axi port=DRAM_mm_w3_25 offset=slave bundle=mem_mm_w3_25
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_26 offset=slave bundle=mem_rms_norm_w1_26
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_26 offset=slave bundle=mem_rms_norm_w2_26
    #pragma HLS interface m_axi port=DRAM_wq_26 offset=slave bundle=mem_wq_26
    #pragma HLS interface m_axi port=DRAM_wk_26 offset=slave bundle=mem_wk_26
    #pragma HLS interface m_axi port=DRAM_wv_26 offset=slave bundle=mem_wv_26
    #pragma HLS interface m_axi port=DRAM_mm_w1_26 offset=slave bundle=mem_mm_w1_26
    #pragma HLS interface m_axi port=DRAM_mm_w2_26 offset=slave bundle=mem_mm_w2_26
    #pragma HLS interface m_axi port=DRAM_mm_w3_26 offset=slave bundle=mem_mm_w3_26
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_27 offset=slave bundle=mem_rms_norm_w1_27
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_27 offset=slave bundle=mem_rms_norm_w2_27
    #pragma HLS interface m_axi port=DRAM_wq_27 offset=slave bundle=mem_wq_27
    #pragma HLS interface m_axi port=DRAM_wk_27 offset=slave bundle=mem_wk_27
    #pragma HLS interface m_axi port=DRAM_wv_27 offset=slave bundle=mem_wv_27
    #pragma HLS interface m_axi port=DRAM_mm_w1_27 offset=slave bundle=mem_mm_w1_27
    #pragma HLS interface m_axi port=DRAM_mm_w2_27 offset=slave bundle=mem_mm_w2_27
    #pragma HLS interface m_axi port=DRAM_mm_w3_27 offset=slave bundle=mem_mm_w3_27
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_28 offset=slave bundle=mem_rms_norm_w1_28
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_28 offset=slave bundle=mem_rms_norm_w2_28
    #pragma HLS interface m_axi port=DRAM_wq_28 offset=slave bundle=mem_wq_28
    #pragma HLS interface m_axi port=DRAM_wk_28 offset=slave bundle=mem_wk_28
    #pragma HLS interface m_axi port=DRAM_wv_28 offset=slave bundle=mem_wv_28
    #pragma HLS interface m_axi port=DRAM_mm_w1_28 offset=slave bundle=mem_mm_w1_28
    #pragma HLS interface m_axi port=DRAM_mm_w2_28 offset=slave bundle=mem_mm_w2_28
    #pragma HLS interface m_axi port=DRAM_mm_w3_28 offset=slave bundle=mem_mm_w3_28
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_29 offset=slave bundle=mem_rms_norm_w1_29
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_29 offset=slave bundle=mem_rms_norm_w2_29
    #pragma HLS interface m_axi port=DRAM_wq_29 offset=slave bundle=mem_wq_29
    #pragma HLS interface m_axi port=DRAM_wk_29 offset=slave bundle=mem_wk_29
    #pragma HLS interface m_axi port=DRAM_wv_29 offset=slave bundle=mem_wv_29
    #pragma HLS interface m_axi port=DRAM_mm_w1_29 offset=slave bundle=mem_mm_w1_29
    #pragma HLS interface m_axi port=DRAM_mm_w2_29 offset=slave bundle=mem_mm_w2_29
    #pragma HLS interface m_axi port=DRAM_mm_w3_29 offset=slave bundle=mem_mm_w3_29
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_30 offset=slave bundle=mem_rms_norm_w1_30
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_30 offset=slave bundle=mem_rms_norm_w2_30
    #pragma HLS interface m_axi port=DRAM_wq_30 offset=slave bundle=mem_wq_30
    #pragma HLS interface m_axi port=DRAM_wk_30 offset=slave bundle=mem_wk_30
    #pragma HLS interface m_axi port=DRAM_wv_30 offset=slave bundle=mem_wv_30
    #pragma HLS interface m_axi port=DRAM_mm_w1_30 offset=slave bundle=mem_mm_w1_30
    #pragma HLS interface m_axi port=DRAM_mm_w2_30 offset=slave bundle=mem_mm_w2_30
    #pragma HLS interface m_axi port=DRAM_mm_w3_30 offset=slave bundle=mem_mm_w3_30
    #pragma HLS interface m_axi port=DRAM_rms_norm_w1_31 offset=slave bundle=mem_rms_norm_w1_31
    #pragma HLS interface m_axi port=DRAM_rms_norm_w2_31 offset=slave bundle=mem_rms_norm_w2_31
    #pragma HLS interface m_axi port=DRAM_wq_31 offset=slave bundle=mem_wq_31
    #pragma HLS interface m_axi port=DRAM_wk_31 offset=slave bundle=mem_wk_31
    #pragma HLS interface m_axi port=DRAM_wv_31 offset=slave bundle=mem_wv_31
    #pragma HLS interface m_axi port=DRAM_mm_w1_31 offset=slave bundle=mem_mm_w1_31
    #pragma HLS interface m_axi port=DRAM_mm_w2_31 offset=slave bundle=mem_mm_w2_31
    #pragma HLS interface m_axi port=DRAM_mm_w3_31 offset=slave bundle=mem_mm_w3_31
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
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_4, BRAM_rms_norm_w1_4);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_4, BRAM_rms_norm_w2_4);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_4, BRAM_wq_4);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_4, BRAM_wk_4);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_4, BRAM_wv_4);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_4, BRAM_mm_w1_4);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_4, BRAM_mm_w2_4);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_4, BRAM_mm_w3_4);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_5, BRAM_rms_norm_w1_5);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_5, BRAM_rms_norm_w2_5);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_5, BRAM_wq_5);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_5, BRAM_wk_5);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_5, BRAM_wv_5);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_5, BRAM_mm_w1_5);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_5, BRAM_mm_w2_5);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_5, BRAM_mm_w3_5);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_6, BRAM_rms_norm_w1_6);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_6, BRAM_rms_norm_w2_6);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_6, BRAM_wq_6);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_6, BRAM_wk_6);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_6, BRAM_wv_6);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_6, BRAM_mm_w1_6);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_6, BRAM_mm_w2_6);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_6, BRAM_mm_w3_6);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_7, BRAM_rms_norm_w1_7);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_7, BRAM_rms_norm_w2_7);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_7, BRAM_wq_7);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_7, BRAM_wk_7);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_7, BRAM_wv_7);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_7, BRAM_mm_w1_7);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_7, BRAM_mm_w2_7);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_7, BRAM_mm_w3_7);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_8, BRAM_rms_norm_w1_8);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_8, BRAM_rms_norm_w2_8);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_8, BRAM_wq_8);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_8, BRAM_wk_8);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_8, BRAM_wv_8);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_8, BRAM_mm_w1_8);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_8, BRAM_mm_w2_8);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_8, BRAM_mm_w3_8);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_9, BRAM_rms_norm_w1_9);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_9, BRAM_rms_norm_w2_9);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_9, BRAM_wq_9);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_9, BRAM_wk_9);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_9, BRAM_wv_9);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_9, BRAM_mm_w1_9);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_9, BRAM_mm_w2_9);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_9, BRAM_mm_w3_9);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_10, BRAM_rms_norm_w1_10);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_10, BRAM_rms_norm_w2_10);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_10, BRAM_wq_10);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_10, BRAM_wk_10);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_10, BRAM_wv_10);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_10, BRAM_mm_w1_10);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_10, BRAM_mm_w2_10);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_10, BRAM_mm_w3_10);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_11, BRAM_rms_norm_w1_11);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_11, BRAM_rms_norm_w2_11);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_11, BRAM_wq_11);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_11, BRAM_wk_11);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_11, BRAM_wv_11);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_11, BRAM_mm_w1_11);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_11, BRAM_mm_w2_11);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_11, BRAM_mm_w3_11);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_12, BRAM_rms_norm_w1_12);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_12, BRAM_rms_norm_w2_12);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_12, BRAM_wq_12);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_12, BRAM_wk_12);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_12, BRAM_wv_12);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_12, BRAM_mm_w1_12);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_12, BRAM_mm_w2_12);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_12, BRAM_mm_w3_12);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_13, BRAM_rms_norm_w1_13);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_13, BRAM_rms_norm_w2_13);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_13, BRAM_wq_13);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_13, BRAM_wk_13);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_13, BRAM_wv_13);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_13, BRAM_mm_w1_13);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_13, BRAM_mm_w2_13);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_13, BRAM_mm_w3_13);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_14, BRAM_rms_norm_w1_14);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_14, BRAM_rms_norm_w2_14);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_14, BRAM_wq_14);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_14, BRAM_wk_14);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_14, BRAM_wv_14);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_14, BRAM_mm_w1_14);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_14, BRAM_mm_w2_14);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_14, BRAM_mm_w3_14);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_15, BRAM_rms_norm_w1_15);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_15, BRAM_rms_norm_w2_15);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_15, BRAM_wq_15);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_15, BRAM_wk_15);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_15, BRAM_wv_15);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_15, BRAM_mm_w1_15);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_15, BRAM_mm_w2_15);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_15, BRAM_mm_w3_15);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_16, BRAM_rms_norm_w1_16);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_16, BRAM_rms_norm_w2_16);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_16, BRAM_wq_16);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_16, BRAM_wk_16);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_16, BRAM_wv_16);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_16, BRAM_mm_w1_16);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_16, BRAM_mm_w2_16);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_16, BRAM_mm_w3_16);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_17, BRAM_rms_norm_w1_17);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_17, BRAM_rms_norm_w2_17);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_17, BRAM_wq_17);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_17, BRAM_wk_17);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_17, BRAM_wv_17);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_17, BRAM_mm_w1_17);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_17, BRAM_mm_w2_17);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_17, BRAM_mm_w3_17);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_18, BRAM_rms_norm_w1_18);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_18, BRAM_rms_norm_w2_18);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_18, BRAM_wq_18);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_18, BRAM_wk_18);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_18, BRAM_wv_18);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_18, BRAM_mm_w1_18);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_18, BRAM_mm_w2_18);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_18, BRAM_mm_w3_18);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_19, BRAM_rms_norm_w1_19);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_19, BRAM_rms_norm_w2_19);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_19, BRAM_wq_19);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_19, BRAM_wk_19);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_19, BRAM_wv_19);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_19, BRAM_mm_w1_19);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_19, BRAM_mm_w2_19);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_19, BRAM_mm_w3_19);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_20, BRAM_rms_norm_w1_20);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_20, BRAM_rms_norm_w2_20);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_20, BRAM_wq_20);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_20, BRAM_wk_20);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_20, BRAM_wv_20);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_20, BRAM_mm_w1_20);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_20, BRAM_mm_w2_20);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_20, BRAM_mm_w3_20);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_21, BRAM_rms_norm_w1_21);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_21, BRAM_rms_norm_w2_21);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_21, BRAM_wq_21);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_21, BRAM_wk_21);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_21, BRAM_wv_21);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_21, BRAM_mm_w1_21);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_21, BRAM_mm_w2_21);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_21, BRAM_mm_w3_21);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_22, BRAM_rms_norm_w1_22);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_22, BRAM_rms_norm_w2_22);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_22, BRAM_wq_22);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_22, BRAM_wk_22);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_22, BRAM_wv_22);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_22, BRAM_mm_w1_22);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_22, BRAM_mm_w2_22);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_22, BRAM_mm_w3_22);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_23, BRAM_rms_norm_w1_23);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_23, BRAM_rms_norm_w2_23);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_23, BRAM_wq_23);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_23, BRAM_wk_23);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_23, BRAM_wv_23);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_23, BRAM_mm_w1_23);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_23, BRAM_mm_w2_23);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_23, BRAM_mm_w3_23);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_24, BRAM_rms_norm_w1_24);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_24, BRAM_rms_norm_w2_24);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_24, BRAM_wq_24);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_24, BRAM_wk_24);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_24, BRAM_wv_24);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_24, BRAM_mm_w1_24);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_24, BRAM_mm_w2_24);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_24, BRAM_mm_w3_24);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_25, BRAM_rms_norm_w1_25);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_25, BRAM_rms_norm_w2_25);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_25, BRAM_wq_25);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_25, BRAM_wk_25);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_25, BRAM_wv_25);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_25, BRAM_mm_w1_25);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_25, BRAM_mm_w2_25);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_25, BRAM_mm_w3_25);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_26, BRAM_rms_norm_w1_26);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_26, BRAM_rms_norm_w2_26);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_26, BRAM_wq_26);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_26, BRAM_wk_26);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_26, BRAM_wv_26);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_26, BRAM_mm_w1_26);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_26, BRAM_mm_w2_26);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_26, BRAM_mm_w3_26);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_27, BRAM_rms_norm_w1_27);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_27, BRAM_rms_norm_w2_27);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_27, BRAM_wq_27);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_27, BRAM_wk_27);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_27, BRAM_wv_27);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_27, BRAM_mm_w1_27);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_27, BRAM_mm_w2_27);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_27, BRAM_mm_w3_27);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_28, BRAM_rms_norm_w1_28);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_28, BRAM_rms_norm_w2_28);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_28, BRAM_wq_28);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_28, BRAM_wk_28);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_28, BRAM_wv_28);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_28, BRAM_mm_w1_28);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_28, BRAM_mm_w2_28);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_28, BRAM_mm_w3_28);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_29, BRAM_rms_norm_w1_29);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_29, BRAM_rms_norm_w2_29);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_29, BRAM_wq_29);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_29, BRAM_wk_29);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_29, BRAM_wv_29);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_29, BRAM_mm_w1_29);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_29, BRAM_mm_w2_29);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_29, BRAM_mm_w3_29);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_30, BRAM_rms_norm_w1_30);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_30, BRAM_rms_norm_w2_30);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_30, BRAM_wq_30);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_30, BRAM_wk_30);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_30, BRAM_wv_30);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_30, BRAM_mm_w1_30);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_30, BRAM_mm_w2_30);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_30, BRAM_mm_w3_30);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w1_31, BRAM_rms_norm_w1_31);
    load_4096_ap_fixed_16_5_(DRAM_rms_norm_w2_31, BRAM_rms_norm_w2_31);
    load_4096_4096_ap_fixed_16_5_(DRAM_wq_31, BRAM_wq_31);
    load_4096_4096_ap_fixed_16_5_(DRAM_wk_31, BRAM_wk_31);
    load_4096_4096_ap_fixed_16_5_(DRAM_wv_31, BRAM_wv_31);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w1_31, BRAM_mm_w1_31);
    load_14336_4096_ap_fixed_16_5_(DRAM_mm_w2_31, BRAM_mm_w2_31);
    load_4096_14336_ap_fixed_16_5_(DRAM_mm_w3_31, BRAM_mm_w3_31);
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
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_4, BRAM_rms_norm_w1_4, BRAM_input_4);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_4, BRAM_wq_4, BRAM_wk_4, BRAM_wv_4, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_4, BRAM_input_4);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_4, BRAM_rms_norm_w2_4, BRAM_input_4);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_4, BRAM_mm_w1_4, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_4, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_4, BRAM_input_5);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_5, BRAM_rms_norm_w1_5, BRAM_input_5);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_5, BRAM_wq_5, BRAM_wk_5, BRAM_wv_5, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_5, BRAM_input_5);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_5, BRAM_rms_norm_w2_5, BRAM_input_5);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_5, BRAM_mm_w1_5, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_5, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_5, BRAM_input_6);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_6, BRAM_rms_norm_w1_6, BRAM_input_6);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_6, BRAM_wq_6, BRAM_wk_6, BRAM_wv_6, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_6, BRAM_input_6);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_6, BRAM_rms_norm_w2_6, BRAM_input_6);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_6, BRAM_mm_w1_6, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_6, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_6, BRAM_input_7);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_7, BRAM_rms_norm_w1_7, BRAM_input_7);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_7, BRAM_wq_7, BRAM_wk_7, BRAM_wv_7, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_7, BRAM_input_7);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_7, BRAM_rms_norm_w2_7, BRAM_input_7);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_7, BRAM_mm_w1_7, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_7, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_7, BRAM_input_8);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_8, BRAM_rms_norm_w1_8, BRAM_input_8);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_8, BRAM_wq_8, BRAM_wk_8, BRAM_wv_8, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_8, BRAM_input_8);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_8, BRAM_rms_norm_w2_8, BRAM_input_8);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_8, BRAM_mm_w1_8, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_8, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_8, BRAM_input_9);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_9, BRAM_rms_norm_w1_9, BRAM_input_9);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_9, BRAM_wq_9, BRAM_wk_9, BRAM_wv_9, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_9, BRAM_input_9);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_9, BRAM_rms_norm_w2_9, BRAM_input_9);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_9, BRAM_mm_w1_9, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_9, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_9, BRAM_input_10);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_10, BRAM_rms_norm_w1_10, BRAM_input_10);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_10, BRAM_wq_10, BRAM_wk_10, BRAM_wv_10, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_10, BRAM_input_10);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_10, BRAM_rms_norm_w2_10, BRAM_input_10);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_10, BRAM_mm_w1_10, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_10, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_10, BRAM_input_11);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_11, BRAM_rms_norm_w1_11, BRAM_input_11);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_11, BRAM_wq_11, BRAM_wk_11, BRAM_wv_11, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_11, BRAM_input_11);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_11, BRAM_rms_norm_w2_11, BRAM_input_11);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_11, BRAM_mm_w1_11, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_11, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_11, BRAM_input_12);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_12, BRAM_rms_norm_w1_12, BRAM_input_12);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_12, BRAM_wq_12, BRAM_wk_12, BRAM_wv_12, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_12, BRAM_input_12);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_12, BRAM_rms_norm_w2_12, BRAM_input_12);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_12, BRAM_mm_w1_12, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_12, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_12, BRAM_input_13);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_13, BRAM_rms_norm_w1_13, BRAM_input_13);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_13, BRAM_wq_13, BRAM_wk_13, BRAM_wv_13, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_13, BRAM_input_13);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_13, BRAM_rms_norm_w2_13, BRAM_input_13);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_13, BRAM_mm_w1_13, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_13, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_13, BRAM_input_14);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_14, BRAM_rms_norm_w1_14, BRAM_input_14);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_14, BRAM_wq_14, BRAM_wk_14, BRAM_wv_14, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_14, BRAM_input_14);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_14, BRAM_rms_norm_w2_14, BRAM_input_14);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_14, BRAM_mm_w1_14, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_14, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_14, BRAM_input_15);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_15, BRAM_rms_norm_w1_15, BRAM_input_15);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_15, BRAM_wq_15, BRAM_wk_15, BRAM_wv_15, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_15, BRAM_input_15);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_15, BRAM_rms_norm_w2_15, BRAM_input_15);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_15, BRAM_mm_w1_15, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_15, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_15, BRAM_input_16);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_16, BRAM_rms_norm_w1_16, BRAM_input_16);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_16, BRAM_wq_16, BRAM_wk_16, BRAM_wv_16, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_16, BRAM_input_16);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_16, BRAM_rms_norm_w2_16, BRAM_input_16);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_16, BRAM_mm_w1_16, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_16, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_16, BRAM_input_17);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_17, BRAM_rms_norm_w1_17, BRAM_input_17);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_17, BRAM_wq_17, BRAM_wk_17, BRAM_wv_17, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_17, BRAM_input_17);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_17, BRAM_rms_norm_w2_17, BRAM_input_17);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_17, BRAM_mm_w1_17, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_17, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_17, BRAM_input_18);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_18, BRAM_rms_norm_w1_18, BRAM_input_18);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_18, BRAM_wq_18, BRAM_wk_18, BRAM_wv_18, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_18, BRAM_input_18);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_18, BRAM_rms_norm_w2_18, BRAM_input_18);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_18, BRAM_mm_w1_18, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_18, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_18, BRAM_input_19);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_19, BRAM_rms_norm_w1_19, BRAM_input_19);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_19, BRAM_wq_19, BRAM_wk_19, BRAM_wv_19, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_19, BRAM_input_19);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_19, BRAM_rms_norm_w2_19, BRAM_input_19);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_19, BRAM_mm_w1_19, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_19, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_19, BRAM_input_20);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_20, BRAM_rms_norm_w1_20, BRAM_input_20);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_20, BRAM_wq_20, BRAM_wk_20, BRAM_wv_20, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_20, BRAM_input_20);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_20, BRAM_rms_norm_w2_20, BRAM_input_20);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_20, BRAM_mm_w1_20, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_20, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_20, BRAM_input_21);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_21, BRAM_rms_norm_w1_21, BRAM_input_21);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_21, BRAM_wq_21, BRAM_wk_21, BRAM_wv_21, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_21, BRAM_input_21);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_21, BRAM_rms_norm_w2_21, BRAM_input_21);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_21, BRAM_mm_w1_21, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_21, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_21, BRAM_input_22);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_22, BRAM_rms_norm_w1_22, BRAM_input_22);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_22, BRAM_wq_22, BRAM_wk_22, BRAM_wv_22, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_22, BRAM_input_22);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_22, BRAM_rms_norm_w2_22, BRAM_input_22);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_22, BRAM_mm_w1_22, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_22, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_22, BRAM_input_23);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_23, BRAM_rms_norm_w1_23, BRAM_input_23);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_23, BRAM_wq_23, BRAM_wk_23, BRAM_wv_23, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_23, BRAM_input_23);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_23, BRAM_rms_norm_w2_23, BRAM_input_23);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_23, BRAM_mm_w1_23, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_23, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_23, BRAM_input_24);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_24, BRAM_rms_norm_w1_24, BRAM_input_24);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_24, BRAM_wq_24, BRAM_wk_24, BRAM_wv_24, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_24, BRAM_input_24);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_24, BRAM_rms_norm_w2_24, BRAM_input_24);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_24, BRAM_mm_w1_24, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_24, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_24, BRAM_input_25);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_25, BRAM_rms_norm_w1_25, BRAM_input_25);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_25, BRAM_wq_25, BRAM_wk_25, BRAM_wv_25, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_25, BRAM_input_25);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_25, BRAM_rms_norm_w2_25, BRAM_input_25);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_25, BRAM_mm_w1_25, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_25, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_25, BRAM_input_26);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_26, BRAM_rms_norm_w1_26, BRAM_input_26);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_26, BRAM_wq_26, BRAM_wk_26, BRAM_wv_26, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_26, BRAM_input_26);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_26, BRAM_rms_norm_w2_26, BRAM_input_26);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_26, BRAM_mm_w1_26, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_26, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_26, BRAM_input_27);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_27, BRAM_rms_norm_w1_27, BRAM_input_27);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_27, BRAM_wq_27, BRAM_wk_27, BRAM_wv_27, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_27, BRAM_input_27);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_27, BRAM_rms_norm_w2_27, BRAM_input_27);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_27, BRAM_mm_w1_27, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_27, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_27, BRAM_input_28);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_28, BRAM_rms_norm_w1_28, BRAM_input_28);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_28, BRAM_wq_28, BRAM_wk_28, BRAM_wv_28, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_28, BRAM_input_28);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_28, BRAM_rms_norm_w2_28, BRAM_input_28);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_28, BRAM_mm_w1_28, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_28, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_28, BRAM_input_29);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_29, BRAM_rms_norm_w1_29, BRAM_input_29);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_29, BRAM_wq_29, BRAM_wk_29, BRAM_wv_29, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_29, BRAM_input_29);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_29, BRAM_rms_norm_w2_29, BRAM_input_29);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_29, BRAM_mm_w1_29, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_29, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_29, BRAM_input_30);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_30, BRAM_rms_norm_w1_30, BRAM_input_30);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_30, BRAM_wq_30, BRAM_wk_30, BRAM_wv_30, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_30, BRAM_input_30);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_30, BRAM_rms_norm_w2_30, BRAM_input_30);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_30, BRAM_mm_w1_30, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_30, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_30, BRAM_input_31);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_31, BRAM_rms_norm_w1_31, BRAM_input_31);
    grouped_multihead_attention_2048_4096_32_128_rope_ap_fixed_16_5_(BRAM_input_31, BRAM_wq_31, BRAM_wk_31, BRAM_wv_31, BRAM_tmp_1, 8);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_31, BRAM_input_31);
    rms_norm_2048_4096_ap_fixed_16_5_(BRAM_input_31, BRAM_rms_norm_w2_31, BRAM_input_31);
    matmul_2048_4096_14336_ap_fixed_16_5_(BRAM_input_31, BRAM_mm_w1_31, BRAM_tmp_2);
    swish_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_2);
    elementwise_mult_2048_14336_ap_fixed_16_5_(BRAM_tmp_2, BRAM_tmp_3, BRAM_tmp_3);
    matmul_2048_14336_4096_ap_fixed_16_5_(BRAM_tmp_3, BRAM_mm_w3_31, BRAM_tmp_1);
    matrix_add_2048_4096_ap_fixed_16_5_(BRAM_tmp_1, BRAM_input_31, BRAM_input_32);
    store_2048_4096_ap_fixed_16_5_(BRAM_input_32, DRAM_output);
}