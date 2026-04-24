
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

data_t BRAM_decode_pos[1];
data_t BRAM_gamma[4096];
data_t BRAM_hidden_a[1][4096];
data_t BRAM_hidden_b[1][4096];
data_t BRAM_matrix_in[1][128];
data_t BRAM_matrix_out[1][256];
data_t BRAM_weight_tile[256][128];
data_t BRAM_gate_chunk[1][128];
data_t BRAM_up_chunk[1][128];
data_t BRAM_ffn_chunk[1][128];
data_t BRAM_q_tile[1][512];
data_t BRAM_k_rope[1][128];
data_t BRAM_ctx_tile[1][512];
data_t BRAM_k_tile[128][128];
data_t BRAM_v_tile[128][128];
data_t BRAM_score_tile[1][4][128];
data_t BRAM_rowmax[1][4];
data_t BRAM_rowsum[1][4];

void load_1_ap_fixed_16_5_(data_t input[1], data_t output[1])
{
    for (int idx0 = 0; idx0 < 1; idx0++) {
        output[idx0] = input[idx0];
    }
}

void embedding_lookup_tile_1_128256_4096_1_ap_fixed_16_5_(
    data_t token_ids[1],
    data_t embedding[128256][4096],
    data_t output[1][4096],
    int token_base,
    int valid_rows
)
{
    for (int r = 0; r < valid_rows; ++r) {
        int token_idx = (int)token_ids[token_base + r];
        if (token_idx < 0 || token_idx >= 128256) token_idx = 0;
        for (int c = 0; c < 4096; ++c) {
            output[r][c] = embedding[token_idx][c];
        }
    }
}

void store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(
    data_t input[1][4096],
    data_t output[1][4096],
    int row_base,
    int col_base,
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[row_base + r][col_base + c] = input[r][c];
        }
    }
}

void load_layer_vector_32_4096_ap_fixed_16_5_(
    data_t input[32][4096],
    data_t output[4096],
    int layer_idx
)
{
    for (int c = 0; c < 4096; ++c) {
        output[c] = input[layer_idx][c];
    }
}

void load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(
    data_t input[1][4096],
    data_t output[1][4096],
    int row_base,
    int col_base
)
{
    for (int r = 0; r < 1; ++r) {
        for (int c = 0; c < 4096; ++c) {
            int src_r = row_base + r;
            int src_c = col_base + c;
            if (src_r < 1 && src_c < 4096) {
                output[r][c] = input[src_r][src_c];
            } else {
                output[r][c] = (data_t)0;
            }
        }
    }
}

void rmsnorm_tile_full_1_4096_ap_fixed_16_5_(
    data_t input[1][4096],
    data_t gamma[4096],
    data_t output[1][4096],
    int valid_rows
)
{
    const acc_t eps = (acc_t)1e-5;
    for (int r = 0; r < valid_rows; ++r) {
        acc_t sum_sq = (acc_t)0;
        for (int c = 0; c < 4096; ++c) {
            sum_sq += (acc_t)input[r][c] * (acc_t)input[r][c];
        }
        acc_t inv_rms = (acc_t)1 / hls::sqrt(sum_sq / (acc_t)4096 + eps);
        for (int c = 0; c < 4096; ++c) {
            output[r][c] = (data_t)((acc_t)input[r][c] * (acc_t)gamma[c] * inv_rms);
        }
    }
}

void clear_matrix_tile_1_256_ap_fixed_16_5_(data_t tile[1][256])
{
    for (int r = 0; r < 1; ++r) {
        for (int c = 0; c < 256; ++c) {
            tile[r][c] = (data_t)0;
        }
    }
}

void load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(
    data_t input[1][4096],
    data_t output[1][128],
    int row_base,
    int col_base
)
{
    for (int r = 0; r < 1; ++r) {
        for (int c = 0; c < 128; ++c) {
            int src_r = row_base + r;
            int src_c = col_base + c;
            if (src_r < 1 && src_c < 4096) {
                output[r][c] = input[src_r][src_c];
            } else {
                output[r][c] = (data_t)0;
            }
        }
    }
}

void load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(
    data_t input[32][4096][4096],
    data_t output[256][128],
    int layer_idx,
    int out_base,
    int in_base
)
{
    for (int o = 0; o < 256; ++o) {
        for (int i = 0; i < 128; ++i) {
            int src_o = out_base + o;
            int src_i = in_base + i;
            if (src_o < 4096 && src_i < 4096) {
                output[o][i] = input[layer_idx][src_o][src_i];
            } else {
                output[o][i] = (data_t)0;
            }
        }
    }
}

void linear_tile_1_128_256_ap_fixed_16_5_(
    data_t input[1][128],
    data_t weight[256][128],
    data_t output[1][256],
    int valid_rows,
    int valid_out,
    int valid_in
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int o = 0; o < valid_out; ++o) {
            acc_t sum = output[r][o];
            for (int i = 0; i < valid_in; ++i) {
                sum += (acc_t)input[r][i] * (acc_t)weight[o][i];
            }
            output[r][o] = (data_t)sum;
        }
    }
}

void store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(
    data_t input[1][256],
    data_t output[1][4096],
    int row_base,
    int col_base,
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[row_base + r][col_base + c] = input[r][c];
        }
    }
}

void load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(
    data_t input[32][1024][4096],
    data_t output[256][128],
    int layer_idx,
    int out_base,
    int in_base
)
{
    for (int o = 0; o < 256; ++o) {
        for (int i = 0; i < 128; ++i) {
            int src_o = out_base + o;
            int src_i = in_base + i;
            if (src_o < 1024 && src_i < 4096) {
                output[o][i] = input[layer_idx][src_o][src_i];
            } else {
                output[o][i] = (data_t)0;
            }
        }
    }
}

void store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(
    data_t input[1][256],
    data_t output[1][1024],
    int row_base,
    int col_base,
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[row_base + r][col_base + c] = input[r][c];
        }
    }
}

void load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(
    data_t input[1][4096],
    data_t output[1][512],
    int row_base,
    int col_base
)
{
    for (int r = 0; r < 1; ++r) {
        for (int c = 0; c < 512; ++c) {
            int src_r = row_base + r;
            int src_c = col_base + c;
            if (src_r < 1 && src_c < 4096) {
                output[r][c] = input[src_r][src_c];
            } else {
                output[r][c] = (data_t)0;
            }
        }
    }
}

void apply_rope_tile_1_4_128_ap_fixed_16_5_(
    data_t input[1][512],
    data_t output[1][512],
    int pos_base,
    int head_base,
    int valid_rows
)
{
    const float rope_theta = 500000.0f;
    for (int r = 0; r < valid_rows; ++r) {
        for (int h = 0; h < 4; ++h) {
            for (int d = 0; d < 128; d += 2) {
                int idx = h * 128 + d;
                int global_head = head_base + h;
                float theta = powf(rope_theta, -((float)d) / 128.0f);
                float angle = (float)(pos_base + r) * theta;
                data_t cos_val = (data_t)hls::cos(angle);
                data_t sin_val = (data_t)hls::sin(angle);
                data_t x0 = input[r][idx];
                data_t x1 = input[r][idx + 1];
                output[r][idx] = x0 * cos_val - x1 * sin_val;
                output[r][idx + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

void store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(
    data_t input[1][512],
    data_t output[1][4096],
    int row_base,
    int col_base,
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[row_base + r][col_base + c] = input[r][c];
        }
    }
}

void load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(
    data_t input[1][1024],
    data_t output[1][128],
    int row_base,
    int col_base
)
{
    for (int r = 0; r < 1; ++r) {
        for (int c = 0; c < 128; ++c) {
            int src_r = row_base + r;
            int src_c = col_base + c;
            if (src_r < 1 && src_c < 1024) {
                output[r][c] = input[src_r][src_c];
            } else {
                output[r][c] = (data_t)0;
            }
        }
    }
}

void apply_rope_tile_1_1_128_ap_fixed_16_5_(
    data_t input[1][128],
    data_t output[1][128],
    int pos_base,
    int head_base,
    int valid_rows
)
{
    const float rope_theta = 500000.0f;
    for (int r = 0; r < valid_rows; ++r) {
        for (int h = 0; h < 1; ++h) {
            for (int d = 0; d < 128; d += 2) {
                int idx = h * 128 + d;
                int global_head = head_base + h;
                float theta = powf(rope_theta, -((float)d) / 128.0f);
                float angle = (float)(pos_base + r) * theta;
                data_t cos_val = (data_t)hls::cos(angle);
                data_t sin_val = (data_t)hls::sin(angle);
                data_t x0 = input[r][idx];
                data_t x1 = input[r][idx + 1];
                output[r][idx] = x0 * cos_val - x1 * sin_val;
                output[r][idx + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

void store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(
    data_t input[1][128],
    data_t output[1][1024],
    int row_base,
    int col_base,
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[row_base + r][col_base + c] = input[r][c];
        }
    }
}

void kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(
    data_t input[1][1024],
    data_t cache[32][2048][8][128],
    int layer_idx,
    int src_row_base,
    int cache_row_base,
    int valid_rows
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int h = 0; h < 8; ++h) {
            for (int d = 0; d < 128; ++d) {
                cache[layer_idx][cache_row_base + r][h][d] = input[src_row_base + r][h * 128 + d];
            }
        }
    }
}

void init_rowmax_tile_1_4_ap_fixed_16_5_(data_t tile[1][4])
{
    for (int r = 0; r < 1; ++r) {
        for (int c = 0; c < 4; ++c) {
            tile[r][c] = (data_t)(-8);
        }
    }
}

void clear_matrix_tile_1_4_ap_fixed_16_5_(data_t tile[1][4])
{
    for (int r = 0; r < 1; ++r) {
        for (int c = 0; c < 4; ++c) {
            tile[r][c] = (data_t)0;
        }
    }
}

void clear_matrix_tile_1_512_ap_fixed_16_5_(data_t tile[1][512])
{
    for (int r = 0; r < 1; ++r) {
        for (int c = 0; c < 512; ++c) {
            tile[r][c] = (data_t)0;
        }
    }
}

void kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(
    data_t cache[32][2048][8][128],
    data_t output[128][128],
    int layer_idx,
    int kv_head_idx,
    int cache_row_base,
    int valid_rows
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int d = 0; d < 128; ++d) {
            output[r][d] = cache[layer_idx][cache_row_base + r][kv_head_idx][d];
        }
    }
}

void attention_score_tile_1_4_128_128_ap_fixed_16_5_(
    data_t q_tile[1][512],
    data_t k_tile[128][128],
    data_t score[1][4][128],
    int valid_q,
    int valid_k,
    int q_index_base,
    int k_index_base
)
{
    const data_t scale = (data_t)1.0 / hls::sqrt((data_t)128);
    for (int qt = 0; qt < valid_q; ++qt) {
        for (int qh = 0; qh < 4; ++qh) {
            for (int kt = 0; kt < valid_k; ++kt) {
                if ((k_index_base + kt) > (q_index_base + qt)) {
                    score[qt][qh][kt] = (data_t)(-8);
                } else {
                    acc_t sum = (acc_t)0;
                    for (int d = 0; d < 128; ++d) {
                        sum += (acc_t)q_tile[qt][qh * 128 + d] * (acc_t)k_tile[kt][d];
                    }
                    score[qt][qh][kt] = (data_t)(sum * scale);
                }
            }
        }
    }
}

void attention_rowmax_tile_1_4_128_ap_fixed_16_5_(
    data_t score[1][4][128],
    data_t rowmax[1][4],
    int valid_q,
    int valid_k
)
{
    for (int qt = 0; qt < valid_q; ++qt) {
        for (int qh = 0; qh < 4; ++qh) {
            for (int kt = 0; kt < valid_k; ++kt) {
                if (score[qt][qh][kt] > rowmax[qt][qh]) {
                    rowmax[qt][qh] = score[qt][qh][kt];
                }
            }
        }
    }
}

void attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(
    data_t score[1][4][128],
    data_t v_tile[128][128],
    data_t rowmax[1][4],
    data_t rowsum[1][4],
    data_t ctx[1][512],
    int valid_q,
    int valid_k
)
{
    for (int qt = 0; qt < valid_q; ++qt) {
        for (int qh = 0; qh < 4; ++qh) {
            for (int kt = 0; kt < valid_k; ++kt) {
                data_t weight = hls::exp(score[qt][qh][kt] - rowmax[qt][qh]);
                rowsum[qt][qh] += weight;
                for (int d = 0; d < 128; ++d) {
                    ctx[qt][qh * 128 + d] += weight * v_tile[kt][d];
                }
            }
        }
    }
}

void attention_finalize_tile_1_4_128_ap_fixed_16_5_(
    data_t ctx[1][512],
    data_t rowsum[1][4],
    int valid_q
)
{
    for (int qt = 0; qt < valid_q; ++qt) {
        for (int qh = 0; qh < 4; ++qh) {
            data_t denom = rowsum[qt][qh];
            if (denom == (data_t)0) denom = (data_t)1;
            for (int d = 0; d < 128; ++d) {
                ctx[qt][qh * 128 + d] = ctx[qt][qh * 128 + d] / denom;
            }
        }
    }
}

void matrix_add_tile_2d_1_4096_ap_fixed_16_5_(
    data_t lhs[1][4096],
    data_t rhs[1][4096],
    data_t output[1][4096],
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[r][c] = lhs[r][c] + rhs[r][c];
        }
    }
}

void load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(
    data_t input[32][14336][4096],
    data_t output[256][128],
    int layer_idx,
    int out_base,
    int in_base
)
{
    for (int o = 0; o < 256; ++o) {
        for (int i = 0; i < 128; ++i) {
            int src_o = out_base + o;
            int src_i = in_base + i;
            if (src_o < 14336 && src_i < 4096) {
                output[o][i] = input[layer_idx][src_o][src_i];
            } else {
                output[o][i] = (data_t)0;
            }
        }
    }
}

void store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(
    data_t input[1][256],
    data_t output[1][14336],
    int row_base,
    int col_base,
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[row_base + r][col_base + c] = input[r][c];
        }
    }
}

void load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(
    data_t input[1][14336],
    data_t output[1][128],
    int row_base,
    int col_base
)
{
    for (int r = 0; r < 1; ++r) {
        for (int c = 0; c < 128; ++c) {
            int src_r = row_base + r;
            int src_c = col_base + c;
            if (src_r < 1 && src_c < 14336) {
                output[r][c] = input[src_r][src_c];
            } else {
                output[r][c] = (data_t)0;
            }
        }
    }
}

void silu_tile_2d_1_128_ap_fixed_16_5_(
    data_t input[1][128],
    data_t output[1][128],
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[r][c] = input[r][c] / ((data_t)1 + hls::exp(-input[r][c]));
        }
    }
}

void elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(
    data_t lhs[1][128],
    data_t rhs[1][128],
    data_t output[1][128],
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[r][c] = lhs[r][c] * rhs[r][c];
        }
    }
}

void store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(
    data_t input[1][128],
    data_t output[1][14336],
    int row_base,
    int col_base,
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[row_base + r][col_base + c] = input[r][c];
        }
    }
}

void load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(
    data_t input[32][4096][14336],
    data_t output[256][128],
    int layer_idx,
    int out_base,
    int in_base
)
{
    for (int o = 0; o < 256; ++o) {
        for (int i = 0; i < 128; ++i) {
            int src_o = out_base + o;
            int src_i = in_base + i;
            if (src_o < 4096 && src_i < 14336) {
                output[o][i] = input[layer_idx][src_o][src_i];
            } else {
                output[o][i] = (data_t)0;
            }
        }
    }
}

void load_4096_ap_fixed_16_5_(data_t input[4096], data_t output[4096])
{
    for (int idx0 = 0; idx0 < 4096; idx0++) {
        output[idx0] = input[idx0];
    }
}

void load_weight_tile_2d_128256_4096_256_128_ap_fixed_16_5_(
    data_t input[128256][4096],
    data_t output[256][128],
    int out_base,
    int in_base
)
{
    for (int o = 0; o < 256; ++o) {
        for (int i = 0; i < 128; ++i) {
            int src_o = out_base + o;
            int src_i = in_base + i;
            if (src_o < 128256 && src_i < 4096) {
                output[o][i] = input[src_o][src_i];
            } else {
                output[o][i] = (data_t)0;
            }
        }
    }
}

void store_matrix_tile_1_128256_1_256_ap_fixed_16_5_(
    data_t input[1][256],
    data_t output[1][128256],
    int row_base,
    int col_base,
    int valid_rows,
    int valid_cols
)
{
    for (int r = 0; r < valid_rows; ++r) {
        for (int c = 0; c < valid_cols; ++c) {
            output[row_base + r][col_base + c] = input[r][c];
        }
    }
}

void top(data_t DRAM_token_id[1], data_t DRAM_decode_pos[1], data_t DRAM_embedding[128256][4096], data_t DRAM_final_norm[4096], data_t DRAM_lm_head[128256][4096], data_t DRAM_attn_norm[32][4096], data_t DRAM_q_proj[32][4096][4096], data_t DRAM_k_proj[32][1024][4096], data_t DRAM_v_proj[32][1024][4096], data_t DRAM_o_proj[32][4096][4096], data_t DRAM_ffn_norm[32][4096], data_t DRAM_gate_proj[32][14336][4096], data_t DRAM_up_proj[32][14336][4096], data_t DRAM_down_proj[32][4096][14336], data_t DRAM_hidden_ping[1][4096], data_t DRAM_hidden_pong[1][4096], data_t DRAM_norm1[1][4096], data_t DRAM_q[1][4096], data_t DRAM_k[1][1024], data_t DRAM_v[1][1024], data_t DRAM_attn[1][4096], data_t DRAM_mid[1][4096], data_t DRAM_norm2[1][4096], data_t DRAM_gate[1][14336], data_t DRAM_up[1][14336], data_t DRAM_ffn[1][14336], data_t DRAM_ffn_out[1][4096], data_t DRAM_k_cache[32][2048][8][128], data_t DRAM_v_cache[32][2048][8][128], data_t DRAM_logits_decode[1][128256])
{
    #pragma HLS interface m_axi port=DRAM_token_id offset=slave bundle=mem_token_id
    #pragma HLS interface m_axi port=DRAM_decode_pos offset=slave bundle=mem_decode_pos
    #pragma HLS interface m_axi port=DRAM_embedding offset=slave bundle=mem_embedding
    #pragma HLS interface m_axi port=DRAM_final_norm offset=slave bundle=mem_final_norm
    #pragma HLS interface m_axi port=DRAM_lm_head offset=slave bundle=mem_lm_head
    #pragma HLS interface m_axi port=DRAM_attn_norm offset=slave bundle=mem_attn_norm
    #pragma HLS interface m_axi port=DRAM_q_proj offset=slave bundle=mem_q_proj
    #pragma HLS interface m_axi port=DRAM_k_proj offset=slave bundle=mem_k_proj
    #pragma HLS interface m_axi port=DRAM_v_proj offset=slave bundle=mem_v_proj
    #pragma HLS interface m_axi port=DRAM_o_proj offset=slave bundle=mem_o_proj
    #pragma HLS interface m_axi port=DRAM_ffn_norm offset=slave bundle=mem_ffn_norm
    #pragma HLS interface m_axi port=DRAM_gate_proj offset=slave bundle=mem_gate_proj
    #pragma HLS interface m_axi port=DRAM_up_proj offset=slave bundle=mem_up_proj
    #pragma HLS interface m_axi port=DRAM_down_proj offset=slave bundle=mem_down_proj
    #pragma HLS interface m_axi port=DRAM_hidden_ping offset=slave bundle=mem_hidden_ping
    #pragma HLS interface m_axi port=DRAM_hidden_pong offset=slave bundle=mem_hidden_pong
    #pragma HLS interface m_axi port=DRAM_norm1 offset=slave bundle=mem_norm1
    #pragma HLS interface m_axi port=DRAM_q offset=slave bundle=mem_q
    #pragma HLS interface m_axi port=DRAM_k offset=slave bundle=mem_k
    #pragma HLS interface m_axi port=DRAM_v offset=slave bundle=mem_v
    #pragma HLS interface m_axi port=DRAM_attn offset=slave bundle=mem_attn
    #pragma HLS interface m_axi port=DRAM_mid offset=slave bundle=mem_mid
    #pragma HLS interface m_axi port=DRAM_norm2 offset=slave bundle=mem_norm2
    #pragma HLS interface m_axi port=DRAM_gate offset=slave bundle=mem_gate
    #pragma HLS interface m_axi port=DRAM_up offset=slave bundle=mem_up
    #pragma HLS interface m_axi port=DRAM_ffn offset=slave bundle=mem_ffn
    #pragma HLS interface m_axi port=DRAM_ffn_out offset=slave bundle=mem_ffn_out
    #pragma HLS interface m_axi port=DRAM_k_cache offset=slave bundle=mem_k_cache
    #pragma HLS interface m_axi port=DRAM_v_cache offset=slave bundle=mem_v_cache
    #pragma HLS interface m_axi port=DRAM_logits_decode offset=slave bundle=mem_logits

    load_1_ap_fixed_16_5_(DRAM_decode_pos, BRAM_decode_pos);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        embedding_lookup_tile_1_128256_4096_1_ap_fixed_16_5_(DRAM_token_id, DRAM_embedding, BRAM_hidden_a, t_base, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 0);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 0, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 0, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 0, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 0, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 0, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 0, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 0, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 0, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 0, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 0);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 0, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 0, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 0, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 1);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 1, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 1, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 1, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 1, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 1, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 1, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 1, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 1, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 1, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 1);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 1, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 1, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 1, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 2);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 2, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 2, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 2, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 2, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 2, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 2, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 2, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 2, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 2, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 2);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 2, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 2, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 2, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 3);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 3, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 3, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 3, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 3, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 3, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 3, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 3, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 3, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 3, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 3);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 3, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 3, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 3, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 4);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 4, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 4, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 4, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 4, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 4, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 4, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 4, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 4, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 4, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 4);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 4, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 4, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 4, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 5);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 5, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 5, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 5, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 5, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 5, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 5, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 5, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 5, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 5, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 5);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 5, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 5, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 5, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 6);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 6, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 6, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 6, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 6, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 6, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 6, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 6, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 6, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 6, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 6);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 6, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 6, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 6, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 7);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 7, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 7, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 7, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 7, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 7, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 7, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 7, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 7, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 7, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 7);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 7, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 7, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 7, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 8);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 8, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 8, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 8, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 8, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 8, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 8, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 8, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 8, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 8, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 8);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 8, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 8, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 8, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 9);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 9, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 9, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 9, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 9, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 9, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 9, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 9, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 9, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 9, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 9);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 9, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 9, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 9, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 10);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 10, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 10, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 10, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 10, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 10, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 10, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 10, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 10, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 10, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 10);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 10, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 10, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 10, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 11);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 11, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 11, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 11, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 11, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 11, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 11, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 11, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 11, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 11, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 11);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 11, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 11, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 11, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 12);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 12, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 12, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 12, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 12, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 12, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 12, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 12, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 12, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 12, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 12);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 12, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 12, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 12, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 13);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 13, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 13, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 13, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 13, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 13, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 13, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 13, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 13, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 13, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 13);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 13, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 13, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 13, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 14);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 14, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 14, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 14, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 14, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 14, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 14, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 14, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 14, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 14, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 14);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 14, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 14, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 14, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 15);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 15, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 15, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 15, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 15, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 15, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 15, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 15, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 15, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 15, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 15);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 15, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 15, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 15, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 16);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 16, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 16, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 16, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 16, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 16, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 16, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 16, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 16, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 16, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 16);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 16, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 16, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 16, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 17);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 17, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 17, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 17, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 17, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 17, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 17, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 17, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 17, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 17, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 17);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 17, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 17, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 17, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 18);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 18, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 18, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 18, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 18, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 18, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 18, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 18, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 18, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 18, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 18);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 18, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 18, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 18, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 19);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 19, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 19, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 19, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 19, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 19, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 19, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 19, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 19, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 19, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 19);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 19, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 19, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 19, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 20);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 20, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 20, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 20, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 20, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 20, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 20, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 20, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 20, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 20, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 20);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 20, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 20, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 20, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 21);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 21, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 21, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 21, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 21, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 21, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 21, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 21, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 21, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 21, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 21);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 21, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 21, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 21, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 22);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 22, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 22, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 22, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 22, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 22, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 22, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 22, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 22, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 22, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 22);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 22, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 22, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 22, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 23);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 23, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 23, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 23, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 23, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 23, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 23, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 23, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 23, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 23, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 23);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 23, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 23, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 23, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 24);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 24, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 24, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 24, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 24, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 24, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 24, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 24, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 24, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 24, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 24);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 24, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 24, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 24, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 25);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 25, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 25, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 25, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 25, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 25, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 25, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 25, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 25, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 25, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 25);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 25, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 25, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 25, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 26);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 26, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 26, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 26, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 26, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 26, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 26, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 26, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 26, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 26, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 26);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 26, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 26, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 26, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 27);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 27, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 27, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 27, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 27, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 27, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 27, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 27, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 27, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 27, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 27);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 27, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 27, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 27, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 28);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 28, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 28, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 28, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 28, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 28, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 28, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 28, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 28, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 28, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 28);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 28, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 28, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 28, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 29);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 29, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 29, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 29, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 29, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 29, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 29, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 29, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 29, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 29, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 29);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 29, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 29, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 29, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 30);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 30, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 30, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 30, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 30, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 30, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 30, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 30, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 30, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 30, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 30);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 30, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 30, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 30, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_pong, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_attn_norm, BRAM_gamma, 31);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_q_proj, BRAM_weight_tile, 31, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_q, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_k_proj, BRAM_weight_tile, 31, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_k, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 1024; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (1024) ? (128) : ((1024) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_1024_4096_256_128_ap_fixed_16_5_(DRAM_v_proj, BRAM_weight_tile, 31, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_1024_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_v, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            apply_rope_tile_1_4_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_q_tile, (int)BRAM_decode_pos[0], qh_base, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_q_tile, DRAM_q, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int kh_base = 0; kh_base < 8; kh_base += 1) {
            int valid_kh = (((kh_base) + (1)) <= (8) ? (1) : ((8) - (kh_base)));
            load_matrix_tile_1_1024_1_128_ap_fixed_16_5_(DRAM_k, BRAM_k_rope, t_base, (kh_base * 128));
            apply_rope_tile_1_1_128_ap_fixed_16_5_(BRAM_k_rope, BRAM_k_rope, (int)BRAM_decode_pos[0], kh_base, valid_t);
            store_matrix_tile_1_1024_1_128_ap_fixed_16_5_(BRAM_k_rope, DRAM_k, t_base, (kh_base * 128), valid_t, 128);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_k, DRAM_k_cache, 31, t_base, (int)BRAM_decode_pos[0], valid_t);
        kv_cache_store_tile_1_2048_8_128_1_1024_ap_fixed_16_5_(DRAM_v, DRAM_v_cache, 31, t_base, (int)BRAM_decode_pos[0], valid_t);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int qh_base = 0; qh_base < 32; qh_base += 4) {
            int valid_qh = (((qh_base) + (4)) <= (32) ? (4) : ((32) - (qh_base)));
            init_rowmax_tile_1_4_ap_fixed_16_5_(BRAM_rowmax);
            clear_matrix_tile_1_4_ap_fixed_16_5_(BRAM_rowsum);
            clear_matrix_tile_1_512_ap_fixed_16_5_(BRAM_ctx_tile);
            load_matrix_tile_1_4096_1_512_ap_fixed_16_5_(DRAM_q, BRAM_q_tile, 0, (qh_base * 128));
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 31, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_rowmax_tile_1_4_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_rowmax, valid_t, valid_k);
            }
            for (int k_base = 0; k_base < ((int)BRAM_decode_pos[0] + 1); k_base += 128) {
                int valid_k = (((k_base) + (128)) <= (((int)BRAM_decode_pos[0] + 1)) ? (128) : ((((int)BRAM_decode_pos[0] + 1)) - (k_base)));
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_k_cache, BRAM_k_tile, 31, (qh_base / 4), k_base, valid_k);
                kv_cache_load_tile_2048_8_128_128_ap_fixed_16_5_(DRAM_v_cache, BRAM_v_tile, 31, (qh_base / 4), k_base, valid_k);
                attention_score_tile_1_4_128_128_ap_fixed_16_5_(BRAM_q_tile, BRAM_k_tile, BRAM_score_tile, valid_t, valid_k, (int)BRAM_decode_pos[0], k_base);
                attention_softmax_context_tile_1_4_128_128_ap_fixed_16_5_(BRAM_score_tile, BRAM_v_tile, BRAM_rowmax, BRAM_rowsum, BRAM_ctx_tile, valid_t, valid_k);
            }
            attention_finalize_tile_1_4_128_ap_fixed_16_5_(BRAM_ctx_tile, BRAM_rowsum, valid_t);
            store_matrix_tile_1_4096_1_512_ap_fixed_16_5_(BRAM_ctx_tile, DRAM_attn, 0, (qh_base * 128), valid_t, 512);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_attn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_4096_256_128_ap_fixed_16_5_(DRAM_o_proj, BRAM_weight_tile, 31, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_attn, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_pong, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_attn, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_mid, t_base, 0, valid_t, 4096);
    }
    load_layer_vector_32_4096_ap_fixed_16_5_(DRAM_ffn_norm, BRAM_gamma, 31);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm2, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_gate_proj, BRAM_weight_tile, 31, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_gate, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 14336; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (14336) ? (128) : ((14336) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm2, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_14336_4096_256_128_ap_fixed_16_5_(DRAM_up_proj, BRAM_weight_tile, 31, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_14336_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_up, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int c_base = 0; c_base < 14336; c_base += 128) {
            int valid_c = (((c_base) + (128)) <= (14336) ? (128) : ((14336) - (c_base)));
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_gate, BRAM_gate_chunk, t_base, c_base);
            load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_up, BRAM_up_chunk, t_base, c_base);
            silu_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_gate_chunk, valid_t, valid_c);
            elementwise_mult_tile_2d_1_128_ap_fixed_16_5_(BRAM_gate_chunk, BRAM_up_chunk, BRAM_ffn_chunk, valid_t, valid_c);
            store_matrix_tile_1_14336_1_128_ap_fixed_16_5_(BRAM_ffn_chunk, DRAM_ffn, t_base, c_base, valid_t, valid_c);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 4096; o_base += 128) {
            int valid_o = (((o_base) + (128)) <= (4096) ? (128) : ((4096) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 14336; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (14336) ? (128) : ((14336) - (i_base)));
                load_matrix_tile_1_14336_1_128_ap_fixed_16_5_(DRAM_ffn, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_layered_32_4096_14336_256_128_ap_fixed_16_5_(DRAM_down_proj, BRAM_weight_tile, 31, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_4096_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_ffn_out, t_base, o_base, valid_t, valid_o);
        }
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_mid, BRAM_hidden_a, t_base, 0);
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_ffn_out, BRAM_hidden_b, t_base, 0);
        matrix_add_tile_2d_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_hidden_b, BRAM_hidden_a, valid_t, 4096);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_a, DRAM_hidden_ping, t_base, 0, valid_t, 4096);
    }
    load_4096_ap_fixed_16_5_(DRAM_final_norm, BRAM_gamma);
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        load_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(DRAM_hidden_ping, BRAM_hidden_a, t_base, 0);
        rmsnorm_tile_full_1_4096_ap_fixed_16_5_(BRAM_hidden_a, BRAM_gamma, BRAM_hidden_b, valid_t);
        store_matrix_tile_1_4096_1_4096_ap_fixed_16_5_(BRAM_hidden_b, DRAM_norm1, t_base, 0, valid_t, 4096);
    }
    for (int t_base = 0; t_base < 1; t_base += 1) {
        int valid_t = (((t_base) + (1)) <= (1) ? (1) : ((1) - (t_base)));
        for (int o_base = 0; o_base < 128256; o_base += 256) {
            int valid_o = (((o_base) + (256)) <= (128256) ? (256) : ((128256) - (o_base)));
            clear_matrix_tile_1_256_ap_fixed_16_5_(BRAM_matrix_out);
            for (int i_base = 0; i_base < 4096; i_base += 128) {
                int valid_i = (((i_base) + (128)) <= (4096) ? (128) : ((4096) - (i_base)));
                load_matrix_tile_1_4096_1_128_ap_fixed_16_5_(DRAM_norm1, BRAM_matrix_in, t_base, i_base);
                load_weight_tile_2d_128256_4096_256_128_ap_fixed_16_5_(DRAM_lm_head, BRAM_weight_tile, o_base, i_base);
                linear_tile_1_128_256_ap_fixed_16_5_(BRAM_matrix_in, BRAM_weight_tile, BRAM_matrix_out, valid_t, valid_o, valid_i);
            }
            store_matrix_tile_1_128256_1_256_ap_fixed_16_5_(BRAM_matrix_out, DRAM_logits_decode, t_base, o_base, valid_t, valid_o);
        }
    }
}