#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16,5> data_t;

void top(data_t DRAM_token_ids[2048], data_t DRAM_prefill_len[1], data_t DRAM_embedding[128256][4096], data_t DRAM_final_norm[4096], data_t DRAM_lm_head[128256][4096], data_t DRAM_attn_norm[32][4096], data_t DRAM_q_proj[32][4096][4096], data_t DRAM_k_proj[32][1024][4096], data_t DRAM_v_proj[32][1024][4096], data_t DRAM_o_proj[32][4096][4096], data_t DRAM_ffn_norm[32][4096], data_t DRAM_gate_proj[32][14336][4096], data_t DRAM_up_proj[32][14336][4096], data_t DRAM_down_proj[32][4096][14336], data_t DRAM_hidden_ping[2048][4096], data_t DRAM_hidden_pong[2048][4096], data_t DRAM_norm1[2048][4096], data_t DRAM_q[2048][4096], data_t DRAM_k[2048][1024], data_t DRAM_v[2048][1024], data_t DRAM_attn[2048][4096], data_t DRAM_mid[2048][4096], data_t DRAM_norm2[2048][4096], data_t DRAM_gate[2048][14336], data_t DRAM_up[2048][14336], data_t DRAM_ffn[2048][14336], data_t DRAM_ffn_out[2048][4096], data_t DRAM_k_cache[32][2048][8][128], data_t DRAM_v_cache[32][2048][8][128], data_t DRAM_logits[2048][128256]);

#endif // TOP_H