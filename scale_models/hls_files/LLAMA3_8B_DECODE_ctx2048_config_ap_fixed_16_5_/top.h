#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16,5> data_t;

void top(data_t DRAM_token_id[1], data_t DRAM_decode_pos[1], data_t DRAM_embedding[128256][4096], data_t DRAM_final_norm[4096], data_t DRAM_lm_head[128256][4096], data_t DRAM_attn_norm[32][4096], data_t DRAM_q_proj[32][4096][4096], data_t DRAM_k_proj[32][1024][4096], data_t DRAM_v_proj[32][1024][4096], data_t DRAM_o_proj[32][4096][4096], data_t DRAM_ffn_norm[32][4096], data_t DRAM_gate_proj[32][14336][4096], data_t DRAM_up_proj[32][14336][4096], data_t DRAM_down_proj[32][4096][14336], data_t DRAM_hidden_ping[1][4096], data_t DRAM_hidden_pong[1][4096], data_t DRAM_norm1[1][4096], data_t DRAM_q[1][4096], data_t DRAM_k[1][1024], data_t DRAM_v[1][1024], data_t DRAM_attn[1][4096], data_t DRAM_mid[1][4096], data_t DRAM_norm2[1][4096], data_t DRAM_gate[1][14336], data_t DRAM_up[1][14336], data_t DRAM_ffn[1][14336], data_t DRAM_ffn_out[1][4096], data_t DRAM_k_cache[32][2048][8][128], data_t DRAM_v_cache[32][2048][8][128], data_t DRAM_logits_decode[1][128256]);

#endif // TOP_H