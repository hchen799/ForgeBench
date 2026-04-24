#include <stdio.h>
#include <stdlib.h>
#include <ap_fixed.h>
#include "top.h"  // Include the top function declaration

typedef ap_fixed<16,5> data_t;

data_t DRAM_token_id[1];
data_t DRAM_decode_pos[1];
data_t DRAM_embedding[128256][4096];
data_t DRAM_final_norm[4096];
data_t DRAM_lm_head[128256][4096];
data_t DRAM_attn_norm[32][4096];
data_t DRAM_q_proj[32][4096][4096];
data_t DRAM_k_proj[32][1024][4096];
data_t DRAM_v_proj[32][1024][4096];
data_t DRAM_o_proj[32][4096][4096];
data_t DRAM_ffn_norm[32][4096];
data_t DRAM_gate_proj[32][14336][4096];
data_t DRAM_up_proj[32][14336][4096];
data_t DRAM_down_proj[32][4096][14336];
data_t DRAM_hidden_ping[1][4096];
data_t DRAM_hidden_pong[1][4096];
data_t DRAM_norm1[1][4096];
data_t DRAM_q[1][4096];
data_t DRAM_k[1][1024];
data_t DRAM_v[1][1024];
data_t DRAM_attn[1][4096];
data_t DRAM_mid[1][4096];
data_t DRAM_norm2[1][4096];
data_t DRAM_gate[1][14336];
data_t DRAM_up[1][14336];
data_t DRAM_ffn[1][14336];
data_t DRAM_ffn_out[1][4096];
data_t DRAM_k_cache[32][8192][8][128];
data_t DRAM_v_cache[32][8192][8][128];
data_t DRAM_logits_decode[1][128256];

void load_txt_to_array(const char *filename, data_t *array, int total_size) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Failed to open %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < total_size; i++) {
        float temp;
        fscanf(fp, "%f", &temp);
        array[i] = (data_t)temp;
    }
    fclose(fp);
}

int main() {
    load_txt_to_array("DRAM_token_id.txt", (data_t*)DRAM_token_id, 1);
    load_txt_to_array("DRAM_decode_pos.txt", (data_t*)DRAM_decode_pos, 1);
    load_txt_to_array("DRAM_embedding.txt", (data_t*)DRAM_embedding, 525336576);
    load_txt_to_array("DRAM_final_norm.txt", (data_t*)DRAM_final_norm, 4096);
    load_txt_to_array("DRAM_lm_head.txt", (data_t*)DRAM_lm_head, 525336576);
    load_txt_to_array("DRAM_attn_norm.txt", (data_t*)DRAM_attn_norm, 131072);
    load_txt_to_array("DRAM_q_proj.txt", (data_t*)DRAM_q_proj, 536870912);
    load_txt_to_array("DRAM_k_proj.txt", (data_t*)DRAM_k_proj, 134217728);
    load_txt_to_array("DRAM_v_proj.txt", (data_t*)DRAM_v_proj, 134217728);
    load_txt_to_array("DRAM_o_proj.txt", (data_t*)DRAM_o_proj, 536870912);
    load_txt_to_array("DRAM_ffn_norm.txt", (data_t*)DRAM_ffn_norm, 131072);
    load_txt_to_array("DRAM_gate_proj.txt", (data_t*)DRAM_gate_proj, 1879048192);
    load_txt_to_array("DRAM_up_proj.txt", (data_t*)DRAM_up_proj, 1879048192);
    load_txt_to_array("DRAM_down_proj.txt", (data_t*)DRAM_down_proj, 1879048192);
    load_txt_to_array("DRAM_hidden_ping.txt", (data_t*)DRAM_hidden_ping, 4096);
    load_txt_to_array("DRAM_hidden_pong.txt", (data_t*)DRAM_hidden_pong, 4096);
    load_txt_to_array("DRAM_norm1.txt", (data_t*)DRAM_norm1, 4096);
    load_txt_to_array("DRAM_q.txt", (data_t*)DRAM_q, 4096);
    load_txt_to_array("DRAM_k.txt", (data_t*)DRAM_k, 1024);
    load_txt_to_array("DRAM_v.txt", (data_t*)DRAM_v, 1024);
    load_txt_to_array("DRAM_attn.txt", (data_t*)DRAM_attn, 4096);
    load_txt_to_array("DRAM_mid.txt", (data_t*)DRAM_mid, 4096);
    load_txt_to_array("DRAM_norm2.txt", (data_t*)DRAM_norm2, 4096);
    load_txt_to_array("DRAM_gate.txt", (data_t*)DRAM_gate, 14336);
    load_txt_to_array("DRAM_up.txt", (data_t*)DRAM_up, 14336);
    load_txt_to_array("DRAM_ffn.txt", (data_t*)DRAM_ffn, 14336);
    load_txt_to_array("DRAM_ffn_out.txt", (data_t*)DRAM_ffn_out, 4096);
    load_txt_to_array("DRAM_k_cache.txt", (data_t*)DRAM_k_cache, 268435456);
    load_txt_to_array("DRAM_v_cache.txt", (data_t*)DRAM_v_cache, 268435456);
    load_txt_to_array("DRAM_logits_decode.txt", (data_t*)DRAM_logits_decode, 128256);

    top(DRAM_token_id, DRAM_decode_pos, DRAM_embedding, DRAM_final_norm, DRAM_lm_head, DRAM_attn_norm, DRAM_q_proj, DRAM_k_proj, DRAM_v_proj, DRAM_o_proj, DRAM_ffn_norm, DRAM_gate_proj, DRAM_up_proj, DRAM_down_proj, DRAM_hidden_ping, DRAM_hidden_pong, DRAM_norm1, DRAM_q, DRAM_k, DRAM_v, DRAM_attn, DRAM_mid, DRAM_norm2, DRAM_gate, DRAM_up, DRAM_ffn, DRAM_ffn_out, DRAM_k_cache, DRAM_v_cache, DRAM_logits_decode);

    // Write contents of DRAM_logits_decode to DRAM_logits_decode_output.txt
    {
        FILE *fp = fopen("DRAM_logits_decode_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 128256; i++) {
                fprintf(fp, "%f ", (float)((data_t*)DRAM_logits_decode)[i]);
            }
            fclose(fp);
        }
    }

    return 0;
}