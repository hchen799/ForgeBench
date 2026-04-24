#include <stdio.h>
#include <stdlib.h>
#include <ap_fixed.h>
#include "top.h"  // Include the top function declaration

typedef ap_fixed<16,5> data_t;

data_t DRAM_input[3][224][224];
data_t DRAM_w_stem[32][3][3][3];
data_t DRAM_bn_stem[4][32];
data_t DRAM_w_dw_0[32][32][3][3];
data_t DRAM_bn_dw_0[4][32];
data_t DRAM_w_proj_0[16][32][1][1];
data_t DRAM_bn_proj_0[4][16];
data_t DRAM_w_exp_1[96][16][1][1];
data_t DRAM_bn_exp_1[4][96];
data_t DRAM_w_dw_1[96][96][3][3];
data_t DRAM_bn_dw_1[4][96];
data_t DRAM_w_proj_1[24][96][1][1];
data_t DRAM_bn_proj_1[4][24];
data_t DRAM_w_exp_2[144][24][1][1];
data_t DRAM_bn_exp_2[4][144];
data_t DRAM_w_dw_2[144][144][3][3];
data_t DRAM_bn_dw_2[4][144];
data_t DRAM_w_proj_2[24][144][1][1];
data_t DRAM_bn_proj_2[4][24];
data_t DRAM_w_exp_3[144][24][1][1];
data_t DRAM_bn_exp_3[4][144];
data_t DRAM_w_dw_3[144][144][3][3];
data_t DRAM_bn_dw_3[4][144];
data_t DRAM_w_proj_3[32][144][1][1];
data_t DRAM_bn_proj_3[4][32];
data_t DRAM_w_exp_4[192][32][1][1];
data_t DRAM_bn_exp_4[4][192];
data_t DRAM_w_dw_4[192][192][3][3];
data_t DRAM_bn_dw_4[4][192];
data_t DRAM_w_proj_4[32][192][1][1];
data_t DRAM_bn_proj_4[4][32];
data_t DRAM_w_exp_5[192][32][1][1];
data_t DRAM_bn_exp_5[4][192];
data_t DRAM_w_dw_5[192][192][3][3];
data_t DRAM_bn_dw_5[4][192];
data_t DRAM_w_proj_5[32][192][1][1];
data_t DRAM_bn_proj_5[4][32];
data_t DRAM_w_exp_6[192][32][1][1];
data_t DRAM_bn_exp_6[4][192];
data_t DRAM_w_dw_6[192][192][3][3];
data_t DRAM_bn_dw_6[4][192];
data_t DRAM_w_proj_6[64][192][1][1];
data_t DRAM_bn_proj_6[4][64];
data_t DRAM_w_exp_7[384][64][1][1];
data_t DRAM_bn_exp_7[4][384];
data_t DRAM_w_dw_7[384][384][3][3];
data_t DRAM_bn_dw_7[4][384];
data_t DRAM_w_proj_7[64][384][1][1];
data_t DRAM_bn_proj_7[4][64];
data_t DRAM_w_exp_8[384][64][1][1];
data_t DRAM_bn_exp_8[4][384];
data_t DRAM_w_dw_8[384][384][3][3];
data_t DRAM_bn_dw_8[4][384];
data_t DRAM_w_proj_8[64][384][1][1];
data_t DRAM_bn_proj_8[4][64];
data_t DRAM_w_exp_9[384][64][1][1];
data_t DRAM_bn_exp_9[4][384];
data_t DRAM_w_dw_9[384][384][3][3];
data_t DRAM_bn_dw_9[4][384];
data_t DRAM_w_proj_9[64][384][1][1];
data_t DRAM_bn_proj_9[4][64];
data_t DRAM_w_exp_10[384][64][1][1];
data_t DRAM_bn_exp_10[4][384];
data_t DRAM_w_dw_10[384][384][3][3];
data_t DRAM_bn_dw_10[4][384];
data_t DRAM_w_proj_10[96][384][1][1];
data_t DRAM_bn_proj_10[4][96];
data_t DRAM_w_exp_11[576][96][1][1];
data_t DRAM_bn_exp_11[4][576];
data_t DRAM_w_dw_11[576][576][3][3];
data_t DRAM_bn_dw_11[4][576];
data_t DRAM_w_proj_11[96][576][1][1];
data_t DRAM_bn_proj_11[4][96];
data_t DRAM_w_exp_12[576][96][1][1];
data_t DRAM_bn_exp_12[4][576];
data_t DRAM_w_dw_12[576][576][3][3];
data_t DRAM_bn_dw_12[4][576];
data_t DRAM_w_proj_12[96][576][1][1];
data_t DRAM_bn_proj_12[4][96];
data_t DRAM_w_exp_13[576][96][1][1];
data_t DRAM_bn_exp_13[4][576];
data_t DRAM_w_dw_13[576][576][3][3];
data_t DRAM_bn_dw_13[4][576];
data_t DRAM_w_proj_13[160][576][1][1];
data_t DRAM_bn_proj_13[4][160];
data_t DRAM_w_exp_14[960][160][1][1];
data_t DRAM_bn_exp_14[4][960];
data_t DRAM_w_dw_14[960][960][3][3];
data_t DRAM_bn_dw_14[4][960];
data_t DRAM_w_proj_14[160][960][1][1];
data_t DRAM_bn_proj_14[4][160];
data_t DRAM_w_exp_15[960][160][1][1];
data_t DRAM_bn_exp_15[4][960];
data_t DRAM_w_dw_15[960][960][3][3];
data_t DRAM_bn_dw_15[4][960];
data_t DRAM_w_proj_15[160][960][1][1];
data_t DRAM_bn_proj_15[4][160];
data_t DRAM_w_exp_16[960][160][1][1];
data_t DRAM_bn_exp_16[4][960];
data_t DRAM_w_dw_16[960][960][3][3];
data_t DRAM_bn_dw_16[4][960];
data_t DRAM_w_proj_16[320][960][1][1];
data_t DRAM_bn_proj_16[4][320];
data_t DRAM_w_final[1280][320][1][1];
data_t DRAM_bn_final[4][1280];
data_t DRAM_out[1000][1][1];
data_t DRAM_fc[1000][1280][1][1];

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
    load_txt_to_array("DRAM_input.txt", (data_t*)DRAM_input, 150528);
    load_txt_to_array("DRAM_w_stem.txt", (data_t*)DRAM_w_stem, 864);
    load_txt_to_array("DRAM_bn_stem.txt", (data_t*)DRAM_bn_stem, 128);
    load_txt_to_array("DRAM_w_dw_0.txt", (data_t*)DRAM_w_dw_0, 9216);
    load_txt_to_array("DRAM_bn_dw_0.txt", (data_t*)DRAM_bn_dw_0, 128);
    load_txt_to_array("DRAM_w_proj_0.txt", (data_t*)DRAM_w_proj_0, 512);
    load_txt_to_array("DRAM_bn_proj_0.txt", (data_t*)DRAM_bn_proj_0, 64);
    load_txt_to_array("DRAM_w_exp_1.txt", (data_t*)DRAM_w_exp_1, 1536);
    load_txt_to_array("DRAM_bn_exp_1.txt", (data_t*)DRAM_bn_exp_1, 384);
    load_txt_to_array("DRAM_w_dw_1.txt", (data_t*)DRAM_w_dw_1, 82944);
    load_txt_to_array("DRAM_bn_dw_1.txt", (data_t*)DRAM_bn_dw_1, 384);
    load_txt_to_array("DRAM_w_proj_1.txt", (data_t*)DRAM_w_proj_1, 2304);
    load_txt_to_array("DRAM_bn_proj_1.txt", (data_t*)DRAM_bn_proj_1, 96);
    load_txt_to_array("DRAM_w_exp_2.txt", (data_t*)DRAM_w_exp_2, 3456);
    load_txt_to_array("DRAM_bn_exp_2.txt", (data_t*)DRAM_bn_exp_2, 576);
    load_txt_to_array("DRAM_w_dw_2.txt", (data_t*)DRAM_w_dw_2, 186624);
    load_txt_to_array("DRAM_bn_dw_2.txt", (data_t*)DRAM_bn_dw_2, 576);
    load_txt_to_array("DRAM_w_proj_2.txt", (data_t*)DRAM_w_proj_2, 3456);
    load_txt_to_array("DRAM_bn_proj_2.txt", (data_t*)DRAM_bn_proj_2, 96);
    load_txt_to_array("DRAM_w_exp_3.txt", (data_t*)DRAM_w_exp_3, 3456);
    load_txt_to_array("DRAM_bn_exp_3.txt", (data_t*)DRAM_bn_exp_3, 576);
    load_txt_to_array("DRAM_w_dw_3.txt", (data_t*)DRAM_w_dw_3, 186624);
    load_txt_to_array("DRAM_bn_dw_3.txt", (data_t*)DRAM_bn_dw_3, 576);
    load_txt_to_array("DRAM_w_proj_3.txt", (data_t*)DRAM_w_proj_3, 4608);
    load_txt_to_array("DRAM_bn_proj_3.txt", (data_t*)DRAM_bn_proj_3, 128);
    load_txt_to_array("DRAM_w_exp_4.txt", (data_t*)DRAM_w_exp_4, 6144);
    load_txt_to_array("DRAM_bn_exp_4.txt", (data_t*)DRAM_bn_exp_4, 768);
    load_txt_to_array("DRAM_w_dw_4.txt", (data_t*)DRAM_w_dw_4, 331776);
    load_txt_to_array("DRAM_bn_dw_4.txt", (data_t*)DRAM_bn_dw_4, 768);
    load_txt_to_array("DRAM_w_proj_4.txt", (data_t*)DRAM_w_proj_4, 6144);
    load_txt_to_array("DRAM_bn_proj_4.txt", (data_t*)DRAM_bn_proj_4, 128);
    load_txt_to_array("DRAM_w_exp_5.txt", (data_t*)DRAM_w_exp_5, 6144);
    load_txt_to_array("DRAM_bn_exp_5.txt", (data_t*)DRAM_bn_exp_5, 768);
    load_txt_to_array("DRAM_w_dw_5.txt", (data_t*)DRAM_w_dw_5, 331776);
    load_txt_to_array("DRAM_bn_dw_5.txt", (data_t*)DRAM_bn_dw_5, 768);
    load_txt_to_array("DRAM_w_proj_5.txt", (data_t*)DRAM_w_proj_5, 6144);
    load_txt_to_array("DRAM_bn_proj_5.txt", (data_t*)DRAM_bn_proj_5, 128);
    load_txt_to_array("DRAM_w_exp_6.txt", (data_t*)DRAM_w_exp_6, 6144);
    load_txt_to_array("DRAM_bn_exp_6.txt", (data_t*)DRAM_bn_exp_6, 768);
    load_txt_to_array("DRAM_w_dw_6.txt", (data_t*)DRAM_w_dw_6, 331776);
    load_txt_to_array("DRAM_bn_dw_6.txt", (data_t*)DRAM_bn_dw_6, 768);
    load_txt_to_array("DRAM_w_proj_6.txt", (data_t*)DRAM_w_proj_6, 12288);
    load_txt_to_array("DRAM_bn_proj_6.txt", (data_t*)DRAM_bn_proj_6, 256);
    load_txt_to_array("DRAM_w_exp_7.txt", (data_t*)DRAM_w_exp_7, 24576);
    load_txt_to_array("DRAM_bn_exp_7.txt", (data_t*)DRAM_bn_exp_7, 1536);
    load_txt_to_array("DRAM_w_dw_7.txt", (data_t*)DRAM_w_dw_7, 1327104);
    load_txt_to_array("DRAM_bn_dw_7.txt", (data_t*)DRAM_bn_dw_7, 1536);
    load_txt_to_array("DRAM_w_proj_7.txt", (data_t*)DRAM_w_proj_7, 24576);
    load_txt_to_array("DRAM_bn_proj_7.txt", (data_t*)DRAM_bn_proj_7, 256);
    load_txt_to_array("DRAM_w_exp_8.txt", (data_t*)DRAM_w_exp_8, 24576);
    load_txt_to_array("DRAM_bn_exp_8.txt", (data_t*)DRAM_bn_exp_8, 1536);
    load_txt_to_array("DRAM_w_dw_8.txt", (data_t*)DRAM_w_dw_8, 1327104);
    load_txt_to_array("DRAM_bn_dw_8.txt", (data_t*)DRAM_bn_dw_8, 1536);
    load_txt_to_array("DRAM_w_proj_8.txt", (data_t*)DRAM_w_proj_8, 24576);
    load_txt_to_array("DRAM_bn_proj_8.txt", (data_t*)DRAM_bn_proj_8, 256);
    load_txt_to_array("DRAM_w_exp_9.txt", (data_t*)DRAM_w_exp_9, 24576);
    load_txt_to_array("DRAM_bn_exp_9.txt", (data_t*)DRAM_bn_exp_9, 1536);
    load_txt_to_array("DRAM_w_dw_9.txt", (data_t*)DRAM_w_dw_9, 1327104);
    load_txt_to_array("DRAM_bn_dw_9.txt", (data_t*)DRAM_bn_dw_9, 1536);
    load_txt_to_array("DRAM_w_proj_9.txt", (data_t*)DRAM_w_proj_9, 24576);
    load_txt_to_array("DRAM_bn_proj_9.txt", (data_t*)DRAM_bn_proj_9, 256);
    load_txt_to_array("DRAM_w_exp_10.txt", (data_t*)DRAM_w_exp_10, 24576);
    load_txt_to_array("DRAM_bn_exp_10.txt", (data_t*)DRAM_bn_exp_10, 1536);
    load_txt_to_array("DRAM_w_dw_10.txt", (data_t*)DRAM_w_dw_10, 1327104);
    load_txt_to_array("DRAM_bn_dw_10.txt", (data_t*)DRAM_bn_dw_10, 1536);
    load_txt_to_array("DRAM_w_proj_10.txt", (data_t*)DRAM_w_proj_10, 36864);
    load_txt_to_array("DRAM_bn_proj_10.txt", (data_t*)DRAM_bn_proj_10, 384);
    load_txt_to_array("DRAM_w_exp_11.txt", (data_t*)DRAM_w_exp_11, 55296);
    load_txt_to_array("DRAM_bn_exp_11.txt", (data_t*)DRAM_bn_exp_11, 2304);
    load_txt_to_array("DRAM_w_dw_11.txt", (data_t*)DRAM_w_dw_11, 2985984);
    load_txt_to_array("DRAM_bn_dw_11.txt", (data_t*)DRAM_bn_dw_11, 2304);
    load_txt_to_array("DRAM_w_proj_11.txt", (data_t*)DRAM_w_proj_11, 55296);
    load_txt_to_array("DRAM_bn_proj_11.txt", (data_t*)DRAM_bn_proj_11, 384);
    load_txt_to_array("DRAM_w_exp_12.txt", (data_t*)DRAM_w_exp_12, 55296);
    load_txt_to_array("DRAM_bn_exp_12.txt", (data_t*)DRAM_bn_exp_12, 2304);
    load_txt_to_array("DRAM_w_dw_12.txt", (data_t*)DRAM_w_dw_12, 2985984);
    load_txt_to_array("DRAM_bn_dw_12.txt", (data_t*)DRAM_bn_dw_12, 2304);
    load_txt_to_array("DRAM_w_proj_12.txt", (data_t*)DRAM_w_proj_12, 55296);
    load_txt_to_array("DRAM_bn_proj_12.txt", (data_t*)DRAM_bn_proj_12, 384);
    load_txt_to_array("DRAM_w_exp_13.txt", (data_t*)DRAM_w_exp_13, 55296);
    load_txt_to_array("DRAM_bn_exp_13.txt", (data_t*)DRAM_bn_exp_13, 2304);
    load_txt_to_array("DRAM_w_dw_13.txt", (data_t*)DRAM_w_dw_13, 2985984);
    load_txt_to_array("DRAM_bn_dw_13.txt", (data_t*)DRAM_bn_dw_13, 2304);
    load_txt_to_array("DRAM_w_proj_13.txt", (data_t*)DRAM_w_proj_13, 92160);
    load_txt_to_array("DRAM_bn_proj_13.txt", (data_t*)DRAM_bn_proj_13, 640);
    load_txt_to_array("DRAM_w_exp_14.txt", (data_t*)DRAM_w_exp_14, 153600);
    load_txt_to_array("DRAM_bn_exp_14.txt", (data_t*)DRAM_bn_exp_14, 3840);
    load_txt_to_array("DRAM_w_dw_14.txt", (data_t*)DRAM_w_dw_14, 8294400);
    load_txt_to_array("DRAM_bn_dw_14.txt", (data_t*)DRAM_bn_dw_14, 3840);
    load_txt_to_array("DRAM_w_proj_14.txt", (data_t*)DRAM_w_proj_14, 153600);
    load_txt_to_array("DRAM_bn_proj_14.txt", (data_t*)DRAM_bn_proj_14, 640);
    load_txt_to_array("DRAM_w_exp_15.txt", (data_t*)DRAM_w_exp_15, 153600);
    load_txt_to_array("DRAM_bn_exp_15.txt", (data_t*)DRAM_bn_exp_15, 3840);
    load_txt_to_array("DRAM_w_dw_15.txt", (data_t*)DRAM_w_dw_15, 8294400);
    load_txt_to_array("DRAM_bn_dw_15.txt", (data_t*)DRAM_bn_dw_15, 3840);
    load_txt_to_array("DRAM_w_proj_15.txt", (data_t*)DRAM_w_proj_15, 153600);
    load_txt_to_array("DRAM_bn_proj_15.txt", (data_t*)DRAM_bn_proj_15, 640);
    load_txt_to_array("DRAM_w_exp_16.txt", (data_t*)DRAM_w_exp_16, 153600);
    load_txt_to_array("DRAM_bn_exp_16.txt", (data_t*)DRAM_bn_exp_16, 3840);
    load_txt_to_array("DRAM_w_dw_16.txt", (data_t*)DRAM_w_dw_16, 8294400);
    load_txt_to_array("DRAM_bn_dw_16.txt", (data_t*)DRAM_bn_dw_16, 3840);
    load_txt_to_array("DRAM_w_proj_16.txt", (data_t*)DRAM_w_proj_16, 307200);
    load_txt_to_array("DRAM_bn_proj_16.txt", (data_t*)DRAM_bn_proj_16, 1280);
    load_txt_to_array("DRAM_w_final.txt", (data_t*)DRAM_w_final, 409600);
    load_txt_to_array("DRAM_bn_final.txt", (data_t*)DRAM_bn_final, 5120);
    load_txt_to_array("DRAM_out.txt", (data_t*)DRAM_out, 1000);
    load_txt_to_array("DRAM_fc.txt", (data_t*)DRAM_fc, 1280000);

    top(DRAM_input, DRAM_w_stem, DRAM_bn_stem, DRAM_w_dw_0, DRAM_bn_dw_0, DRAM_w_proj_0, DRAM_bn_proj_0, DRAM_w_exp_1, DRAM_bn_exp_1, DRAM_w_dw_1, DRAM_bn_dw_1, DRAM_w_proj_1, DRAM_bn_proj_1, DRAM_w_exp_2, DRAM_bn_exp_2, DRAM_w_dw_2, DRAM_bn_dw_2, DRAM_w_proj_2, DRAM_bn_proj_2, DRAM_w_exp_3, DRAM_bn_exp_3, DRAM_w_dw_3, DRAM_bn_dw_3, DRAM_w_proj_3, DRAM_bn_proj_3, DRAM_w_exp_4, DRAM_bn_exp_4, DRAM_w_dw_4, DRAM_bn_dw_4, DRAM_w_proj_4, DRAM_bn_proj_4, DRAM_w_exp_5, DRAM_bn_exp_5, DRAM_w_dw_5, DRAM_bn_dw_5, DRAM_w_proj_5, DRAM_bn_proj_5, DRAM_w_exp_6, DRAM_bn_exp_6, DRAM_w_dw_6, DRAM_bn_dw_6, DRAM_w_proj_6, DRAM_bn_proj_6, DRAM_w_exp_7, DRAM_bn_exp_7, DRAM_w_dw_7, DRAM_bn_dw_7, DRAM_w_proj_7, DRAM_bn_proj_7, DRAM_w_exp_8, DRAM_bn_exp_8, DRAM_w_dw_8, DRAM_bn_dw_8, DRAM_w_proj_8, DRAM_bn_proj_8, DRAM_w_exp_9, DRAM_bn_exp_9, DRAM_w_dw_9, DRAM_bn_dw_9, DRAM_w_proj_9, DRAM_bn_proj_9, DRAM_w_exp_10, DRAM_bn_exp_10, DRAM_w_dw_10, DRAM_bn_dw_10, DRAM_w_proj_10, DRAM_bn_proj_10, DRAM_w_exp_11, DRAM_bn_exp_11, DRAM_w_dw_11, DRAM_bn_dw_11, DRAM_w_proj_11, DRAM_bn_proj_11, DRAM_w_exp_12, DRAM_bn_exp_12, DRAM_w_dw_12, DRAM_bn_dw_12, DRAM_w_proj_12, DRAM_bn_proj_12, DRAM_w_exp_13, DRAM_bn_exp_13, DRAM_w_dw_13, DRAM_bn_dw_13, DRAM_w_proj_13, DRAM_bn_proj_13, DRAM_w_exp_14, DRAM_bn_exp_14, DRAM_w_dw_14, DRAM_bn_dw_14, DRAM_w_proj_14, DRAM_bn_proj_14, DRAM_w_exp_15, DRAM_bn_exp_15, DRAM_w_dw_15, DRAM_bn_dw_15, DRAM_w_proj_15, DRAM_bn_proj_15, DRAM_w_exp_16, DRAM_bn_exp_16, DRAM_w_dw_16, DRAM_bn_dw_16, DRAM_w_proj_16, DRAM_bn_proj_16, DRAM_w_final, DRAM_bn_final, DRAM_out, DRAM_fc);

    // Write contents of DRAM_out to DRAM_out_output.txt
    {
        FILE *fp = fopen("DRAM_out_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%f ", (float)((data_t*)DRAM_out)[i]);
            }
            fclose(fp);
        }
    }

    return 0;
}