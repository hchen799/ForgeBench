#include <stdio.h>
#include <stdlib.h>
#include <ap_fixed.h>
#include "top.h"  // Include the top function declaration

typedef ap_fixed<16,5> data_t;

data_t DRAM_input[3][224][224];
data_t DRAM_w_stem[64][3][7][7];
data_t DRAM_bn_stem[4][64];
data_t DRAM_w_s1_b0_1[64][64][3][3];
data_t DRAM_bn_s1_b0_1[4][64];
data_t DRAM_w_s1_b0_2[64][64][3][3];
data_t DRAM_bn_s1_b0_2[4][64];
data_t DRAM_w_s1_b1_1[64][64][3][3];
data_t DRAM_bn_s1_b1_1[4][64];
data_t DRAM_w_s1_b1_2[64][64][3][3];
data_t DRAM_bn_s1_b1_2[4][64];
data_t DRAM_w_s1_b2_1[64][64][3][3];
data_t DRAM_bn_s1_b2_1[4][64];
data_t DRAM_w_s1_b2_2[64][64][3][3];
data_t DRAM_bn_s1_b2_2[4][64];
data_t DRAM_w_s2_b0_1[128][64][3][3];
data_t DRAM_bn_s2_b0_1[4][128];
data_t DRAM_w_s2_b0_2[128][128][3][3];
data_t DRAM_bn_s2_b0_2[4][128];
data_t DRAM_w_s2_b0_down[128][64][1][1];
data_t DRAM_w_s2_b1_1[128][128][3][3];
data_t DRAM_bn_s2_b1_1[4][128];
data_t DRAM_w_s2_b1_2[128][128][3][3];
data_t DRAM_bn_s2_b1_2[4][128];
data_t DRAM_w_s2_b2_1[128][128][3][3];
data_t DRAM_bn_s2_b2_1[4][128];
data_t DRAM_w_s2_b2_2[128][128][3][3];
data_t DRAM_bn_s2_b2_2[4][128];
data_t DRAM_w_s2_b3_1[128][128][3][3];
data_t DRAM_bn_s2_b3_1[4][128];
data_t DRAM_w_s2_b3_2[128][128][3][3];
data_t DRAM_bn_s2_b3_2[4][128];
data_t DRAM_w_s3_b0_1[256][128][3][3];
data_t DRAM_bn_s3_b0_1[4][256];
data_t DRAM_w_s3_b0_2[256][256][3][3];
data_t DRAM_bn_s3_b0_2[4][256];
data_t DRAM_w_s3_b0_down[256][128][1][1];
data_t DRAM_w_s3_b1_1[256][256][3][3];
data_t DRAM_bn_s3_b1_1[4][256];
data_t DRAM_w_s3_b1_2[256][256][3][3];
data_t DRAM_bn_s3_b1_2[4][256];
data_t DRAM_w_s3_b2_1[256][256][3][3];
data_t DRAM_bn_s3_b2_1[4][256];
data_t DRAM_w_s3_b2_2[256][256][3][3];
data_t DRAM_bn_s3_b2_2[4][256];
data_t DRAM_w_s3_b3_1[256][256][3][3];
data_t DRAM_bn_s3_b3_1[4][256];
data_t DRAM_w_s3_b3_2[256][256][3][3];
data_t DRAM_bn_s3_b3_2[4][256];
data_t DRAM_w_s3_b4_1[256][256][3][3];
data_t DRAM_bn_s3_b4_1[4][256];
data_t DRAM_w_s3_b4_2[256][256][3][3];
data_t DRAM_bn_s3_b4_2[4][256];
data_t DRAM_w_s3_b5_1[256][256][3][3];
data_t DRAM_bn_s3_b5_1[4][256];
data_t DRAM_w_s3_b5_2[256][256][3][3];
data_t DRAM_bn_s3_b5_2[4][256];
data_t DRAM_w_s4_b0_1[512][256][3][3];
data_t DRAM_bn_s4_b0_1[4][512];
data_t DRAM_w_s4_b0_2[512][512][3][3];
data_t DRAM_bn_s4_b0_2[4][512];
data_t DRAM_w_s4_b0_down[512][256][1][1];
data_t DRAM_w_s4_b1_1[512][512][3][3];
data_t DRAM_bn_s4_b1_1[4][512];
data_t DRAM_w_s4_b1_2[512][512][3][3];
data_t DRAM_bn_s4_b1_2[4][512];
data_t DRAM_w_s4_b2_1[512][512][3][3];
data_t DRAM_bn_s4_b2_1[4][512];
data_t DRAM_w_s4_b2_2[512][512][3][3];
data_t DRAM_bn_s4_b2_2[4][512];
data_t DRAM_out[1000][1][1];
data_t DRAM_fc[1000][512][1][1];

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
    load_txt_to_array("DRAM_w_stem.txt", (data_t*)DRAM_w_stem, 9408);
    load_txt_to_array("DRAM_bn_stem.txt", (data_t*)DRAM_bn_stem, 256);
    load_txt_to_array("DRAM_w_s1_b0_1.txt", (data_t*)DRAM_w_s1_b0_1, 36864);
    load_txt_to_array("DRAM_bn_s1_b0_1.txt", (data_t*)DRAM_bn_s1_b0_1, 256);
    load_txt_to_array("DRAM_w_s1_b0_2.txt", (data_t*)DRAM_w_s1_b0_2, 36864);
    load_txt_to_array("DRAM_bn_s1_b0_2.txt", (data_t*)DRAM_bn_s1_b0_2, 256);
    load_txt_to_array("DRAM_w_s1_b1_1.txt", (data_t*)DRAM_w_s1_b1_1, 36864);
    load_txt_to_array("DRAM_bn_s1_b1_1.txt", (data_t*)DRAM_bn_s1_b1_1, 256);
    load_txt_to_array("DRAM_w_s1_b1_2.txt", (data_t*)DRAM_w_s1_b1_2, 36864);
    load_txt_to_array("DRAM_bn_s1_b1_2.txt", (data_t*)DRAM_bn_s1_b1_2, 256);
    load_txt_to_array("DRAM_w_s1_b2_1.txt", (data_t*)DRAM_w_s1_b2_1, 36864);
    load_txt_to_array("DRAM_bn_s1_b2_1.txt", (data_t*)DRAM_bn_s1_b2_1, 256);
    load_txt_to_array("DRAM_w_s1_b2_2.txt", (data_t*)DRAM_w_s1_b2_2, 36864);
    load_txt_to_array("DRAM_bn_s1_b2_2.txt", (data_t*)DRAM_bn_s1_b2_2, 256);
    load_txt_to_array("DRAM_w_s2_b0_1.txt", (data_t*)DRAM_w_s2_b0_1, 73728);
    load_txt_to_array("DRAM_bn_s2_b0_1.txt", (data_t*)DRAM_bn_s2_b0_1, 512);
    load_txt_to_array("DRAM_w_s2_b0_2.txt", (data_t*)DRAM_w_s2_b0_2, 147456);
    load_txt_to_array("DRAM_bn_s2_b0_2.txt", (data_t*)DRAM_bn_s2_b0_2, 512);
    load_txt_to_array("DRAM_w_s2_b0_down.txt", (data_t*)DRAM_w_s2_b0_down, 8192);
    load_txt_to_array("DRAM_w_s2_b1_1.txt", (data_t*)DRAM_w_s2_b1_1, 147456);
    load_txt_to_array("DRAM_bn_s2_b1_1.txt", (data_t*)DRAM_bn_s2_b1_1, 512);
    load_txt_to_array("DRAM_w_s2_b1_2.txt", (data_t*)DRAM_w_s2_b1_2, 147456);
    load_txt_to_array("DRAM_bn_s2_b1_2.txt", (data_t*)DRAM_bn_s2_b1_2, 512);
    load_txt_to_array("DRAM_w_s2_b2_1.txt", (data_t*)DRAM_w_s2_b2_1, 147456);
    load_txt_to_array("DRAM_bn_s2_b2_1.txt", (data_t*)DRAM_bn_s2_b2_1, 512);
    load_txt_to_array("DRAM_w_s2_b2_2.txt", (data_t*)DRAM_w_s2_b2_2, 147456);
    load_txt_to_array("DRAM_bn_s2_b2_2.txt", (data_t*)DRAM_bn_s2_b2_2, 512);
    load_txt_to_array("DRAM_w_s2_b3_1.txt", (data_t*)DRAM_w_s2_b3_1, 147456);
    load_txt_to_array("DRAM_bn_s2_b3_1.txt", (data_t*)DRAM_bn_s2_b3_1, 512);
    load_txt_to_array("DRAM_w_s2_b3_2.txt", (data_t*)DRAM_w_s2_b3_2, 147456);
    load_txt_to_array("DRAM_bn_s2_b3_2.txt", (data_t*)DRAM_bn_s2_b3_2, 512);
    load_txt_to_array("DRAM_w_s3_b0_1.txt", (data_t*)DRAM_w_s3_b0_1, 294912);
    load_txt_to_array("DRAM_bn_s3_b0_1.txt", (data_t*)DRAM_bn_s3_b0_1, 1024);
    load_txt_to_array("DRAM_w_s3_b0_2.txt", (data_t*)DRAM_w_s3_b0_2, 589824);
    load_txt_to_array("DRAM_bn_s3_b0_2.txt", (data_t*)DRAM_bn_s3_b0_2, 1024);
    load_txt_to_array("DRAM_w_s3_b0_down.txt", (data_t*)DRAM_w_s3_b0_down, 32768);
    load_txt_to_array("DRAM_w_s3_b1_1.txt", (data_t*)DRAM_w_s3_b1_1, 589824);
    load_txt_to_array("DRAM_bn_s3_b1_1.txt", (data_t*)DRAM_bn_s3_b1_1, 1024);
    load_txt_to_array("DRAM_w_s3_b1_2.txt", (data_t*)DRAM_w_s3_b1_2, 589824);
    load_txt_to_array("DRAM_bn_s3_b1_2.txt", (data_t*)DRAM_bn_s3_b1_2, 1024);
    load_txt_to_array("DRAM_w_s3_b2_1.txt", (data_t*)DRAM_w_s3_b2_1, 589824);
    load_txt_to_array("DRAM_bn_s3_b2_1.txt", (data_t*)DRAM_bn_s3_b2_1, 1024);
    load_txt_to_array("DRAM_w_s3_b2_2.txt", (data_t*)DRAM_w_s3_b2_2, 589824);
    load_txt_to_array("DRAM_bn_s3_b2_2.txt", (data_t*)DRAM_bn_s3_b2_2, 1024);
    load_txt_to_array("DRAM_w_s3_b3_1.txt", (data_t*)DRAM_w_s3_b3_1, 589824);
    load_txt_to_array("DRAM_bn_s3_b3_1.txt", (data_t*)DRAM_bn_s3_b3_1, 1024);
    load_txt_to_array("DRAM_w_s3_b3_2.txt", (data_t*)DRAM_w_s3_b3_2, 589824);
    load_txt_to_array("DRAM_bn_s3_b3_2.txt", (data_t*)DRAM_bn_s3_b3_2, 1024);
    load_txt_to_array("DRAM_w_s3_b4_1.txt", (data_t*)DRAM_w_s3_b4_1, 589824);
    load_txt_to_array("DRAM_bn_s3_b4_1.txt", (data_t*)DRAM_bn_s3_b4_1, 1024);
    load_txt_to_array("DRAM_w_s3_b4_2.txt", (data_t*)DRAM_w_s3_b4_2, 589824);
    load_txt_to_array("DRAM_bn_s3_b4_2.txt", (data_t*)DRAM_bn_s3_b4_2, 1024);
    load_txt_to_array("DRAM_w_s3_b5_1.txt", (data_t*)DRAM_w_s3_b5_1, 589824);
    load_txt_to_array("DRAM_bn_s3_b5_1.txt", (data_t*)DRAM_bn_s3_b5_1, 1024);
    load_txt_to_array("DRAM_w_s3_b5_2.txt", (data_t*)DRAM_w_s3_b5_2, 589824);
    load_txt_to_array("DRAM_bn_s3_b5_2.txt", (data_t*)DRAM_bn_s3_b5_2, 1024);
    load_txt_to_array("DRAM_w_s4_b0_1.txt", (data_t*)DRAM_w_s4_b0_1, 1179648);
    load_txt_to_array("DRAM_bn_s4_b0_1.txt", (data_t*)DRAM_bn_s4_b0_1, 2048);
    load_txt_to_array("DRAM_w_s4_b0_2.txt", (data_t*)DRAM_w_s4_b0_2, 2359296);
    load_txt_to_array("DRAM_bn_s4_b0_2.txt", (data_t*)DRAM_bn_s4_b0_2, 2048);
    load_txt_to_array("DRAM_w_s4_b0_down.txt", (data_t*)DRAM_w_s4_b0_down, 131072);
    load_txt_to_array("DRAM_w_s4_b1_1.txt", (data_t*)DRAM_w_s4_b1_1, 2359296);
    load_txt_to_array("DRAM_bn_s4_b1_1.txt", (data_t*)DRAM_bn_s4_b1_1, 2048);
    load_txt_to_array("DRAM_w_s4_b1_2.txt", (data_t*)DRAM_w_s4_b1_2, 2359296);
    load_txt_to_array("DRAM_bn_s4_b1_2.txt", (data_t*)DRAM_bn_s4_b1_2, 2048);
    load_txt_to_array("DRAM_w_s4_b2_1.txt", (data_t*)DRAM_w_s4_b2_1, 2359296);
    load_txt_to_array("DRAM_bn_s4_b2_1.txt", (data_t*)DRAM_bn_s4_b2_1, 2048);
    load_txt_to_array("DRAM_w_s4_b2_2.txt", (data_t*)DRAM_w_s4_b2_2, 2359296);
    load_txt_to_array("DRAM_bn_s4_b2_2.txt", (data_t*)DRAM_bn_s4_b2_2, 2048);
    load_txt_to_array("DRAM_out.txt", (data_t*)DRAM_out, 1000);
    load_txt_to_array("DRAM_fc.txt", (data_t*)DRAM_fc, 512000);

    top(DRAM_input, DRAM_w_stem, DRAM_bn_stem, DRAM_w_s1_b0_1, DRAM_bn_s1_b0_1, DRAM_w_s1_b0_2, DRAM_bn_s1_b0_2, DRAM_w_s1_b1_1, DRAM_bn_s1_b1_1, DRAM_w_s1_b1_2, DRAM_bn_s1_b1_2, DRAM_w_s1_b2_1, DRAM_bn_s1_b2_1, DRAM_w_s1_b2_2, DRAM_bn_s1_b2_2, DRAM_w_s2_b0_1, DRAM_bn_s2_b0_1, DRAM_w_s2_b0_2, DRAM_bn_s2_b0_2, DRAM_w_s2_b0_down, DRAM_w_s2_b1_1, DRAM_bn_s2_b1_1, DRAM_w_s2_b1_2, DRAM_bn_s2_b1_2, DRAM_w_s2_b2_1, DRAM_bn_s2_b2_1, DRAM_w_s2_b2_2, DRAM_bn_s2_b2_2, DRAM_w_s2_b3_1, DRAM_bn_s2_b3_1, DRAM_w_s2_b3_2, DRAM_bn_s2_b3_2, DRAM_w_s3_b0_1, DRAM_bn_s3_b0_1, DRAM_w_s3_b0_2, DRAM_bn_s3_b0_2, DRAM_w_s3_b0_down, DRAM_w_s3_b1_1, DRAM_bn_s3_b1_1, DRAM_w_s3_b1_2, DRAM_bn_s3_b1_2, DRAM_w_s3_b2_1, DRAM_bn_s3_b2_1, DRAM_w_s3_b2_2, DRAM_bn_s3_b2_2, DRAM_w_s3_b3_1, DRAM_bn_s3_b3_1, DRAM_w_s3_b3_2, DRAM_bn_s3_b3_2, DRAM_w_s3_b4_1, DRAM_bn_s3_b4_1, DRAM_w_s3_b4_2, DRAM_bn_s3_b4_2, DRAM_w_s3_b5_1, DRAM_bn_s3_b5_1, DRAM_w_s3_b5_2, DRAM_bn_s3_b5_2, DRAM_w_s4_b0_1, DRAM_bn_s4_b0_1, DRAM_w_s4_b0_2, DRAM_bn_s4_b0_2, DRAM_w_s4_b0_down, DRAM_w_s4_b1_1, DRAM_bn_s4_b1_1, DRAM_w_s4_b1_2, DRAM_bn_s4_b1_2, DRAM_w_s4_b2_1, DRAM_bn_s4_b2_1, DRAM_w_s4_b2_2, DRAM_bn_s4_b2_2, DRAM_out, DRAM_fc);

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