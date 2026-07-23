#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ap_fixed.h>
#include "top.h"  // Include the top function declaration

#ifndef VERIF_RTOL
#define VERIF_RTOL 1e-3
#endif
#ifndef VERIF_ATOL
#define VERIF_ATOL 1e-5
#endif

typedef float data_t;

data_t DRAM_image_input[64][14][14];
data_t DRAM_conv_weight[64][64][3][3];
data_t DRAM_conv_bias[64];
data_t DRAM_image_output[64][14][14];

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
    load_txt_to_array("DRAM_image_input.txt", (data_t*)DRAM_image_input, 12544);
    load_txt_to_array("DRAM_conv_weight.txt", (data_t*)DRAM_conv_weight, 36864);
    load_txt_to_array("DRAM_conv_bias.txt", (data_t*)DRAM_conv_bias, 64);
    load_txt_to_array("DRAM_image_output.txt", (data_t*)DRAM_image_output, 12544);

    top(DRAM_image_input, DRAM_conv_weight, DRAM_conv_bias, DRAM_image_output);

    double verif_max_abs = 0.0, verif_max_rel = 0.0;
    long verif_n_mismatch = 0, verif_n_total = 0, verif_n_checked = 0;

    // Write contents of DRAM_image_output to DRAM_image_output_output.txt
    {
        FILE *fp = fopen("DRAM_image_output_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 12544; i++) {
                fprintf(fp, "%f ", (float)((data_t*)DRAM_image_output)[i]);
            }
            fclose(fp);
        }
    }
    // Compare DRAM_image_output against DRAM_image_output.golden.txt if present
    {
        FILE *gf = fopen("DRAM_image_output.golden.txt", "r");
        if (gf != NULL) {
            for (int i = 0; i < 12544; i++) {
                float gv;
                if (fscanf(gf, "%f", &gv) != 1) break;
                double a = (double)(float)((data_t*)DRAM_image_output)[i];
                double b = (double)gv;
                double abs_err = fabs(a - b);
                double rel_err = abs_err / (fabs(b) + 1e-12);
                double tol = VERIF_ATOL + VERIF_RTOL * fabs(b);
                int bad = (!(abs_err <= tol)) || (!isfinite(a));
                if (abs_err > verif_max_abs) verif_max_abs = abs_err;
                if (rel_err > verif_max_rel) verif_max_rel = rel_err;
                if (bad) verif_n_mismatch++;
                verif_n_total++;
            }
            fclose(gf);
            verif_n_checked++;
        }
    }

    if (verif_n_checked == 0) {
        printf("VERIFICATION: SKIP (no golden files found)\n");
        return 0;
    }
    if (verif_n_mismatch == 0) {
        printf("VERIFICATION: PASS (max_abs=%g, max_rel=%g, n=%ld)\n", verif_max_abs, verif_max_rel, verif_n_total);
        return 0;
    }
    printf("VERIFICATION: FAIL (max_abs=%g, max_rel=%g, n_mismatch=%ld/%ld)\n", verif_max_abs, verif_max_rel, verif_n_mismatch, verif_n_total);
    return 1;
}