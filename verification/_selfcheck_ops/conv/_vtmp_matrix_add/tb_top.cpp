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
#ifndef VERIF_ATOL_SCALE
#define VERIF_ATOL_SCALE 5e-5
#endif

typedef float data_t;

data_t DRAM_input_1[8][16][16];
data_t DRAM_input_2[8][16][16];
data_t DRAM_output[8][16][16];

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
    load_txt_to_array("DRAM_input_1.txt", (data_t*)DRAM_input_1, 2048);
    load_txt_to_array("DRAM_input_2.txt", (data_t*)DRAM_input_2, 2048);
    load_txt_to_array("DRAM_output.txt", (data_t*)DRAM_output, 2048);

    top(DRAM_input_1, DRAM_input_2, DRAM_output);

    double verif_max_abs = 0.0, verif_max_rel = 0.0;
    long verif_n_mismatch = 0, verif_n_total = 0, verif_n_checked = 0;

    // Write contents of DRAM_output to DRAM_output_output.txt
    {
        FILE *fp = fopen("DRAM_output_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 2048; i++) {
                fprintf(fp, "%f ", (float)((data_t*)DRAM_output)[i]);
            }
            fclose(fp);
        }
    }
    // Compare DRAM_output against DRAM_output.golden.txt if present
    {
        FILE *gf = fopen("DRAM_output.golden.txt", "r");
        if (gf != NULL) {
            static float verif_golden[2048];
            long gn = 0;
            for (int i = 0; i < 2048; i++) {
                float gv;
                if (fscanf(gf, "%f", &gv) != 1) break;
                verif_golden[i] = gv; gn++;
            }
            double verif_scale = 0.0;
            for (long i = 0; i < gn; i++) {
                double m = fabs((double)verif_golden[i]);
                if (m > verif_scale) verif_scale = m;
            }
            double verif_atol = VERIF_ATOL + VERIF_ATOL_SCALE * verif_scale;
            for (long i = 0; i < gn; i++) {
                double a = (double)(float)((data_t*)DRAM_output)[i];
                double b = (double)verif_golden[i];
                double abs_err = fabs(a - b);
                double rel_err = abs_err / (fabs(b) + 1e-12);
                double tol = verif_atol + VERIF_RTOL * fabs(b);
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