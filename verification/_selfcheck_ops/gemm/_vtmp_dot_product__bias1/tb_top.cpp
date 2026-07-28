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

data_t DRAM_1[64];
data_t DRAM_2[64];
data_t DRAM_3[1];
data_t DRAM_4[1];

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
    load_txt_to_array("DRAM_1.txt", (data_t*)DRAM_1, 64);
    load_txt_to_array("DRAM_2.txt", (data_t*)DRAM_2, 64);
    load_txt_to_array("DRAM_3.txt", (data_t*)DRAM_3, 1);
    load_txt_to_array("DRAM_4.txt", (data_t*)DRAM_4, 1);

    top(DRAM_1, DRAM_2, DRAM_3, DRAM_4);

    double verif_max_abs = 0.0, verif_max_rel = 0.0;
    long verif_n_mismatch = 0, verif_n_total = 0, verif_n_checked = 0;

    // Write contents of DRAM_4 to DRAM_4_output.txt
    {
        FILE *fp = fopen("DRAM_4_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1; i++) {
                fprintf(fp, "%f ", (float)((data_t*)DRAM_4)[i]);
            }
            fclose(fp);
        }
    }
    // Compare DRAM_4 against DRAM_4.golden.txt if present
    {
        FILE *gf = fopen("DRAM_4.golden.txt", "r");
        if (gf != NULL) {
            static float verif_golden[1];
            long gn = 0;
            for (int i = 0; i < 1; i++) {
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
                double a = (double)(float)((data_t*)DRAM_4)[i];
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