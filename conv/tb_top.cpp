#include <stdio.h>
#include <stdlib.h>
#include <ap_fixed.h>
#include "top.h"  // Include the top function declaration

typedef ap_fixed<16, 5> data_t;

data_t DRAM_1[2][4][4];
data_t DRAM_2[2][2][3][3];
data_t DRAM_3[2];
data_t DRAM_4[4][2];
data_t DRAM_5[2][2][2];

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
    load_txt_to_array("DRAM_1.txt", (data_t*)DRAM_1, 32);
    load_txt_to_array("DRAM_2.txt", (data_t*)DRAM_2, 36);
    load_txt_to_array("DRAM_3.txt", (data_t*)DRAM_3, 2);
    load_txt_to_array("DRAM_4.txt", (data_t*)DRAM_4, 8);
    load_txt_to_array("DRAM_5.txt", (data_t*)DRAM_5, 8);

    top(DRAM_1, DRAM_2, DRAM_3, DRAM_4, DRAM_5);

    // Write contents of DRAM_4 to DRAM_4_output.txt
    {
        FILE *fp = fopen("DRAM_4_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 8; i++) {
                fprintf(fp, "%f ", (float)((data_t*)DRAM_4)[i]);
            }
            fclose(fp);
        }
    }

    // Write contents of DRAM_5 to DRAM_5_output.txt
    {
        FILE *fp = fopen("DRAM_5_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 8; i++) {
                fprintf(fp, "%f ", (float)((data_t*)DRAM_5)[i]);
            }
            fclose(fp);
        }
    }

    return 0;
}