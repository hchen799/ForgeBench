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

typedef ap_fixed<16, 5> data_t;

#define C 64
#define H 28
#define W 28

data_t FM_buffer_1[C][H][W];
data_t FM_buffer_2[C][H][W];
data_t FM_buffer_3[C][H][W];
data_t FM_buffer_4[C][H][W];
data_t FM_buffer_5[C][H][W];


void load_feature_map(data_t input_dram[C][H][W], data_t input_buffer[C][H][W])
{
    #pragma HLS inline off
    for (int c = 0; c < C; c++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                input_buffer[c][h][w] = input_dram[c][h][w];
            }
        }
    }
}

void store_feature_map(data_t output_buffer[C][H][W], data_t output_dram[C][H][W])
{
    #pragma HLS inline off
    for (int c = 0; c < C; c++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                output_dram[c][h][w] = output_buffer[c][h][w];
            }
        }
    }
}

void compute_exp(data_t input[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = hls::exp(input[i][j][k]);
            }
        }
    }

}

void negative(data_t input[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = -input[i][j][k];
            }
        }
    }
}

void compute_div(data_t input_1[C][H][W], data_t input_2[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = input_1[i][j][k]/input_2[i][j][k];
            }
        }
    }
}

void compute_add(data_t input_1[C][H][W], data_t input_2[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = input_1[i][j][k] + input_2[i][j][k];
            }
        }
    }
}

void compute_mul(data_t input_1[C][H][W], data_t input_2[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = input_1[i][j][k] * input_2[i][j][k];
            }
        }
    }
}

void set_value(data_t bram[C][H][W], data_t value)
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                bram[i][j][k] = value;
            }
        }
    }
}


void select(data_t input_1[C][H][W], data_t input_2[C][H][W], data_t output[C][H][W],  data_t threshold)
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                if (input_1[i][j][k] >= threshold) {
                    output[i][j][k] = input_1[i][j][k];
                } else {
                    output[i][j][k] = input_2[i][j][k];
                }
            }
        }
    }
}

void absolute(data_t input[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = (input[i][j][k] < (data_t) 0) ? (data_t) -input[i][j][k] : input[i][j][k];
            }
        }
    }
}

// output = (sign >= 0) ? pos : neg
void select_sign(data_t sign[C][H][W], data_t pos[C][H][W], data_t neg[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = (sign[i][j][k] >= (data_t) 0) ? pos[i][j][k] : neg[i][j][k];
            }
        }
    }
}

// Numerically stable: exp is only taken of -|x| (<= 0), so it cannot overflow the format.
// sigmoid(x) = 1 / (1 + e) for x >= 0 and e / (1 + e) for x < 0, with e = exp(-|x|)
void sigmoid (data_t input[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1);                   // x
    absolute(FM_buffer_1, FM_buffer_2);
    negative(FM_buffer_2, FM_buffer_3);                     // -|x|
    compute_exp(FM_buffer_3, FM_buffer_2);                  // e
    set_value(FM_buffer_3, (data_t) 1);
    compute_add(FM_buffer_3, FM_buffer_2, FM_buffer_4);     // 1 + e
    compute_div(FM_buffer_3, FM_buffer_4, FM_buffer_5);     // sigmoid(|x|)
    compute_div(FM_buffer_2, FM_buffer_4, FM_buffer_3);     // sigmoid(-|x|)
    select_sign(FM_buffer_1, FM_buffer_5, FM_buffer_3, FM_buffer_2);
    store_feature_map(FM_buffer_2, output);
}

// tanh(x) = sign(x) * (1 - t) / (1 + t), with t = exp(-2|x|) = exp(-|x|)^2 (2|x| itself could overflow the format)
void tanh (data_t input[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1);                   // x
    absolute(FM_buffer_1, FM_buffer_2);
    negative(FM_buffer_2, FM_buffer_3);                     // -|x|
    compute_exp(FM_buffer_3, FM_buffer_2);                  // exp(-|x|)
    load_feature_map(FM_buffer_2, FM_buffer_3);             // copy
    compute_mul(FM_buffer_2, FM_buffer_3, FM_buffer_4);     // t
    set_value(FM_buffer_5, (data_t) 1);
    compute_add(FM_buffer_5, FM_buffer_4, FM_buffer_2);     // 1 + t
    negative(FM_buffer_4, FM_buffer_3);
    compute_add(FM_buffer_5, FM_buffer_3, FM_buffer_4);     // 1 - t
    compute_div(FM_buffer_4, FM_buffer_2, FM_buffer_3);     // tanh(|x|)
    negative(FM_buffer_3, FM_buffer_4);                     // tanh(-|x|)
    select_sign(FM_buffer_1, FM_buffer_3, FM_buffer_4, FM_buffer_2);
    store_feature_map(FM_buffer_2, output);
}

void elu(data_t input[C][H][W], data_t output[C][H][W], data_t alpha)
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1);
    compute_exp(FM_buffer_1, FM_buffer_2);
    set_value(FM_buffer_3, (data_t) -1);
    compute_add(FM_buffer_2, FM_buffer_3, FM_buffer_4);
    set_value(FM_buffer_3, alpha);
    compute_mul(FM_buffer_3, FM_buffer_4, FM_buffer_2);
    select(FM_buffer_1, FM_buffer_2, FM_buffer_3, (data_t) 0);
    store_feature_map(FM_buffer_3, output);
}




void top(
    data_t input_sigmoid[C][H][W],
    data_t output_sigmoid[C][H][W], 
    data_t input_tanh[C][H][W],
    data_t output_tanh[C][H][W],
    data_t input_elu[C][H][W],
    data_t output_elu[C][H][W]
)
{
    #pragma HLS interface m_axi port=input_sigmoid offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_sigmoid offset=slave bundle=mem2

    #pragma HLS interface m_axi port=input_tanh offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_tanh offset=slave bundle=mem2

    #pragma HLS interface m_axi port=input_elu offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_elu offset=slave bundle=mem2


    #pragma HLS allocation function instances= load_feature_map limit=1
    #pragma HLS allocation function instances= store_feature_map limit=1
    #pragma HLS allocation function instances= compute_exp limit=1
    #pragma HLS allocation function instances= negative limit=1
    #pragma HLS allocation function instances= compute_div limit=1
    #pragma HLS allocation function instances= compute_add limit=1
    #pragma HLS allocation function instances= compute_mul limit=1
    #pragma HLS allocation function instances= set_value limit=1
    #pragma HLS allocation function instances= select limit=1
    #pragma HLS allocation function instances= absolute limit=1
    #pragma HLS allocation function instances= select_sign limit=1
   
    // //sigmoid
    // load_feature_map(input_sigmoid, FM_buffer_1);
    // negative(FM_buffer_1, FM_buffer_2);
    // compute_exp(FM_buffer_2, FM_buffer_1);
    // set_value(FM_buffer_3, (data_t) 1);
    // compute_add(FM_buffer_3, FM_buffer_1, FM_buffer_2);
    // compute_div(FM_buffer_3, FM_buffer_2, FM_buffer_1);
    // store_feature_map(FM_buffer_1, output_sigmoid);

    // //tanh
    // load_feature_map(input_tanh, FM_buffer_1);
    // compute_exp(FM_buffer_1, FM_buffer_2); //exp(x)
    // negative(FM_buffer_1, FM_buffer_3);
    // compute_exp(FM_buffer_3, FM_buffer_1); //exp(-x)
    // compute_add(FM_buffer_2, FM_buffer_1, FM_buffer_3);
    // negative(FM_buffer_1, FM_buffer_4);
    // compute_add(FM_buffer_2, FM_buffer_4, FM_buffer_1);
    // compute_div(FM_buffer_1, FM_buffer_3, FM_buffer_2);
    // store_feature_map(FM_buffer_2, output_tanh);

    // //elu
    // load_feature_map(input_elu, FM_buffer_1);
    // compute_exp(FM_buffer_1, FM_buffer_2);
    // set_value(FM_buffer_3, (data_t) -1);
    // compute_add(FM_buffer_2, FM_buffer_3, FM_buffer_4);
    // set_value(FM_buffer_3, (data_t) 0.5);
    // compute_mul(FM_buffer_3, FM_buffer_4, FM_buffer_2);
    // select(FM_buffer_1, FM_buffer_2, FM_buffer_3, (data_t) 0);
    // store_feature_map(FM_buffer_3, output_elu);
    // sigmoid(input_sigmoid, output_sigmoid);
    tanh(input_tanh, output_tanh);
    // elu(input_elu, output_elu, (data_t) 0.5);
}
