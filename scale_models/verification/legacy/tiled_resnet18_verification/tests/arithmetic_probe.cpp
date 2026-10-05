// Built against the installed AMD headers, never the float verification shim.
#include "top.cpp"
#include <iostream>
#include <string>
#include <cstdint>

data_t from_code(long long raw) {
    data_t x;
    x.range(15, 0) = raw;
    return x;
}
long long code(data_t x) { return static_cast<ap_int<16> >(x.range(15, 0)).to_int64(); }
long long acc_code(acc_t x) { return static_cast<ap_int<32> >(x.range(31, 0)).to_int64(); }
static data_t feature[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W];
static data_t result[MAX_FEAT_C][MAX_FEAT_H][MAX_FEAT_W];
static data_t bn_params[4][1];
static data_t fc_input[512], fc_weights[1][512], fc_output[1];

int main() {
    std::string operation;
    while (std::cin >> operation) {
        if (operation == "quant") {
            double x; std::cin >> x;
            std::cout << code((data_t)x) << '\n';
        } else if (operation == "shift") {
            long long raw; int shift; std::cin >> raw >> shift;
            acc_t value; value.range(31, 0) = raw;
            std::cout << acc_code(apply_power_of_two_shift(value, shift)) << '\n';
        } else if (operation == "sqrt") {
            long long raw; std::cin >> raw;
            acc_t var = (acc_t)from_code(raw);
            auto root = hls::sqrt(var + (acc_t)1e-5);
            std::cout << static_cast<ap_int<33> >(root.range(32, 0)).to_int64() << '\n';
        } else if (operation == "bn") {
            long long x, g, b, m, v; std::cin >> x >> g >> b >> m >> v;
            feature[0][0][0] = from_code(x);
            bn_params[0][0] = from_code(g); bn_params[1][0] = from_code(b);
            bn_params[2][0] = from_code(m); bn_params[3][0] = from_code(v);
            batchnorm_tiled_runtime<1, MAX_FEAT_C, MAX_FEAT_H, MAX_FEAT_W>(1, 1, feature, bn_params, result);
            std::cout << code(result[0][0][0]) << '\n';
        } else if (operation == "gap") {
            long long raw; std::cin >> raw;
            for (int h = 0; h < 7; ++h) for (int w = 0; w < 7; ++w) feature[0][h][w] = from_code(raw);
            data_t pooled[1]; global_avgpool_runtime<1>(feature, 7, 7, pooled);
            std::cout << code(pooled[0]) << '\n';
        } else if (operation == "fc") {
            int mode; std::cin >> mode;
            for (int i = 0; i < 512; ++i) {
                fc_input[i] = (data_t)1;
                fc_weights[0][i] = (data_t)(i < 128 ? (mode == 0 ? 3 : 8) : (i < 256 && mode == 0 ? -1 : 0));
            }
            fc_tiled_runtime<1, 512>(fc_input, fc_weights, mode == 0 ? 5 : 7, fc_output);
            std::cout << code(fc_output[0]) << '\n';
        } else if (operation == "fc_tail") {
            static data_t inputs[256], weights[1000][256], outputs[1000];
            for (int i = 0; i < 256; ++i) inputs[i] = (data_t)1;
            for (int o = 0; o < 1000; ++o)
                for (int i = 0; i < 256; ++i)
                    weights[o][i] = o == 0 ? (data_t)(i < 128 ? 3 : -1)
                                          : from_code(i < 128 ? 0 : o % 7 - 3);
            fc_tiled_runtime<1000, 256>(inputs, weights, 0, outputs);
            for (int o = 0; o < 1000; ++o) std::cout << code(outputs[o]) << '\n';
        } else if (operation == "conv") {
            // Tile edges, a partial input chunk, both strides, and cancellation.
            int stride; std::cin >> stride;
            static data_t weights[2][129][3][3];
            for (int c = 0; c < 129; ++c)
                for (int h = 0; h < 17; ++h)
                    for (int w = 0; w < 17; ++w)
                        feature[c][h][w] = from_code((c * 7 + h * 13 + w * 3) % 4097 - 2048);
            for (int o = 0; o < 2; ++o)
                for (int c = 0; c < 129; ++c)
                    for (int h = 0; h < 3; ++h)
                        for (int w = 0; w < 3; ++w)
                            weights[o][c][h][w] = from_code((o * 31 + c * 3 + h * 7 + w) % 129 - 64);
            conv3x3_tiled_runtime<129, 2>(17, 17, stride, feature, weights, 0, result);
            int size = out_dim(17, 1, stride, 3);
            for (int c = 0; c < 2; ++c) for (int h = 0; h < size; ++h) for (int w = 0; w < size; ++w)
                std::cout << code(result[c][h][w]) << '\n';
        } else {
            return 2;
        }
    }
}
