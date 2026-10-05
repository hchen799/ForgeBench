"""Simulation-only C++ support. No PyTorch dependency in the HLS generator."""


def is_input(name):
    return name == "DRAM_input" or name == "DRAM_fc" or name.startswith(("DRAM_w_", "DRAM_bn_"))


def make_trace_header():
    return r'''#pragma once
#if defined(RESNET18_VERIFY_TRACE) && !defined(__SYNTHESIS__)
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdint>
namespace verification {
extern std::string case_dir, context, operation;
extern std::ofstream exponents;
inline void write_code(std::ostream &f, const data_t &v) {
    unsigned bits = v.range(15, 0).to_uint();
    f.put(static_cast<char>(bits & 255));
    f.put(static_cast<char>((bits >> 8) & 255));
    if (!f) throw std::runtime_error("cannot write fixed-point data");
}
inline data_t read_code(std::istream &f) {
    int lo = f.get(), hi = f.get();
    if (lo == EOF || hi == EOF) throw std::runtime_error("truncated fixed-point data");
    data_t value;
    value.range(15, 0) = static_cast<unsigned>(lo | (hi << 8));
    return value;
}
template <typename T> void load_recursive(std::istream &f, T &v) { v = read_code(f); }
template <typename T, unsigned long N> void load_recursive(std::istream &f, T (&v)[N]) {
    for (unsigned long i = 0; i < N; ++i) load_recursive(f, v[i]);
}
template <typename T> void load(const std::string &name, T &v) {
    std::ifstream f((case_dir + "/inputs/" + name + ".bin").c_str(), std::ios::binary);
    if (!f) throw std::runtime_error("cannot open input " + name);
    load_recursive(f, v);
    if (f.peek() != EOF) throw std::runtime_error("extra input data: " + name);
}
inline std::ofstream output(const char *label) {
    std::string path = case_dir + "/csim/" + context + "." + label + ".bin";
    std::ofstream f(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot open output " + path);
    return f;
}
template <int HMAX, int WMAX>
void tensor(const char *label, const data_t (*v)[HMAX][WMAX], int c, int h, int w) {
    std::ofstream f = output(label);
    for (int ic = 0; ic < c; ++ic)
        for (int ih = 0; ih < h; ++ih)
            for (int iw = 0; iw < w; ++iw) write_code(f, v[ic][ih][iw]);
}
inline void vector(const char *label, const data_t *v, int n) {
    std::ofstream f = output(label);
    for (int i = 0; i < n; ++i) write_code(f, v[i]);
}
inline void set_context(const char *label) {
    context = label;
    std::cout << "CHECKPOINT: " << label << std::endl;
}
inline void exponent(int co, int row, int col, int ci, int exp) {
    exponents << context << "." << operation << ' ' << co << ' ' << row << ' '
              << col << ' ' << ci << ' ' << exp << '\n';
    if (!exponents) throw std::runtime_error("cannot write exponent trace");
}
}
#define VERIFY_CONTEXT(label) verification::set_context(label)
#define VERIFY_OP(label) (verification::operation = label)
#define VERIFY_TENSOR(label, value, c, h, w) verification::tensor(label, value, c, h, w)
#define VERIFY_VECTOR(label, value, n) verification::vector(label, value, n)
#define VERIFY_EXP(co, row, col, ci, exp) verification::exponent(co, row, col, ci, exp)
#else
#define VERIFY_CONTEXT(label) ((void)0)
#define VERIFY_OP(label) ((void)0)
#define VERIFY_TENSOR(label, value, c, h, w) ((void)0)
#define VERIFY_VECTOR(label, value, n) ((void)0)
#define VERIFY_EXP(co, row, col, ci, exp) ((void)0)
#endif
'''


def make_testbench(params):
    declarations = "\n".join("static data_t " + p.name + "".join(f"[{d}]" for d in p.dims) + ";" for p in params)
    loads = "\n".join(f'        verification::load("{p.name}", {p.name});' for p in params if is_input(p.name))
    arguments = ", ".join(p.name for p in params)
    return r'''#include "top.h"
#include "verification_trace.h"
#include <iomanip>
#include <exception>
namespace verification {
std::string case_dir, context, operation;
std::ofstream exponents;
}
''' + declarations + r'''
int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: csim.exe ABSOLUTE_CASE_DIRECTORY");
        verification::case_dir = argv[1];
        verification::exponents.open((verification::case_dir + "/csim/exponents.txt").c_str());
        if (!verification::exponents) throw std::runtime_error("cannot create exponent trace");
''' + loads + f"\n        top({arguments});\n" + r'''
        verification::exponents.close();
        std::ifstream expected((verification::case_dir + "/fixed/head.logits.bin").c_str(), std::ios::binary);
        if (!expected) throw std::runtime_error("missing fixed-point golden logits");
        std::ofstream readable((verification::case_dir + "/csim/logits.txt").c_str());
        if (!readable) throw std::runtime_error("cannot create readable logits");
        readable << std::setprecision(17);
        int mismatches = 0;
        for (int i = 0; i < 1000; ++i) {
            data_t ref = verification::read_code(expected);
            if (ref.range(15, 0) != DRAM_out[i].range(15, 0)) ++mismatches;
            readable << DRAM_out[i].to_double() << '\n';
        }
        if (expected.peek() != EOF) throw std::runtime_error("extra golden logits");
        if (!readable) throw std::runtime_error("cannot write readable logits");
        std::cout << "IMPLEMENTATION_MATCH: " << (mismatches ? "FAIL" : "PASS")
                  << " mismatches=" << mismatches << "/1000" << std::endl;
        return mismatches ? 1 : 0;
    } catch (const std::exception &e) {
        std::cerr << "VERIFICATION ERROR: " << e.what() << std::endl;
        return 2;
    }
}
'''


def make_csim_tcl(part, period):
    return f'''# Dedicated fixed-point C simulation; no synthesis or RTL co-simulation.
if {{![info exists ::env(RESNET18_CASE_DIR)]}} {{ error "RESNET18_CASE_DIR is required" }}
set script_dir [file dirname [file normalize [info script]]]
cd $script_dir
open_project verification_project
set_top top
add_files top.cpp -cflags "-std=c++14 -O3 -DNDEBUG -DRESNET18_VERIFY_TRACE"
add_files -tb tb_top.cpp -cflags "-std=c++14 -O3 -DNDEBUG -DRESNET18_VERIFY_TRACE"
open_solution "solution1"
set_part {part}
create_clock -period {period} -name default
csim_design -O -argv [list $::env(RESNET18_CASE_DIR)]
exit
'''
