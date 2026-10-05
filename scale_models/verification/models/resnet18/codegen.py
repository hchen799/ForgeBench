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
    ap_int<DATA_BITS> raw;
    raw.range(DATA_BITS - 1, 0) = v.range(DATA_BITS - 1, 0);
    uint64_t bits = static_cast<uint64_t>(raw.to_int64());
    for (int b = 0; b < DATA_BYTES; ++b) f.put(static_cast<char>((bits >> (8 * b)) & 255));
    if (!f) throw std::runtime_error("cannot write fixed-point data");
}
inline data_t read_code(std::istream &f) {
    uint64_t bits = 0;
    for (int b = 0; b < DATA_BYTES; ++b) {
        int byte = f.get();
        if (byte == EOF) throw std::runtime_error("truncated fixed-point data");
        bits |= uint64_t(byte) << (8 * b);
    }
    data_t value;
    value.range(DATA_BITS - 1, 0) = bits;
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
            if (ref.range(DATA_BITS - 1, 0) != DRAM_out[i].range(DATA_BITS - 1, 0)) ++mismatches;
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


def emit_project(destination, config):
    """Reuse the existing graph emitter, with explicit precision and BN sqrt."""
    import json
    from pathlib import Path
    import generate_tiled_resnet18 as gen
    c = config.validate()
    p = c.precision
    changes = dict(DATA_TYPE=f"ap_fixed<{p.word_bits},{p.integer_bits},AP_RND,AP_SAT>",
                   ACC_TYPE=f"ap_fixed<{c.acc_word_bits},{c.acc_integer_bits},AP_RND,AP_SAT>",
                   TILE_C=c.tile_c, TILE_H=c.tile_h, TILE_W=c.tile_w,
                   MAX_PATCH=2 * (max(c.tile_h, c.tile_w) - 1) + 7)
    previous = {key: getattr(gen, key) for key in changes}
    try:
        for key, value in changes.items():
            setattr(gen, key, value)
        blocks = gen.build_blocks()
        params = gen.build_params(blocks)
        header = gen.make_top_h(params) + f"\nstatic const int DATA_BITS = {p.word_bits};\nstatic const int DATA_BYTES = {p.itemsize};\n"
        source = gen.make_top_cpp(blocks, params)
        source = source.replace(f"typedef {gen.ACC_TYPE} acc_t;", f"typedef {gen.ACC_TYPE} acc_t;\n" + sqrt_helper(c))
        source = source.replace("hls::sqrt(var + (acc_t)1e-5)", "verification_sqrt(var)")
        scales = gen.make_scale_header(blocks)
        zeros = ", ".join("0" for _ in c.shifts)
        values = ", ".join(str(c.shifts[name]) for name in gen.build_shift_names(blocks))
        scales = scales.replace("{" + zeros + "}", "{" + values + "}").replace("kRenormGuard = 256", f"kRenormGuard = {c.guard}")
        files = {"top.h": header, "top.cpp": source, "resnet18_tiled_scales.h": scales,
                 "verification_trace.h": make_trace_header(), "tb_top.cpp": make_testbench(params),
                 "run_csim.tcl": make_csim_tcl(gen.FPGA_NAME, gen.CLOCK_PERIOD), "run_hls.tcl": gen.make_tcl(),
                 "config.json": json.dumps(c.to_dict(), indent=2) + "\n"}
    finally:
        for key, value in previous.items():
            setattr(gen, key, value)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (destination / name).write_text(content)


def sqrt_helper(c):
    import math
    frac = 2 * c.precision.fractional_bits
    epsilon = max(1, math.floor(1e-5 * (1 << frac) + .5))
    return f'''// The vendor sqrt returns zero for F > 32. Use exact integer logic.
static ap_fixed<{c.acc_word_bits + 1},{c.acc_integer_bits + 1}> verification_sqrt(acc_t variance) {{
    ap_uint<128> raw = variance.range({c.acc_word_bits - 1}, 0);
    ap_uint<128> radicand = (raw + {epsilon}ULL) << {frac};
    ap_uint<128> remainder = radicand, root = 0;
    ap_uint<128> bit = ap_uint<128>(1) << 126;
    for (int step = 0; step < 64; ++step) {{
        if (remainder >= root + bit) {{
            remainder -= root + bit;
            root = (root >> 1) + bit;
        }} else {{ root >>= 1; }}
        bit >>= 2;
    }}
    if (radicand - root * root > root) ++root;
    ap_fixed<{c.acc_word_bits + 1},{c.acc_integer_bits + 1}> result;
    result.range({c.acc_word_bits}, 0) = root;
    return result;
}}
'''
