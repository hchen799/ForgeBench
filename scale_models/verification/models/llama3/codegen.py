"""Emit synthesizable integer datapaths and a host-only mmap testbench."""
import json
import math
from pathlib import Path

from .config import scratch_layout, segments


def tables(c):
    p = c.precision
    exp = [int(math.floor(math.exp(-i / 2048.0) * (1 << c.exp_frac) + .5)) for i in range(65537)]
    silu = [max(p.minimum, min(p.maximum, int(math.floor((i / 2048.0) / (1 + math.exp(-i / 2048.0)) * p.scale + .5)))) for i in range(-65536, 65537)]
    return exp, silu


def make_config_header(c):
    layout, count = scratch_layout(c)
    weights = segments(c)
    mapping = {s["name"]: s for s in weights}
    lines = ["#pragma once", "#include <cstdint>"]
    wide_bits = 64 if c.word_bits <= 16 and c.frac <= 11 else 128
    lines += [f"using data_t = {c.precision.cpp_type};", "#if defined(__SYNTHESIS__)",
              "#include <ap_int.h>", f"using wide_t = ap_int<{wide_bits}>;", "using uwide_t = ap_uint<128>;",
              "#else", f"using wide_t = {'int64_t' if wide_bits == 64 else '__int128'};", "using uwide_t = unsigned __int128;", "#endif",
              f"static const int DATA_BITS = {c.word_bits};", f"static const int FRAC = {c.frac};",
              f"static const int EXP_FRAC = {c.exp_frac};",
              f"static const int64_t DATA_MIN = {c.precision.minimum}LL;",
              f"static const int64_t DATA_MAX = {c.precision.maximum}LL;",
              f"static const uint64_t EPSILON_Q = {max(1, math.floor(c.epsilon * (1 << (2*c.frac)) + .5))}ULL;",
              f"static const uint64_t SCORE_SCALE_Q = {math.floor((1 << (2*c.frac)) / math.sqrt(c.head_dim) + .5)}ULL;"]
    constants = dict(LAYERS=c.layers, HIDDEN=c.hidden, FFN=c.ffn, Q_HEADS=c.q_heads, KV_HEADS=c.kv_heads,
                     HEAD_DIM=c.head_dim, KV_DIM=c.kv_dim, VOCAB=c.vocab, MAX_CTX=c.max_ctx,
                     PREFILL_TILE=c.prefill_tile, TILE_IN=c.tile_in, TILE_OUT=c.tile_out)
    lines += [f"static const int {name} = {value};" for name, value in constants.items()]
    embedding = mapping["embedding"]["count"]
    stride = sum(s["count"] for s in weights if s["name"].startswith("layers.0."))
    wide = dict(EMBEDDING_COUNT=embedding, LAYER_WEIGHT_COUNT=stride,
                FINAL_NORM_OFFSET=mapping["final_norm"]["offset"], LM_HEAD_OFFSET=mapping["lm_head"]["offset"],
                WEIGHT_COUNT=weights[-1]["offset"] + weights[-1]["count"],
                CACHE_COUNT=c.layers * c.max_ctx * c.kv_dim, SCRATCH_COUNT=count)
    for s in weights:
        if s["name"].startswith("layers.0."):
            wide["OFFSET_" + s["name"].split(".")[-1].upper()] = s["offset"] - embedding
    wide.update({"SCRATCH_" + k.upper(): v for k, v in layout.items()})
    lines += [f"static const uint64_t {name} = {value}ULL;" for name, value in wide.items()]
    return "\n".join(lines) + "\n"


def make_top_header():
    return '''#pragma once
#include "model_config.h"
int llama_prefill(const int32_t tokens[PREFILL_TILE], int token_count, int start_pos,
                  const data_t weights[WEIGHT_COUNT], const data_t rope_table[MAX_CTX * HEAD_DIM],
                  data_t key_cache[CACHE_COUNT], data_t value_cache[CACHE_COUNT],
                  data_t scratch[SCRATCH_COUNT], data_t logits[PREFILL_TILE * VOCAB]);
int llama_decode(int32_t token, int position,
                 const data_t weights[WEIGHT_COUNT], const data_t rope_table[MAX_CTX * HEAD_DIM],
                 data_t key_cache[CACHE_COUNT], data_t value_cache[CACHE_COUNT],
                 data_t scratch[SCRATCH_COUNT], data_t logits[VOCAB]);
'''


def trace_header():
    return r'''#pragma once
#if defined(LLAMA_VERIFY) && !defined(__SYNTHESIS__)
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
namespace llama_trace {
extern std::string folder;
extern int call, layer;
extern bool all_ops;
extern std::ofstream index;
inline std::string name(const char *operation) {
    std::ostringstream out;
    out << "call" << std::setw(4) << std::setfill('0') << call << '.';
    if (layer >= 0) out << "layer" << std::setw(2) << layer << '.';
    out << operation;
    return out.str();
}
inline void write(const char *operation, const data_t *data, int rows, int cols, bool essential) {
    if (!all_ops && !essential) return;
    std::string id = name(operation);
    std::ofstream out((folder + "/" + id + ".bin").c_str(), std::ios::binary);
    out.write(reinterpret_cast<const char *>(data), uint64_t(rows) * cols * sizeof(data_t));
    if (!out) throw std::runtime_error("trace write failed: " + id);
    index << id << ' ' << rows << ' ' << cols << '\n';
    if (!index) throw std::runtime_error("trace index write failed");
}
}
#define LLAMA_TRACE(op, data, rows, cols, essential) llama_trace::write(op, data, rows, cols, essential)
#define LLAMA_LAYER(value) (llama_trace::layer = value)
#else
#define LLAMA_TRACE(op, data, rows, cols, essential) ((void)0)
#define LLAMA_LAYER(value) ((void)0)
#endif
'''


def testbench():
    return r'''#include "top.h"
#include "verification_trace.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <string>
namespace llama_trace {
std::string folder;
int call = 0, layer = -1;
bool all_ops = true;
std::ofstream index;
}
class ModelMap {
    int descriptor;
    void *mapping;
public:
    explicit ModelMap(const std::string &path) {
        descriptor = open(path.c_str(), O_RDONLY);
        if (descriptor < 0) throw std::runtime_error("cannot open model weights");
        struct stat info;
        if (fstat(descriptor, &info) != 0 || uint64_t(info.st_size) != WEIGHT_COUNT * sizeof(data_t))
            throw std::runtime_error("wrong weight file length");
        mapping = mmap(nullptr, WEIGHT_COUNT * sizeof(data_t), PROT_READ, MAP_SHARED, descriptor, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("weight mmap failed");
    }
    ~ModelMap() { munmap(mapping, WEIGHT_COUNT * sizeof(data_t)); close(descriptor); }
    const data_t *data() const { return static_cast<const data_t *>(mapping); }
};
template <class T> std::vector<T> load(const std::string &path, uint64_t count) {
    std::ifstream in(path.c_str(), std::ios::binary | std::ios::ate);
    if (!in || uint64_t(in.tellg()) != count * sizeof(T)) throw std::runtime_error("bad input file: " + path);
    in.seekg(0);
    std::vector<T> result(count);
    in.read(reinterpret_cast<char *>(result.data()), count * sizeof(T));
    if (!in) throw std::runtime_error("read failed: " + path);
    return result;
}
int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: csim.exe CASE_DIRECTORY MODEL_DIRECTORY");
        uint16_t endian = 1;
        if (*reinterpret_cast<unsigned char *>(&endian) != 1) throw std::runtime_error("testbench requires little endian host");
        std::string directory = argv[1], model = argv[2];
        std::ifstream control((directory + "/case.txt").c_str());
        int total, calls, trace_all;
        if (!(control >> total >> calls >> trace_all)) throw std::runtime_error("invalid case control file");
        if (total < 1 || total > MAX_CTX || calls < 1) throw std::runtime_error("invalid token/call count");
        auto tokens = load<int32_t>(directory + "/tokens.bin", total);
        auto rope_table = load<data_t>(model + "/rope.bin", uint64_t(MAX_CTX) * HEAD_DIM);
        ModelMap weights(model + "/weights.bin");
        std::vector<data_t> keys(CACHE_COUNT, 0), values(CACHE_COUNT, 0), scratch(SCRATCH_COUNT, 0), logits(uint64_t(PREFILL_TILE) * VOCAB);
        llama_trace::folder = directory + "/csim";
        llama_trace::all_ops = trace_all != 0;
        llama_trace::index.open((llama_trace::folder + "/index.txt").c_str());
        if (!llama_trace::index) throw std::runtime_error("cannot write trace index");
        int consumed = 0;
        uint64_t mismatches = 0;
        for (int call = 0; call < calls; ++call) {
            int start, count, decode;
            if (!(control >> start >> count >> decode) || start != consumed || count < 1 || count > PREFILL_TILE || count > total - start || (decode && count != 1))
                throw std::runtime_error("invalid call schedule");
            llama_trace::call = call;
            llama_trace::layer = -1;
            std::cout << "CALL " << call << " position=" << start << " tokens=" << count << (decode ? " decode" : " prefill") << std::endl;
            int status = decode ? llama_decode(tokens[start], start, weights.data(), rope_table.data(), keys.data(), values.data(), scratch.data(), logits.data())
                                : llama_prefill(tokens.data() + start, count, start, weights.data(), rope_table.data(), keys.data(), values.data(), scratch.data(), logits.data());
            if (status) throw std::runtime_error("kernel rejected arguments: " + std::to_string(status));
            auto expected = load<data_t>(directory + "/fixed/" + llama_trace::name("logits") + ".bin", uint64_t(count) * VOCAB);
            uint64_t bad = 0;
            for (uint64_t i = 0; i < expected.size(); ++i) bad += expected[i] != logits[i];
            mismatches += bad;
            std::cout << "CALL_RESULT " << call << " mismatches=" << bad << '/' << expected.size() << std::endl;
            if (bad) return 1;
            consumed += count;
        }
        if (consumed != total) throw std::runtime_error("call schedule does not consume all tokens");
        llama_trace::index.close();
        std::cout << "IMPLEMENTATION_MATCH: PASS mismatches=" << mismatches << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "VERIFICATION ERROR: " << error.what() << std::endl;
        return 2;
    }
}
'''


def emit_project(destination, config):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = {"model_config.h": make_config_header(config), "top.h": make_top_header(),
             "top.cpp": Path(__file__).with_name("kernel.cpp.in").read_text(),
             "verification_trace.h": trace_header(), "tb_top.cpp": testbench()}
    exp, silu = tables(config)
    table_lines = ["#pragma once", '#include "model_config.h"']
    for name, dtype, values in [("EXP_NEG", "uint32_t", exp), ("SILU", "data_t", silu)]:
        table_lines.append(f"static const {dtype} {name}[{len(values)}] = {{")
        table_lines += [", ".join(map(str, values[i:i + 32])) + "," for i in range(0, len(values), 32)]
        table_lines.append("};")
    files["nonlinear_tables.h"] = "\n".join(table_lines) + "\n"
    for mode in ("prefill", "decode"):
        files[f"run_{mode}_hls.tcl"] = f'''cd [file dirname [file normalize [info script]]]
open_project project_{mode}
set_top llama_{mode}
add_files top.cpp
open_solution solution1
set_part xczu9eg-ffvb1156-2-e
create_clock -period 10 -name default
csynth_design
exit
'''
    files["run_csim.tcl"] = '''cd [file dirname [file normalize [info script]]]
if {![info exists ::env(LLAMA_CASE_DIR)] || ![info exists ::env(LLAMA_MODEL_DIR)]} {error "case/model directory required"}
open_project verification_project
set_top llama_prefill
add_files top.cpp -cflags "-std=c++14 -O3 -march=native -DLLAMA_VERIFY"
add_files -tb tb_top.cpp -cflags "-std=c++14 -O3 -march=native -DLLAMA_VERIFY"
open_solution solution1
set_part xczu9eg-ffvb1156-2-e
create_clock -period 10 -name default
csim_design -O -argv [list $::env(LLAMA_CASE_DIR) $::env(LLAMA_MODEL_DIR)]
exit
'''
    files["config.json"] = json.dumps(config.to_dict(), indent=2) + "\n"
    for name, content in files.items():
        (destination / name).write_text(content)
