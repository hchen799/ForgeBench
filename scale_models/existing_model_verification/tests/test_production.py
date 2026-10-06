"""Execute repaired production operator emitters with real Vitis types."""
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

SCALE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCALE_DIR))
import auto_generate_json as graph
import generate_code as gen
from production_types import annotate_storage, numeric_support, specialize
from existing_model_verification.references.repaired import LlamaOps, ResNetOps, exp_negative
from existing_model_verification.references.arithmetic import wrap

INCLUDE = Path(
    os.environ.get(
        "VITIS_HLS_INCLUDE", "/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1/include"
    )
)


def compile_probes(root, word=16, integer=5):
    if not (INCLUDE / "ap_fixed.h").exists():
        pytest.skip("Vitis headers unavailable")
    dtype = f"ap_fixed<{word},{integer}>"
    code = [
        "#include <ap_fixed.h>\n#include <hls_math.h>\n#include <cassert>\n#include <cstdint>\n#include <algorithm>\n"
        + f"typedef {dtype} data_t;",
        numeric_support(dtype),
    ]

    def emit(result, types):
        body, name = result
        parameters = [f"p{i}" for i in range(len(types))]
        body, name = specialize(
            body, name, {"args": parameters}, dict(zip(parameters, types))
        )
        code.append(body)
        return name

    linear = emit(
        gen.generate_linear_tile_function(dtype, 3, 33, 19),
        ["data_t", "data_t", "acc_t", "int", "int", "int"],
    )
    acc = emit(
        gen.generate_rmsnorm_accumulate_tile_function(dtype, 2, 128),
        ["data_t", "acc_t", "int", "int"],
    )
    finalize = emit(
        gen.generate_rmsnorm_finalize_rows_function(dtype, 2, 4096),
        ["acc_t", "acc_t", "int"],
    )
    apply = emit(
        gen.generate_rmsnorm_apply_tile_function(dtype, 2, 128),
        ["data_t", "data_t", "acc_t", "data_t", "int", "int"],
    )
    bn = emit(
        gen.generate_batchnorm_tile_function(dtype, 4, 3, 3),
        ["data_t", "data_t", "data_t", "int", "int", "int"],
    )
    silu = emit(
        gen.generate_activation_tile_2d_function(dtype, 3, 137),
        ["data_t", "data_t", "int", "int"],
    )
    mul = emit(
        gen.generate_elementwise_mult_tile_2d_function(dtype, 3, 137),
        ["data_t", "data_t", "data_t", "int", "int"],
    )
    score = emit(
        gen.generate_attention_score_tile_function(dtype, 3, 4, 3, 128, 1),
        ["data_t", "data_t", "acc_t", "int", "int", "int", "int"],
    )
    rowmax = emit(
        gen.generate_attention_rowmax_tile_function(dtype, 3, 4, 3),
        ["acc_t", "acc_t", "int", "int"],
    )
    context = emit(
        gen.generate_attention_softmax_context_tile_function(dtype, 3, 4, 3, 128, 1),
        ["acc_t", "data_t", "acc_t", "acc_t", "acc_t", "int", "int", "int", "int"],
    )
    finish = emit(
        gen.generate_attention_finalize_tile_function(dtype, 3, 4, 128),
        ["acc_t", "acc_t", "int"],
    )
    code.append(
        r"""
data_t read(int64_t x) {data_t v;v.range(15,0)=x;return v;}
int64_t raw(data_t x) {ap_int<16> v=x.range(15,0);return v.to_int64();}
extern "C" {
void primitives(int64_t *a,int64_t *b,int64_t *out,int n,int kind) {
 for(int i=0;i<n;++i) {acc_t x=fb_from_code(a[i]),y=fb_from_code(b[i]);
  out[i]=fb_code(kind==0?fb_sqrt(x):kind==1?fb_div(x,y):fb_exp_negative(x)).to_int64();
 }
}
int storage_bytes() {return sizeof(data_t);}
"""
    )
    code.append(
        f"""
void linear_probe(int64_t *x,int64_t *w,int64_t *out) {{
 data_t a[3][33],b[19][33];acc_t y[3][19];
 for(int r=0;r<3;++r)for(int c=0;c<19;++c)y[r][c]=0;
 for(int base=0;base<137;base+=33) {{int valid=std::min(33,137-base);
  for(int r=0;r<3;++r)for(int c=0;c<valid;++c)a[r][c]=read(x[r*137+base+c]);
  for(int r=0;r<19;++r)for(int c=0;c<valid;++c)b[r][c]=read(w[r*137+base+c]);
  {linear}(a,b,y,3,19,valid);
 }}
 for(int r=0;r<3;++r)for(int c=0;c<19;++c)out[r*19+c]=raw(data_t(y[r][c]));
}}
void norm_probe(int64_t *x,int64_t *gamma,int64_t *out) {{
 data_t a[2][128],g[128],y[2][128];acc_t sum[2],inv[2];
 for(int r=0;r<2;++r)sum[r]=0;
 for(int base=0;base<4096;base+=128) {{
  for(int r=0;r<2;++r)for(int c=0;c<128;++c)a[r][c]=read(x[r*4096+base+c]);
  {acc}(a,sum,2,128);
 }}
 {finalize}(sum,inv,2);
 for(int base=0;base<4096;base+=128) {{
  for(int r=0;r<2;++r)for(int c=0;c<128;++c)a[r][c]=read(x[r*4096+base+c]);
  for(int c=0;c<128;++c)g[c]=read(gamma[base+c]);
  {apply}(a,g,inv,y,2,128);
  for(int r=0;r<2;++r)for(int c=0;c<128;++c)out[r*4096+base+c]=raw(y[r][c]);
 }}
}}
void bn_probe(int64_t *x,int64_t *params,int64_t *out) {{
 data_t a[4][3][3],p[4][4],y[4][3][3];
 for(int i=0;i<36;++i)((data_t*)a)[i]=read(x[i]);
 for(int i=0;i<16;++i)((data_t*)p)[i]=read(params[i]);
 {bn}(a,p,y,4,3,3);
 for(int i=0;i<36;++i)out[i]=raw(((data_t*)y)[i]);
}}
void swiglu_probe(int64_t *x,int64_t *up,int64_t *out) {{
 data_t a[3][137],b[3][137],y[3][137];
 for(int i=0;i<411;++i){{((data_t*)a)[i]=read(x[i]);((data_t*)b)[i]=read(up[i]);}}
 {silu}(a,a,3,137); {mul}(a,b,y,3,137);
 for(int i=0;i<411;++i)out[i]=raw(((data_t*)y)[i]);
}}
void attention_probe(int64_t *q,int64_t *k,int64_t *v,int64_t *out) {{
 data_t qt[3][512],kt[3][128],vt[3][128];
 acc_t scores[3][4][3],maxima[3][4],sums[3][4],ctx[3][512];
 for(int i=0;i<12;++i)((acc_t*)sums)[i]=0;
 for(int i=0;i<1536;++i)((acc_t*)ctx)[i]=0;
 for(int i=0;i<1536;++i)((data_t*)qt)[i]=read(q[i]);
 for(int i=0;i<12;++i)((acc_t*)maxima)[i]=FB_MIN_SCORE;
 for(int pass=0;pass<2;++pass)for(int base=0;base<7;base+=3){{int valid=std::min(3,7-base);
  for(int r=0;r<valid;++r)for(int d=0;d<128;++d){{kt[r][d]=read(k[(base+r)*128+d]);vt[r][d]=read(v[(base+r)*128+d]);}}
  {score}(qt,kt,scores,3,valid,4,base);
  if(pass==0){rowmax}(scores,maxima,3,valid);
  else {context}(scores,vt,maxima,sums,ctx,3,valid,4,base);
 }}
 {finish}(ctx,sums,3);
 for(int i=0;i<1536;++i)out[i]=raw(data_t(((acc_t*)ctx)[i]));
}}
}}
"""
    )
    source = root / "probe.cpp"
    source.write_text(
        "\n".join(code)
        .replace("range(15,0)", f"range({word-1},0)")
        .replace("ap_int<16>", f"ap_int<{word}>")
    )
    library = root / "probe.so"
    libs = INCLUDE.parent / "lnx64/lib/csim"
    fpo = INCLUDE.parent / "lnx64/tools/fpo_v7_1"
    compiled = subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++14",
            "-O2",
            "-I" + str(INCLUDE),
            str(source),
            "-o",
            str(library),
            "-L" + str(libs),
            "-Wl,--disable-new-dtags",
            "-Wl,-rpath," + str(libs),
            "-Wl,-rpath," + str(fpo),
            "-lhlsmc++-GCC46",
            "-lhlsm-GCC46",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stderr
    lib = ctypes.CDLL(
        str(library), mode=os.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0)
    )
    ptr = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    for name in ("linear_probe", "norm_probe", "bn_probe", "swiglu_probe"):
        getattr(lib, name).argtypes = [ptr, ptr, ptr]
    lib.attention_probe.argtypes = [ptr] * 4
    lib.primitives.argtypes = [ptr, ptr, ptr, ctypes.c_int, ctypes.c_int]
    return lib


@pytest.fixture(scope="module")
def production(tmp_path_factory):
    return compile_probes(tmp_path_factory.mktemp("production_v2"))


def test_production_primitives(production):
    rng = np.random.default_rng(81)
    a = rng.integers(0, 1 << 45, 4000, dtype=np.int64)
    b = rng.integers(1, 1 << 25, 4000, dtype=np.int64)
    out = np.empty_like(a)
    production.primitives(a, b, out, len(a), 0)
    expected = []
    for value in a:
        n = int(value) << 22
        r = math.isqrt(n)
        expected.append(r + int(n - r * r > r))
    np.testing.assert_array_equal(out, expected)
    a = rng.integers(-(1 << 35), 1 << 35, 4000, dtype=np.int64)
    b[::2] *= -1
    production.primitives(a, b, out, len(a), 1)
    np.testing.assert_array_equal(out, [(int(x) << 22) // int(y) for x, y in zip(a, b)])
    a = rng.integers(-20 * (1 << 22), 1 << 22, 4000, dtype=np.int64)
    production.primitives(a, b, out, len(a), 2)
    np.testing.assert_array_equal(out, exp_negative(torch.from_numpy(a)))


def test_wide_linear_partial_tiles(production):
    rng = np.random.default_rng(82)
    x = rng.integers(-32768, 32768, (3, 137), dtype=np.int64)
    w = rng.integers(-32768, 32768, (19, 137), dtype=np.int64)
    out = np.empty((3, 19), dtype=np.int64)
    production.linear_probe(x, w, out)
    np.testing.assert_array_equal(
        out, LlamaOps().linear(torch.from_numpy(x), torch.from_numpy(w))
    )


def test_actual_4096_rms_reduction(production):
    rng = np.random.default_rng(83)
    x = rng.integers(-3072, 3072, (2, 4096), dtype=np.int64)
    gamma = rng.integers(1900, 2200, 4096, dtype=np.int64)
    out = np.empty_like(x)
    production.norm_probe(x, gamma, out)
    np.testing.assert_array_equal(
        out, LlamaOps().rmsnorm(torch.from_numpy(x), torch.from_numpy(gamma))
    )
    x.fill(0)
    production.norm_probe(x, gamma, out)
    assert not out.any()


def test_wide_batchnorm_and_stable_swiglu(production):
    rng = np.random.default_rng(84)
    for _ in range(20):
        x = rng.integers(-32768, 32768, (4, 3, 3), dtype=np.int64)
        p = rng.integers(-3000, 3000, (4, 4), dtype=np.int64)
        p[3] = rng.integers(1, 32768, 4, dtype=np.int64)
        out = np.empty_like(x)
        production.bn_probe(x, p, out)
        np.testing.assert_array_equal(
            out, ResNetOps().bn(torch.from_numpy(x), torch.from_numpy(p))
        )
    x = rng.integers(-32768, 32768, (3, 137), dtype=np.int64)
    up = rng.integers(-32768, 32768, (3, 137), dtype=np.int64)
    out = np.empty_like(x)
    production.swiglu_probe(x, up, out)
    np.testing.assert_array_equal(
        out, LlamaOps().swiglu(torch.from_numpy(x), torch.from_numpy(up))
    )


def test_causal_attention_across_key_tiles(production):
    rng = np.random.default_rng(85)
    q = rng.integers(-4096, 4096, (3, 512), dtype=np.int64)
    k = rng.integers(-4096, 4096, (7, 128), dtype=np.int64)
    v = rng.integers(-4096, 4096, (7, 128), dtype=np.int64)
    out = np.empty_like(q)
    production.attention_probe(q, k, v, out)
    np.testing.assert_array_equal(
        out,
        LlamaOps().attention(
            torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), 4
        ),
    )
    baseline = out.copy()
    v[5:] = 32767
    production.attention_probe(q, k, v, out)
    np.testing.assert_array_equal(out[0], baseline[0])


@pytest.mark.parametrize("depth", [18, 50, 101, 152])
def test_full_resnet_parameter_loads(depth):
    config = json.loads(graph.generate_resnet_config_txt(depth))
    loads = {o["args"][0] for o in config["ops"].values() if o["func_name"] == "load"}
    parameters = {
        d["name"]
        for d in config["drams"]
        if d["name"].startswith(("DRAM_w_", "DRAM_bn_")) or d["name"] == "DRAM_fc"
    }
    assert parameters <= loads


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_llama_production_graph_and_types(mode):
    config = json.loads(
        getattr(graph, "generate_llama3_8b_" + mode + "_config_text")(2048)
    )
    types = {
        d["name"]: d.get("dtype", "data_t") for d in config["drams"] + config["brams"]
    }
    assert (
        types["DRAM_token_ids" if mode == "prefill" else "DRAM_token_id"] == "int32_t"
    )
    assert (
        types["BRAM_matrix_out"]
        == types["BRAM_rms_sumsq"]
        == types["BRAM_rowsum"]
        == "acc_t"
    )
    for name, op in config["ops"].items():
        if name.endswith("_oproj_load_in"):
            store = config["ops"][name.removesuffix("_load_in") + "_store"]
            assert op["args"][0] != store["args"][1]
        if op["func_name"] == "attention_softmax_context_tile":
            assert len(op["args"]) == 9


def test_reject_unread_parameters():
    config = json.loads(graph.generate_resnet_config_txt(50))
    config["ops"] = {
        n: o for n, o in config["ops"].items() if not n.startswith("load_param_")
    }
    with pytest.raises(ValueError, match="unread model parameters"):
        annotate_storage(config["brams"], config["drams"], config["ops"])


@pytest.mark.parametrize("fault", ["missing", "undersized"])
def test_bottleneck_buffer_validation(monkeypatch, fault):
    monkeypatch.chdir(SCALE_DIR)
    config = json.loads(graph.generate_resnet_tiled_config_txt(50))
    if fault == "missing":
        config["brams"] = [
            b for b in config["brams"] if b["name"] != "BRAM_in_patch_stride1_k1"
        ]
    else:
        next(d for d in config["drams"] if d["name"] == "DRAM_s2_pre_downsample")[
            "dims"
        ] = [128, 28, 28]
    with pytest.raises(ValueError, match="undeclared array|needs"):
        gen.generate_top_function(
            config["brams"], config["drams"], config["ops"], config["data_type"]
        )


def test_saved_projects_match_production_json(monkeypatch):
    monkeypatch.chdir(SCALE_DIR)
    # auto_generated_configs is intentionally git-ignored; the emitter saves
    # a resolved JSON alongside each committed production project.
    paths = sorted((SCALE_DIR / "hls_files").glob("*/resolved_config.json"))
    assert len(paths) >= 6
    for path in paths:
        config = json.loads(path.read_text())
        code = gen.generate_top_function(
            config["brams"],
            config["drams"],
            config["ops"],
            config["data_type"],
            config["top_func_name"],
        )
        assert (path.parent / "top.cpp").read_text() == code, path.parent.name
        assert (path.parent / "top.h").read_text() == gen.generate_top_h(
            config["drams"], config["data_type"], config["top_func_name"]
        ), path.parent.name
