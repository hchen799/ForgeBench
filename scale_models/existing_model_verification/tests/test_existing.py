"""Tests use vendor types, independently supplied inputs, and actual saved HLS."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from existing_model_verification.references.arithmetic import (
    DATA,
    ArithmeticFault,
    exp_codes,
    product_sum,
    sqrt_codes,
    trunc_div,
    wrap,
)
from existing_model_verification.references.resnet import FixedOps
from existing_model_verification.references.project import Project, sha
from existing_model_verification.references.llama import FixedOps as LlamaFixedOps, LlamaReference
from existing_model_verification.references.runner import (
    compare_arrays,
    llama_preflight,
    validate_inputs,
)

INCLUDE = Path(
    os.environ.get(
        "VITIS_HLS_INCLUDE", "/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1/include"
    )
)
SCALE = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def vendor(tmp_path_factory):
    if not (INCLUDE / "ap_fixed.h").exists():
        pytest.skip("real Vitis headers unavailable")
    folder = tmp_path_factory.mktemp("existing_arithmetic")
    source = folder / "probe.cpp"
    source.write_text(
        r"""
#include <ap_fixed.h>
#include <hls_math.h>
#include <cstdint>
typedef ap_fixed<16,5> D;
D read(int64_t n){D d;d.range(15,0)=n;return d;}
template<typename T> int64_t code(T x){ap_int<T::width> n=x.range(T::width-1,0);return n.to_int64();}
extern "C" {
void constants(int64_t *out){
 out[0]=code(ap_fixed<32,10>(4096));
 out[1]=code(D(128));
 out[2]=code(hls::sqrt(D(128)));
 out[3]=code(D(49));
 out[4]=code(D(16));
}
void probe(int64_t *x,int64_t *out,int n,int op){
 for(int i=0;i<n;++i){
   D a=read(x[5*i]),b=read(x[5*i+1]),c=read(x[5*i+2]),d=read(x[5*i+3]),e=read(x[5*i+4]);
   if(op==0){D norm=(a-c)/hls::sqrt(e+(D)0.00001);out[i]=code(D(b*norm+d));}
   if(op==1){ap_fixed<17,6> v;v.range(16,0)=x[5*i];out[i]=code(hls::sqrt(v));}
   if(op==2){ap_fixed<17,6> v;v.range(16,0)=x[5*i];out[i]=code(hls::exp(v));}
   if(op==3){out[i]=code(D(a/b));}
   if(op==4){D y=0; y+=a*b; y+=c*d;out[i]=code(y);}
   if(op==5){out[i]=code(D(a/(D)49));}
   if(op==6){D s=a/((D)1+hls::exp(-a));out[i]=code(D(s*b));}
 }
}
void linear(int64_t *x,int64_t *w,int64_t *y,int rows,int ins,int outs,int tile){
 using A=ap_fixed<32,10>;
 for(int r=0;r<rows;++r) for(int o=0;o<outs;++o){D out=0;
  for(int base=0;base<ins;base+=tile){A sum=out;
   for(int i=base;i<ins && i<base+tile;++i) sum+=(A)read(x[r*ins+i])*(A)read(w[o*ins+i]);
   out=(D)sum;
  } y[r*outs+o]=code(out);
 }
}
void norm(int64_t *x,int64_t *g,int64_t *y,int rows,int cols,int tile){
 using A=ap_fixed<32,10>;
 for(int r=0;r<rows;++r){D sumsq=0;
  for(int base=0;base<cols;base+=tile){A sum=sumsq;
   for(int c=base;c<cols && c<base+tile;++c) sum+=(A)read(x[r*cols+c])*(A)read(x[r*cols+c]);
   sumsq=(D)sum;
  }
  D inv=(D)((A)1/hls::sqrt((A)sumsq/(A)cols+(A)1e-5));
  for(int c=0;c<cols;++c) y[r*cols+c]=code(D((A)read(x[r*cols+c])*(A)read(g[c])*(A)inv));
 }
}
void rope(int64_t *x,int64_t *y,int rows,int start){
 for(int r=0;r<rows;++r)for(int d=0;d<128;d+=2){
  float theta=powf(500000.0f,-((float)d)/128.0f);
  float angle=(float)(start+r)*theta;
  D c=(D)hls::cos(angle),s=(D)hls::sin(angle),a=read(x[r*128+d]),b=read(x[r*128+d+1]);
  y[r*128+d]=code(D(a*c-b*s));y[r*128+d+1]=code(D(a*s+b*c));
 }
}
}"""
    )
    lib = folder / "probe.so"
    mathlib = INCLUDE.parent / "lnx64/lib/csim"
    fpolib = INCLUDE.parent / "lnx64/tools/fpo_v7_1"
    subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++14",
            "-O2",
            "-I" + str(INCLUDE),
            str(source),
            "-o",
            str(lib),
            "-L" + str(mathlib),
            "-Wl,-rpath," + str(mathlib),
            "-Wl,--disable-new-dtags",
            "-Wl,-rpath," + str(fpolib),
            "-lhlsmc++-GCC46",
            "-lhlsm-GCC46",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dll = ctypes.CDLL(str(lib), mode=os.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0))
    pointer = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    dll.probe.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int]
    dll.constants.argtypes = [pointer]
    dll.linear.argtypes = [pointer, pointer, pointer] + [ctypes.c_int] * 4
    dll.norm.argtypes = [pointer, pointer, pointer] + [ctypes.c_int] * 3
    dll.rope.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int]

    def run(x, op):
        x = np.ascontiguousarray(x, dtype=np.int64)
        out = np.empty(x.shape[0], dtype=np.int64)
        dll.probe(x, out, len(out), op)
        return out

    run.dll = dll
    return run


def test_quantization():
    assert DATA.quantize(
        torch.tensor([-0.0001, 16.0, 49.0, 128.0, 2048.0])
    ).tolist() == [-1, -32768, -30720, 0, 0]
    assert trunc_div(torch.tensor([-5, 5, -5]), torch.tensor([2, -2, -2])).tolist() == [
        -2,
        -2,
        2,
    ]
    with pytest.raises(ArithmeticFault, match="zero"):
        trunc_div(torch.tensor([3]), 0)


def test_vendor_arithmetic(vendor):
    torch.set_num_threads(4)
    rng = np.random.default_rng(723)
    values = rng.integers(-32768, 32768, (5000, 5), dtype=np.int64)
    values[:, 4] = rng.integers(1, 32768, 5000)
    t = torch.from_numpy(values)
    bn = FixedOps().bn(
        t[:, 0, None, None], torch.stack((t[:, 1], t[:, 3], t[:, 2], t[:, 4]))
    )
    np.testing.assert_array_equal(bn.flatten().numpy(), vendor(values, 0))
    np.testing.assert_array_equal(
        wrap(trunc_div(t[:, 0] * 2048, t[:, 1])).numpy(), vendor(values, 3)
    )
    np.testing.assert_array_equal(
        wrap((t[:, 0] * t[:, 1] >> 11) + (t[:, 2] * t[:, 3] >> 11)).numpy(),
        vendor(values, 4),
    )
    np.testing.assert_array_equal(
        wrap(trunc_div(t[:, 0] * 2048, -30720)).numpy(), vendor(values, 5)
    )
    np.testing.assert_array_equal(
        LlamaFixedOps().swiglu(t[:, 0], t[:, 1]).numpy(), vendor(values, 6)
    )


def test_saved_constant_casts_with_vendor_types(vendor):
    out = np.empty(5, dtype=np.int64)
    vendor.dll.constants(out)
    np.testing.assert_array_equal(out, [0, 0, 0, -30720, -32768])


def test_vendor_sqrt_exhaustive(vendor):
    values = np.zeros((1 << 17, 5), dtype=np.int64)
    values[:, 0] = np.arange(-(1 << 16), 1 << 16)
    np.testing.assert_array_equal(
        sqrt_codes(torch.from_numpy(values[:, 0])).numpy(), vendor(values, 1)
    )


def test_vendor_exp_exhaustive(vendor):
    values = np.zeros((1 << 17, 5), dtype=np.int64)
    values[:, 0] = np.arange(-(1 << 16), 1 << 16)
    np.testing.assert_array_equal(
        exp_codes(torch.from_numpy(values[:, 0])).numpy(), vendor(values, 2)
    )


def test_product_quantization_before_reduction():
    a = torch.tensor([[1, -1], [1, -1]])
    b = torch.tensor([[1, 1]])
    assert product_sum(a, b).tolist() == [[0, -2]]


@pytest.mark.parametrize(
    "depth,name,outputs",
    [
        (18, "RESNET18", 12),
        (18, "RESNET18_TILED", 12),
        (34, "RESNET34", 20),
        (34, "RESNET34_TILED", 20),
        (50, "RESNET50", 20),
        (50, "RESNET50_TILED", 20),
        (101, "RESNET101", 37),
        (101, "RESNET101_TILED", 37),
        (152, "RESNET152", 54),
        (152, "RESNET152_TILED", 54),
    ],
)
def test_saved_resnet_interface(depth, name, outputs):
    project = Project(
        SCALE / "hls_files" / (name + "_config_ap_fixed_16_5_"), f"resnet{depth}"
    )
    assert len(project.outputs) == outputs
    assert project.outputs["logits"][1] == (1000,)
    assert project.contract_version == 2
    assert not project.findings
    project.assert_unchanged()


@pytest.mark.parametrize("context", [2048, 8192])
def test_saved_llama_interface(context):
    folder = SCALE / f"hls_files/LLAMA3_8B_PREFILL_ctx{context}_config_ap_fixed_16_5_"
    if not folder.exists():
        pytest.skip("production Llama project absent")
    p = Project(folder, "llama3")
    assert p.variant == "llama3-prefill"
    assert p.max_ctx == context
    assert p.contract_version == 2
    assert not p.findings
    assert p.port_types["DRAM_token_ids"] == "int32_t"


def test_llama_linear_and_norm(vendor):
    rng = np.random.default_rng(412)
    x = rng.integers(-32768, 32768, (3, 137), dtype=np.int64)
    w = rng.integers(-32768, 32768, (19, 137), dtype=np.int64)
    y = np.empty((3, 19), dtype=np.int64)
    vendor.dll.linear(x, w, y, 3, 137, 19, 33)
    np.testing.assert_array_equal(
        LlamaFixedOps(tile_in=33).linear(torch.from_numpy(x), torch.from_numpy(w)), y
    )
    x = rng.integers(-256, 256, (7, 128), dtype=np.int64)
    gamma = rng.integers(1900, 2200, 128, dtype=np.int64)
    y = np.empty_like(x)
    vendor.dll.norm(x, gamma, y, 7, 128, 31)
    np.testing.assert_array_equal(
        LlamaFixedOps(hidden_chunk=31).rmsnorm(
            torch.from_numpy(x), torch.from_numpy(gamma)
        ),
        y,
    )
    with pytest.raises(ArithmeticFault, match="4096"):
        LlamaFixedOps().rmsnorm(
            torch.zeros((1, 4096), dtype=torch.int64),
            torch.ones(4096, dtype=torch.int64),
        )


def test_llama_rope_all_context_positions(vendor):
    rng = np.random.default_rng(128)
    x = rng.integers(-32768, 32768, (2048, 128), dtype=np.int64)
    y = np.empty_like(x)
    vendor.dll.rope(x, y, 2048, 0)
    np.testing.assert_array_equal(LlamaFixedOps().rope(torch.from_numpy(x), 0, 1), y)


def test_separate_comparison_verdicts():
    actual = np.array([2048, -2048], dtype=np.int16)
    result = compare_arrays(actual, actual, np.array([2.0, -2.0]))
    assert result["implementation_verdict"] == "PASS"
    assert result["mathematical_verdict"] == "FAIL"
    result = compare_arrays(actual, actual + 1, actual.astype(float) / 2048)
    assert result["implementation_verdict"] == "FAIL"
    assert result["mathematical_verdict"] == "PASS"
    assert result["max_code_error"] == 1
    # The requested 5% relative-L2 rule is decisive even when max_abs > 0.1.
    large = np.array([20480], dtype=np.int16)
    result = compare_arrays(large, large, np.array([9.8]))
    assert result["max_abs"] > 0.1
    assert result["relative_l2"] < 0.05
    assert result["mathematical_verdict"] == "PASS"
    result = compare_arrays(large, large, np.array([9.4]))
    assert result["relative_l2"] > 0.05
    assert result["mathematical_verdict"] == "FAIL"
    zero = np.zeros(2, dtype=np.int16)
    assert (
        compare_arrays(zero, zero, zero.astype(float))["mathematical_verdict"] == "PASS"
    )
    with pytest.raises(ValueError, match="nonfinite"):
        compare_arrays(actual, actual, np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="shapes"):
        compare_arrays(actual, actual[:1], np.zeros(2))


def test_interface_changes_are_not_silently_accepted(tmp_path):
    original = SCALE / "hls_files/RESNET18_TILED_config_ap_fixed_16_5_"
    for name in ("top.cpp", "top.h"):
        (tmp_path / name).write_bytes((original / name).read_bytes())
    (tmp_path / "top.h").write_text(
        (tmp_path / "top.h").read_text().replace("[3][224][224]", "[4][224][224]")
    )
    with pytest.raises(ValueError, match="port mismatch"):
        Project(tmp_path, "resnet18")
    (tmp_path / "top.h").write_bytes((original / "top.h").read_bytes())
    with (tmp_path / "top.cpp").open("a") as f:
        f.write("\n// changed source version\n")
    with pytest.raises(ValueError, match="unregistered"):
        Project(tmp_path, "resnet18")


def test_corrupted_input(tmp_path):
    path = tmp_path / "DRAM_input.bin"
    path.write_bytes(b"\0\0")
    entry = {"DRAM_input": dict(shape=[1], sha256=sha(path))}
    validate_inputs(tmp_path, entry)
    path.write_bytes(b"\1\0")
    with pytest.raises(ValueError, match="changed"):
        validate_inputs(tmp_path, entry)
    path.write_bytes(b"\0")
    with pytest.raises(ValueError, match="truncated"):
        validate_inputs(tmp_path, entry)


def test_llama_preflight_and_inspection_do_not_execute(tmp_path):
    prefill = SCALE / "hls_files/LLAMA3_8B_PREFILL_ctx2048_config_ap_fixed_16_5_"
    decode = SCALE / "hls_files/LLAMA3_8B_DECODE_ctx2048_config_ap_fixed_16_5_"
    if not prefill.exists() or not decode.exists():
        pytest.skip("production Llama pair absent")
    old_prefill = Project(prefill, "llama3")
    old_decode = Project(decode, "llama3")
    old_prefill.contract_version = old_decode.contract_version = 1
    _, failures = llama_preflight(
        old_prefill,
        old_decode,
        SimpleNamespace(prefill=4, decode=2, tokens="1,2,3,4,5,6", seed=42),
    )
    assert len(failures) == 2
    assert "RMSNorm" in failures[0] and "attention" in failures[1]
    output = tmp_path / "inspected_llama"
    result = subprocess.run(
        [
            sys.executable,
            str(SCALE / "existing_model_verification/verify.py"),
            "--family",
            "llama3",
            "--project",
            str(prefill),
            "--decode-project",
            str(decode),
            "--tokens",
            "128000,43,160,2000,31,27",
            "--inspect-only",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "INSPECTED_NOT_EXECUTED"
    assert (
        report["implementation_verdict"] == report["mathematical_verdict"] == "NOT_RUN"
    )
    assert not report["preflight_failures"]
    assert len(report["reference_pairs"]) == 2
    assert report["original_sources_unchanged"]
    assert not report["accelerator_executed"]
    assert not (output / "inputs").exists()


def test_inspect_does_not_claim_numerical_pass(tmp_path):
    output = tmp_path / "inspection"
    result = subprocess.run(
        [
            sys.executable,
            str(SCALE / "existing_model_verification/verify.py"),
            "--family",
            "resnet18",
            "--inspect-only",
            "--project",
            str(SCALE / "hls_files/RESNET18_TILED_config_ap_fixed_16_5_"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "INSPECTED_NOT_EXECUTED"
    assert (
        report["implementation_verdict"] == report["mathematical_verdict"] == "NOT_RUN"
    )
    assert not report["accelerator_executed"]


def test_vendor_rope_loads_in_fresh_process():
    if not (INCLUDE / "ap_fixed.h").exists():
        pytest.skip("real Vitis headers unavailable")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from existing_model_verification.references.vendor_math import VendorRope; "
            f"p = VendorRope({str(INCLUDE.parent)!r}); "
            "t = p.coefficients(0, 1); "
            "assert t.shape == (1, 64, 2); "
            "assert (t[:,:,0] == 2048).all() and (t[:,:,1] == 0).all()",
        ],
        cwd=SCALE,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_llama_prefill_decode_reference_cache():
    # Reduced fixture tests the reference's state machinery, not an 8B claim.
    shape = (2, 8, 1, 128)
    project = SimpleNamespace(
        tile_in=33,
        hidden_chunk=31,
        max_ctx=8,
        ports={"DRAM_k_cache": shape, "DRAM_embedding": (20, 512)},
    )
    rng = np.random.default_rng(512)
    params = {}
    for name, dims in {
        "embedding": (20, 512),
        "lm_head": (20, 512),
        "final_norm": (512,),
        "attn_norm": (2, 512),
        "ffn_norm": (2, 512),
        "q_proj": (2, 512, 512),
        "k_proj": (2, 128, 512),
        "v_proj": (2, 128, 512),
        "o_proj": (2, 512, 512),
        "gate_proj": (2, 640, 512),
        "up_proj": (2, 640, 512),
        "down_proj": (2, 512, 640),
    }.items():
        params["DRAM_" + name] = torch.from_numpy(
            rng.integers(-64, 65, dims, dtype=np.int64)
        )
        if "norm" in name:
            params["DRAM_" + name] += 2048

    class Store:
        def get(self, name, layer=None):
            return params[name] if layer is None else params[name][layer]

    one = LlamaReference(project, Store(), False)
    split = LlamaReference(project, Store(), False)
    fixed = LlamaReference(project, Store(), True)
    assert fixed.keys.data_ptr() != split.keys.data_ptr()
    tokens = [3, 8, 15, 1, 19]
    all_logits = one.forward(tokens, 0, lambda n, v: None)
    prefill = split.forward(tokens[:3], 0, lambda n, v: None)
    decode1 = split.forward(tokens[3:4], 3, lambda n, v: None)
    decode2 = split.forward(tokens[4:], 4, lambda n, v: None)
    torch.testing.assert_close(
        torch.cat((prefill, decode1, decode2)), all_logits, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(one.keys, split.keys, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(one.values, split.values, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(fixed.keys) == 0
    with pytest.raises(ValueError, match="position"):
        split.forward([1], 0, lambda n, v: None)
