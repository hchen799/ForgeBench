"""Directed format, exact-arithmetic, real C++ and CLI regression coverage."""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from verification.precision import FixedFormat, exact_dot, objects, round_shift, clamp_integer
from verification.models.llama3.config import make_config
from verification.models.llama3.codegen import emit_project
from verification.models.llama3.io import prepare_model, prepare_case, WeightStore
from verification.models.llama3.model import FixedOps, LlamaReference
from verification.models.llama3.runner import reference, csim, compare
from verification.verify import main, resolve_config, finish
from verification.examples.make_generic_fixture import create


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize("w,i", [(16, 5), (32, 10), (24, 8), (12, 6), (8, 4), (32, 8)])
def test_quantization_formats(w, i):
    p = FixedFormat(w, i)
    x = torch.tensor([-1e5, -1.5 / p.scale, -.5 / p.scale, .5 / p.scale, 1.5 / p.scale, 1e5], dtype=torch.float64)
    assert p.quantize(x).tolist() == [p.minimum, -1, 0, 1, 2, p.maximum]
    codes = np.array([p.minimum, -1, 0, p.maximum], dtype=p.dtype)
    assert np.frombuffer(codes.tobytes(), dtype=p.dtype).tolist() == codes.tolist()
    assert p.itemsize == (2 if w <= 16 else 4)


@pytest.mark.parametrize("options", [dict(word_bits=33), dict(word_bits=7), dict(integer_bits=16),
                                     dict(word_bits=32, integer_bits=7), dict(rounding="truncate"),
                                     dict(overflow="wrap"), dict(word_bits=16.0)])
def test_rejected_formats(options):
    with pytest.raises(ValueError):
        FixedFormat(**options)


def test_exact_dot_extreme_and_cancellation():
    a = torch.tensor([[-2**31] * 8, [2**31 - 1, -2**31] * 4], dtype=torch.int64)
    b = torch.tensor([[-2**31, 2**31 - 1], [2**31 - 1, 2**31 - 1]] * 4, dtype=torch.int64)
    expected = np.array([[sum(int(x) * int(y) for x, y in zip(row, col)) for col in b.T] for row in a], dtype=object)
    np.testing.assert_array_equal(exact_dot(a, b), expected)
    assert abs(expected[1, 0]) > 2**63
    for bits in (1, 11, 22, 63, 72):
        x = torch.tensor([-2**63, -3, -1, 0, 1, 3, 2**63 - 1])
        np.testing.assert_array_equal(round_shift(x, bits), [(int(v) + 2**(bits - 1)) // 2**bits for v in x])
    with pytest.raises(ValueError):
        exact_dot(torch.tensor([[.5]]), torch.ones((1, 1)))


def test_clamp_preserves_scalar_and_array_int64_endpoints():
    lo, hi = -2**63, 2**63-1
    values = [lo*2, lo-1, lo, lo+1, -1, 0, 1, hi-1, hi, hi+1, hi*2, 2**127]
    expected = [min(hi, max(lo, v)) for v in values]
    assert [clamp_integer(v, lo, hi).item() for v in values] == expected
    assert clamp_integer(values, lo, hi).tolist() == expected
    assert clamp_integer(np.array(values, dtype=object), lo, hi).tolist() == expected


def test_llama_scalar_wide_saturation():
    c = make_config("tiny", word_bits=32, integer_bits=10)
    ops = FixedOps(c, np.zeros((c.max_ctx, c.head_dim), dtype=np.int32))
    assert ops.commit(2**70).item() == c.precision.maximum
    assert ops.commit(-2**70).item() == c.precision.minimum
    assert sum(ops.saturations.values()) == 2


@pytest.mark.parametrize("w,i,synthesis", [(16, 5, False), (32, 10, False), (24, 8, False),
                                          (12, 6, False), (8, 4, False), (32, 8, False), (32, 10, True)])
def test_real_cpp_extreme_primitives(tmp_path, w, i, synthesis):
    c = make_config("tiny", word_bits=w, integer_bits=i)
    p = c.precision
    emit_project(tmp_path, c)
    source = tmp_path / "probe.cpp"
    # Load the vendor's HOST implementation before selecting the kernel's
    # synthesis aliases. Defining __SYNTHESIS__ for the vendor headers themselves
    # needs Vitis compiler intrinsics and cannot be done by ordinary g++.
    prefix = '#include <ap_int.h>\n#define __SYNTHESIS__\n' if synthesis else ''
    source.write_text(prefix + '''#include "top.cpp"
extern "C" {
void linear_probe(const data_t *x,const data_t *w,data_t *y,int rows) { linear(x,w,y,rows,HIDDEN,FFN); }
void norm_probe(const data_t *x,const data_t *g,data_t *y,int rows) { rmsnorm(x,g,y,rows); }
void rope_probe(data_t *x,const data_t *table,int rows,int start) { rope(x,table,rows,Q_HEADS,start); }
void attention_probe(const data_t *q,const data_t *k,const data_t *v,data_t *y,int rows,int start) { attention(q,k,v,y,rows,start); }
void silu_probe(const data_t *x,data_t *y,int count) { for(int i=0;i<count;++i) y[i]=silu_value(x[i]); }
}
''')
    command = ["g++", "-shared", "-fPIC", "-std=c++14", "-O2", str(source), "-o", str(tmp_path / "probe.so")]
    if synthesis:
        include = Path(os.environ.get("VITIS_HLS_INCLUDE", "/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1/include"))
        if not (include / "ap_int.h").is_file():
            pytest.skip("vendor headers unavailable for synthesis-branch arithmetic probe")
        command += ["-I" + str(include)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    library = ctypes.CDLL(str(tmp_path / "probe.so"))
    pointer = np.ctypeslib.ndpointer(dtype=np.dtype(p.dtype), flags="C_CONTIGUOUS")
    integer = ctypes.c_int
    library.linear_probe.argtypes = [pointer, pointer, pointer, integer]
    library.norm_probe.argtypes = [pointer, pointer, pointer, integer]
    library.rope_probe.argtypes = [pointer, pointer, integer, integer]
    library.attention_probe.argtypes = [pointer, pointer, pointer, pointer, integer, integer]
    library.silu_probe.argtypes = [pointer, pointer, integer]
    rng = np.random.default_rng(97)
    raw = lambda shape: rng.integers(p.minimum, p.maximum + 1, shape, dtype=np.dtype(p.dtype))
    tensor = lambda x: torch.from_numpy(x.astype(np.int64))
    rope = raw((c.max_ctx, c.head_dim))
    ops = FixedOps(c, rope)
    rows, start = 3, c.max_ctx - 3
    x, weights = raw((rows, c.hidden)), raw((c.ffn, c.hidden))
    x[0] = p.minimum
    weights[0] = p.minimum
    y = np.empty((rows, c.ffn), dtype=p.dtype)
    library.linear_probe(x, weights, y, rows)
    np.testing.assert_array_equal(y, ops.linear(tensor(x), tensor(weights)))
    gamma = raw((c.hidden,))
    x[1] = 0
    y = np.empty_like(x)
    library.norm_probe(x, gamma, y, rows)
    np.testing.assert_array_equal(y, ops.rmsnorm(tensor(x), tensor(gamma)))
    y = x.copy()
    library.rope_probe(y, rope, rows, start)
    np.testing.assert_array_equal(y, ops.rope(tensor(x), start, c.q_heads))
    keys, values = raw((c.max_ctx, c.kv_dim)), raw((c.max_ctx, c.kv_dim))
    library.attention_probe(x, keys, values, y, rows, start)
    np.testing.assert_array_equal(y, ops.attention(tensor(x), tensor(keys), tensor(values), start))
    library.silu_probe(x, y, x.size)
    np.testing.assert_array_equal(y, ops.silu(tensor(x)))
    assert sum(ops.saturations.values()) > 0


@pytest.mark.parametrize("w,i", [(16, 5), (32, 10), (24, 8), (12, 6)])
def test_partial_tiles_end_to_end(tmp_path, w, i):
    c = make_config("tiny", word_bits=w, integer_bits=i, hidden=60, q_heads=3, head_dim=20,
                    ffn=131, vocab=259, tile_in=17, tile_out=19, max_ctx=9, prefill_tile=3)
    emit_project(tmp_path / "design", c)
    prepare_model(tmp_path / "design", tmp_path / "model", seed=29)
    prepare_case(tmp_path / "model", tmp_path / "case", prefill=7, decode=2)
    reference(tmp_path / "case", threads=2, cache_gib=.1)
    csim(tmp_path / "case", backend="native")
    report = compare(tmp_path / "case")
    assert report["implementation_match"] and report["logit_errors"]["count"] == 9 * c.vocab
    assert report["precision"]["storage_bytes"] == c.precision.itemsize
    if w == 32:
        original = (tmp_path / "case/comparison.json").read_bytes()
        command = ["compare", "--run-dir", str(tmp_path / "case")]
        assert main(command + ["--output", str(tmp_path / "replay")]) == 0
        assert main(command + ["--float-atol", "0", "--float-rtol", "0", "--output", str(tmp_path / "negative")]) == 1
        assert (tmp_path / "case/comparison.json").read_bytes() == original


def test_same_underlying_random_draws_between_formats(tmp_path):
    stores = []
    for w, i in [(16, 5), (32, 10)]:
        c = make_config("tiny", word_bits=w, integer_bits=i)
        directory = tmp_path / str(w)
        emit_project(directory / "design", c)
        metadata = prepare_model(directory / "design", directory / "model", seed=42)
        stores.append((WeightStore(directory / "model", metadata), c.precision))
    for name in stores[0][0].entries:
        a, b = [store.view(name).astype(np.float64) / p.scale for store, p in stores]
        assert np.max(np.abs(a - b)) <= .5 / stores[0][1].scale + .5 / stores[1][1].scale


def test_cli_generic_logs_and_failure_codes(tmp_path):
    manifest = create(tmp_path / "fixture", 32, 10)
    output = tmp_path / "pass"
    assert main(["compare-tensors", "--manifest", str(manifest), "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["saturation_count"] is None
    assert "logit_errors" not in summary
    log = (output / "verification.log").read_text()
    assert all(text in log for text in ["GENERIC COMPARISON CONTRACT", "matmul.output", "integer_mismatches=0", "max_abs=", '"word_bits": 32', "NOT produced", "sha256", "Exit code: 0"])
    actual = np.load(manifest.parent / "actual.npy")
    actual[0, 0] += 1
    np.save(manifest.parent / "actual.npy", actual)
    assert main(["compare-tensors", "--manifest", str(manifest), "--output", str(tmp_path / "fail")]) == 1
    assert main(["compare-tensors", "--manifest", str(manifest), "--output", str(output)]) == 2


@pytest.mark.parametrize("damage", ["shape", "dtype", "nan", "checksum", "format", "tolerance", "unknown"])
def test_generic_invalid_inputs_fail_with_logs(tmp_path, damage):
    path = create(tmp_path / "fixture")
    manifest = json.loads(path.read_text())
    entry = manifest["checkpoints"][0]
    if damage == "shape":
        entry["shape"] = [8]
    elif damage == "dtype":
        entry["floating"]["dtype"] = "<f4"
    elif damage == "nan":
        value = np.load(path.parent / "floating.npy")
        value[0, 0] = np.nan
        np.save(path.parent / "floating.npy", value)
    elif damage == "checksum":
        entry["actual"]["sha256"] = "bad"
    elif damage == "format":
        manifest["precision"]["overflow"] = "wrap"
    elif damage == "tolerance":
        manifest["float_atol"] = .1
    else:
        manifest["ignored_typo"] = True
    path.write_text(json.dumps(manifest))
    output = tmp_path / "result"
    assert main(["compare-tensors", "--manifest", str(path), "--output", str(output)]) == 2
    assert "VERIFICATION ERROR" in (output / "verification.log").read_text()
    assert (output / "error.json").exists()


def test_generic_tolerance_policy(tmp_path):
    path = create(tmp_path / "fixture")
    manifest = json.loads(path.read_text())
    manifest.update(float_atol=0, float_rtol=0)
    path.write_text(json.dumps(manifest))
    output = tmp_path / "result"
    assert main(["compare-tensors", "--manifest", str(path), "--output", str(output)]) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["implementation_verdict"] == "PASS" and summary["mathematical_verdict"] == "FAIL"


@pytest.mark.parametrize("w,i", [(16, 5), (32, 10), (12, 6)])
def test_generic_raw_binary_and_truncation(tmp_path, w, i):
    path = create(tmp_path / "fixture", w, i)
    manifest = json.loads(path.read_text())
    for kind in ("actual", "fixed", "floating"):
        data = np.load(path.parent / f"{kind}.npy")
        data.tofile(path.parent / f"{kind}.bin")
        manifest["checkpoints"][0][kind] = dict(path=f"{kind}.bin", dtype=data.dtype.str)
    path.write_text(json.dumps(manifest))
    assert main(["compare-tensors", "--manifest", str(path), "--output", str(tmp_path / "pass")]) == 0
    actual = path.parent / "actual.bin"
    actual.write_bytes(actual.read_bytes()[:-1])
    assert main(["compare-tensors", "--manifest", str(path), "--output", str(tmp_path / "truncated")]) == 2


def test_optional_saturation_failure_policy(tmp_path):
    path = create(tmp_path / "fixture")
    output = tmp_path / "result"
    assert main(["compare-tensors", "--manifest", str(path), "--output", str(output)]) == 0
    report = json.loads((output / "comparison.json").read_text())
    report.pop("saturation_observability")
    report["saturation_count"] = 1
    assert finish(report, output, dict(fail_on_saturation=True)) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["implementation_verdict"] == "PASS" and summary["overall_verdict"] == "FAIL"


def test_config_validation():
    args = argparse.Namespace(word_bits=None, integer_bits=None)
    raw = dict(schema_version=1, family="llama3", model=dict(profile="tiny"))
    config, c = resolve_config(raw, args)
    assert c.word_bits == 16 and config["test"]["prefill"] == 4
    with pytest.raises(ValueError):
        resolve_config(dict(raw, model=dict(hidden=3)), args)
    with pytest.raises(ValueError):
        resolve_config(dict(raw, execution=dict(typo=1)), args)
    with pytest.raises(ValueError):
        resolve_config(dict(raw, test=dict(prefill=100)), args)
    with pytest.raises(ValueError):
        resolve_config(dict(raw, family="resnet18", model={}, execution=dict(backend="native")), args)
