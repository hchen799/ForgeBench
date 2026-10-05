"""Run with: python -m pytest -q llama3_verification/tests."""
import ctypes
import subprocess

import numpy as np
import pytest
import torch

from llama3_verification.codegen import emit_project
from llama3_verification.config import make_config, segments
from llama3_verification.io import WeightStore, prepare_case, prepare_model, validate_case
from llama3_verification.model import FixedOps, FloatOps, LlamaReference, round_ratio, round_shift
from llama3_verification.runner import compare, csim, reference


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    root = tmp_path_factory.mktemp("llama")
    config = make_config("tiny", prefill_tile=32)
    emit_project(root / "project", config)
    metadata = prepare_model(root / "project", root / "model", seed=17)
    torch.set_num_threads(2)
    return root, config, metadata


def test_full_architecture():
    c = make_config(max_ctx=8192)
    assert (c.layers, c.hidden, c.ffn, c.q_heads, c.kv_heads, c.head_dim, c.vocab) == (32, 4096, 14336, 32, 8, 128, 128256)
    entries = segments(c)
    assert sum(s["count"] for s in entries) == 8030261248
    assert all(a["offset"] + a["count"] == b["offset"] for a, b in zip(entries, entries[1:]))
    assert c.epsilon == 1e-5 and c.rope_theta == 500000


@pytest.mark.parametrize("options", [dict(hidden=65), dict(kv_heads=3), dict(max_ctx=8193), dict(head_dim=15), dict(prefill_tile=33), dict(ffn=65536)])
def test_invalid_architecture(options):
    with pytest.raises(ValueError):
        make_config("tiny", **options)


def test_rounding():
    values = torch.arange(-32768, 32769, dtype=torch.int64)
    for bits in (1, 11, 22, 33):
        expected = np.floor(values.numpy() / (1 << bits) + .5).astype(np.int64)
        np.testing.assert_array_equal(round_shift(values, bits), expected)
    for denominator in (1, 2, 3, 2048, 8192 * (1 << 22)):
        scaled = values * (denominator // 3 + 1)
        expected = torch.tensor([(2 * int(x) + denominator) // (2 * denominator) for x in scaled])
        assert torch.equal(round_ratio(scaled, denominator), expected)


def test_nonlinear_and_zero_norm(model):
    root, c, _ = model
    rope = np.fromfile(root / "model/rope.bin", dtype="<i2").reshape(c.max_ctx, c.head_dim)
    ops = FixedOps(c, rope)
    x = torch.arange(-32768, 32768, dtype=torch.int64)
    mathematical = x.double() / (1 + torch.exp(-x.double() / 2048))
    assert torch.equal(ops.silu(x), torch.floor(mathematical + .5).clamp(-32768, 32767).long())
    assert ops.exp_table[0] == 1 << 22
    assert bool(torch.all(ops.exp_table[1:] <= ops.exp_table[:-1]))
    assert torch.count_nonzero(ops.rmsnorm(torch.zeros((3, c.hidden), dtype=torch.int64), torch.full((c.hidden,), 2048))) == 0


@pytest.mark.parametrize("fixed", [False, True])
def test_cache_chunking_and_decode_equivalence(model, fixed):
    root, c, metadata = model
    weights = WeightStore(root / "model", metadata, cache_gib=.1)
    rope = np.fromfile(root / "model/rope.bin", dtype="<i2").reshape(c.max_ctx, c.head_dim)
    full = LlamaReference(c, weights, fixed, rope)
    chunked = LlamaReference(c, weights, fixed, rope)
    tokens = np.random.default_rng(88).integers(0, c.vocab, 32, dtype=np.int32)
    tokens[0] = c.vocab - 1
    sink = lambda *args: None
    expected = full.forward(tokens, 0, 0, sink)
    actual = torch.cat([chunked.forward(tokens[a:b], a, i, sink) for i, (a, b) in enumerate(((0, 4), (4, 13), (13, 30), (30, 31), (31, 32)))])
    if fixed:
        assert torch.equal(actual, expected)
        assert torch.equal(full.keys, chunked.keys)
        assert torch.equal(full.values, chunked.values)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(full.keys, chunked.keys, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError):
        chunked.forward(tokens[:1], 32, 9, sink)


@pytest.mark.parametrize("fixed", [False, True])
def test_gqa_and_strict_causality(model, fixed):
    root, c, _ = model
    rope = np.fromfile(root / "model/rope.bin", dtype="<i2").reshape(c.max_ctx, c.head_dim)
    ops = FixedOps(c, rope) if fixed else FloatOps(c)
    dtype = torch.int64 if fixed else torch.float64
    q = torch.zeros((4, c.hidden), dtype=dtype)
    k = torch.zeros((4, c.kv_dim), dtype=dtype)
    v = torch.arange(1, 5, dtype=dtype)[:, None].expand(4, c.kv_dim) * 2048
    result = ops.attention(q, k, v, 0)
    # Uniform scores: every head sees the mean of only its causal prefix.
    expected = torch.arange(1, 5, dtype=torch.float64).add(1).mul(1024)
    torch.testing.assert_close(result.double(), expected[:, None].expand_as(result), atol=0, rtol=0)
    v[-1] = -32768
    changed = ops.attention(q, k, v, 0)
    assert torch.equal(changed[:3], result[:3])


def test_native_cpp_primitives_and_rejected_arguments(model):
    root, _, _ = model
    source = root / "project/primitives.cpp"
    source.write_text(r'''#include "top.cpp"
#include <cassert>
#include <limits>
int main() {
    assert(round_data(1024, 11) == 1);
    assert(round_data(-1024, 11) == 0);
    assert(round_data(-3072, 11) == -1);
    assert(saturate_data(100000) == 32767 && saturate_data(-100000) == -32768);
    for (int64_t n = -1000; n <= 1000; ++n) for (int64_t d = 1; d < 30; ++d) {
        int64_t q = n / d, r = n % d;
        if (r < 0) { --q; r += d; }
        assert(rounded_ratio(n, d) == q + (2 * r >= d));
    }
    uint64_t value = 3;
    for (int i = 0; i < 10000; ++i) {
        value ^= value << 13; value ^= value >> 7; value ^= value << 17;
        uint64_t r = integer_sqrt(value);
        assert(r == 0 || r <= value / r);
        assert(r == 0xffffffffULL || r + 1 > value / (r + 1));
    }
    assert(integer_sqrt(0) == 0 && integer_sqrt(1) == 1);
    assert(integer_sqrt(std::numeric_limits<uint64_t>::max()) == 0xffffffffULL);
    assert(llama_prefill(nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == 1);
    assert(llama_decode(0, MAX_CTX, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == 1);
    assert(llama_decode(-1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == 2);
    assert(llama_decode(VOCAB, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == 2);
}
''')
    subprocess.run(["g++", "-std=c++14", "-O2", str(source), "-o", str(root / "primitives")], check=True)
    subprocess.run([str(root / "primitives")], check=True)


def test_end_to_end_and_corruption(model, tmp_path):
    root, _, _ = model
    run = tmp_path / "run"
    prepare_case(root / "model", run, prefill=30, decode=2, chunk=7)
    reference(run, threads=2, cache_gib=.1)
    csim(run, backend="native")
    result = compare(run)
    assert result["implementation_match"]
    assert result["saturation_count"] == 0
    assert result["logit_errors"]["count"] == 32 * 257
    with (run / "tokens.bin").open("r+b") as stream:
        stream.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="corrupted"):
        validate_case(run)


def test_non_multiple_tiles(tmp_path):
    c = make_config("tiny", hidden=60, q_heads=3, kv_heads=1, head_dim=20, ffn=131,
                    vocab=259, prefill_tile=3, max_ctx=9)
    emit_project(tmp_path / "project", c)
    prepare_model(tmp_path / "project", tmp_path / "model", seed=29)
    prepare_case(tmp_path / "model", tmp_path / "case", prefill=7, decode=2)
    reference(tmp_path / "case", threads=2, cache_gib=.1)
    csim(tmp_path / "case", backend="native")
    assert compare(tmp_path / "case")["implementation_match"]


def test_cpp_extreme_datapath_values(model):
    root, c, _ = model
    wrapper = root / "project/probes.cpp"
    wrapper.write_text('''#include "top.cpp"
extern "C" {
void probe_linear(const int16_t *x, const int16_t *w, int16_t *y, int rows) { linear(x, w, y, rows, HIDDEN, FFN); }
void probe_norm(const int16_t *x, const int16_t *g, int16_t *y, int rows) { rmsnorm(x, g, y, rows); }
void probe_rope(int16_t *x, const int16_t *table, int rows, int start) { rope(x, table, rows, Q_HEADS, start); }
void probe_attention(const int16_t *q, const int16_t *k, const int16_t *v, int16_t *y, int rows, int start) { attention(q, k, v, y, rows, start); }
}
''')
    library = root / "probes.so"
    subprocess.run(["g++", "-shared", "-fPIC", "-std=c++14", "-O2", str(wrapper), "-o", str(library)], check=True)
    probes = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.int16, flags="C_CONTIGUOUS")
    integer = ctypes.c_int
    probes.probe_linear.argtypes = [pointer, pointer, pointer, integer]
    probes.probe_norm.argtypes = [pointer, pointer, pointer, integer]
    probes.probe_rope.argtypes = [pointer, pointer, integer, integer]
    probes.probe_attention.argtypes = [pointer, pointer, pointer, pointer, integer, integer]
    rng = np.random.default_rng(97)
    raw = lambda shape: rng.integers(-32768, 32768, shape, dtype=np.int16)
    tensor = lambda x: torch.from_numpy(x.astype(np.int64))
    rope = np.fromfile(root / "model/rope.bin", dtype="<i2").reshape(c.max_ctx, c.head_dim)
    ops = FixedOps(c, rope)
    rows, start = 3, c.max_ctx - 3
    x, w = raw((rows, c.hidden)), raw((c.ffn, c.hidden))
    x[0] = -32768
    w[0] = -32768
    y = np.empty((rows, c.ffn), dtype=np.int16)
    probes.probe_linear(x, w, y, rows)
    np.testing.assert_array_equal(y, ops.linear(tensor(x), tensor(w).double()))
    gamma = raw((c.hidden,))
    x[1] = 0
    z = np.empty_like(x)
    probes.probe_norm(x, gamma, z, rows)
    np.testing.assert_array_equal(z, ops.rmsnorm(tensor(x), tensor(gamma)))
    z = x.copy()
    probes.probe_rope(z, rope, rows, start)
    np.testing.assert_array_equal(z, ops.rope(tensor(x), start, c.q_heads))
    k, v = raw((c.max_ctx, c.kv_dim)), raw((c.max_ctx, c.kv_dim))
    probes.probe_attention(x, k, v, z, rows, start)
    np.testing.assert_array_equal(z, ops.attention(tensor(x), tensor(k), tensor(v), start))
    assert sum(ops.saturations.values()) > 0
