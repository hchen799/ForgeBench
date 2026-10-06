"""Q32.10 reference checks against actual production operators and Vitis types."""
import math

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from test_production import compile_probes
from verification.existing.wide import LlamaOps, ResNetOps, exp_negative, dot
from verification.existing.arithmetic import Format
from verification.existing.runner import prepare_inputs, compare_arrays


@pytest.fixture(scope="module")
def production32(tmp_path_factory):
    torch.set_num_threads(4)
    return compile_probes(tmp_path_factory.mktemp("production32"), 32, 10)


def test_limb_reduction_exceeds_int64():
    x = torch.tensor([[-(2**31), 2**31 - 1] * 65])
    assert dot(x, x.T)[0, 0] == sum(int(v) ** 2 for v in x[0])


def test_precision_changes_preserve_underlying_random_draws(tmp_path):
    inputs = {
        "DRAM_input": (3, 4, 4),
        "DRAM_w_test": (2, 3, 3, 3),
        "DRAM_bn_test": (4, 2),
    }
    for word, integer in ((16, 5), (32, 10)):
        project = SimpleNamespace(
            format=Format(word, integer), family="resnet18", inputs=inputs
        )
        prepare_inputs(project, tmp_path / str(word), 42)
    for name in inputs:
        old = np.fromfile(tmp_path / "16" / (name + ".bin"), dtype="<i2")
        new = np.fromfile(tmp_path / "32" / (name + ".bin"), dtype="<i4")
        np.testing.assert_array_equal(new >> 11, old)
    actual = np.array([1 << 22, -(1 << 22)], dtype=np.int64)
    result = compare_arrays(actual, actual, np.array([1.0, -1.0]), fmt=Format(32, 10))
    assert result["implementation_verdict"] == result["mathematical_verdict"] == "PASS"


def test_primitives32(production32):
    rng = np.random.default_rng(101)
    a = rng.integers(0, 1 << 58, 4000, dtype=np.int64)
    b = rng.integers(1 << 35, 1 << 45, 4000, dtype=np.int64)
    out = np.empty_like(a)
    production32.primitives(a, b, out, len(a), 0)
    expected = []
    for value in a:
        n = int(value) << 44
        r = math.isqrt(n)
        expected.append(r + int(n - r * r > r))
    np.testing.assert_array_equal(out, expected)
    a = rng.integers(-(1 << 45), 1 << 45, 4000, dtype=np.int64)
    b[::2] *= -1
    production32.primitives(a, b, out, len(a), 1)
    np.testing.assert_array_equal(out, [(int(x) << 44) // int(y) for x, y in zip(a, b)])
    a = rng.integers(-20 * (1 << 44), 1 << 44, 4000, dtype=np.int64)
    production32.primitives(a, b, out, len(a), 2)
    np.testing.assert_array_equal(
        out, exp_negative(torch.from_numpy(a)).astype(np.int64)
    )


def test_full_range_partial_linear32(production32):
    rng = np.random.default_rng(102)
    x = rng.integers(-(1 << 31), 1 << 31, (3, 137), dtype=np.int64)
    w = rng.integers(-(1 << 31), 1 << 31, (19, 137), dtype=np.int64)
    out = np.empty((3, 19), dtype=np.int64)
    production32.linear_probe(x, w, out)
    np.testing.assert_array_equal(
        out, LlamaOps().linear(torch.from_numpy(x), torch.from_numpy(w))
    )


def test_full_range_4096_rms32(production32):
    rng = np.random.default_rng(103)
    x = rng.integers(-(1 << 31), 1 << 31, (2, 4096), dtype=np.int64)
    gamma = rng.integers(3800000, 4500000, 4096, dtype=np.int64)
    out = np.empty_like(x)
    production32.norm_probe(x, gamma, out)
    np.testing.assert_array_equal(
        out, LlamaOps().rmsnorm(torch.from_numpy(x), torch.from_numpy(gamma))
    )
    x.fill(0)
    production32.norm_probe(x, gamma, out)
    assert not out.any()


def test_bn_swiglu32(production32):
    rng = np.random.default_rng(104)
    for _ in range(10):
        x = rng.integers(-(1 << 31), 1 << 31, (4, 3, 3), dtype=np.int64)
        p = rng.integers(-6000000, 6000000, (4, 4), dtype=np.int64)
        p[3] = rng.integers(0, 1 << 31, 4, dtype=np.int64)
        p[3, 0] = 0
        out = np.empty_like(x)
        production32.bn_probe(x, p, out)
        np.testing.assert_array_equal(
            out, ResNetOps().bn(torch.from_numpy(x), torch.from_numpy(p))
        )
    x = rng.integers(-(1 << 31), 1 << 31, (3, 137), dtype=np.int64)
    up = rng.integers(-(1 << 31), 1 << 31, (3, 137), dtype=np.int64)
    out = np.empty_like(x)
    production32.swiglu_probe(x, up, out)
    np.testing.assert_array_equal(
        out, LlamaOps().swiglu(torch.from_numpy(x), torch.from_numpy(up))
    )


def test_attention32(production32):
    rng = np.random.default_rng(105)
    q = rng.integers(-(1 << 23), 1 << 23, (3, 512), dtype=np.int64)
    k = rng.integers(-(1 << 23), 1 << 23, (7, 128), dtype=np.int64)
    v = rng.integers(-(1 << 23), 1 << 23, (7, 128), dtype=np.int64)
    out = np.empty_like(q)
    production32.attention_probe(q, k, v, out)
    np.testing.assert_array_equal(
        out,
        LlamaOps().attention(
            torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), 4
        ),
    )
    prior = out[0].copy()
    v[5:] = (1 << 31) - 1
    production32.attention_probe(q, k, v, out)
    np.testing.assert_array_equal(out[0], prior)
