import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

import generate_tiled_resnet18 as gen
from tiled_resnet18_verification.fixed import FixedOps, quantize, sqrt_codes
from tiled_resnet18_verification.io import configuration, digest, prepare, read_codes, validate_case, write_codes, write_json, reference_fingerprint
from tiled_resnet18_verification.model import FloatOps
from tiled_resnet18_verification.runner import numerical_error, validate_reference


@pytest.fixture(scope="session", autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def probe(tmp_path_factory):
    tool = shutil.which("vitis_hls")
    include = Path(os.environ.get("VITIS_HLS_INCLUDE", str(Path(tool).resolve().parent.parent / "include") if tool else ""))
    if not (include / "ap_fixed.h").is_file():
        pytest.skip("real Vitis headers unavailable; set VITIS_HLS_INCLUDE")
    directory = tmp_path_factory.mktemp("vitis_probe")
    gen.emit_project(str(directory), str(directory))
    source = Path(__file__).with_name("arithmetic_probe.cpp")
    executable = directory / "probe"
    subprocess.run(["g++", "-std=c++14", "-O2", "-DNDEBUG", "-Wno-unknown-pragmas", "-I" + str(include),
                    "-I" + str(directory), str(source), "-o", str(executable)], check=True, capture_output=True, text=True)

    def run(commands):
        completed = subprocess.run([str(executable)], input="\n".join(commands) + "\n", text=True, capture_output=True, check=True)
        return torch.tensor([int(x) for x in completed.stdout.split()], dtype=torch.int64)
    return run


def test_assignment_rounding_and_saturation(probe):
    x = torch.tensor([-17, -16, -1.5 / 2048, -0.5 / 2048, 0.5 / 2048, 1.5 / 2048, 15.99951171875, 16, 17], dtype=torch.float64)
    assert torch.equal(probe([f"quant {v:.17g}" for v in x.tolist()]), quantize(x))


def test_division_and_shift_saturation(probe):
    pairs = [(v, s) for v in [-2147483648, -7, -3, -1, 0, 1, 3, 7, 2147483647] for s in [-3, -1, 0, 1, 3]]
    expected = torch.tensor([FixedOps().shift(torch.tensor(v), s).item() for v, s in pairs])
    assert torch.equal(probe([f"shift {v} {s}" for v, s in pairs]), expected)


def test_sqrt_entire_stored_variance_domain(probe):
    variance = torch.arange(32768, dtype=torch.int64)
    assert torch.equal(probe([f"sqrt {v}" for v in variance.tolist()]), sqrt_codes(variance))


def test_bn_random_and_extreme_parameters(probe):
    rng = np.random.default_rng(17)
    rows = rng.integers(-32768, 32768, size=(512, 5))
    rows[:, 4] = rng.integers(0, 32768, size=512)
    rows[:8, 4] = [0, 1, 2, 1843, 2048, 2253, 32766, 32767]
    x = torch.from_numpy(rows[:, 0].copy()).reshape(-1, 1, 1)
    params = torch.from_numpy(rows[:, 1:].T.copy())
    expected = FixedOps().bn(x, params).flatten()
    assert torch.equal(probe(["bn " + " ".join(map(str, row)) for row in rows]), expected)


@pytest.mark.parametrize("stride", [1, 2])
def test_tiled_convolution_and_fast_path(probe, stride):
    c, h, w = torch.meshgrid(torch.arange(129), torch.arange(17), torch.arange(17), indexing="ij")
    x = (c * 7 + h * 13 + w * 3) % 4097 - 2048
    o, c, h, w = torch.meshgrid(torch.arange(2), torch.arange(129), torch.arange(3), torch.arange(3), indexing="ij")
    weights = (o * 31 + c * 3 + h * 7 + w) % 129 - 64
    fast = FixedOps().conv(x, weights, stride=stride, padding=1)
    slow = FixedOps(fast=False).conv(x, weights, stride=stride, padding=1)
    assert torch.equal(fast, slow)
    assert torch.equal(probe([f"conv {stride}"]), slow.flatten())


def test_known_exponent_and_saturation_defects(probe):
    x = torch.full((512,), 2048, dtype=torch.int64)
    w = torch.zeros((1, 512), dtype=torch.int64)
    w[:, :128] = 3 * 2048
    w[:, 128:256] = -2048
    result = FixedOps().fc(x, w, 5)
    assert result.item() == 4 * 2048
    assert FloatOps().fc(x.double() / 2048, w.double() / 2048, 5).item() == 8
    assert torch.equal(probe(["fc 0"]), result)
    w.zero_()
    w[:, :128] = 8 * 2048
    overflow = FixedOps().fc(x, w, 7)
    assert overflow.item() == 4 * 2048
    assert torch.equal(probe(["fc 1"]), overflow)
    for value in [0, 1, 6, 12, -12]:
        result = FixedOps().gap(torch.full((1, 7, 7), value * 2048, dtype=torch.int64))
        assert torch.equal(probe([f"gap {value * 2048}"]), result)
        if abs(value) == 12:
            assert result.item() != value * 2048


def test_pool_zero_padding_and_residual_saturation():
    x = torch.full((1, 4, 4), -2048, dtype=torch.int64)
    result = FixedOps().pool(x)
    assert result.tolist() == [[[0, 0], [0, -2048]]]
    assert FixedOps().add(torch.tensor([32767, -32768]), torch.tensor([1, -1])).tolist() == [32767, -32768]


def test_fc_partial_output_tile_and_shared_exponent(probe):
    x = torch.full((256,), 2048, dtype=torch.int64)
    weights = torch.zeros((1000, 256), dtype=torch.int64)
    weights[:, 128:] = (torch.arange(1000) % 7 - 3)[:, None]
    weights[0, :128] = 3 * 2048
    weights[0, 128:] = -2048
    fast_ops, slow_ops = FixedOps(), FixedOps(fast=False)
    fast, slow = fast_ops.fc(x, weights), slow_ops.fc(x, weights)
    assert torch.equal(fast, slow)
    assert fast_ops.events == slow_ops.events
    assert torch.equal(probe(["fc_tail"]), fast)
    # Channel 1 shares channel 0's exponent; the next output tile does not.
    assert fast[1].item() == -512
    assert fast[134].item() == -256
    assert fast.shape == (1000,)


def test_zero_and_impulse_cross_correlation():
    ops = FixedOps()
    weights = torch.arange(1, 10, dtype=torch.int64).reshape(1, 1, 3, 3) * 2048
    x = torch.zeros((1, 17, 17), dtype=torch.int64)
    assert not ops.conv(x, weights, padding=1).any()
    x[0, 14, 14] = 2048
    result = ops.conv(x, weights, padding=1)
    assert torch.equal(result[0, 13:16, 13:16], weights[0, 0].flip((0, 1)))
    assert result.count_nonzero() == 9


def test_configuration_reads_actual_shifts_and_rejects_mismatch(tmp_path):
    gen.emit_project(str(tmp_path), str(tmp_path))
    assert configuration(tmp_path)["shifts"]["SHIFT_STEM"] == 0
    path = tmp_path / "resnet18_tiled_scales.h"
    path.write_text(path.read_text().replace("kConvOutputShift[21] = {0, 0", "kConvOutputShift[21] = {2, -1"))
    assert configuration(tmp_path)["shifts"]["SHIFT_STEM"] == 2
    assert configuration(tmp_path)["shifts"]["SHIFT_S1_B0_1"] == -1
    path = tmp_path / "top.cpp"
    path.write_text(path.read_text().replace("TILE_C = 128", "TILE_C = 64"))
    with pytest.raises(ValueError, match="TILE_C"):
        configuration(tmp_path)


def test_stale_and_corrupted_reference_detection(tmp_path):
    (tmp_path / "fixed").mkdir()
    write_json(tmp_path / "manifest.json", {"seed": 42})
    path = tmp_path / "fixed/head.logits.bin"
    write_codes(path, torch.tensor([1, 2, 3]))
    (tmp_path / "fixed/exponents.txt").write_text("")
    meta = {"manifest_sha256": digest(tmp_path / "manifest.json"), "implementation": reference_fingerprint(),
            "checkpoints": [{"name": "head.logits", "shape": [3], "sha256": digest(path)}],
            "exponents_sha256": digest(tmp_path / "fixed/exponents.txt")}
    write_json(tmp_path / "fixed/metadata.json", meta)
    validate_reference(tmp_path, "fixed")
    write_codes(path, torch.tensor([1, 2, 4]))
    with pytest.raises(ValueError, match="corrupted"):
        validate_reference(tmp_path, "fixed")
    write_codes(path, torch.tensor([1, 2, 3]))
    write_json(tmp_path / "manifest.json", {"seed": 43})
    with pytest.raises(ValueError, match="different inputs"):
        validate_reference(tmp_path, "fixed")


def test_seed_reproducibility_and_corrupt_input_detection(tmp_path):
    project = tmp_path / "project"
    gen.emit_project(str(tmp_path), str(project))
    first = prepare(project, tmp_path / "first", 42)
    second = prepare(project, tmp_path / "second", 42)
    assert first["tensors"] == second["tensors"]
    assert first["sources"] == second["sources"]
    validate_case(tmp_path / "first")
    for entry in first["tensors"]:
        if entry["name"].startswith("DRAM_bn_"):
            codes = read_codes(tmp_path / "first/inputs" / (entry["name"] + ".bin"), entry["shape"])
            assert (codes[3] > 0).all()
    with pytest.raises(ValueError, match="not empty"):
        prepare(project, tmp_path / "first", 43)
    source = tmp_path / "first/inputs/DRAM_input.bin"
    source.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(ValueError, match="corrupted or changed"):
        validate_case(tmp_path / "first")


def test_relative_error_near_zero_and_explicit_tolerance():
    actual = torch.tensor([0.0, 1 / 2048, 2.0], dtype=torch.float64)
    expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    result = numerical_error(actual, expected, atol=1 / 2048, rtol=0)
    assert result["max_relative"] == 1
    assert result["tolerance_mismatches"] == 1


def test_binary_exchange_and_length_validation(tmp_path):
    values = torch.tensor([[[-32768, -1, 0, 1, 32767]]])
    path = tmp_path / "codes.bin"
    write_codes(path, values)
    assert path.read_bytes() == bytes([0, 128, 255, 255, 0, 0, 1, 0, 255, 127])
    assert torch.equal(read_codes(path, values.shape), values)
    with pytest.raises(ValueError, match="byte count"):
        read_codes(path, (4,))
