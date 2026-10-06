"""Reduced integration test of the actual JSON -> production HLS workflow.

Dimensions are reduced only in this test, never in the saved 8B projects.
17 prefill tokens cross the 16-token tile; decode positions and IDs exceed 15.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCALE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCALE))
import auto_generate_json as graph
import gen_configs
from existing_model_verification.references.llama import LlamaReference, Weights
from existing_model_verification.references.arithmetic import Format
from existing_model_verification.references.project import parameters, testbench as make_testbench
from existing_model_verification.references.runner import prepare_inputs, compare_arrays
from existing_model_verification.references.vendor_math import VendorRope


@pytest.mark.parametrize("word,integer", [(16, 5), (32, 10)])
def test_production_prefill_decode(tmp_path, monkeypatch, word, integer):
    fmt = Format(word, integer)
    dtype = "<i2" if word == 16 else "<i4"
    Reference, Provider = LlamaReference, VendorRope
    if word == 32:
        from existing_model_verification.references.wide import (
            LlamaReference32 as Reference,
            Rope32 as Provider,
        )
    include = Path(
        os.environ.get(
            "VITIS_HLS_INCLUDE",
            "/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1/include",
        )
    )
    if not (include / "ap_fixed.h").exists():
        pytest.skip("Vitis headers unavailable")
    torch.set_num_threads(4)
    for key, value in dict(
        LAYERS=2, HIDDEN=512, FFN=640, Q_HEADS=4, KV_HEADS=1, KV_DIM=128, VOCAB=257
    ).items():
        monkeypatch.setattr(graph, "LLAMA3_8B_" + key, value)
    monkeypatch.chdir(SCALE)  # Production templates resolve from scale_models.
    designs = {}
    for mode in ("prefill", "decode"):
        config = json.loads(
            getattr(graph, "generate_llama3_8b_" + mode + "_config_text")(
                20, data_type=f"ap_fixed<{word},{integer}>"
            )
        )
        path = tmp_path / (mode + ".json")
        path.write_text(json.dumps(config))
        gen_configs.run_hls_flow(str(path), str(tmp_path))
        ports = {d["name"]: tuple(d["dims"]) for d in config["drams"]}
        design = SimpleNamespace(
            ports=ports,
            port_types={d["name"]: d.get("dtype", "data_t") for d in config["drams"]},
            brams={},
            inputs=parameters("llama3", ports),
            top_name=config["top_func_name"],
            format=fmt,
            family="llama3",
            max_ctx=20,
            tile_in=128,
            hidden_chunk=128,
            contract_version=2,
            vendor_root=include.parent,
        )
        designs[mode] = design
        outputs = {
            "logits": (
                "DRAM_logits" if mode == "prefill" else "DRAM_logits_decode",
                (17 if mode == "prefill" else 1, 257),
            ),
            "k_cache": ("DRAM_k_cache", ports["DRAM_k_cache"]),
            "v_cache": ("DRAM_v_cache", ports["DRAM_v_cache"]),
        }
        names = (
            set(design.inputs) | {"DRAM_token_ids", "DRAM_prefill_len"}
            if mode == "prefill"
            else set(design.inputs)
            | {"DRAM_token_id", "DRAM_decode_pos", "DRAM_k_cache", "DRAM_v_cache"}
        )
        root = tmp_path / mode
        (root / "tb_existing.cpp").write_text(make_testbench(design, names, outputs))
        libs = include.parent / "lnx64/lib/csim"
        fpo = include.parent / "lnx64/tools/fpo_v7_1"
        result = subprocess.run(
            [
                "g++",
                "-std=c++14",
                "-O2",
                "-I" + str(include),
                str(root / "top.cpp"),
                str(root / "tb_existing.cpp"),
                "-o",
                str(root / "run"),
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
        assert result.returncode == 0, result.stderr
    inputs = tmp_path / "inputs"
    prepare_inputs(designs["prefill"], inputs, 42)
    weights = Weights(inputs, designs["prefill"].inputs, dtype)
    refs = {
        mode: Reference(designs["prefill"], weights, mode == "fixed")
        for mode in ("fixed", "fp64")
    }
    refs["fixed"].ops.rope_provider = Provider(include.parent)
    tokens = np.random.default_rng(91).integers(16, 257, 19)
    previous = None
    for call, (start, rows) in enumerate(((0, 17), (17, 1), (18, 1))):
        mode = "prefill" if call == 0 else "decode"
        case = tmp_path / f"call{call}"
        (case / "actual").mkdir(parents=True)
        token_name = "DRAM_token_ids" if call == 0 else "DRAM_token_id"
        control_name = "DRAM_prefill_len" if call == 0 else "DRAM_decode_pos"
        ids = np.zeros(designs[mode].ports[token_name], dtype="<i4")
        ids[:rows] = tokens[start : start + rows]
        ids.tofile(case / (token_name + ".bin"))
        np.array([rows if call == 0 else start], dtype="<i4").tofile(
            case / (control_name + ".bin")
        )
        if previous:
            for label in ("k_cache", "v_cache"):
                shutil.copyfile(
                    previous / "actual" / (label + ".bin"),
                    case / ("DRAM_" + label + ".bin"),
                )
        result = subprocess.run(
            [str(tmp_path / mode / "run"), str(inputs), str(case)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        expected = {}
        for reference_name, reference in refs.items():
            expected[reference_name] = {}
            reference.forward(
                tokens[start : start + rows].tolist(),
                start,
                lambda n, v: expected[reference_name].update({n: v.clone()}),
            )
        for label in ("logits", "k_cache", "v_cache"):
            actual = np.fromfile(case / "actual" / (label + ".bin"), dtype=dtype)
            actual = (
                actual.reshape(rows, 257)
                if label == "logits"
                else actual.reshape(2, 20, 1, 128)[:, : start + rows]
            )
            np.testing.assert_array_equal(
                actual, expected["fixed"][label], err_msg=f"call {call} {label}"
            )
            # FP64 remains independent of the fixed operator implementation.
            assert torch.isfinite(expected["fp64"][label]).all()
            if word == 32:
                report = compare_arrays(
                    actual, expected["fixed"][label], expected["fp64"][label], fmt=fmt
                )
                assert report["mathematical_verdict"] == "PASS", (call, label, report)
        previous = case
