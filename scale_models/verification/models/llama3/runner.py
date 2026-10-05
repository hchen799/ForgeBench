"""Reference generation, real C++/Vitis simulation, and tensor-level comparison."""
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .io import WeightStore, fingerprint, read_json, sha, validate_case, write_json
from .model import LlamaReference


def reference(run_dir, threads=16, cache_gib=64):
    run_dir = Path(run_dir).resolve()
    case, model, tokens = validate_case(run_dir)
    c = Config(**model["config"])
    if threads < 1 or cache_gib < 0:
        raise ValueError("threads must be positive; cache size must be nonnegative")
    for kind in ("fixed", "float"):
        if any((run_dir / kind).iterdir()):
            raise ValueError(f"reference output already exists: {run_dir / kind}")
    torch.set_num_threads(threads)
    store = WeightStore(case["model_dir"], model, cache_gib)
    rope = np.fromfile(Path(case["model_dir"]) / "rope.bin", dtype=c.precision.dtype).reshape(c.max_ctx, c.head_dim)
    fixed = LlamaReference(c, store, fixed=True, rope=rope)
    floating = LlamaReference(c, store)
    records = {"fixed": {}, "float": {}}

    def checkpoint(kind):
        def save(name, value, essential):
            if case["trace"] != "ops" and not essential:
                return
            dtype = c.precision.dtype if kind == "fixed" else "<f8"
            data = value.detach().cpu().numpy().astype(dtype)
            path = run_dir / kind / (name + ".bin")
            data.tofile(path)
            records[kind][name] = dict(shape=list(data.shape), sha256=sha(path))
        return save

    started = time.monotonic()
    for number, step in enumerate(case["schedule"]):
        start, count = step["start"], step["count"]
        sample = tokens[start:start + count]
        floating.forward(sample, start, number, checkpoint("float"))
        fixed.forward(sample, start, number, checkpoint("fixed"))
        print(f"REFERENCE {number + 1}/{len(case['schedule'])}: position={start}, tokens={count}, elapsed={time.monotonic() - started:.1f}s", flush=True)
    result = dict(version=1, case_sha256=sha(run_dir / "case.json"), implementation=fingerprint(),
                  torch_version=torch.__version__, threads=threads, cache_gib=cache_gib,
                  elapsed_seconds=time.monotonic() - started, tensors=records,
                  saturations={k: v for k, v in fixed.ops.saturations.items() if v},
                  saturation_count=sum(fixed.ops.saturations.values()), precision=c.precision.describe())
    write_json(run_dir / "reference.json", result)
    return result


def validate_reference(run_dir, check_files=True):
    run_dir = Path(run_dir)
    ref = read_json(run_dir / "reference.json")
    if ref["case_sha256"] != sha(run_dir / "case.json") or ref["implementation"] != fingerprint():
        raise ValueError("reference provenance is stale; generate a new reference case")
    if not ref["tensors"]["fixed"] or ref["tensors"]["fixed"].keys() != ref["tensors"]["float"].keys():
        raise ValueError("incomplete reference tensor manifest")
    for kind, entries in ref["tensors"].items():
        size = ref["precision"]["storage_bytes"] if kind == "fixed" else 8
        for name, entry in entries.items():
            path = run_dir / kind / (name + ".bin")
            if path.stat().st_size != int(np.prod(entry["shape"])) * size:
                raise ValueError(f"wrong reference size: {path}")
            if check_files and sha(path) != entry["sha256"]:
                raise ValueError(f"reference checksum mismatch: {path}")
    return ref


def csim(run_dir, backend="vitis", vitis=None):
    run_dir = Path(run_dir).resolve()
    case, model, _ = validate_case(run_dir)
    validate_reference(run_dir)
    if any((run_dir / "csim").iterdir()):
        raise ValueError("C-simulation output already exists; use a new case directory")
    build = run_dir / "build"
    shutil.copytree(Path(case["model_dir"]) / "source", build)
    started = time.monotonic()
    log_path = run_dir / "csim.log"
    env = os.environ.copy()
    env.update(LLAMA_CASE_DIR=str(run_dir), LLAMA_MODEL_DIR=case["model_dir"])
    if backend == "native":
        compiler = shutil.which("g++")
        if not compiler:
            raise ValueError("g++ not found")
        command = [compiler, "-std=c++14", "-O3", "-march=native", "-DLLAMA_VERIFY", "top.cpp", "tb_top.cpp", "-o", "csim.exe"]
        subprocess.run(command, cwd=build, check=True)
        command = [str(build / "csim.exe"), str(run_dir), case["model_dir"]]
    else:
        executable = vitis or shutil.which("vitis_hls")
        fallback = "/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1/bin/vitis_hls"
        if not executable and Path(fallback).is_file():
            executable = fallback
        if not executable:
            raise ValueError("vitis_hls not found; pass --vitis PATH or explicitly select --backend native")
        command = [str(executable), "-f", "run_csim.tcl"]
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=build, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    result = dict(backend=backend, command=command, returncode=code, elapsed_seconds=time.monotonic() - started,
                  case_sha256=sha(run_dir / "case.json"), reference_sha256=sha(run_dir / "reference.json"),
                  log_sha256=sha(log_path), sources=model["sources"])
    if code == 0 and "IMPLEMENTATION_MATCH: PASS" not in log_path.read_text():
        result["returncode"] = 2
    write_json(run_dir / "csim.json", result)
    if result["returncode"]:
        raise RuntimeError(f"C simulation failed; see {log_path}")
    return result


class Errors:
    def __init__(self, floor=1 / 2048):
        self.floor = floor
        self.count = 0
        self.maximum = self.absolute = self.squared = self.relative = 0.0

    def add(self, actual, expected):
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError("non-finite comparison tensor")
        delta = np.abs(actual - expected)
        self.count += delta.size
        self.maximum = max(self.maximum, float(delta.max()))
        self.absolute += float(delta.sum(dtype=np.float64))
        self.squared += float(np.square(delta).sum(dtype=np.float64))
        self.relative = max(self.relative, float((delta / np.maximum(np.abs(expected), self.floor)).max()))

    def summary(self):
        return dict(count=self.count, max_abs=self.maximum, mae=self.absolute / self.count,
                    rmse=(self.squared / self.count) ** .5, max_relative=self.relative,
                    relative_denominator_floor=self.floor)


def compare(run_dir, atol=None, rtol=None):
    run_dir = Path(run_dir).resolve()
    case, model, _ = validate_case(run_dir)
    precision = Config(**model["config"]).precision
    ref = validate_reference(run_dir)
    simulation = read_json(run_dir / "csim.json")
    if simulation["returncode"] or simulation["case_sha256"] != sha(run_dir / "case.json") or simulation["reference_sha256"] != sha(run_dir / "reference.json"):
        raise ValueError("simulation missing, failed, or stale")
    if simulation["sources"] != model["sources"] or simulation["log_sha256"] != sha(run_dir / "csim.log"):
        raise ValueError("simulation provenance changed")
    index = {}
    for line in (run_dir / "csim/index.txt").read_text().splitlines():
        name, rows, cols = line.split()
        if name in index:
            raise ValueError(f"duplicate trace: {name}")
        index[name] = [int(rows), int(cols)]
    if index.keys() != ref["tensors"]["fixed"].keys():
        raise ValueError("C simulation checkpoint set differs from reference")
    records, mismatch_count, total_codes = {}, 0, 0
    logits = Errors(1 / precision.scale)
    tolerance_mismatches = 0
    for name, shape in index.items():
        if shape != ref["tensors"]["fixed"][name]["shape"] or shape != ref["tensors"]["float"][name]["shape"]:
            raise ValueError(f"checkpoint shape mismatch: {name}")
        actual_path = run_dir / "csim" / (name + ".bin")
        actual = np.fromfile(actual_path, dtype=precision.dtype).astype(np.int64)
        fixed = np.fromfile(run_dir / "fixed" / (name + ".bin"), dtype=precision.dtype).astype(np.int64)
        floating = np.fromfile(run_dir / "float" / (name + ".bin"), dtype="<f8")
        if actual.size != fixed.size:
            raise ValueError(f"C simulation tensor truncated: {name}")
        bad = int(np.count_nonzero(actual != fixed))
        mismatch_count += bad
        total_codes += actual.size
        actual_real = actual.astype(np.float64) / precision.scale
        errors = Errors(1 / precision.scale)
        errors.add(actual_real, floating)
        records[name] = dict(shape=shape, integer_mismatches=bad, max_lsb_error=int(np.abs(actual - fixed).max()),
                             mathematical_errors=errors.summary(), csim_sha256=sha(actual_path))
        if bad:
            i = int(np.flatnonzero(actual != fixed)[0])
            records[name]["first_mismatch"] = dict(flat_index=i, expected=int(fixed[i]), actual=int(actual[i]))
        if atol is not None:
            failures = int(np.count_nonzero(np.abs(actual_real - floating) > atol + rtol * np.abs(floating)))
            records[name]["mathematical_tolerance_mismatches"] = failures
            tolerance_mismatches += failures
        if name.endswith(".logits"):
            logits.add(actual_real, floating)
    expected_logits = (case["prefill"] + case["decode"]) * model["config"]["vocab"]
    if logits.count != expected_logits:
        raise ValueError("not all tokens/vocabulary logits were compared")
    result = dict(implementation_match=mismatch_count == 0, integer_mismatches=mismatch_count,
                  compared_integer_codes=total_codes, checkpoints=len(records), backend=simulation["backend"],
                  profile=model["config"]["profile"], prefill=case["prefill"], decode=case["decode"],
                  logit_errors=logits.summary(), saturation_count=ref["saturation_count"], precision=precision.describe(),
                  model_config=model["config"], float_atol=atol, float_rtol=rtol,
                  mathematical_verdict="DIAGNOSTIC_ONLY" if atol is None else ("FAIL" if tolerance_mismatches else "PASS"),
                  mathematical_tolerance_mismatches=tolerance_mismatches,
                  mathematical_comparison="diagnostic; no uncalibrated accuracy threshold imposed",
                  parameter_source="deterministic synthetic random weights, not pretrained Meta weights",
                  checkpoints_detail=records)
    write_json(run_dir / "comparison.json", result)
    print(f"IMPLEMENTATION_MATCH: {'PASS' if not mismatch_count else 'FAIL'}; {len(records)} checkpoints, {total_codes} codes, {mismatch_count} mismatches")
    print(f"Logit errors vs mathematical FP64: {result['logit_errors']}")
    print(f"Fixed-point saturations: {result['saturation_count']}")
    if mismatch_count or tolerance_mismatches:
        raise RuntimeError("verification failed; inspect comparison.json for exact and mathematical verdicts")
    return result
