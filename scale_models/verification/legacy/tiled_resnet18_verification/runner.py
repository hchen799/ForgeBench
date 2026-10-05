"""Reference execution, real Vitis C simulation, and checkpoint comparisons."""
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .fixed import FixedOps, DATA_SCALE, DATA_MIN, DATA_MAX
from .model import FloatOps, forward
from .io import (digest, load_tensors, read_codes, read_json, reference_fingerprint,
                 validate_case, write_codes, write_json, SOURCE_FILES)


def statistics(x):
    x = x.to(torch.float64)
    return {"min": x.min().item(), "max": x.max().item(), "std": x.std(unbiased=False).item(),
            "zero_fraction": (x == 0).double().mean().item()}


@torch.inference_mode()
def reference(run_dir, mode="both", fast=True):
    run_dir = Path(run_dir).resolve()
    manifest = validate_case(run_dir)
    tensors = load_tensors(run_dir, manifest)
    for mode_name in (["float", "fixed"] if mode == "both" else [mode]):
        started = time.monotonic()
        folder = run_dir / mode_name
        folder.mkdir(exist_ok=True)
        # Remove the completion marker first, so an interrupted rerun cannot
        # leave apparently valid results from an earlier implementation.
        marker = folder / "metadata.json"
        if marker.exists():
            marker.unlink()
        ops = FixedOps(manifest["config"]["guard"], fast=fast) if mode_name == "fixed" else FloatOps()
        values = tensors if mode_name == "fixed" else {k: v.double() / DATA_SCALE for k, v in tensors.items()}
        checkpoints = []

        def save(name, tensor):
            if mode_name == "fixed":
                path = folder / (name + ".bin")
                write_codes(path, tensor)
                stats = statistics(tensor.double() / DATA_SCALE)
                stats["endpoint_fraction"] = ((tensor == DATA_MIN) | (tensor == DATA_MAX)).double().mean().item()
            else:
                path = folder / (name + ".npy")
                np.save(path, tensor.cpu().numpy(), allow_pickle=False)
                stats = statistics(tensor)
            checkpoints.append({"name": name, "shape": list(tensor.shape), "sha256": digest(path), "statistics": stats})
            if name.endswith(("pool", ".out", "logits")):
                print(f"{mode_name}: {name} shape={tuple(tensor.shape)} range=[{stats['min']:.6g}, {stats['max']:.6g}]", flush=True)

        logits = forward(values, manifest["config"]["shifts"], ops, save)
        np.savetxt(folder / "logits.txt", (logits.double() / DATA_SCALE if mode_name == "fixed" else logits).numpy(), fmt="%.17g")
        meta = {"manifest_sha256": digest(run_dir / "manifest.json"), "implementation": reference_fingerprint(),
                "checkpoints": checkpoints, "elapsed_seconds": time.monotonic() - started}
        if mode_name == "fixed":
            with (folder / "exponents.txt").open("w") as f:
                for row in ops.events:
                    f.write(" ".join(map(str, row)) + "\n")
            meta["exponents_sha256"] = digest(folder / "exponents.txt")
            meta["arithmetic"] = ops.stats
            meta["nominal_warnings"] = []
            if any(v.get("storage_saturations", 0) or v.get("accumulator_saturations", 0) for v in ops.stats.values()):
                meta["nominal_warnings"].append("saturation occurred; inspect arithmetic statistics")
            if logits.unique().numel() < 2:
                meta["nominal_warnings"].append("logits collapsed to a constant")
        write_json(marker, meta)


def validate_reference(run_dir, mode):
    run_dir = Path(run_dir)
    meta = read_json(run_dir / mode / "metadata.json")
    if meta["manifest_sha256"] != digest(run_dir / "manifest.json"):
        raise ValueError(f"{mode} reference belongs to different inputs or sources")
    if meta["implementation"] != reference_fingerprint():
        raise ValueError(f"{mode} reference implementation changed; rerun reference")
    for checkpoint in meta["checkpoints"]:
        suffix = ".bin" if mode == "fixed" else ".npy"
        if digest(run_dir / mode / (checkpoint["name"] + suffix)) != checkpoint["sha256"]:
            raise ValueError(f"corrupted {mode} checkpoint: {checkpoint['name']}")
    if mode == "fixed" and digest(run_dir / mode / "exponents.txt") != meta["exponents_sha256"]:
        raise ValueError("corrupted exponent reference")
    return meta


def csim(run_dir, vitis="vitis_hls", cache_dir=None, rebuild=False):
    run_dir = Path(run_dir).resolve()
    manifest = validate_case(run_dir)
    validate_reference(run_dir, "fixed")
    tool = shutil.which(vitis)
    if not tool:
        raise ValueError(f"Vitis HLS executable not found: {vitis}")
    tool = str(Path(tool).resolve())
    # Tool identity includes the installed path and launcher checksum; saved logs
    # additionally contain Vitis's full version/build identification.
    identity = {"sources": manifest["sources"], "tool": tool, "tool_sha256": digest(tool)}
    import json
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir).resolve() if cache_dir else run_dir.parent / ".csim_cache"
    build = cache / key
    build.mkdir(parents=True, exist_ok=True)
    executable = build / "verification_project/solution1/csim/build/csim.exe"
    cache_marker = build / "compiled.json"
    reusable = (not rebuild and executable.is_file() and cache_marker.is_file()
                and read_json(cache_marker).get("executable_sha256") == digest(executable))
    output = run_dir / "csim"
    if output.exists() and any(output.iterdir()):
        output.rename(run_dir / ("csim.previous." + str(time.time_ns())))
    output.mkdir(exist_ok=True)
    started = time.monotonic()
    if reusable:
        command = [str(executable), str(run_dir)]
    else:
        for name in SOURCE_FILES:
            shutil.copy2(run_dir / "source" / name, build / name)
        command = [tool, "-f", "run_csim.tcl"]
    env = os.environ.copy()
    env["RESNET18_CASE_DIR"] = str(run_dir)
    print(f"C simulation ({'cached executable' if reusable else 'Vitis build'}); log: {output / 'simulation.log'}", flush=True)
    with (output / "simulation.log").open("w") as log:
        process = subprocess.Popen(command, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(output / "running.json", {"pid": process.pid, "command": command, "build_dir": str(build)})
        returncode = process.wait()
    if executable.is_file() and (returncode == 0 or "IMPLEMENTATION_MATCH:" in (output / "simulation.log").read_text(errors="replace")):
        write_json(cache_marker, {"identity": identity, "executable_sha256": digest(executable)})
    meta = {"manifest_sha256": digest(run_dir / "manifest.json"), "identity": identity,
            "cached_executable": reusable, "returncode": returncode, "elapsed_seconds": time.monotonic() - started,
            "outputs": {p.name: digest(p) for p in output.glob("*.bin")},
            "exponents_sha256": digest(output / "exponents.txt") if (output / "exponents.txt").exists() else None}
    write_json(output / "metadata.json", meta)
    (output / "running.json").unlink()
    if returncode:
        print(f"C simulation returned {returncode}; retained outputs and log for diagnosis", flush=True)
    return returncode


def numerical_error(actual, expected, atol=None, rtol=None):
    error = (actual - expected).abs()
    relative = error / expected.abs().clamp_min(1.0 / DATA_SCALE)
    result = {"max_abs": error.max().item(), "mae": error.mean().item(), "rmse": error.square().mean().sqrt().item(),
              "max_relative": relative.max().item(), "relative_denominator_floor": 1.0 / DATA_SCALE}
    if atol is not None:
        result["tolerance_mismatches"] = int((error > atol + rtol * expected.abs()).sum())
    return result


@torch.inference_mode()
def compare(run_dir, atol=None, rtol=None):
    run_dir = Path(run_dir).resolve()
    validate_case(run_dir)
    fixed = validate_reference(run_dir, "fixed")
    floating = validate_reference(run_dir, "float")
    simulation = read_json(run_dir / "csim/metadata.json")
    if simulation["manifest_sha256"] != digest(run_dir / "manifest.json"):
        raise ValueError("C-simulation results belong to another case")
    float_shapes = {p["name"]: p["shape"] for p in floating["checkpoints"]}
    if float_shapes != {p["name"]: p["shape"] for p in fixed["checkpoints"]}:
        raise ValueError("reference checkpoint schemas differ")
    comparisons = []
    first = None
    for checkpoint in fixed["checkpoints"]:
        name, shape = checkpoint["name"], checkpoint["shape"]
        path = run_dir / "csim" / (name + ".bin")
        if not path.is_file():
            raise ValueError(f"C simulation did not produce {name}; inspect simulation.log")
        if simulation["outputs"].get(path.name) != digest(path):
            raise ValueError(f"C-simulation output changed: {name}")
        actual = read_codes(path, shape)
        expected = read_codes(run_dir / "fixed" / (name + ".bin"), shape)
        mathematical = torch.from_numpy(np.load(run_dir / "float" / (name + ".npy"), allow_pickle=False))
        if list(mathematical.shape) != shape or not torch.isfinite(mathematical).all():
            raise ValueError(f"invalid mathematical checkpoint: {name}")
        mask = actual != expected
        result = {"name": name, "shape": shape, "mismatches": int(mask.sum()),
                  "max_lsb_error": int((actual - expected).abs().max()),
                  "mathematical_error": numerical_error(actual.double() / DATA_SCALE, mathematical, atol, rtol)}
        if mask.any():
            index = tuple(mask.nonzero()[0].tolist())
            result["first_mismatch"] = {"index": list(index), "expected_code": int(expected[index]), "actual_code": int(actual[index])}
            if first is None:
                first = {"name": name, **result["first_mismatch"]}
        comparisons.append(result)
    expected_names = {c["name"] + ".bin" for c in fixed["checkpoints"]}
    if set(simulation["outputs"]) != expected_names:
        raise ValueError("unexpected or missing C-simulation checkpoint files")
    exponent_path = run_dir / "csim/exponents.txt"
    if digest(exponent_path) != simulation["exponents_sha256"]:
        raise ValueError("C-simulation exponent trace changed")
    exponent_match = exponent_path.read_text().splitlines() == (run_dir / "fixed/exponents.txt").read_text().splitlines()
    exact = first is None and exponent_match and simulation["returncode"] == 0
    numerical_pass = None if atol is None else all(c["mathematical_error"]["tolerance_mismatches"] == 0 for c in comparisons)
    report = {"implementation_match": "PASS" if exact else "FAIL", "exponent_match": exponent_match,
              "first_mismatch": first, "checkpoint_count": len(comparisons), "checkpoints": comparisons,
              "mathematical_verdict": "DIAGNOSTIC_ONLY" if numerical_pass is None else ("PASS" if numerical_pass else "FAIL"),
              "float_atol": atol, "float_rtol": rtol, "nominal_warnings": fixed["nominal_warnings"],
              "arithmetic": fixed["arithmetic"], "note": "Implementation agreement does not establish mathematical correctness; see directed arithmetic regressions."}
    write_json(run_dir / "report.json", report)
    lines = [f"Implementation agreement: {report['implementation_match']}",
             f"Checkpoints: {len(comparisons)}; exponent trace agreement: {exponent_match}",
             f"First mismatch: {first}", f"Mathematical comparison: {report['mathematical_verdict']}",
             "Logit errors: " + str(comparisons[-1]["mathematical_error"]),
             "Nominal warnings: " + str(fixed["nominal_warnings"]), report["note"]]
    (run_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return exact and numerical_pass is not False and not fixed["nominal_warnings"]
