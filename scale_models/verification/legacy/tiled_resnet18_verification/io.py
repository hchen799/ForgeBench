"""Versioned, checksummed exchange of raw fixed-point tensor codes."""
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import torch

import generate_tiled_resnet18 as gen
from .codegen import is_input
from .fixed import quantize

SOURCE_FILES = ("top.cpp", "top.h", "resnet18_tiled_scales.h", "verification_trace.h", "tb_top.cpp", "run_csim.tcl")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_codes(path, tensor):
    tensor.cpu().numpy().astype("<i2").tofile(path)


def read_codes(path, shape):
    path = Path(path)
    if path.stat().st_size != math.prod(shape) * 2:
        raise ValueError(f"incorrect byte count: {path}")
    return torch.from_numpy(np.fromfile(path, dtype="<i2").astype(np.int64).reshape(shape))


def configuration(project):
    project = Path(project)
    header = (project / "top.h").read_text()
    code = (project / "top.cpp").read_text()
    if not re.search(r"typedef\s+ap_fixed<16,\s*5,\s*AP_RND,\s*AP_SAT>\s+data_t", header):
        raise ValueError("unsupported storage type")
    if not re.search(r"typedef\s+ap_fixed<32,\s*10,\s*AP_RND,\s*AP_SAT>\s+acc_t", code):
        raise ValueError("unsupported accumulator type")
    for name, expected in [("TILE_C", 128), ("TILE_H", 14), ("TILE_W", 14), ("MAX_FEAT_C", 512), ("MAX_FEAT_H", 56), ("MAX_FEAT_W", 56)]:
        match = re.search(rf"\b{name}\s*=\s*(\d+)\s*;", code)
        if not match or int(match[1]) != expected:
            raise ValueError(f"unsupported {name}; expected {expected}")
    for p in gen.build_params(gen.build_blocks()):
        dimensions = "".join(r"\[\s*" + str(d) + r"\s*\]" for d in p.dims)
        if not re.search(r"\b" + p.name + dimensions, header):
            raise ValueError(f"incompatible top port: {p.name}")
    scales = re.sub(r"//[^\n]*", "", (project / "resnet18_tiled_scales.h").read_text())
    names = gen.build_shift_names(gen.build_blocks())
    for i, name in enumerate(names):
        if not re.search(rf"\b{name}\s*=\s*{i}\b", scales):
            raise ValueError("shift enum does not match the graph")
    match = re.search(r"kConvOutputShift\[21\]\s*=\s*\{([^}]+)\}", scales)
    if not match:
        raise ValueError("expected 21 literal output shifts")
    shifts = [int(v.strip()) for v in match[1].split(",")]
    if len(shifts) != 21 or any(abs(s) > 31 for s in shifts):
        raise ValueError("expected 21 shifts in [-31,31]")
    guard = int(re.search(r"kRenormGuard\s*=\s*(\d+)", scales)[1])
    if not 0 < guard < 512:
        raise ValueError("unsupported renormalization guard")
    return {"shifts": dict(zip(names, shifts)), "guard": guard, "tile": [128, 14, 14], "data_fractional_bits": 11, "acc_fractional_bits": 22}


def prepare(project, run_dir, seed):
    project, run_dir = Path(project).resolve(), Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"run directory is not empty: {run_dir}; choose a new directory")
    config = configuration(project)
    for name in SOURCE_FILES:
        if not (project / name).is_file():
            raise ValueError(f"missing {name}; regenerate with the updated generator")
    for folder in ("inputs", "source", "fixed", "float", "csim"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(project / name, run_dir / "source" / name)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    entries = []
    for param in gen.build_params(gen.build_blocks()):
        if not is_input(param.name):
            continue
        values = torch.empty(param.dims, dtype=torch.float64)
        if param.name.startswith("DRAM_bn_"):
            ranges = [(0.9, 1.1), (-0.05, 0.05), (-0.05, 0.05), (0.9, 1.1)]
            for row, (lo, hi) in enumerate(ranges):
                values[row].uniform_(lo, hi, generator=generator)
            distribution = {"kind": "uniform", "rows": ranges}
        else:
            bound = 1.0 if param.name == "DRAM_input" else math.sqrt(3.0 / math.prod(param.dims[1:]))
            values.uniform_(-bound, bound, generator=generator)
            distribution = {"kind": "uniform", "low": -bound, "high": bound}
        codes = quantize(values)
        if param.name.startswith("DRAM_bn_") and not (codes[3] > 0).all():
            raise ValueError("BN variance is not positive after quantization")
        path = run_dir / "inputs" / (param.name + ".bin")
        write_codes(path, codes)
        entries.append({"name": param.name, "shape": list(param.dims), "sha256": digest(path), "distribution": distribution})
    manifest = {"version": 1, "seed": seed, "format": "little-endian-int16-q5.11", "config": config,
                "project_dir": str(project), "sources": {name: digest(project / name) for name in SOURCE_FILES},
                "torch_version": torch.__version__, "numpy_version": np.__version__, "tensors": entries}
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def validate_case(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    if manifest["version"] != 1 or manifest["format"] != "little-endian-int16-q5.11":
        raise ValueError("unsupported manifest format")
    expected = {p.name: list(p.dims) for p in gen.build_params(gen.build_blocks()) if is_input(p.name)}
    actual = {p["name"]: p["shape"] for p in manifest["tensors"]}
    if actual != expected or len(manifest["tensors"]) != len(expected):
        raise ValueError("manifest tensors do not match the accelerator interface")
    if set(manifest["sources"]) != set(SOURCE_FILES):
        raise ValueError("incomplete source manifest")
    for name, sha in manifest["sources"].items():
        if digest(run_dir / "source" / name) != sha:
            raise ValueError(f"source changed since preparation: {name}")
    if configuration(run_dir / "source") != manifest["config"]:
        raise ValueError("configuration differs from source snapshot")
    for entry in manifest["tensors"]:
        path = run_dir / "inputs" / (entry["name"] + ".bin")
        if path.stat().st_size != math.prod(entry["shape"]) * 2 or digest(path) != entry["sha256"]:
            raise ValueError(f"input corrupted or changed: {entry['name']}")
    return manifest


def load_tensors(run_dir, manifest):
    return {e["name"]: read_codes(Path(run_dir) / "inputs" / (e["name"] + ".bin"), e["shape"]) for e in manifest["tensors"]}


def reference_fingerprint():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
