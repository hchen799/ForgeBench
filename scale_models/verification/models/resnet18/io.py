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
from .config import Config
from ...precision import FixedFormat

SOURCE_FILES = ("top.cpp", "top.h", "resnet18_tiled_scales.h", "verification_trace.h", "tb_top.cpp", "run_csim.tcl", "config.json")


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


def write_codes(path, tensor, precision=None):
    tensor.cpu().numpy().astype((precision or FixedFormat()).dtype).tofile(path)


def read_codes(path, shape, precision=None):
    precision = precision or FixedFormat()
    path = Path(path)
    if path.stat().st_size != math.prod(shape) * precision.itemsize:
        raise ValueError(f"incorrect byte count: {path}")
    return torch.from_numpy(np.fromfile(path, dtype=precision.dtype).astype(np.int64).reshape(shape))


def configuration(project):
    project = Path(project)
    c = Config(**read_json(project / "config.json")).validate()
    header = (project / "top.h").read_text()
    source = (project / "top.cpp").read_text()
    if f"ap_fixed<{c.word_bits},{c.integer_bits},AP_RND,AP_SAT> data_t" not in header:
        raise ValueError("storage precision differs from generated config")
    if f"ap_fixed<{c.acc_word_bits},{c.acc_integer_bits},AP_RND,AP_SAT> acc_t" not in source:
        raise ValueError("accumulator precision differs from generated config")
    for name, value in (("TILE_C", c.tile_c), ("TILE_H", c.tile_h), ("TILE_W", c.tile_w)):
        if not re.search(rf"\b{name}\s*=\s*{value}\s*;", source):
            raise ValueError(f"{name} differs from generated config")
    scales = (project / "resnet18_tiled_scales.h").read_text()
    match = re.search(r"kConvOutputShift\[21\]\s*=\s*\{([^}]+)\}", scales)
    if not match or [int(v) for v in match[1].split(",")] != list(c.shifts.values()):
        raise ValueError("output shifts differ from config")
    if not re.search(rf"kRenormGuard\s*=\s*{c.guard}\s*;", scales):
        raise ValueError("renormalization guard differs from config")
    return c.to_dict()


def prepare(project, run_dir, seed):
    project, run_dir = Path(project).resolve(), Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"run directory is not empty: {run_dir}; choose a new directory")
    config = configuration(project)
    precision = Config(**config).validate().precision
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
        codes = quantize(values, precision)
        if param.name.startswith("DRAM_bn_") and not (codes[3] > 0).all():
            raise ValueError("BN variance is not positive after quantization")
        path = run_dir / "inputs" / (param.name + ".bin")
        write_codes(path, codes, precision)
        entries.append({"name": param.name, "shape": list(param.dims), "sha256": digest(path), "distribution": distribution})
    manifest = {"version": 2, "seed": seed, "format": precision.describe(), "config": config,
                "project_dir": str(project), "sources": {name: digest(project / name) for name in SOURCE_FILES},
                "torch_version": torch.__version__, "numpy_version": np.__version__, "tensors": entries}
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def validate_case(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    if manifest["version"] != 2:
        raise ValueError("unsupported manifest format")
    precision = Config(**manifest["config"]).validate().precision
    if manifest["format"] != precision.describe():
        raise ValueError("precision metadata differs from configuration")
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
        if path.stat().st_size != math.prod(entry["shape"]) * precision.itemsize or digest(path) != entry["sha256"]:
            raise ValueError(f"input corrupted or changed: {entry['name']}")
    return manifest


def load_tensors(run_dir, manifest):
    return {e["name"]: read_codes(Path(run_dir) / "inputs" / (e["name"] + ".bin"), e["shape"], Config(**manifest["config"]).validate().precision) for e in manifest["tensors"]}


def reference_fingerprint():
    folder = Path(__file__).parent
    result = {p.name: digest(p) for p in sorted(folder.glob("*.py"))}
    result["shared_precision.py"] = digest(folder.parents[1] / "precision.py")
    result["graph_generator.py"] = digest(gen.__file__)
    return result
