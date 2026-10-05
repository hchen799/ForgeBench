"""Compare exported tensors from any accelerator against supplied golden tensors."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .precision import FixedFormat
from .reporting import write_json


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_tolerances(atol, rtol):
    if (atol is None) != (rtol is None):
        raise ValueError("provide both float_atol and float_rtol, or neither")
    if any(x is not None and (not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0) for x in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")


def load_file(base, specification, shape, fixed=False, precision=None):
    allowed = {"path", "dtype", "sha256"}
    if set(specification) - allowed or "path" not in specification:
        raise ValueError("tensor file specification requires path and optional dtype/sha256")
    path = (base / specification["path"]).resolve()
    if path.suffix == ".npy":
        data = np.load(path, allow_pickle=False)
        if "dtype" in specification and data.dtype != np.dtype(specification["dtype"]):
            raise ValueError(f"declared dtype differs from NPY header: {path}")
        if tuple(data.shape) != tuple(shape):
            raise ValueError(f"shape mismatch: {path}: {data.shape} != {shape}")
    else:
        if not fixed and "dtype" not in specification:
            raise ValueError("raw floating tensors require an explicit dtype")
        dtype = np.dtype(specification.get("dtype", precision.dtype if precision else None))
        if dtype.kind not in "ifu" or dtype.itemsize > 8:
            raise ValueError("only ordinary numeric binary dtypes are supported")
        if path.stat().st_size != math.prod(shape) * dtype.itemsize:
            raise ValueError(f"byte count does not match shape/dtype: {path}")
        data = np.fromfile(path, dtype=dtype).reshape(shape)
    if fixed:
        if data.dtype.kind != "i" or data.dtype.itemsize != precision.itemsize:
            raise ValueError(f"fixed codes must use signed {precision.itemsize * 8}-bit containers: {path}")
        if data.min() < precision.minimum or data.max() > precision.maximum:
            raise ValueError(f"code outside declared logical precision: {path}")
    elif data.dtype.kind != "f" or data.dtype.itemsize not in (2, 4, 8):
        raise ValueError("mathematical references must be FP16, FP32, or FP64 arrays")
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite tensor: {path}")
    digest = sha(path)
    if specification.get("sha256", digest) != digest:
        raise ValueError(f"checksum mismatch: {path}")
    return data, dict(path=str(path), dtype=data.dtype.str, sha256=digest)


def compare_manifest(manifest_path, destination):
    manifest_path, destination = Path(manifest_path).resolve(), Path(destination)
    manifest = json.loads(manifest_path.read_text())
    allowed = {"schema_version", "model", "precision", "checkpoints", "float_atol", "float_rtol", "notes"}
    if manifest.get("schema_version") != 1 or set(manifest) - allowed:
        raise ValueError("unsupported generic manifest schema or unknown fields")
    p = FixedFormat(**manifest["precision"])
    atol, rtol = manifest.get("float_atol"), manifest.get("float_rtol")
    validate_tolerances(atol, rtol)
    if not isinstance(manifest.get("checkpoints"), list) or not manifest["checkpoints"]:
        raise ValueError("at least one explicit checkpoint is required")
    print("GENERIC COMPARISON CONTRACT")
    print("Model label:", manifest.get("model", "user-supplied accelerator"))
    print("Precision:", json.dumps(p.describe()))
    print("A: supplied accelerator raw codes versus supplied fixed-point golden raw codes (bit-exact).")
    print("B: accelerator codes / scale versus supplied floating golden values.")
    print("This command does NOT run an accelerator or generate/verify the origin of the supplied golden model.")
    print("Manifest:", manifest_path, "SHA256:", sha(manifest_path))
    print("Resolved manifest:\n" + json.dumps(manifest, indent=2))
    records, mismatch_count, tolerance_count = {}, 0, 0
    for entry in manifest["checkpoints"]:
        if set(entry) != {"name", "shape", "actual", "fixed", "floating"}:
            raise ValueError("each checkpoint requires name, shape, actual, fixed, and floating")
        name, shape = entry["name"], entry["shape"]
        if not isinstance(name, str) or not name or name in records:
            raise ValueError("checkpoint names must be nonempty and unique")
        if not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("checkpoint shapes must list positive integer dimensions")
        a, am = load_file(manifest_path.parent, entry["actual"], shape, True, p)
        f, fm = load_file(manifest_path.parent, entry["fixed"], shape, True, p)
        floating, gm = load_file(manifest_path.parent, entry["floating"], shape)
        print(name, "files:\n" + json.dumps(dict(actual=am, fixed=fm, floating=gm), indent=2))
        a, f = a.astype(np.int64), f.astype(np.int64)
        mask = a != f
        bad = int(np.count_nonzero(mask))
        mismatch_count += bad
        actual = a.astype(np.float64) / p.scale
        expected = floating.astype(np.float64)
        delta = np.abs(actual - expected)
        error = dict(count=a.size, max_abs=float(delta.max()), mae=float(delta.mean()), rmse=float(np.sqrt(np.square(delta).mean())),
                     max_relative=float((delta / np.maximum(np.abs(expected), 1 / p.scale)).max()), relative_denominator_floor=1 / p.scale)
        item = dict(shape=shape, integer_mismatches=bad, max_lsb_error=int(np.abs(a - f).max()), mathematical_errors=error,
                    files=dict(actual=am, fixed=fm, floating=gm))
        if bad:
            index = tuple(int(i) for i in np.argwhere(mask)[0])
            item["first_mismatch"] = dict(index=index, actual_code=int(a[index]), expected_code=int(f[index]))
        if atol is not None:
            failures = int(np.count_nonzero(delta > atol + rtol * np.abs(expected)))
            item["mathematical_tolerance_mismatches"] = failures
            tolerance_count += failures
        records[name] = item
    report = dict(implementation_match=mismatch_count == 0,
                  mathematical_verdict="DIAGNOSTIC_ONLY" if atol is None else ("PASS" if tolerance_count == 0 else "FAIL"),
                  float_atol=atol, float_rtol=rtol, precision=p.describe(),
                  checkpoints_detail=records, manifest_sha256=sha(manifest_path),
                  saturation_count=None, saturation_observability="not observable from exported tensors alone")
    write_json(destination / "comparison.json", report)
    return report
