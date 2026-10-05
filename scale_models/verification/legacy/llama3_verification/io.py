"""Shared, checksummed packed models and independent token/call datasets."""
import hashlib
import json
import math
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .config import Config, segments

SOURCE_FILES = ("config.json", "model_config.h", "top.h", "top.cpp", "nonlinear_tables.h",
                "verification_trace.h", "tb_top.cpp", "run_csim.tcl")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint():
    folder = Path(__file__).parent
    return {p.name: sha(p) for p in sorted(folder.iterdir()) if p.suffix in (".py", ".in")}


def config_of(directory):
    return Config(**read_json(Path(directory) / "config.json")).validate()


def prepare_model(project, model_dir, seed=42):
    project, model_dir = Path(project).resolve(), Path(model_dir).resolve()
    c = config_of(project)
    if model_dir.exists() and any(model_dir.iterdir()):
        raise ValueError(f"model directory is not empty: {model_dir}")
    (model_dir / "source").mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(project / name, model_dir / "source" / name)
    entries = segments(c)
    total = entries[-1]["offset"] + entries[-1]["count"]
    if shutil.disk_usage(model_dir).free < total * 2 + (1 << 30):
        raise ValueError(f"insufficient disk for {total * 2 / 2**30:.2f} GiB parameter file")
    digest = hashlib.sha256()
    with (model_dir / "weights.bin").open("wb") as output:
        for entry in entries:
            name, kind = entry["name"], entry["kind"]
            if kind == "norm":
                low, high = math.ceil(.9 * 2048), math.floor(1.1 * 2048)
            else:
                bound = .5 if kind == "embedding" else math.sqrt(3.0 / entry["fan_in"])
                if kind == "residual":
                    bound /= math.sqrt(2 * c.layers)
                high = max(1, math.floor(bound * 2048))
                low = -high
            entry["raw_uniform_inclusive"] = [low, high]
            local_seed = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(local_seed)
            for offset in range(0, entry["count"], 1 << 20):
                size = min(1 << 20, entry["count"] - offset)
                chunk = rng.integers(low, high + 1, size=size, dtype=np.int16).astype("<i2", copy=False).tobytes()
                output.write(chunk)
                digest.update(chunk)
            if name.endswith("down_proj") or name in ("embedding", "lm_head"):
                print(f"Parameters: {name} ({(entry['offset'] + entry['count']) * 2 / 2**30:.2f} GiB)", flush=True)
    frequencies = c.rope_theta ** (-np.arange(0, c.head_dim, 2, dtype=np.float64) / c.head_dim)
    angles = np.arange(c.max_ctx, dtype=np.float64)[:, None] * frequencies[None, :]
    rope = np.empty((c.max_ctx, c.head_dim), dtype=np.float64)
    rope[:, 0::2], rope[:, 1::2] = np.cos(angles), np.sin(angles)
    np.clip(np.floor(rope * 2048 + .5), -32768, 32767).astype("<i2").tofile(model_dir / "rope.bin")
    manifest = dict(version=1, config=c.to_dict(), seed=seed, format="little-endian-int16-q5.11",
                    weights_count=total, weights_sha256=digest.hexdigest(), rope_sha256=sha(model_dir / "rope.bin"),
                    sources={name: sha(model_dir / "source" / name) for name in SOURCE_FILES}, segments=entries,
                    numpy_version=np.__version__, torch_version=torch.__version__, generator=fingerprint())
    write_json(model_dir / "model.json", manifest)
    return manifest


def validate_model(model_dir, check_weights=True):
    model_dir = Path(model_dir).resolve()
    manifest = read_json(model_dir / "model.json")
    if manifest["version"] != 1 or manifest["format"] != "little-endian-int16-q5.11":
        raise ValueError("unsupported model format")
    c = Config(**manifest["config"]).validate()
    expected = segments(c)
    if [{k: s[k] for k in e} for s, e in zip(manifest["segments"], expected)] != expected or len(manifest["segments"]) != len(expected):
        raise ValueError("model layout does not match architecture")
    count = expected[-1]["offset"] + expected[-1]["count"]
    if count != manifest["weights_count"] or (model_dir / "weights.bin").stat().st_size != count * 2:
        raise ValueError("incorrect model weight length")
    if check_weights and sha(model_dir / "weights.bin") != manifest["weights_sha256"]:
        raise ValueError("model weights checksum mismatch")
    if (model_dir / "rope.bin").stat().st_size != c.max_ctx * c.head_dim * 2 or sha(model_dir / "rope.bin") != manifest["rope_sha256"]:
        raise ValueError("RoPE table corruption")
    if set(manifest["sources"]) != set(SOURCE_FILES):
        raise ValueError("incomplete source snapshot")
    for name, digest in manifest["sources"].items():
        if sha(model_dir / "source" / name) != digest:
            raise ValueError(f"source changed: {name}")
    if config_of(model_dir / "source") != c:
        raise ValueError("source configuration differs from model")
    return manifest


def prepare_case(model_dir, run_dir, prefill=4, decode=2, seed=43, trace="ops", chunk=None):
    model_dir, run_dir = Path(model_dir).resolve(), Path(run_dir).resolve()
    manifest = validate_model(model_dir)
    c = Config(**manifest["config"])
    if prefill < 1 or decode < 0 or prefill + decode > c.max_ctx:
        raise ValueError("require positive prefill, nonnegative decode, and total <= max_ctx")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"run directory is not empty: {run_dir}")
    chunk = c.prefill_tile if chunk is None else chunk
    if not 1 <= chunk <= c.prefill_tile or trace not in ("ops", "layers"):
        raise ValueError("invalid chunk size or trace mode")
    for folder in ("fixed", "float", "csim"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    tokens = np.random.default_rng(seed).integers(0, c.vocab, size=prefill + decode, dtype=np.int32)
    tokens[0] = c.vocab - 1  # Explicitly exercise IDs outside the activation range.
    if len(tokens) > 1:
        tokens[1] = min(17, c.vocab - 1)
    tokens.astype("<i4").tofile(run_dir / "tokens.bin")
    schedule = [dict(start=i, count=min(chunk, prefill - i), decode=False) for i in range(0, prefill, chunk)]
    schedule += [dict(start=prefill + i, count=1, decode=True) for i in range(decode)]
    lines = [f"{len(tokens)} {len(schedule)} {int(trace == 'ops')}"]
    lines += [f"{s['start']} {s['count']} {int(s['decode'])}" for s in schedule]
    (run_dir / "case.txt").write_text("\n".join(lines) + "\n")
    case = dict(version=1, model_dir=str(model_dir), model_sha256=sha(model_dir / "model.json"),
                seed=seed, prefill=prefill, decode=decode, trace=trace, schedule=schedule,
                tokens_sha256=sha(run_dir / "tokens.bin"), control_sha256=sha(run_dir / "case.txt"))
    write_json(run_dir / "case.json", case)
    return case


def validate_case(run_dir, check_weights=True):
    run_dir = Path(run_dir).resolve()
    case = read_json(run_dir / "case.json")
    if case["version"] != 1 or sha(Path(case["model_dir"]) / "model.json") != case["model_sha256"]:
        raise ValueError("case model metadata changed")
    model = validate_model(case["model_dir"], check_weights)
    c = Config(**model["config"])
    count = case["prefill"] + case["decode"]
    if (run_dir / "tokens.bin").stat().st_size != count * 4 or sha(run_dir / "tokens.bin") != case["tokens_sha256"]:
        raise ValueError("token file corrupted")
    if sha(run_dir / "case.txt") != case["control_sha256"]:
        raise ValueError("case schedule corrupted")
    expected = [f"{count} {len(case['schedule'])} {int(case['trace'] == 'ops')}"]
    consumed = 0
    for s in case["schedule"]:
        if s["start"] != consumed or not 1 <= s["count"] <= c.prefill_tile or (s["decode"] and s["count"] != 1):
            raise ValueError("invalid cache-contiguous schedule")
        consumed += s["count"]
        expected.append(f"{s['start']} {s['count']} {int(s['decode'])}")
    if consumed != count or count > c.max_ctx or (run_dir / "case.txt").read_text().splitlines() != expected:
        raise ValueError("case schedule inconsistent")
    tokens = np.fromfile(run_dir / "tokens.bin", dtype="<i4")
    if (tokens < 0).any() or (tokens >= c.vocab).any():
        raise ValueError("token ID outside vocabulary")
    return case, model, tokens


class WeightStore:
    """Memory-map int16 parameters; optionally cache float64 integer codes.

    Both references use the same cached matrices: the mathematical reference
    divides the linear result by 2048, while the implementation reference
    rounds an exact integer dot product. No second 60-GiB weight copy is needed.
    """
    def __init__(self, model_dir, model, cache_gib=64):
        self.raw = np.memmap(Path(model_dir) / "weights.bin", mode="r", dtype="<i2")
        self.entries = {s["name"]: s for s in model["segments"]}
        self.cache = OrderedDict()
        self.limit, self.size = int(cache_gib * 2**30), 0

    def view(self, name):
        s = self.entries[name]
        return self.raw[s["offset"]:s["offset"] + s["count"]].reshape(s["shape"])

    def tensor(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        result = torch.from_numpy(np.array(self.view(name), dtype=np.float64, copy=True))
        nbytes = result.numel() * 8
        if nbytes <= self.limit:
            while self.cache and self.size + nbytes > self.limit:
                _, previous = self.cache.popitem(last=False)
                self.size -= previous.numel() * 8
            self.cache[name] = result
            self.size += nbytes
        return result

    def embedding(self, tokens):
        return torch.from_numpy(np.array(self.view("embedding")[tokens], dtype=np.int64, copy=True))
