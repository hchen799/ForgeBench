#!/usr/bin/env python3
"""Unified, logged architecture verification and exported-tensor comparison."""
import argparse
import datetime
import importlib
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification.generic import compare_manifest, sha, validate_tolerances
from verification.precision import FixedFormat
from verification.reporting import contract, logged, render_report, write_json


def fields(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"{label}: expected an object with only fields {sorted(allowed)}")
    return dict(value)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def resolve_config(raw, args):
    raw = fields(raw, {"schema_version", "family", "model", "precision", "parameters", "test", "execution", "comparison"}, "configuration")
    if raw.get("schema_version") != 1 or raw.get("family") not in ("llama3", "resnet18"):
        raise ValueError("require schema_version=1 and family=llama3 or resnet18")
    precision = fields(raw.get("precision", {}), {"word_bits", "integer_bits", "rounding", "overflow"}, "precision")
    for key in ("word_bits", "integer_bits"):
        if getattr(args, key, None) is not None:
            precision[key] = getattr(args, key)
    p = FixedFormat(**precision)
    model = dict(raw.get("model", {}))
    if {"word_bits", "integer_bits"} & model.keys():
        raise ValueError("specify storage widths in precision, not model")
    if raw["family"] == "llama3":
        from verification.models.llama3.config import Config, make_config
        fields(model, set(Config.__dataclass_fields__) - {"word_bits", "integer_bits"}, "model")
        c = make_config(model.pop("profile", "tiny"), **model, word_bits=p.word_bits, integer_bits=p.integer_bits)
        for key in ("layers", "hidden", "ffn", "q_heads", "kv_heads", "head_dim", "vocab", "max_ctx", "prefill_tile", "tile_in", "tile_out"):
            integer(getattr(c, key), key, 1)
        test = dict(seed=43, prefill=4, decode=2, trace="ops", chunk=c.prefill_tile)
        test.update(fields(raw.get("test", {}), test.keys(), "test"))
        for key in ("prefill", "decode"):
            if getattr(args, key, None) is not None:
                test[key] = getattr(args, key)
        integer(test["seed"], "test.seed")
        integer(test["prefill"], "test.prefill", 1)
        integer(test["decode"], "test.decode")
        integer(test["chunk"], "test.chunk", 1)
        if test["trace"] not in ("ops", "layers") or test["chunk"] > c.prefill_tile or test["prefill"] + test["decode"] > c.max_ctx:
            raise ValueError("invalid trace/chunk/context: total tokens must fit max_ctx and chunk <= prefill_tile")
    else:
        from verification.models.resnet18.config import Config
        fields(model, set(Config.__dataclass_fields__) - {"word_bits", "integer_bits"}, "model")
        c = Config(**model, word_bits=p.word_bits, integer_bits=p.integer_bits).validate()
        for key in ("acc_word_bits", "acc_integer_bits", "tile_c", "tile_h", "tile_w", "guard"):
            integer(getattr(c, key), key, 1)
        test = fields(raw.get("test", {}), set(), "ResNet test (fixed input shape 3x224x224)")
        if getattr(args, "prefill", None) is not None or getattr(args, "decode", None) is not None:
            raise ValueError("prefill/decode only apply to Llama")
    parameters = dict(seed=42)
    parameters.update(fields(raw.get("parameters", {}), {"seed"}, "parameters"))
    integer(parameters["seed"], "parameters.seed")
    execution = dict(backend="vitis", threads=8, cache_gib=64, vitis=None)
    execution.update(fields(raw.get("execution", {}), execution.keys(), "execution"))
    for key in ("backend", "threads", "vitis"):
        if getattr(args, key, None) is not None:
            execution[key] = getattr(args, key)
    integer(execution["threads"], "execution.threads", 1)
    if not isinstance(execution["cache_gib"], (float, int)) or not math.isfinite(execution["cache_gib"]) or execution["cache_gib"] < 0:
        raise ValueError("cache_gib must be finite and nonnegative")
    if execution["backend"] not in ("native", "vitis") or (raw["family"] == "resnet18" and execution["backend"] != "vitis"):
        raise ValueError("Llama supports native/vitis; ResNet requires vitis for vendor ap_fixed semantics")
    comparison = dict(float_atol=None, float_rtol=None, fail_on_saturation=False)
    comparison.update(fields(raw.get("comparison", {}), comparison.keys(), "comparison"))
    validate_tolerances(comparison["float_atol"], comparison["float_rtol"])
    if type(comparison["fail_on_saturation"]) is not bool:
        raise ValueError("fail_on_saturation must be boolean")
    return dict(schema_version=1, family=raw["family"], model=c.to_dict(), precision=p.describe(),
                parameters=parameters, test=test, execution=execution, comparison=comparison), c


def vitis_path(requested):
    found = shutil.which(requested or "vitis_hls")
    fallback = Path("/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1/bin/vitis_hls")
    if not found and not requested and fallback.is_file():
        found = str(fallback)
    if not found:
        raise ValueError("Vitis HLS not found: add it to PATH or provide --vitis /path/to/vitis_hls")
    return str(Path(found).resolve())


def compare_architecture(case, family, legacy=False, atol=None, rtol=None):
    module = ("verification.legacy.llama3_verification" if family == "llama3" else "verification.legacy.tiled_resnet18_verification") if legacy else "verification.models." + family
    runner = importlib.import_module(module + ".runner")
    filename = "comparison.json" if family == "llama3" else "report.json"
    try:
        if legacy and family == "llama3":
            if atol is not None:
                raise ValueError("archived Llama comparison has diagnostic-only FP64 metrics; use compare-tensors for new tolerances")
            runner.compare(case)
        else:
            runner.compare(case, atol, rtol)
    except RuntimeError:
        # A completed comparison writes its failure report before raising.
        # Only consume a freshly generated report, never a stale success.
        report_path = case / filename
        if not report_path.exists() or json.loads(report_path.read_text()).get("implementation_match") in (True, "PASS") and json.loads(report_path.read_text()).get("mathematical_verdict") != "FAIL":
            raise
    return json.loads((case / filename).read_text())


def finish(report, output, comparison=None):
    if "arithmetic" in report:
        print("ARITHMETIC COUNTERS (reference-model instrumentation):\n" + json.dumps(report["arithmetic"], indent=2))
    summary = render_report(report)
    strict = bool((comparison or {}).get("fail_on_saturation", False))
    passed = summary["implementation_verdict"] == "PASS" and summary["mathematical_verdict"] != "FAIL"
    passed &= not (strict and summary["saturation_count"])
    summary.update(overall_verdict="PASS" if passed else "FAIL", fail_on_saturation=strict)
    write_json(output / "comparison.json", report)
    write_json(output / "summary.json", summary)
    print("Overall verdict:", summary["overall_verdict"], "(including requested tolerance/saturation policy)")
    return 0 if passed else 1


def run(args, output):
    import numpy as np
    import torch
    config, c = resolve_config(json.loads(Path(args.config).read_text()), args)
    print("PyTorch:", torch.__version__, "NumPy:", np.__version__)
    execution = config["execution"]
    if execution["backend"] == "vitis":
        execution["vitis"] = vitis_path(execution["vitis"])
    elif not shutil.which("g++"):
        raise ValueError("g++ is required for the native Llama backend")
    write_json(output / "resolved_config.json", config)
    print("Configuration source:", Path(args.config).resolve(), "SHA256:", sha(args.config))
    paths = dict(design=output / "design", case=output / "case")
    family = config["family"]
    if family == "llama3":
        paths["model"] = Path(args.model_dir).resolve() if args.model_dir else output / "model"
        accumulator = ("signed 64-bit" if c.word_bits <= 16 and c.frac <= 11 else "signed 128-bit") + "; RMS square root uses unsigned 128-bit; no chunk renormalization"
    else:
        if args.model_dir:
            raise ValueError("--model-dir reuse is only supported for Llama")
        accumulator = f"ap_fixed<{c.acc_word_bits},{c.acc_integer_bits},AP_RND,AP_SAT>; vendor expression temporaries can be wider; shared-exponent trace also compared"
    contract(family, config, c.precision.describe(), execution["backend"], paths, accumulator)
    print("Random data are deterministic synthetic parameters, NOT pretrained weights.")
    torch.set_num_threads(execution["threads"])
    package = "verification.models." + family
    codegen, io, runner = [importlib.import_module(package + "." + name) for name in ("codegen", "io", "runner")]
    timings = {}

    def phase(label, fn):
        print("\nPHASE:", label, flush=True)
        started = time.monotonic()
        result = fn()
        timings[label] = time.monotonic() - started
        print(f"PHASE COMPLETE: {label}: {timings[label]:.3f} seconds", flush=True)
        write_json(output / "timings.json", timings)
        return result

    phase("generate C/C++ HLS project", lambda: codegen.emit_project(paths["design"], c))
    seed = config["parameters"]["seed"]
    if family == "llama3":
        if (paths["model"] / "model.json").exists():
            model = phase("validate reused parameters", lambda: io.validate_model(paths["model"]))
            if model["config"] != c.to_dict() or model["seed"] != seed or model["sources"] != {name: sha(paths["design"] / name) for name in io.SOURCE_FILES}:
                raise ValueError("reused model config/seed/generated source differs; use a new model directory")
        else:
            model = phase("generate random parameters", lambda: io.prepare_model(paths["design"], paths["model"], seed))
        print("PARAMETER MANIFEST:\n" + json.dumps(model, indent=2))
        case = phase("generate token input", lambda: io.prepare_case(paths["model"], paths["case"], **config["test"]))
        print("CASE MANIFEST:\n" + json.dumps(case, indent=2))
        phase("PyTorch fixed and FP64 references", lambda: runner.reference(paths["case"], execution["threads"], execution["cache_gib"]))
        phase("actual C/C++ simulation", lambda: runner.csim(paths["case"], execution["backend"], execution["vitis"]))
    else:
        manifest = phase("generate input and random parameters", lambda: io.prepare(paths["design"], paths["case"], seed))
        print("INPUT/PARAMETER MANIFEST:\n" + json.dumps(manifest, indent=2))
        phase("PyTorch fixed and FP64 references", lambda: runner.reference(paths["case"]))
        phase("actual Vitis C/C++ simulation", lambda: runner.csim(paths["case"], execution["vitis"]))
        print("COMPILER/SIMULATOR LOG:\n" + (paths["case"] / "csim/simulation.log").read_text(errors="replace"))
    policy = config["comparison"]
    reference_file = paths["case"] / ("reference.json" if family == "llama3" else "fixed/metadata.json")
    provenance = json.loads(reference_file.read_text())
    provenance = {k: v for k, v in provenance.items() if k not in ("tensors", "checkpoints", "arithmetic")}
    print("REFERENCE PROVENANCE AND SATURATION OBSERVATIONS:\n" + json.dumps(provenance, indent=2))
    report = phase("all-checkpoint comparison", lambda: compare_architecture(paths["case"], family, atol=policy["float_atol"], rtol=policy["float_rtol"]))
    print("PHASE TIMINGS:\n" + json.dumps(timings, indent=2))
    return finish(report, output, policy)


def replay(args, output):
    import torch
    torch.set_num_threads(args.threads)
    case = Path(args.run_dir).resolve()
    validate_tolerances(args.float_atol, args.float_rtol)
    if (case / "case.json").exists():
        family = "llama3"
        case_manifest = json.loads((case / "case.json").read_text())
        metadata = json.loads((Path(case_manifest["model_dir"]) / "model.json").read_text())
    else:
        family = "resnet18"
        metadata = json.loads((case / "manifest.json").read_text())
    legacy = metadata["version"] == 1
    precision = FixedFormat(metadata["config"].get("word_bits", 16), metadata["config"].get("integer_bits", 5))
    contract(family, metadata["config"], precision.describe(), "recomparison of saved simulation", {"case": case},
             "see saved source snapshot; frozen legacy implementation" if legacy else "see resolved model configuration and source snapshot")
    print("Archived legacy path:", legacy)
    print("INPUT/PARAMETER MANIFEST:\n" + json.dumps(metadata, indent=2))
    if family == "llama3":
        print("CASE MANIFEST:\n" + json.dumps(case_manifest, indent=2))
    # Existing adapters write comparison reports beside their input artifacts.
    # Give them a lightweight view under the NEW output directory so replay
    # never overwrites a historical case's reports (including tolerance reruns).
    view = output / "saved_case_view"
    view.mkdir()
    for item in case.iterdir():
        if item.name not in ("comparison.json", "report.json", "report.txt"):
            (view / item.name).symlink_to(item, target_is_directory=item.is_dir())
    report = compare_architecture(view, family, legacy, args.float_atol, args.float_rtol)
    # Older Llama reports did not duplicate shape into each comparison row.
    if family == "llama3":
        ref = json.loads((case / "reference.json").read_text())
        for name, item in report["checkpoints_detail"].items():
            item.setdefault("shape", ref["tensors"]["fixed"][name]["shape"])
    return finish(report, output, dict(fail_on_saturation=args.fail_on_saturation))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="generate, simulate, and compare ResNet/Llama")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output", required=True, help="NEW output directory; existing paths are never overwritten")
    run_parser.add_argument("--word-bits", type=int)
    run_parser.add_argument("--integer-bits", type=int)
    run_parser.add_argument("--backend", choices=("native", "vitis"))
    run_parser.add_argument("--vitis")
    run_parser.add_argument("--threads", type=int)
    run_parser.add_argument("--prefill", type=int)
    run_parser.add_argument("--decode", type=int)
    run_parser.add_argument("--model-dir", help="Llama parameter store to create or reuse with exact config/seed/source matching")
    compare_parser = commands.add_parser("compare", help="integrity-check and recompare saved architecture outputs, including legacy runs")
    compare_parser.add_argument("--run-dir", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--float-atol", type=float)
    compare_parser.add_argument("--float-rtol", type=float)
    compare_parser.add_argument("--fail-on-saturation", action="store_true")
    compare_parser.add_argument("--threads", type=int, default=8)
    generic_parser = commands.add_parser("compare-tensors", help="compare raw/NPY exports from an unrelated accelerator")
    generic_parser.add_argument("--manifest", required=True)
    generic_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"ERROR: output already exists; choose a new directory: {output}", file=sys.stderr)
        return 2
    with logged(output / "verification.log"):
        try:
            if args.command == "run":
                result = run(args, output)
            elif args.command == "compare":
                result = replay(args, output)
            else:
                result = finish(compare_manifest(args.manifest, output), output)
        except Exception as error:
            print("VERIFICATION ERROR:", str(error), flush=True)
            traceback.print_exc()
            write_json(output / "error.json", dict(type=type(error).__name__, message=str(error)))
            result = 2
        print("Finished UTC:", datetime.datetime.now(datetime.timezone.utc).isoformat(), "Exit code:", result)
        return result


if __name__ == "__main__":
    raise SystemExit(main())
