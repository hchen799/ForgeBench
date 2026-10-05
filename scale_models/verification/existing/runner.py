"""Reproducible fixtures, direct C simulation, and two independent verdicts."""
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

from .arithmetic import DATA, ArithmeticFault
from .project import sha, testbench, write_json
from .resnet import ResNetReference
from .llama import LlamaReference, Weights


def save_tensor(path, value, fixed):
    array = value.detach().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError("nonfinite reference: " + str(path))
    # NPY preserves shape. Fixed references remain int64; actual codes are i16.
    np.save(path, array.astype("<i8" if fixed else "<f8"), allow_pickle=False)


def prepare_inputs(project, folder, seed):
    folder.mkdir()
    needed = sum(math.prod(s) * 2 for s in project.inputs.values())
    if shutil.disk_usage(folder).free < needed + (1 << 30):
        raise ValueError("insufficient space for parameter store")
    entries = {}
    for name, shape in project.inputs.items():
        # Tensor-specific seed makes the result independent of port order and
        # keeps prefill/decode weights identical without another full copy.
        import hashlib

        key = int.from_bytes(
            hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little"
        )
        rng = np.random.default_rng(key)
        count = math.prod(shape)
        path = folder / (name + ".bin")
        bn = name.startswith("DRAM_bn_")
        norm = "norm" in name and project.family == "llama3"
        fan_in = math.prod(shape[1:]) if project.family == "resnet18" else shape[-1]
        bound = (
            1.0
            if name == "DRAM_input"
            else 0.5
            if name == "DRAM_embedding"
            else math.sqrt(3 / fan_in)
        )
        with path.open("wb") as f:
            for start in range(0, count, 1 << 20):
                n = min(1 << 20, count - start)
                if bn:
                    # BN stores gamma, beta, mean, variance, each one full row.
                    row = np.arange(start, start + n) // shape[1]
                    low = np.choose(row, [0.9, -0.05, -0.05, 0.9])
                    high = np.choose(row, [1.1, 0.05, 0.05, 1.1])
                    real = rng.uniform(low, high)
                else:
                    real = (
                        rng.uniform(0.9, 1.1, n)
                        if norm
                        else rng.uniform(-bound, bound, n)
                    )
                codes = DATA.quantize(real).numpy().astype("<i2")
                f.write(codes.tobytes())
        entries[name] = dict(
            shape=shape,
            dtype="<i2",
            sha256=sha(path),
            seed=key,
            distribution="bn_rows" if bn else "norm_uniform" if norm else "uniform",
            bound=bound if not (bn or norm) else None,
        )
        print("Prepared", name, shape, flush=True)
    write_json(
        folder / "manifest.json",
        dict(seed=seed, arithmetic=DATA.describe(), tensors=entries),
    )
    return entries


def validate_inputs(folder, entries):
    for name, entry in entries.items():
        path = folder / (name + ".bin")
        if (
            path.stat().st_size != math.prod(entry["shape"]) * 2
            or sha(path) != entry["sha256"]
        ):
            raise ValueError("input changed or truncated: " + name)


def find_vitis(requested=None):
    path = requested or shutil.which("vitis_hls")
    if not path:
        candidate = "/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1/bin/vitis_hls"
        path = candidate if Path(candidate).exists() else None
    if not path or not Path(path).is_file():
        raise ValueError("Vitis HLS not found; supply --vitis")
    return Path(path).resolve()


def simulate(project, source, inputs, case, input_names, outputs, vitis, timeout):
    (case / "actual").mkdir(exist_ok=True)
    (source / "tb_existing.cpp").write_text(testbench(project, input_names, outputs))
    # Never run the project's original synthesis TCL or output-only testbench.
    (source / "run_existing_csim.tcl").write_text(
        f"""cd [file dirname [file normalize [info script]]]
open_project existing_csim
set_top {project.top_name}
add_files top.cpp -cflags "-std=c++14 -O3 -DNDEBUG"
add_files -tb tb_existing.cpp -cflags "-std=c++14 -O3 -DNDEBUG"
open_solution solution1
set_part xczu9eg-ffvb1156-2-e
create_clock -period 10 -name default
csim_design -O -argv [list $::env(FORGEBENCH_EXISTING_INPUTS) $::env(FORGEBENCH_EXISTING_CASE)]
exit
"""
    )
    env = os.environ.copy()
    env["FORGEBENCH_EXISTING_INPUTS"] = str(inputs)
    env["FORGEBENCH_EXISTING_CASE"] = str(case)
    command = [str(vitis), "-f", "run_existing_csim.tcl"]
    start = time.monotonic()
    print(
        "Compiling/running saved",
        project.variant,
        "log:",
        case / "simulation.log",
        flush=True,
    )
    with (case / "simulation.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f"C simulation exceeded {timeout} seconds")
    text = (case / "simulation.log").read_text(errors="replace")
    success = rc == 0 and "ACCELERATOR_EXECUTED" in text
    for name, digest in project.files.items():
        if sha(source / name) != digest:
            raise ValueError("simulation source snapshot changed: " + name)
    result = dict(
        returncode=rc,
        executed=success,
        elapsed_seconds=time.monotonic() - start,
        command=command,
        tool=str(vitis),
        tool_sha256=sha(vitis),
        log_sha256=sha(case / "simulation.log"),
        testbench_sha256=sha(source / "tb_existing.cpp"),
        tcl_sha256=sha(source / "run_existing_csim.tcl"),
    )
    executable = source / "existing_csim/solution1/csim/build/csim.exe"
    if executable.is_file():
        result["executable_sha256"] = sha(executable)
        if not success:
            # Vitis sometimes suppresses the terminating signal. Replay the
            # exact binary once for a useful error, never substituting outputs
            # from a failed invocation into a successful comparison.
            try:
                probe = subprocess.run(
                    [str(executable), str(inputs), str(case)],
                    cwd=source,
                    capture_output=True,
                    timeout=min(timeout, 60),
                )
                result["diagnostic_replay_returncode"] = probe.returncode
                if probe.returncode < 0:
                    result["signal"] = signal.Signals(-probe.returncode).name
                (case / "diagnostic_replay.log").write_bytes(
                    probe.stdout + probe.stderr
                )
            except subprocess.TimeoutExpired:
                result["diagnostic_replay"] = "timed out"
    write_json(case / "simulation.json", result)
    if not success:
        raise RuntimeError(
            "C simulation failed"
            + (" (" + result["signal"] + ")" if "signal" in result else "")
            + "; see "
            + str(case / "simulation.log")
        )
    return result


def compare_arrays(actual, fixed, floating, max_abs=0.1, relative_l2=0.01):
    actual, fixed, floating = (
        np.asarray(actual),
        np.asarray(fixed),
        np.asarray(floating),
    )
    if actual.shape != fixed.shape or actual.shape != floating.shape:
        raise ValueError("actual/fixed/FP64 tensor shapes differ")
    if not actual.size or not all(
        np.isfinite(x).all() for x in (actual, fixed, floating)
    ):
        raise ValueError("empty or nonfinite comparison tensor")
    if actual.dtype.kind != "i" or fixed.dtype.kind != "i":
        raise ValueError("actual/fixed values must be raw signed integer codes")
    if any(np.any((x < -32768) | (x > 32767)) for x in (actual, fixed)):
        raise ValueError("raw code outside ap_fixed<16,5> range")
    a = actual.astype(np.int64)
    delta_code = a - fixed.astype(np.int64)
    delta = a.astype(np.float64) / 2048 - floating
    reference_norm = float(np.linalg.norm(floating.reshape(-1)))
    error_norm = float(np.linalg.norm(delta.reshape(-1)))
    rel = error_norm / reference_norm if reference_norm else None
    maximum = float(np.abs(delta).max())
    mismatch = int(np.count_nonzero(delta_code))
    mathematical_pass = maximum <= max_abs and (rel is None or rel <= relative_l2)
    report = dict(
        elements=actual.size,
        implementation_verdict="PASS" if not mismatch else "FAIL",
        mismatching_codes=mismatch,
        max_code_error=int(np.abs(delta_code).max()),
        mathematical_verdict="PASS" if mathematical_pass else "FAIL",
        max_abs=maximum,
        mae=float(np.abs(delta).mean()),
        rmse=float(np.sqrt(np.square(delta).mean())),
        relative_l2=rel,
        zero_reference=reference_norm == 0,
    )
    if mismatch:
        idx = tuple(int(i) for i in np.argwhere(delta_code != 0)[0])
        report["first_mismatch"] = dict(
            index=idx, actual=int(a[idx]), fixed=int(fixed[idx])
        )
    return report


def compare_case(case, outputs):
    records = {}
    for label, (_, shape) in outputs.items():
        path = case / "actual" / (label + ".bin")
        if path.stat().st_size != math.prod(shape) * 2:
            raise ValueError("incorrect accelerator output size: " + str(path))
        a = np.fromfile(path, dtype="<i2").reshape(shape)
        f = np.load(case / "fixed" / (label + ".npy"), allow_pickle=False)
        golden = np.load(case / "fp64" / (label + ".npy"), allow_pickle=False)
        records[label] = compare_arrays(a, f, golden)
        records[label]["sha256"] = dict(
            actual=sha(path),
            fixed=sha(case / "fixed" / (label + ".npy")),
            fp64=sha(case / "fp64" / (label + ".npy")),
        )
        print(
            label,
            records[label]["implementation_verdict"],
            records[label]["mathematical_verdict"],
            "max_abs=",
            records[label]["max_abs"],
            flush=True,
        )
    write_json(case / "comparison.json", records)
    return records


def run_resnet(project, output, args):
    inputs = output / "inputs"
    entries = prepare_inputs(project, inputs, args.seed)
    case = output / "case"
    case.mkdir()
    tensors = {
        n: torch.from_numpy(
            np.fromfile(inputs / (n + ".bin"), dtype="<i2").reshape(s).astype(np.int64)
        )
        for n, s in project.inputs.items()
    }
    for fixed, mode in ((False, "fp64"), (True, "fixed")):
        folder = case / mode
        folder.mkdir()

        def checkpoint(name, value):
            print(mode, name, flush=True)
            if name in project.outputs:
                save_tensor(folder / (name + ".npy"), value, fixed)

        ResNetReference(project.variant, fixed).run(tensors, checkpoint)
    validate_inputs(inputs, entries)
    simulation = simulate(
        project,
        output / "source",
        inputs,
        case,
        project.inputs,
        project.outputs,
        args.vitis,
        args.timeout,
    )
    validate_inputs(inputs, entries)
    records = compare_case(case, project.outputs)
    return dict(
        cases={"resnet18": records}, simulation=simulation, input_manifest=entries
    )


def llama_preflight(project, decode, args):
    if project.variant != "llama3-prefill" or decode.variant != "llama3-decode":
        raise ValueError("Llama requires a prefill project and a decode project")
    if project.inputs != decode.inputs or project.max_ctx != decode.max_ctx:
        raise ValueError("prefill/decode parameter or cache layouts differ")
    if (
        args.prefill < 1
        or args.decode < 0
        or args.prefill + args.decode > project.max_ctx
    ):
        raise ValueError("invalid Llama token schedule")
    if args.tokens:
        tokens = [int(t) for t in args.tokens.split(",")]
        if len(tokens) != args.prefill + args.decode:
            raise ValueError("--tokens must provide exactly prefill+decode IDs")
    else:
        tokens = (
            np.random.default_rng(args.seed + 1)
            .integers(0, 128256, args.prefill + args.decode)
            .tolist()
        )
    failures = []
    for role, values in (
        ("token IDs", tokens),
        ("prefill length", [args.prefill]),
        ("decode positions", list(range(args.prefill, args.prefill + args.decode))),
    ):
        if any(int(DATA.quantize(v)) != v * 2048 for v in values):
            failures.append(
                role + " cannot be represented by the existing data_t interface"
            )
    if any(t < 0 or t >= 128256 for t in tokens):
        failures.append("token ID outside vocabulary")
    # Exercise the independent fixed reference on the actual constant/divisor,
    # before allocating 8B weights. Invalid arithmetic cannot get a PASS.
    from .llama import FixedOps

    try:
        FixedOps(project.tile_in, project.hidden_chunk).rmsnorm(
            torch.zeros((1, 4096), dtype=torch.int64),
            torch.full((4096,), 2048, dtype=torch.int64),
        )
    except ArithmeticFault as exc:
        failures.append(str(exc))
    try:
        FixedOps(project.tile_in, project.hidden_chunk).attention(
            torch.zeros((1, 4096), dtype=torch.int64),
            torch.zeros((1, 1024), dtype=torch.int64),
            torch.zeros((1, 1024), dtype=torch.int64),
            0,
        )
    except ArithmeticFault as exc:
        failures.append(str(exc))
    return tokens, failures


def run_llama(project, decode, output, args, tokens):
    from .llama import FixedOps
    from .vendor_math import VendorRope

    vendor_root = args.vitis.parent.parent
    provider = VendorRope(vendor_root)
    dependencies = provider.provenance()
    write_json(output / "fixed_reference_dependencies.json", dependencies)
    project.vendor_root = decode.vendor_root = vendor_root
    inputs = output / "inputs"
    entries = prepare_inputs(project, inputs, args.seed)
    validate_inputs(inputs, entries)
    weights = Weights(inputs, project.inputs)
    refs = {
        "fixed": LlamaReference(project, weights, True),
        "fp64": LlamaReference(project, weights, False),
    }
    cases, simulations, coefficient_files = {}, {}, {}
    previous = None
    schedule = [(0, args.prefill)] + [
        (p, 1) for p in range(args.prefill, args.prefill + args.decode)
    ]
    for call, (start, rows) in enumerate(schedule):
        design = project if call == 0 else decode
        case = output / f"call{call:04d}"
        case.mkdir()
        # Use this accelerator version's arithmetic without resetting its
        # independent reference cache at the prefill/decode boundary.
        refs["fixed"].ops = FixedOps(design.tile_in, design.hidden_chunk, vendor_root)
        refs["fixed"].ops.rope_provider = provider
        coefficients = case / "fixed_rope_coefficients.npy"
        save_tensor(coefficients, provider.coefficients(start, rows), True)
        coefficient_files[f"call{call:04d}"] = sha(coefficients)
        for mode, ref in refs.items():
            (case / mode).mkdir()
            ref.forward(
                tokens[start : start + rows],
                start,
                lambda n, v: save_tensor(
                    case / mode / (n + ".npy"), v, mode == "fixed"
                ),
            )
        token_name = "DRAM_token_ids" if call == 0 else "DRAM_token_id"
        control_name = "DRAM_prefill_len" if call == 0 else "DRAM_decode_pos"
        ids = np.zeros(design.ports[token_name], dtype="<i2")
        ids[:rows] = np.array(tokens[start : start + rows]) * 2048
        ids.tofile(case / (token_name + ".bin"))
        np.array([(rows if call == 0 else start) * 2048], dtype="<i2").tofile(
            case / (control_name + ".bin")
        )
        input_names = set(design.inputs) | {token_name, control_name}
        if previous:
            for name, label in (
                ("DRAM_k_cache", "k_cache_full"),
                ("DRAM_v_cache", "v_cache_full"),
            ):
                shutil.copyfile(
                    previous / "actual" / (label + ".bin"), case / (name + ".bin")
                )
                input_names.add(name)
        logit_name = "DRAM_logits" if call == 0 else "DRAM_logits_decode"
        outputs = {
            "logits": (logit_name, (rows, 128256)),
            "k_cache_full": ("DRAM_k_cache", design.ports["DRAM_k_cache"]),
            "v_cache_full": ("DRAM_v_cache", design.ports["DRAM_v_cache"]),
        }
        source = output / ("source" if call == 0 else "decode_source")
        simulations[f"call{call:04d}"] = simulate(
            design, source, inputs, case, input_names, outputs, args.vitis, args.timeout
        )
        for label in ("k_cache", "v_cache"):
            shape = design.ports["DRAM_" + label]
            array = np.memmap(
                case / "actual" / (label + "_full.bin"),
                mode="r",
                dtype="<i2",
                shape=shape,
            )
            np.ascontiguousarray(array[:, : start + rows]).tofile(
                case / "actual" / (label + ".bin")
            )
        selected = {
            "logits": outputs["logits"],
            **{
                n: ("DRAM_" + n, (32, start + rows, 8, 128))
                for n in ("k_cache", "v_cache")
            },
        }
        cases[f"call{call:04d}"] = compare_case(case, selected)
        previous = case
    validate_inputs(inputs, entries)
    return dict(
        cases=cases,
        input_manifest=entries,
        simulations=simulations,
        fixed_reference_dependencies=dependencies,
        fixed_rope_coefficient_sha256=coefficient_files,
    )
