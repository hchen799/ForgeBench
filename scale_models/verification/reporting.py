"""Console/file logging and a model-independent, reviewable comparison report."""
import contextlib
import datetime
import json
import math
import platform
import sys
from pathlib import Path


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.file.flush()


@contextlib.contextmanager
def logged(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            print("FORGEBENCH VERIFICATION LOG")
            print("UTC:", datetime.datetime.now(datetime.timezone.utc).isoformat())
            print("Python:", sys.version.split()[0], "Platform:", platform.platform())
            print("Command:", json.dumps(sys.argv))
            print("Log:", path.resolve())
            yield


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def contract(family, config, precision, backend, paths, accumulator):
    print("\nCOMPARISON CONTRACT")
    print("Model family:", family)
    print("Resolved configuration:\n" + json.dumps(config, indent=2))
    print("Stored signed fixed point:\n" + json.dumps(precision, indent=2))
    print("Accumulator:", accumulator)
    print("Execution:", backend, "(host C/C++ simulation; NOT FPGA-board execution or RTL cosimulation)")
    print("A: actual generated C/C++ checkpoint codes versus exact fixed-point PyTorch checkpoint codes.")
    print("A requires zero mismatching integer codes at EVERY selected checkpoint; no tolerance is applied.")
    print("B: dequantized C/C++ values versus mathematical FP64 PyTorch, using the SAME stored input parameters.")
    print("B measures accumulated quantization/nonlinear approximation error, not pretrained model quality.")
    print("B is diagnostic unless both absolute and relative tolerances are supplied.")
    print("Paths:\n" + json.dumps({k: str(Path(v).resolve()) for k, v in paths.items()}, indent=2))


def render_report(report):
    """Log every tensor, not just the final logit or an abbreviated top-k."""
    print("\nPER-CHECKPOINT COMPARISONS")
    records = report.get("checkpoints_detail")
    if records is not None:
        entries = records.items()
    else:
        entries = [(item["name"], item) for item in report["checkpoints"]]
    count = mismatches = codes = 0
    last_errors = None
    for name, entry in entries:
        error = entry.get("mathematical_errors", entry.get("mathematical_error"))
        bad = entry.get("integer_mismatches", entry.get("mismatches", 0))
        shape = entry.get("shape")
        elements = math.prod(shape) if shape else error.get("count", 0)
        count += 1
        mismatches += bad
        codes += elements
        last_errors = error
        print(f"{name} shape={shape or 'see reference manifest'} elements={elements} "
              f"integer_mismatches={bad} max_lsb_error={entry.get('max_lsb_error', 0 if bad == 0 else 'not_recorded')} "
              f"max_abs={error['max_abs']:.12g} mae={error['mae']:.12g} rmse={error['rmse']:.12g} "
              f"max_relative={error['max_relative']:.12g} relative_floor={error['relative_denominator_floor']:.12g}")
        if "first_mismatch" in entry:
            print("  First mismatch:", json.dumps(entry["first_mismatch"]))
        if "mathematical_tolerance_mismatches" in entry:
            print("  Mathematical tolerance failures:", entry["mathematical_tolerance_mismatches"])
        if "tolerance_mismatches" in error:
            print("  Mathematical tolerance failures:", error["tolerance_mismatches"])
    implementation = report["implementation_match"] in (True, "PASS")
    verdict = report.get("mathematical_verdict", "DIAGNOSTIC_ONLY")
    saturation_count = report.get("saturation_count")
    if saturation_count is None and "saturation_observability" not in report:
        saturation_count = sum(value for stats in report.get("arithmetic", {}).values()
                               for name, value in stats.items() if name.endswith("saturations"))
    summary = dict(implementation_verdict="PASS" if implementation else "FAIL",
                   mathematical_verdict=verdict, checkpoints=count, compared_codes=codes,
                   integer_mismatches=mismatches, saturation_count=saturation_count,
                   float_atol=report.get("float_atol"), float_rtol=report.get("float_rtol"))
    if "saturation_observability" in report:
        summary["saturation_observability"] = report["saturation_observability"]
        summary["last_checkpoint_errors"] = last_errors
    else:
        summary["logit_errors"] = report.get("logit_errors", last_errors)
    if "exponent_match" in report:
        summary["exponent_match"] = report["exponent_match"]
    print("\nFINAL COMPARISON SUMMARY\n" + json.dumps(summary, indent=2))
    print("Relative errors near zero can be large; inspect absolute errors and the denominator floor.")
    if saturation_count:
        print("WARNING: clipping occurred. Bit-exact agreement does not imply saturation-free arithmetic.")
    print("No pretrained-model accuracy, RTL correctness, board timing, or throughput claim follows from this result.")
    return summary
