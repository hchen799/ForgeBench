"""License-free syntax check for the Catapult backend (review concern E1).

Generates each design with `tool="catapult"` and compiles the emitted `top.cpp`
against the real Catapult `ac_fixed`/`ac_math` headers using a stock g++. This
catches type and math-binding errors -- the bulk of the porting risk -- without
consuming a Catapult license or waiting on synthesis.

It does NOT check scheduling, area, or the `hls_*` pragmas: g++ ignores unknown
pragmas. Those need a real `catapult -shell -f run_hls.tcl` run.

Mirrors the subprocess+cwd pattern in `verification/gcc_selfcheck.py`, which
exists because the four domain generators are only importable with the domain
directory as cwd.

CLI:
  python -m verification.catapult_syntax_check gemm                 # all test_case_configs
  python -m verification.catapult_syntax_check gemm mlp diff_dims_p1
  python -m verification.catapult_syntax_check gemm --auto 20       # sample auto configs
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MGC_HOME = os.environ.get(
    "MGC_HOME", "/tools/software/siemens/catapult/latest/Mgc_home"
)
AC_INCLUDE = os.path.join(MGC_HOME, "shared", "include")


def _domain_dir(domain):
    return os.path.join(REPO_ROOT, domain)


def generate_catapult_design(domain, config_path, out_base):
    """Generate one design with the Catapult backend; return its run dir."""
    with open(config_path) as f:
        config = json.load(f)
    config["tool"] = "catapult"

    stem = os.path.splitext(os.path.basename(config_path))[0]
    run_name = f"_ctmp_{stem}"
    domain_dir = _domain_dir(domain)
    tmp_cfg = os.path.join(domain_dir, f"{run_name}.json")
    with open(tmp_cfg, "w") as f:
        json.dump(config, f)

    try:
        code = (
            "import gen_configs; "
            f"gen_configs.run_hls_flow({os.path.basename(tmp_cfg)!r}, "
            f"base_dir={out_base!r}, tool='catapult')"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=domain_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)

    return os.path.join(out_base, run_name)


def syntax_check(run_dir):
    """Returns (ok, message)."""
    cp = subprocess.run(
        [
            "g++", "-std=c++11", "-fsyntax-only",
            f"-I{run_dir}", f"-I{AC_INCLUDE}",
            os.path.join(run_dir, "top.cpp"),
        ],
        capture_output=True,
        text=True,
    )
    return cp.returncode == 0, cp.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("domain", help="gemm | conv | llm | scale_models")
    ap.add_argument("configs", nargs="*", help="config stems (default: all test cases)")
    ap.add_argument("--auto", type=int, default=0,
                    help="also sample N configs from auto_generated_configs")
    args = ap.parse_args()

    if not os.path.isdir(AC_INCLUDE):
        sys.exit(f"Catapult headers not found at {AC_INCLUDE}; set MGC_HOME.")

    domain_dir = _domain_dir(args.domain)
    if args.configs:
        paths = [
            os.path.join(domain_dir, "test_case_configs",
                         c if c.endswith(".json") else c + ".json")
            for c in args.configs
        ]
    else:
        paths = sorted(glob.glob(os.path.join(domain_dir, "test_case_configs", "*.json")))

    if args.auto:
        auto = sorted(glob.glob(os.path.join(domain_dir, "auto_generated_configs", "*.json")))
        step = max(1, len(auto) // args.auto)
        paths += auto[::step][: args.auto]

    failures = []
    with tempfile.TemporaryDirectory(prefix="fb_catapult_") as tmp:
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            try:
                run_dir = generate_catapult_design(args.domain, p, tmp)
            except subprocess.CalledProcessError as e:
                failures.append(name)
                print(f"GEN FAIL  {name}\n{e.stderr}")
                continue
            ok, err = syntax_check(run_dir)
            print(f"{'ok       ' if ok else 'COMPILE  '}{name}")
            if not ok:
                failures.append(name)
                print(err)

    print(f"\n{len(paths) - len(failures)}/{len(paths)} designs compiled against ac_ headers")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
