"""Table 7 modularized ("merged") designs checked against their programs: one testbench per merged design that drives the synthesized
`top` with every program's inputs at once and checks each program's outputs against that program's golden and bound.

A merged design (modular_data/<domain>/hls_files/<name>_module*) merges the programs P1..P3 of one Table 7 row into one top function
whose ports are the programs' ports one program after the other (each program's DRAMs in its config order). The check reuses, per
program, exactly what verified the program itself (verification/results_paper/<programs run>/configs/designs__<domain>__<P>__<tag>.json):

  inputs    the program's effective design JSON (per-DRAM input ranges incl. fixed weight / norm ranges), refresh_inputs with the
            program's report seeds (seed_base + report_seed_offset + k), so trial k of the merged design sees the inputs of trial k
            of the program
  golden    compute_goldens on the program's config (float64 on the quantized inputs)
  bound     bound_for with the program's accumulation length and compute stages (as functional_verification for a whole design)

Ports: the merged design's `top` signature is parsed (dims through its #defines). Each program DRAM is mapped, in order, to the next
`top` parameter; parameters named in the spec's `zero` set (ports the program does not have, e.g. a conv bias of a bias-free
program) are zero-filled and not read back. A merged port may be larger than the program's array (designs with MAX_C/MAX_H/MAX_W
buffers and runtime sizes): the program's data sits in the leading corner, the rest is zero, outputs are read back from the corner.

Types: data_t / acc_t / math_t rebound and every spelled-out ap_fixed<16,5> replaced (as prepare_table7_synth.rebind), so the merged
design runs in the format its programs were verified in.

  python3 -m verification.merged_designs --programs-run verification/results_paper/C_32_10__table7r2_pm1 [<more runs>] \
      --out verification/results_paper/C_32_10__table7r2_merged [--designs conv_module ...] [--jobs 6] [--n N]

-> <out>/summary.csv (one row per merged design x program), <out>/trials/<design>__<program>.csv.gz, <out>/<design>.log
"""
import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from verification import manual_designs as M
from verification.csim_runner import csim_build_dir, refresh_inputs, rerun_csim_exe, run_vitis
from verification.fixed_runner import accum_length, read_outputs
from verification.fixedpoint import DType, error_metrics, quantize
from verification.functional_verification import bound_for, compute_ops
from verification.golden_ref import compute_goldens

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_TYPE = re.compile(r"ap_fixed\s*<\s*16\s*,\s*5\s*>")
_LOCK = threading.Lock()

# name -> domain, merged-design directory, programs (in port order), ports of `top` the programs do not have (zero-filled)
SPECS = {
    "diff_dims_module_small": {"domain": "gemm", "programs": ["diff_dims_p1", "diff_dims_p2", "diff_dims_p3"]},
    "diff_orders_module": {"domain": "gemm", "programs": ["diff_orders_p1", "diff_orders_p2", "diff_orders_p3"]},
    "mult_op_module_dot": {"domain": "gemm", "programs": ["mult_op_p1", "mult_op_p2", "mult_op_p3"]},
    "mult_op_module_mm": {"domain": "gemm", "programs": ["mult_op_p1", "mult_op_p2", "mult_op_p3"]},
    "mult_op_module_mmv": {"domain": "gemm", "programs": ["mult_op_p1", "mult_op_p2", "mult_op_p3"]},
    "vec_mtx_module": {"domain": "gemm", "programs": ["vec_mtx_p1", "vec_mtx_p2"]},
    "activation_module": {"domain": "conv", "programs": ["activation_op1", "activation_op2", "activation_op3"]},
    "conv_module": {"domain": "conv", "programs": ["conv_A", "conv_B", "conv_C"]},
    "conv_block_module": {"domain": "conv", "programs": ["conv_block_op1", "conv_block_op2", "conv_block_op3"],
                          "zero": {"conv_bias_1_A", "conv_bias_2_A", "conv_bias_3_A", "conv_bias_4_A", "conv_bias_1_B", "conv_bias_2_B",
                                   "conv_bias_1_C", "conv_bias_2_C", "conv_bias_3_C"}},
    "tiled_attn_module": {"domain": "llm", "programs": ["tiled_attn_p1", "tiled_attn_p2"]},
    "attn_breakdown_module": {"domain": "llm", "programs": ["attn_breakdown_op1", "attn_breakdown_op2"]},
    "Llama_GPT_module": {"domain": "llm", "programs": ["gpt_transformer_p1", "llama_transformer_p2"]},
}


def src_dir(name):
    return os.path.join(REPO, "modular_data", SPECS[name]["domain"], "hls_files", name)


# ----------------------------------------------------------------------------- top signature
def _strip_comments(text):
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def _defines(text):
    d = {}
    for m in re.finditer(r"^\s*#define\s+(\w+)\s+([^\n]+)$", _strip_comments(text), flags=re.M):
        d[m.group(1)] = m.group(2).strip()
    return d


def _eval_dim(expr, defs, depth=0):
    e = expr
    for _ in range(10):
        e2 = re.sub(r"\b([A-Za-z_]\w*)\b", lambda m: f"({defs[m.group(1)]})" if m.group(1) in defs else m.group(1), e)
        if e2 == e:
            break
        e = e2
    if re.search(r"[A-Za-z_]", e):
        raise ValueError(f"cannot evaluate array extent {expr!r} -> {e!r}")
    return int(eval(e, {"__builtins__": {}}))


def top_params(run_dir):
    """-> [(name, [dims])] of `void top(...)` in run_dir/top.cpp (dims evaluated through the #defines of top.h and top.cpp)."""
    text = "".join(open(os.path.join(run_dir, f)).read() + "\n" for f in ("top.h", "top.cpp") if os.path.isfile(os.path.join(run_dir, f)))
    defs = _defines(text)
    src = _strip_comments(open(os.path.join(run_dir, "top.cpp")).read())
    m = re.search(r"\bvoid\s+top\s*\(([^)]*)\)\s*\{", src)
    if not m:
        raise ValueError("no `void top(...) {` definition in top.cpp")
    out = []
    for p in m.group(1).split(","):
        p = p.strip()
        if not p:
            continue
        pm = re.match(r"(?:const\s+)?\w+\s+(\w+)\s*((?:\[[^\]]+\]\s*)+)$", p)
        if not pm:
            raise ValueError(f"unsupported parameter {p!r}")
        out.append((pm.group(1), [_eval_dim(x, defs) for x in re.findall(r"\[([^\]]+)\]", pm.group(2))]))
    return out


# ----------------------------------------------------------------------------- program side (as verified)
class Program:
    def __init__(self, domain, name, programs_run, dt, tol):
        self.domain, self.name, self.dt = domain, name, dt
        # programs_run: one or more verification output dirs; a later dir overrides an earlier one (e.g. a single re-verified program)
        runs = [programs_run] if isinstance(programs_run, str) else list(programs_run)
        cands = [r for r in runs if os.path.isfile(os.path.join(r, "configs", f"designs__{domain}__{name}__{dt.tag()}.json"))]
        if not cands:
            raise FileNotFoundError(f"no verified config of {domain}/{name} ({dt.tag()}) in {runs}")
        programs_run = cands[-1]
        self.source_run = programs_run
        p = os.path.join(programs_run, "configs", f"designs__{domain}__{name}__{dt.tag()}.json")
        self.config = json.load(open(p))
        ver = self.config["_verification"]
        self.seed0 = ver["seed_base"] + ver.get("report_seed_offset", 0)
        self.n_trials = ver["n_trials"]
        self.outputs = self.config["output_dram_names"]
        ops = compute_ops(self.config)
        op = max(ops, key=lambda o: accum_length(o["func_name"], o.get("dims", [])))
        self.L = accum_length(op["func_name"], op.get("dims", []))
        bcfg = {"tolerance": {**tol, "stages": tol["stages"] + len(ops) - 1}}
        _, self.boundfn, _ = bound_for(bcfg, dt, self.L)
        self.trials_file = os.path.join(programs_run, "trials", f"{domain}__{name}__{dt.tag()}.csv.gz")

    def program_trials(self):
        if not os.path.isfile(self.trials_file):
            return {}
        with gzip.open(self.trials_file, "rt") as f:
            return {int(r["seed"]): r for r in csv.DictReader(f)}


# ----------------------------------------------------------------------------- merged design side
def _ports(name, progs, params):
    """-> list of (param name, param dims, program or None, dram dict or None, is_output)."""
    zero = SPECS[name].get("zero", set())
    queue = [(p, d) for p in progs for d in p.config["drams"]]
    out = []
    for pname, pdims in params:
        if pname in zero:
            out.append((pname, pdims, None, None, False))
            continue
        if not queue:
            raise ValueError(f"{name}: top has more ports than its programs ({pname})")
        p, d = queue.pop(0)
        if len(d["dims"]) != len(pdims) or any(a > b for a, b in zip(d["dims"], pdims)):
            raise ValueError(f"{name}: port {pname}{pdims} cannot hold {p.name}.{d['name']}{d['dims']}")
        out.append((pname, pdims, p, d, d["name"] in p.outputs))
    if queue:
        raise ValueError(f"{name}: programs have {len(queue)} more DRAMs than top has ports: {[(p.name, d['name']) for p, d in queue]}")
    for p in progs:
        for o in p.outputs:
            if not any(pp is p and dd["name"] == o for _, _, pp, dd, _ in out):
                raise ValueError(f"{name}: output {p.name}.{o} not mapped")
    return out


def _cd(dims):
    return "".join(f"[{d}]" for d in dims)


def _corner_loops(pdims, mdims, body):
    """C loops over the program extents; body gets (program flat index, merged flat index) expressions."""
    idx = [f"i{k}" for k in range(len(pdims))]
    pflat = " + ".join(f"{i}*{int(np.prod(pdims[k + 1:]))}L" for k, i in enumerate(idx))
    mflat = " + ".join(f"{i}*{int(np.prod(mdims[k + 1:]))}L" for k, i in enumerate(idx))
    head = " ".join(f"for (long {i} = 0; {i} < {d}; {i}++)" for i, d in zip(idx, pdims))
    return f"    {head} {body(pflat, mflat)}"


def testbench(ports, data_t):
    L = ["#include <cstdio>", "#include <cstdlib>", "#include <fstream>", "#include <ap_fixed.h>", "", f"typedef {data_t} data_t;", "",
         "void top(" + ", ".join(f"data_t {n}{_cd(d)}" for n, d, *_ in ports) + ");", ""]
    L += [f"static data_t {n}{_cd(d)};" for n, d, *_ in ports]
    L += ["static double tmp_buf[%d];" % max(int(np.prod(p[3]["dims"])) for p in ports if p[2] is not None), "",
          "static void rd(const char *f, long n) {",
          "    std::ifstream in(f); if (!in) { printf(\"missing %s\\n\", f); exit(2); }",
          "    for (long i = 0; i < n; i++) if (!(in >> tmp_buf[i])) { printf(\"short %s\\n\", f); exit(2); }", "}", "",
          "int main() {"]
    for n, mdims, p, d, is_out in ports:
        if p is None:
            continue
        L.append(f"    rd(\"{p.name}__{d['name']}.txt\", {int(np.prod(d['dims']))}L);")
        L.append(_corner_loops(d["dims"], mdims, lambda pf, mf, n=n: f"((data_t *){n})[{mf}] = (data_t)tmp_buf[{pf}];"))
    L.append("    top(" + ", ".join(n for n, *_ in ports) + ");")
    for n, mdims, p, d, is_out in ports:
        if p is None or not is_out:
            continue
        L += ["    {", f"        FILE *o = fopen(\"{p.name}__{d['name']}_output.txt\", \"w\");"]
        L.append(_corner_loops(d["dims"], mdims, lambda pf, mf, n=n: f"fprintf(o, \"%.17g\\n\", (double)((data_t *){n})[{mf}]);"))
        L += ["        fclose(o);", "    }"]
    L += ["    return 0;", "}"]
    return "\n".join(L) + "\n"


def prepare(name, src, run_dir, dtype_text, progs):
    """Copy the merged design with types rebound, add the merged testbench and a CSIM run_hls.tcl. -> ports."""
    os.makedirs(run_dir, exist_ok=True)
    tys = M.types(dtype_text)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if not os.path.isfile(p) or not fn.endswith((".cpp", ".h", ".hpp")) or fn.startswith("tb_"):
            continue
        text = open(p).read()
        for tname, c in tys.items():
            text = M._typedef_re(tname).sub(f"typedef {c} {tname};", text)
        text = OLD_TYPE.sub(tys["data_t"], text)
        open(os.path.join(run_dir, fn), "w").write(text)
    ports = _ports(name, progs, top_params(run_dir))
    open(os.path.join(run_dir, "tb_merged.cpp"), "w").write(testbench(ports, tys["data_t"]))
    srcs = [f for f in sorted(os.listdir(run_dir)) if f.endswith(".cpp") and f != "tb_merged.cpp"]
    tcl = ["open_project -reset project_1", "set_top top"] + [f"add_files {f}" for f in srcs] + [
        "add_files -tb tb_merged.cpp"] + [f"add_files -tb {p.name}__{d['name']}.txt" for _, _, p, d, _ in ports if p is not None] + [
        "open_solution -reset solution1", "set_part xczu9eg-ffvb1156-2-e",
        "create_clock -period 10 -name default", "csim_design", "exit"]
    open(os.path.join(run_dir, "run_hls.tcl"), "w").write("\n".join(tcl) + "\n")
    return ports


def verify(name, programs_run, out, dt, tol, n_override=None, work=None, keep=False, src=None):
    sp = SPECS[name]
    dom = sp["domain"]
    t0 = time.time()
    log = open(os.path.join(out, f"{name}.log"), "w")

    def say(msg):
        log.write(msg + "\n")
        log.flush()

    progs = [Program(dom, p, programs_run, dt, tol) for p in sp["programs"]]
    rows = {p.name: {"merged_design": f"modular/{dom}/{name}", "program": f"designs/{dom}/{p.name}", "datatype": dt.text, "status": "ok",
                     "n_trials": 0, "trials_within_bound": 0, "max_abs_err": 0.0, "rel_l2": "", "same_as_program": 0,
                     "program_trials_compared": 0, "accum_len": p.L, "program_run": os.path.relpath(p.source_run, REPO), "notes": ""}
            for p in progs}
    run_dir = os.path.join(work or os.path.join(out, "_work"), name)
    shutil.rmtree(run_dir, ignore_errors=True)
    pdirs = {p.name: os.path.join(run_dir, "_inputs", p.name) for p in progs}
    for d in pdirs.values():
        os.makedirs(d, exist_ok=True)
    acc = {p.name: {"err": 0.0, "sig": 0.0} for p in progs}
    trials = {p.name: [] for p in progs}
    try:
        ports = prepare(name, src or src_dir(name), run_dir, dt.text, progs)
        say("ports: " + "; ".join(f"{n}{d} <- " + (f"{p.name}.{dd['name']}{dd['dims']}{' (out)' if o else ''}" if p else "zero")
                                   for n, d, p, dd, o in ports))
        N = n_override or min(p.n_trials for p in progs)
        build = None
        ptrials = {p.name: p.program_trials() for p in progs}
        qf = (lambda x: quantize(x, dt))
        for k in range(N):
            for p in progs:
                refresh_inputs(dom, p.config, pdirs[p.name], seed=p.seed0 + k)
                for d in p.config["drams"]:
                    shutil.copy2(os.path.join(pdirs[p.name], f"{d['name']}.txt"), os.path.join(run_dir, f"{p.name}__{d['name']}.txt"))
            names = [f"{p.name}__{o}" for p in progs for o in p.outputs]
            if build is None:
                rc, text = run_vitis(run_dir, log_name="vitis_trial0.log")
                build = csim_build_dir(run_dir)
                if build is None:
                    raise RuntimeError(f"CSIM build failed (rc={rc}); see {run_dir}/vitis_trial0.log: "
                                       + " | ".join(l.strip() for l in text.splitlines() if "error" in l.lower())[:600])
                if not all(os.path.isfile(os.path.join(build, f"{n}_output.txt")) for n in names):
                    raise RuntimeError(f"trial 0 produced no outputs (rc={rc}); see {run_dir}/vitis_trial0.log")
            else:
                for n in names:
                    if os.path.isfile(os.path.join(build, f"{n}_output.txt")):
                        os.remove(os.path.join(build, f"{n}_output.txt"))
                rc, _ = rerun_csim_exe(run_dir, build)
                if rc != 0:
                    raise RuntimeError(f"csim.exe exited with {rc} at trial {k}")
            for p in progs:
                seed = p.seed0 + k
                o = read_outputs(build, [f"{p.name}__{x}" for x in p.outputs])
                g = compute_goldens(p.config, pdirs[p.name], dom, input_transform=qf)
                gq = np.concatenate([g[x].reshape(-1) for x in p.outputs])
                m = error_metrics(o, gq, atol=0.0)
                within = int(np.all(np.isfinite(o)) and np.all(np.abs(o - gq) <= p.boundfn(gq)))
                r = rows[p.name]
                r["n_trials"] += 1
                r["trials_within_bound"] += within
                r["max_abs_err"] = max(r["max_abs_err"], m["max_abs_err"])
                acc[p.name]["err"] += float(((o - gq) ** 2).sum())
                acc[p.name]["sig"] += float((gq * gq).sum())
                pt = ptrials[p.name].get(seed)
                same = ""
                if pt is not None and pt.get("crash", "0") == "0":
                    same = int(float(pt["max_abs_err"]) == m["max_abs_err"] and float(pt["rmse"]) == m["rmse"])
                    r["program_trials_compared"] += 1
                    r["same_as_program"] += same
                trials[p.name].append({"seed": seed, "max_abs_err": m["max_abs_err"], "rmse": m["rmse"], "within_bound": within,
                                       "same_as_program": same})
            say(f"trial {k}: " + ", ".join(f"{p.name} max_abs={trials[p.name][-1]['max_abs_err']:.3g} "
                                          f"within={trials[p.name][-1]['within_bound']}" for p in progs))
    except Exception as e:
        for r in rows.values():
            r["status"] = "error"
            r["notes"] += f"{type(e).__name__}: {e}"
        say(f"ERROR {type(e).__name__}: {e}")
    for p in progs:
        r = rows[p.name]
        if acc[p.name]["sig"] > 0:
            r["rel_l2"] = math.sqrt(acc[p.name]["err"] / acc[p.name]["sig"])
        if r["status"] == "ok" and r["trials_within_bound"] < r["n_trials"]:
            r["status"] = "outside_bound"
        r["total_s"] = f"{time.time() - t0:.1f}"
    tdir = os.path.join(out, "trials")
    os.makedirs(tdir, exist_ok=True)
    for p in progs:
        if trials[p.name]:
            with gzip.open(os.path.join(tdir, f"{name}__{p.name}.csv.gz"), "wt") as f:
                w = csv.DictWriter(f, fieldnames=list(trials[p.name][0]), lineterminator="\n")
                w.writeheader()
                w.writerows(trials[p.name])
    rdir = os.path.join(out, "rows")
    os.makedirs(rdir, exist_ok=True)
    json.dump(list(rows.values()), open(os.path.join(rdir, f"{name}.json"), "w"), indent=1, default=str)
    if not keep and all(r["status"] == "ok" for r in rows.values()):
        shutil.rmtree(run_dir, ignore_errors=True)
    with _LOCK:
        for r in rows.values():
            print(f"  {name} <- {r['program'].rsplit('/', 1)[1]}: {r['status']} within={r['trials_within_bound']}/{r['n_trials']} "
                  f"max_abs={r['max_abs_err']:.3g} same_as_program={r['same_as_program']}/{r['program_trials_compared']} "
                  f"({r['total_s']}s) {r['notes'][:200]}", flush=True)
    log.close()
    return list(rows.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--programs-run", required=True, nargs="+",
                    help="functional_verification output dir(s) of the programs (configs/, trials/); a later dir overrides an earlier one")
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", nargs="*", default=sorted(SPECS))
    ap.add_argument("--src-root", help="take merged designs from <src-root>/<name>/ instead of modular_data/<domain>/hls_files/<name>/")
    ap.add_argument("--datatype", default="fixed<32,10>")
    ap.add_argument("--n", type=int, help="trials per design (default: the smallest N of its programs)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    dt = DType(a.datatype)
    tol = json.load(open(os.path.join(a.programs_run[0], "resolved_config.json")))["tolerance"]
    print(f"{len(a.designs)} merged designs, {dt.text}, programs from {a.programs_run}", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda n: verify(n, a.programs_run, out, dt, tol, a.n, keep=a.keep,
                                           src=os.path.join(a.src_root, n) if a.src_root else None), a.designs))
    rows = [r for rs in res for r in rs]
    allrows = []
    for fn in sorted(os.listdir(os.path.join(out, "rows"))):           # summary over every design run into this folder so far
        allrows += json.load(open(os.path.join(out, "rows", fn)))
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(allrows)
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"DONE {len(rows)} design x program rows; {len(bad)} not ok -> {out}/summary.csv", flush=True)


if __name__ == "__main__":
    main()
