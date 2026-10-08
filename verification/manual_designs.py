"""Hand-written (non-generated) designs of the paper's modularization study (Table 7), verified with the same engine as generated ones.

These designs have no JSON config and no testbench: they were written by hand (`modular_data/<domain>/hls_files/<name>/top.cpp`).
Each entry of SPECS describes one design's top-level ports (declared C++ extents and the region the design actually uses), the input
ranges, and an independent float64 golden of what the design is meant to compute. The harness

  * copies the design, rebinds `typedef ... data_t;` to the format under test (float / any ap_fixed spelling) and `acc_t` / `math_t`
    (accumulators / hls::sqrt+exp operands) to the config's op_params acc_type / acc_rounding / acc_overflow (default: data_t),
  * writes a generic testbench (reads `<port>.txt` for every input port, calls `top`, writes `<port>_output.txt` for every output port),
  * builds it once with Vitis CSIM and re-runs the built csim.exe for every trial (same scheme as the generated designs),

and presents the interface of `functional_verification.Harness`, so range search / N trials / error bounds are shared.

Variant ids: designs/<domain>/<name> with operator "manual_design".
"""
import json
import os
import re
import shutil

import numpy as np

from verification.csim_runner import csim_build_dir, rerun_csim_exe, run_vitis
from verification.fixedpoint import quantize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- float64 goldens (independent of the C++)
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _elu(x, alpha=0.5):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1.0))


def _conv(x, w, b, pad):
    """x [Cin,H,W], w [Cout,Cin,K,K], stride 1, zero padding."""
    cin, h, wd = x.shape
    cout, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    oh, ow = h + 2 * pad - k + 1, wd + 2 * pad - k + 1
    out = np.zeros((cout, oh, ow))
    for kh in range(k):
        for kw in range(k):
            patch = xp[:, kh:kh + oh, kw:kw + ow].reshape(cin, -1)          # [Cin, OH*OW]
            out += (w[:, :, kh, kw] @ patch).reshape(cout, oh, ow)
    return out + b[:, None, None]


def _bn(x, p, eps=1e-5):
    """p [4, C] = gamma, beta, mean, var."""
    g, be, m, v = (p[i][:, None, None] for i in range(4))
    return g * (x - m) / np.sqrt(v + eps) + be


def _relu(x):
    return np.maximum(x, 0.0)


def _maxpool2(x):
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def _attention(x, wq, wk, wv, heads, rope):
    """GPT-style multi-head attention core (no output projection, no mask). Weights are [out][in]: Q = x @ Wq^T."""
    q, k, v = x @ wq.T, x @ wk.T, x @ wv.T
    seq, dim = x.shape
    hd = dim // heads
    if rope:
        q, k = _rope(q, heads, hd), _rope(k, heads, hd)
    out = np.zeros_like(q)
    for h in range(heads):
        s = slice(h * hd, (h + 1) * hd)
        sc = q[:, s] @ k[:, s].T / np.sqrt(hd)
        sc = np.exp(sc - sc.max(axis=1, keepdims=True))
        out[:, s] = (sc / sc.sum(axis=1, keepdims=True)) @ v[:, s]
    return out


def _rope(x, heads, hd):
    """Interleaved-pair RoPE: (x[2i], x[2i+1]) of each head rotated by pos * 10000^(-2i/hd)."""
    out = x.copy()
    pos = np.arange(x.shape[0])[:, None]
    for h in range(heads):
        for d in range(0, hd, 2):
            ang = pos[:, 0] * 10000.0 ** (-d / hd)
            a, b = x[:, h * hd + d], x[:, h * hd + d + 1]
            out[:, h * hd + d] = a * np.cos(ang) - b * np.sin(ang)
            out[:, h * hd + d + 1] = a * np.sin(ang) + b * np.cos(ang)
    return out


# ----------------------------------------------------------------------------- design specs
FM, WK3, WK1, BIAS, BN = [256, 56, 56], [256, 256, 3, 3], [256, 256, 1, 1], [256], [4, 256]
BN_RANGE = [0.25, 1.0]          # gamma, beta, mean, var all in [0.25, 1] (variance floor), as for the batchnorm operator


def _w(fan_in):
    """Weight range +-sqrt(3/fan_in): unit-variance weight sums (each layer keeps its input's scale), so a chain of layers stays
    within the data format for +-1 inputs instead of growing by sqrt(fan_in) * 0.1 per layer."""
    a = float(np.sqrt(3.0 / fan_in))
    return (-a, a)


def _p(name, decl, active=None, out=False, rng=(-1.0, 1.0), fixed=False):
    return {"name": name, "decl": list(decl), "active": list(active or decl), "out": out, "input_range": list(rng), "fixed_range": fixed}


def _act_ports():
    return [_p(f"{d}_{a}", [64, 28, 28], out=(d == "output")) for a in ("sigmoid", "tanh", "elu") for d in ("input", "output")]


def _conv_block_ports():
    A = [_p("input_A", FM, [128, 56, 56])] + [q for i in range(1, 5) for q in (
        _p(f"conv_weight_{i}_A", WK3, [256, 128 if i == 1 else 256, 3, 3], rng=_w(9 * (128 if i == 1 else 256)), fixed=True),
        _p(f"conv_bias_{i}_A", BIAS, rng=(-0.1, 0.1), fixed=True))] + [_p("output_A", FM, [256, 28, 28], out=True)]
    B = [_p("input_B", FM, [256, 14, 14])] + [q for i in (1, 2) for q in (
        _p(f"conv_weight_{i}_B", WK3, rng=_w(9 * 256), fixed=True), _p(f"conv_bias_{i}_B", BIAS, rng=(-0.1, 0.1), fixed=True),
        _p(f"batch_norm_weight_{i}_B", BN, rng=BN_RANGE, fixed=True))] + [_p("output_B", FM, [256, 14, 14], out=True)]
    C = [_p("input_C", FM, [256, 56, 56]),
         _p("conv_weight_1_C", WK1, [64, 256, 1, 1], rng=_w(256), fixed=True), _p("conv_bias_1_C", BIAS, [64], rng=(-0.1, 0.1), fixed=True),
         _p("batch_norm_weight_1_C", BN, [4, 64], rng=BN_RANGE, fixed=True),
         _p("conv_weight_2_C", WK3, [64, 64, 3, 3], rng=_w(9 * 64), fixed=True), _p("conv_bias_2_C", BIAS, [64], rng=(-0.1, 0.1), fixed=True),
         _p("batch_norm_weight_2_C", BN, [4, 64], rng=BN_RANGE, fixed=True),
         _p("conv_weight_3_C", WK1, [256, 64, 1, 1], rng=_w(64), fixed=True), _p("conv_bias_3_C", BIAS, [256], rng=(-0.1, 0.1), fixed=True),
         _p("batch_norm_weight_3_C", BN, [4, 256], rng=BN_RANGE, fixed=True),
         _p("output_C", FM, [256, 56, 56], out=True)]
    return A, B, C


def _golden_block(which, t):
    if which == "A":
        x = t["input_A"]
        for i in range(1, 5):
            x = _relu(_conv(x, t[f"conv_weight_{i}_A"], t[f"conv_bias_{i}_A"], 1))
        return {"output_A": _maxpool2(x)}
    if which == "B":
        x = t["input_B"]
        y = _relu(_bn(_conv(x, t["conv_weight_1_B"], t["conv_bias_1_B"], 1), t["batch_norm_weight_1_B"]))
        y = _bn(_conv(y, t["conv_weight_2_B"], t["conv_bias_2_B"], 1), t["batch_norm_weight_2_B"])
        return {"output_B": _relu(x + y)}
    x = t["input_C"]
    y = _relu(_bn(_conv(x, t["conv_weight_1_C"], t["conv_bias_1_C"], 0), t["batch_norm_weight_1_C"]))
    y = _relu(_bn(_conv(y, t["conv_weight_2_C"], t["conv_bias_2_C"], 1), t["batch_norm_weight_2_C"]))
    y = _bn(_conv(y, t["conv_weight_3_C"], t["conv_bias_3_C"], 0), t["batch_norm_weight_3_C"])
    return {"output_C": _relu(x + y)}


def _attn_ports():
    return [q for s in ("A", "B") for q in (
        _p(f"input_dram_{s}", [8, 32]), _p(f"Q_weight_dram_{s}", [32, 32], rng=_w(32), fixed=True),
        _p(f"K_weight_dram_{s}", [32, 32], rng=_w(32), fixed=True), _p(f"V_weight_dram_{s}", [32, 32], rng=_w(32), fixed=True),
        _p(f"output_dram_{s}", [8, 32], out=True))]


def _golden_attn(s, t):
    return {f"output_dram_{s}": _attention(t[f"input_dram_{s}"], t[f"Q_weight_dram_{s}"], t[f"K_weight_dram_{s}"], t[f"V_weight_dram_{s}"],
                                           heads=8, rope=(s == "B"))}


_A, _B, _C = _conv_block_ports()
_ACT = {"sigmoid": _sigmoid, "tanh": np.tanh, "elu": _elu}

# name -> spec. `used`: the input/output ports this design reads/writes (other ports exist in the signature but are untouched);
# `accum`: longest accumulation of one output element (tolerance); `ops`: number of dependent compute stages (tolerance stages).
SPECS = {}
for i, a in enumerate(("sigmoid", "tanh", "elu"), 1):
    SPECS[f"conv/activation_op{i}"] = dict(dir="modular_data/conv/hls_files/activation_op%d" % i, ports=_act_ports(),
                                           used=[f"input_{a}", f"output_{a}"], golden=(lambda t, a=a: {f"output_{a}": _ACT[a](t[f"input_{a}"])}),
                                           accum=1, ops={"sigmoid": 4, "tanh": 6, "elu": 4}[a])
for i, (blk, ports) in enumerate((("A", _A), ("B", _B), ("C", _C)), 1):
    SPECS[f"conv/conv_block_op{i}"] = dict(dir="modular_data/conv/hls_files/conv_block_op%d" % i, ports=_A + _B + _C,
                                           used=[p["name"] for p in ports], golden=(lambda t, b=blk: _golden_block(b, t)),
                                           accum={"A": 256 * 9, "B": 256 * 9, "C": 256}[blk], ops={"A": 9, "B": 7, "C": 10}[blk])
for i, s in enumerate(("A", "B"), 1):
    SPECS[f"llm/attn_breakdown_op{i}"] = dict(dir="modular_data/llm/hls_files/attn_breakdown_op%d" % i, ports=_attn_ports(),
                                              used=[f"{n}_{s}" for n in ("input_dram", "Q_weight_dram", "K_weight_dram", "V_weight_dram", "output_dram")],
                                              golden=(lambda t, s=s: _golden_attn(s, t)), accum=32, ops=5 + (s == "B"))


def variants():
    return [{"domain": k.split("/")[0], "operator": "manual_design", "variant": k.split("/")[1], "stem": k.split("/")[1], "id": f"designs/{k}",
             "path": os.path.join(REPO, v["dir"])} for k, v in sorted(SPECS.items())]


# ----------------------------------------------------------------------------- testbench / build
def _typedef_re(name):
    return re.compile(r"typedef\s+ap_fixed\s*<[^;]*>\s*" + name + r"\s*;")


def _cdims(dims):
    return "".join(f"[{d}]" for d in dims)


def testbench(spec):
    ports = spec["ports"]
    L = ["#include <cstdio>", "#include <fstream>", "#include <ap_fixed.h>", "#include <hls_math.h>", "", "@TYPEDEF@", ""]
    L.append("void top(" + ", ".join(f"data_t {p['name']}{_cdims(p['decl'])}" for p in ports) + ");")
    L += [f"static data_t {p['name']}{_cdims(p['decl'])};" for p in ports]
    L += ["", "static void rd(const char *f, data_t *a, long n) {", "    std::ifstream in(f); if (!in) return; double v;",
          "    for (long i = 0; i < n && (in >> v); i++) a[i] = (data_t)v;", "}",
          "static void wr(const char *f, data_t *a, long n) {", "    FILE *o = fopen(f, \"w\");",
          "    for (long i = 0; i < n; i++) fprintf(o, \"%.17g\\n\", (double)a[i]);", "    fclose(o);", "}", "", "int main() {"]
    for p in ports:
        if not p["out"] and p["name"] in spec["used"]:
            L.append(f"    rd(\"{p['name']}.txt\", (data_t *){p['name']}, {int(np.prod(p['decl']))}L);")
    L.append("    top(" + ", ".join(p["name"] for p in ports) + ");")
    for p in ports:
        if p["out"] and p["name"] in spec["used"]:
            L.append(f"    wr(\"{p['name']}_output.txt\", (data_t *){p['name']}, {int(np.prod(p['decl']))}L);")
    L += ["    return 0;", "}"]
    return "\n".join(L) + "\n"


def types(dtype_text, op_params=None):
    """-> {typedef name: C type}. data_t is the format under test; acc_t / math_t (accumulators; operand of hls::sqrt/exp) follow
    op_params acc_type / acc_rounding / acc_overflow exactly as a generated operator's ACC / ACCM do, else they are data_t."""
    from backends import current
    if dtype_text == "float":
        return {"data_t": "float", "acc_t": "float", "math_t": "float"}
    acc, accm = current().acc_decls(dtype_text, op_params or {})
    data = _ctype(dtype_text)
    return {"data_t": data, "acc_t": data if acc == "data_t" else acc, "math_t": data if accm == "data_t" else accm}


def prepare(spec, dtype_text, run_dir, tasks=("csim",), trials_tcl="", op_params=None):
    """Copy the design into run_dir with data_t / acc_t / math_t rebound and a testbench + run_hls.tcl added."""
    src = os.path.join(REPO, spec["dir"])
    os.makedirs(run_dir, exist_ok=True)
    tys = types(dtype_text, op_params)
    ctype = tys["data_t"]
    for fn in ("top.cpp", "top.h"):
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            text = open(p).read()
            for name, c in tys.items():
                text = _typedef_re(name).sub(f"typedef {c} {name};", text)
            open(os.path.join(run_dir, fn), "w").write(text)
    open(os.path.join(run_dir, "tb_top.cpp"), "w").write(testbench(spec).replace("@TYPEDEF@", f"typedef {ctype} data_t;"))
    steps = {"csim": "csim_design", "csynth": "csynth_design", "cosim": "cosim_design"}
    tcl = ["open_project -reset project_1", "set_top top", "add_files top.cpp", "add_files -tb tb_top.cpp", trials_tcl,
           "open_solution -reset solution1", "set_part xczu9eg-ffvb1156-2-e", "create_clock -period 10 -name default"]
    tcl += [steps[t] for t in tasks] + ["exit"]
    open(os.path.join(run_dir, "run_hls.tcl"), "w").write("\n".join(tcl) + "\n")


def _ctype(dtype_text):
    from backends import current
    return current().type_decl(dtype_text)


# ----------------------------------------------------------------------------- harness (functional_verification.Harness interface)
class ManualHarness:
    def __init__(self, key, dt, work, op_params=None):
        self.key, self.spec, self.dt = key, SPECS[key], dt
        self.cfg_path = None
        self.run_dir = os.path.join(work, key.replace("/", "__"))
        self.types = types(dt.text, op_params)
        shutil.rmtree(self.run_dir, ignore_errors=True)
        prepare(self.spec, dt.text, self.run_dir, op_params=op_params)
        used = set(self.spec["used"])
        ins = [p for p in self.spec["ports"] if not p["out"] and p["name"] in used]
        self.outs = [p for p in self.spec["ports"] if p["out"] and p["name"] in used]
        # a design-config look-alike: ranged_config()/explicit_config() rescale drams[*].input_range unless fixed_range
        self.config = {"input_range": [-1.0, 1.0], "output_dram_names": [p["name"] for p in self.outs],
                       "drams": [{"name": p["name"], "dims": p["active"], "input_range": list(p["input_range"]), **({"fixed_range": True} if p["fixed_range"] else {})}
                                 for p in ins]}
        self.ins = {p["name"]: p for p in ins}
        self._last = None
        self._write_inputs(self.config, 0)
        rc, _ = run_vitis(self.run_dir, log_name="vitis_trial0.log")
        self.build = csim_build_dir(self.run_dir)
        if self.build is None:
            raise RuntimeError(f"CSIM build failed (rc={rc}); see {self.run_dir}/vitis_trial0.log")

    def _write_inputs(self, cfg, seed):
        rng = np.random.default_rng(seed)
        glob_lo, glob_hi = cfg.get("input_range", [-1.0, 1.0])
        vals = {}
        for d in cfg["drams"]:
            p = self.ins[d["name"]]
            lo, hi = d.get("input_range", [glob_lo, glob_hi])
            a = rng.uniform(lo, hi, size=p["active"])
            full = np.zeros(p["decl"])
            full[tuple(slice(0, n) for n in p["active"])] = a
            np.savetxt(os.path.join(self.run_dir, f"{p['name']}.txt"), full.reshape(-1), fmt="%.17g")
            vals[p["name"]] = a
        self._last = vals

    def _golden(self, quantized):
        t = {k: (quantize(v, self.dt) if quantized else v) for k, v in self._last.items()}
        g = self.spec["golden"](t)
        return np.concatenate([np.asarray(g[p["name"]], dtype=np.float64).reshape(-1) for p in self.outs])

    def run(self, cfg_ranged, seed, static_dir=None):
        from verification.functional_verification import CrashError
        self._write_inputs(cfg_ranged, seed)
        for p in self.outs:
            f = os.path.join(self.build, f"{p['name']}_output.txt")
            if os.path.isfile(f):
                os.remove(f)
        rc, _ = rerun_csim_exe(self.run_dir, self.build)
        if rc != 0:
            raise CrashError(rc)
        parts = []
        for p in self.outs:
            full = np.loadtxt(os.path.join(self.build, f"{p['name']}_output.txt"), dtype=np.float64).reshape(p["decl"])
            parts.append(full[tuple(slice(0, n) for n in p["active"])].reshape(-1))
        return np.concatenate(parts), self._golden(True)

    def raw_golden(self):
        return self._golden(False)

    def close(self, keep=False):
        if not keep:
            shutil.rmtree(self.run_dir, ignore_errors=True)
