"""quantize() must agree bit-for-bit with the real Vitis ap_fixed conversion (needs $XILINX_HLS or VITIS_HLS_INCLUDE)."""
import os
import subprocess
import tempfile

import numpy as np
import pytest

from verification.fixedpoint import DType, quantize

INC = os.environ.get("VITIS_HLS_INCLUDE") or "/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1/include"
TYPES = ["ap_fixed<16,5>", "ap_fixed<32,10>", "ap_fixed<16,5,AP_RND,AP_SAT>", "ap_fixed<16,5,AP_TRN,AP_SAT>",
         "ap_fixed<16,5,AP_RND,AP_WRAP>", "ap_fixed<24,8,AP_RND,AP_SAT>"]
GENERIC = {"fixed<16,5>": "ap_fixed<16,5,AP_RND,AP_SAT>", "fixed<32,10>": "ap_fixed<32,10,AP_RND,AP_SAT>",
           "fixed<16,5,trn,wrap>": "ap_fixed<16,5>"}
PROBE = r'''
#include <cstdio>
#include <ap_fixed.h>
typedef %(T)s data_t;
int main() { double x; while (scanf("%%lf", &x) == 1) { data_t v = x; printf("%%.17g\n", (double)v); } }
'''


def test_generic_spelling_means_the_lowered_type():
    for generic, raw in GENERIC.items():
        a, b = DType(generic), DType(raw)
        assert (a.W, a.I, a.q, a.o) == (b.W, b.I, b.q, b.o), (generic, raw)


@pytest.mark.skipif(not os.path.isdir(INC), reason="Vitis ap_fixed headers not found")
@pytest.mark.parametrize("t", TYPES)
def test_quantize_matches_vitis(t):
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.uniform(-8, 8, 4000), rng.uniform(-70, 70, 4000),          # in range and overflowing
                        np.arange(-40, 40) / 64.0 + 2.0 ** -13,                         # near rounding ties
                        np.arange(-40, 40) / 4096.0, [0.0, -0.0, 15.999, -16.0, 16.0, 1e3, -1e3]])
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "p.cpp"), os.path.join(d, "p")
        open(src, "w").write(PROBE % {"T": t})
        subprocess.run(["g++", "-std=c++14", f"-I{INC}", src, "-o", exe], check=True, capture_output=True)
        r = subprocess.run([exe], input="\n".join(repr(float(v)) for v in x), capture_output=True, text=True, check=True)
    got = np.array([float(v) for v in r.stdout.split()])
    want = quantize(x, DType(t))
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, f"{t}: {bad.size}/{x.size} mismatches, e.g. x={x[bad[0]]!r} vitis={got[bad[0]]!r} emulated={want[bad[0]]!r}"
