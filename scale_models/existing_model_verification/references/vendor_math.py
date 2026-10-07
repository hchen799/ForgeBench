"""Versioned RoPE constants for the implementation reference only.

Vitis float pow/sin/cos do not round identically to libm or torch. The fixed
reference shares these primitive constants, like a ROM table, with the DUT.
No accelerator outputs are used as golden values. FP64 never uses this module.
This limited shared dependency is explicitly recorded in the run manifest.
"""
import ctypes
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .arithmetic import DATA
from .project import sha

DEFAULT_ROOT = Path("/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1")
BRIDGE = r"""
#include <hls_math.h>
extern "C" {
float rope_pow(float x, float y) { return powf(x, y); }
float rope_cos(float x) { return hls::cos(x); }
float rope_sin(float x) { return hls::sin(x); }
}
"""


class VendorRope:
    def __init__(self, root=None):
        self.root = Path(
            root or os.environ.get("VITIS_HLS_ROOT", DEFAULT_ROOT)
        ).resolve()
        mathlib = self.root / "lnx64/lib/csim"
        fpolib = self.root / "lnx64/tools/fpo_v7_1"
        self.paths = [
            fpolib / "libgmp.so.11",
            fpolib / "libmpfr.so.4",
            fpolib / "libIp_floating_point_v7_1_bitacc_cmodel.so",
            mathlib / "libhlsmc++-GCC46.so",
            mathlib / "libhlsm-GCC46.so",
        ]
        for path in self.paths:
            if not path.is_file():
                raise ValueError("Vitis RoPE primitive library missing: " + str(path))
        # RPATH locates transitive Vitis libraries without relying on ambient
        # LD_LIBRARY_PATH. This bridge contains only primitive calls, no graph.
        self.folder = tempfile.TemporaryDirectory(prefix="forgebench_rope_")
        source = Path(self.folder.name) / "rope.cpp"
        self.library = source.with_suffix(".so")
        source.write_text(BRIDGE)
        self.command = [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++14",
            "-O2",
            "-I" + str(self.root / "include"),
            str(source),
            "-o",
            str(self.library),
            "-L" + str(mathlib),
            "-Wl,--disable-new-dtags",
            "-Wl,-rpath," + str(mathlib),
            "-Wl,-rpath," + str(fpolib),
            "-lhlsmc++-GCC46",
            "-lhlsm-GCC46",
        ]
        result = subprocess.run(
            self.command, capture_output=True, text=True, timeout=60
        )
        if result.returncode:
            raise RuntimeError(
                "Vitis primitive bridge compilation failed: " + result.stderr
            )
        # Match standalone C simulation's symbol resolution, including powf.
        mode = os.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0)
        self.handle = ctypes.CDLL(str(self.library), mode=mode)
        self.cos, self.sin, self.pow = (
            self.handle.rope_cos,
            self.handle.rope_sin,
            self.handle.rope_pow,
        )
        for func in (self.cos, self.sin):
            func.argtypes = [ctypes.c_float]
            func.restype = ctypes.c_float
        self.pow.argtypes = [ctypes.c_float, ctypes.c_float]
        self.pow.restype = ctypes.c_float

    def provenance(self):
        return dict(
            kind="shared_vendor_primitive_constants",
            scope="fixed-reference RoPE coefficients only; FP64 is independent",
            libraries={str(p): sha(p) for p in self.paths},
            bridge_source=BRIDGE,
            bridge_library_sha256=sha(self.library),
            bridge_command=self.command,
        )

    @lru_cache(maxsize=8)
    def coefficients(self, start, rows, head_dim=128):
        values = np.empty((rows, head_dim // 2, 2), dtype=np.float64)
        frequency = [
            np.float32(self.pow(500000.0, -d / head_dim)) for d in range(0, head_dim, 2)
        ]
        for r in range(rows):
            for d, freq in enumerate(frequency):
                angle = float(np.float32(start + r) * freq)
                values[r, d] = self.cos(angle), self.sin(angle)
        return DATA.quantize(values)
