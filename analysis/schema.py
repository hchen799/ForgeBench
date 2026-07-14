"""Tool-agnostic PPA metrics record shared by extractors, plots, and tables.

Every extractor (Vitis now, Catapult later) returns `PPAMetrics`, and every
plot/table consumes it, so adding a tool means implementing one extractor with
no changes elsewhere.

Fields that Vitis csynth cannot provide (power, achieved clock / Fmax, timing
closure, throughput) are left as None until an implementation ("impl") report is
parsed; see extractors/vitis.py.
"""
from dataclasses import dataclass, asdict, field
from typing import Optional

# The unified metrics.csv column order (also the DataFrame schema plots/tables use).
CSV_COLUMNS = [
    "design", "domain", "tool", "flow", "datatype",
    "bram", "dsp", "ff", "lut",
    "bram_avail", "dsp_avail", "ff_avail", "lut_avail",
    "bram_util", "dsp_util", "ff_util", "lut_util",
    "cycles", "latency_ns",
    "ii", "target_clock_ns", "achieved_clock_ns", "fmax_mhz",
    "power_w", "power_dyn_w", "power_static_w",
    "timing_met", "throughput", "notes",
]


@dataclass
class PPAMetrics:
    design: str
    domain: str = ""                 # gemm | conv | llm | scale | modular
    tool: str = "vitis"             # vitis | catapult
    flow: str = "csynth"            # csynth | impl
    datatype: str = ""

    # Area
    bram: Optional[int] = None
    dsp: Optional[int] = None
    ff: Optional[int] = None
    lut: Optional[int] = None
    bram_avail: Optional[int] = None
    dsp_avail: Optional[int] = None
    ff_avail: Optional[int] = None
    lut_avail: Optional[int] = None

    # Performance
    cycles: Optional[int] = None
    latency_ns: Optional[float] = None
    ii: Optional[int] = None

    # Timing / power (impl-only; None from csynth)
    target_clock_ns: Optional[float] = None
    achieved_clock_ns: Optional[float] = None
    fmax_mhz: Optional[float] = None
    power_w: Optional[float] = None
    power_dyn_w: Optional[float] = None
    power_static_w: Optional[float] = None
    timing_met: Optional[bool] = None
    throughput: Optional[float] = None

    notes: str = ""

    def _util(self, used, avail):
        if used is None or not avail:
            return None
        return round(100.0 * used / avail, 4)

    @property
    def bram_util(self):
        return self._util(self.bram, self.bram_avail)

    @property
    def dsp_util(self):
        return self._util(self.dsp, self.dsp_avail)

    @property
    def ff_util(self):
        return self._util(self.ff, self.ff_avail)

    @property
    def lut_util(self):
        return self._util(self.lut, self.lut_avail)

    def to_row(self):
        d = asdict(self)
        d["bram_util"] = self.bram_util
        d["dsp_util"] = self.dsp_util
        d["ff_util"] = self.ff_util
        d["lut_util"] = self.lut_util
        return {k: d.get(k) for k in CSV_COLUMNS}
