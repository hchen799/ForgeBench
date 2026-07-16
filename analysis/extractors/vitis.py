"""Vitis HLS report extractor.

Consolidates the five duplicated per-domain parsers
(gemm/conv/llm/scale_models/modular_data `parse_*_resource_utlization.py`).

Report locations under each design dir:
  csynth: project_1/solution1/syn/report/csynth.xml          (AreaEstimates)
  impl:   project_1/solution1/impl/report/verilog/export_impl.xml (AreaReport + TimingReport)
  power:  project_1/solution1/impl/verilog/project.runs/impl_1/*_power_routed.rpt

csynth gives area + latency. The impl bundle gives post-P&R area, achieved
clock / Fmax and timing closure (all in export_impl.xml's <TimingReport>), plus
power in the Vivado *_power_routed.rpt. impl latency/throughput are derived from
the csynth cycle count (export_impl.xml has no PerformanceEstimates) times the
target clock the design is constrained to. Tag paths were locked against a real
Vitis 2024.1.2 / Vivado impl bundle (ZCU102, xczu9eg).

Unit note (reviewer B3): csynth reports BRAM as BRAM_18K (18-Kb blocks); the
impl AreaReport reports BRAM as RAMB36 (36-Kb) tile count — a different unit.
DSP/FF/LUT are directly comparable across flows.
"""
import glob
import os
import re
import xml.etree.ElementTree as ET

from analysis.extractors.base import Extractor, register
from analysis.schema import PPAMetrics

CSYNTH_REL = "project_1/solution1/syn/report/csynth.xml"
IMPL_REL = "project_1/solution1/impl/report/verilog/export_impl.xml"
# Vivado routed power report (name is <bd>_wrapper_power_routed.rpt); glob to be
# robust to the top/wrapper name.
POWER_GLOB = "project_1/solution1/impl/verilog/project.runs/impl_1/*_power_routed.rpt"


def _int(node, tag):
    if node is None:
        return None
    el = node.find(tag)
    return int(el.text) if el is not None and el.text is not None else None


def _ftext(node, tag):
    """Float value of a child tag, or None."""
    if node is None:
        return None
    el = node.find(tag)
    if el is None or not el.text:
        return None
    try:
        return float(el.text)
    except ValueError:
        return None


def _parse_power_rpt(path):
    """Return (total_w, dyn_w, static_w) from a Vivado routed power report.

    Rows look like:  | Total On-Chip Power (W)  | 0.763        |
    """
    total = dyn = static = None
    pat = {
        "power_w": re.compile(r"Total On-Chip Power \(W\)\s*\|\s*([\d.]+)"),
        "power_dyn_w": re.compile(r"^\|\s*Dynamic \(W\)\s*\|\s*([\d.]+)"),
        "power_static_w": re.compile(r"Device Static \(W\)\s*\|\s*([\d.]+)"),
    }
    with open(path, errors="ignore") as f:
        for line in f:
            for m in pat["power_w"].finditer(line):
                total = float(m.group(1))
            for m in pat["power_dyn_w"].finditer(line):
                dyn = float(m.group(1))
            for m in pat["power_static_w"].finditer(line):
                static = float(m.group(1))
    return total, dyn, static


def _latency_ns(perf):
    """Worst-case latency in ns from a PerformanceEstimates node (real time is a
    human string like '0.664 ms'; prefer cycles*clock if we can, else parse)."""
    if perf is None:
        return None, None
    summ = perf.find("SummaryOfOverallLatency")
    if summ is None:
        return None, None
    cyc_el = summ.find("Worst-caseLatency")
    cycles = None
    if cyc_el is not None and cyc_el.text and cyc_el.text.strip().isdigit():
        cycles = int(cyc_el.text)
    return cycles, None  # latency_ns derived from cycles*target_clock in caller


@register
class VitisExtractor(Extractor):
    tool = "vitis"

    def discover(self, results_dir, flow="csynth"):
        rel = CSYNTH_REL if flow == "csynth" else IMPL_REL
        for sub in sorted(os.listdir(results_dir)):
            design_dir = os.path.join(results_dir, sub)
            if os.path.exists(os.path.join(design_dir, rel)):
                yield sub, design_dir

    def parse(self, design_name, design_dir, flow="csynth", domain="", datatype=""):
        if flow == "csynth":
            return self._parse_csynth(design_name, design_dir, domain, datatype)
        return self._parse_impl(design_name, design_dir, domain, datatype)

    def _parse_csynth(self, design_name, design_dir, domain, datatype):
        xml_path = os.path.join(design_dir, CSYNTH_REL)
        if not os.path.exists(xml_path):
            return None
        root = ET.parse(xml_path).getroot()
        area = root.find("AreaEstimates")
        if area is None:
            return None
        res = area.find("Resources")
        avail = area.find("AvailableResources")
        perf = root.find("PerformanceEstimates")

        m = PPAMetrics(
            design=design_name, domain=domain, tool="vitis", flow="csynth",
            datatype=datatype,
            bram=_int(res, "BRAM_18K"), dsp=_int(res, "DSP"),
            ff=_int(res, "FF"), lut=_int(res, "LUT"),
            bram_avail=_int(avail, "BRAM_18K"), dsp_avail=_int(avail, "DSP"),
            ff_avail=_int(avail, "FF"), lut_avail=_int(avail, "LUT"),
        )
        m.cycles, _ = _latency_ns(perf)
        # Target clock (ns) if present, to derive latency_ns.
        if perf is not None:
            clk = perf.find("SummaryOfTimingAnalysis/EstimatedClockPeriod")
            if clk is not None and clk.text:
                try:
                    m.target_clock_ns = float(clk.text)
                except ValueError:
                    pass
        if m.cycles is not None and m.target_clock_ns:
            m.latency_ns = m.cycles * m.target_clock_ns
        return m

    def _parse_impl(self, design_name, design_dir, domain, datatype):
        xml_path = os.path.join(design_dir, IMPL_REL)
        if not os.path.exists(xml_path):
            return None
        root = ET.parse(xml_path).getroot()
        area = root.find("AreaReport")
        res = area.find("Resources") if area is not None else None
        avail = area.find("AvailableResources") if area is not None else None

        # impl AreaReport uses <BRAM> (RAMB36 tile count), not <BRAM_18K>.
        m = PPAMetrics(
            design=design_name, domain=domain, tool="vitis", flow="impl",
            datatype=datatype,
            bram=_int(res, "BRAM"), dsp=_int(res, "DSP"),
            ff=_int(res, "FF"), lut=_int(res, "LUT"),
            bram_avail=_int(avail, "BRAM"), dsp_avail=_int(avail, "DSP"),
            ff_avail=_int(avail, "FF"), lut_avail=_int(avail, "LUT"),
        )
        self._parse_impl_ppa(m, design_dir, root)
        return m

    def _parse_impl_ppa(self, m, design_dir, impl_root):
        """Fill post-P&R timing/power/latency/throughput for the impl flow.

        Sources (locked against a real Vitis 2024.1.2 impl bundle):
          - Timing/Fmax: export_impl.xml <TimingReport> — TargetClockPeriod,
            AchievedClockPeriod (== target - WNS), TIMING_MET, WNS_FINAL.
            fmax_mhz = 1000 / achieved_clock_ns.
          - Power: Vivado *_power_routed.rpt — total / dynamic / device-static W.
          - Latency/throughput: cycles come from the sibling csynth.xml
            (export_impl.xml has no PerformanceEstimates); latency_ns = cycles *
            target_clock_ns (the constraint the design is built to meet, and
            meets); throughput = 1e9 / latency_ns (end-to-end designs/sec).
        Any source that is missing is left as None and noted.
        """
        notes = []

        # --- Timing / Fmax (from export_impl.xml) ---
        tr = impl_root.find("TimingReport")
        if tr is not None:
            m.target_clock_ns = _ftext(tr, "TargetClockPeriod") or _ftext(tr, "CP_TARGET")
            m.achieved_clock_ns = _ftext(tr, "AchievedClockPeriod") or _ftext(tr, "CP_FINAL")
            met = tr.find("TIMING_MET")
            if met is not None and met.text:
                m.timing_met = met.text.strip().upper() == "TRUE"
            else:
                wns = _ftext(tr, "WNS_FINAL")
                if wns is not None:
                    m.timing_met = wns >= 0
            if m.achieved_clock_ns:
                m.fmax_mhz = round(1000.0 / m.achieved_clock_ns, 3)
        else:
            notes.append("no <TimingReport> in export_impl.xml")

        # --- Power (from Vivado routed power report) ---
        power_files = glob.glob(os.path.join(design_dir, POWER_GLOB))
        if power_files:
            m.power_w, m.power_dyn_w, m.power_static_w = _parse_power_rpt(power_files[0])
        else:
            notes.append("no *_power_routed.rpt found")

        # --- Latency / throughput (cycles from sibling csynth.xml) ---
        csynth_path = os.path.join(design_dir, CSYNTH_REL)
        if os.path.exists(csynth_path):
            perf = ET.parse(csynth_path).getroot().find("PerformanceEstimates")
            m.cycles, _ = _latency_ns(perf)
            if perf is not None:
                summ = perf.find("SummaryOfOverallLatency")
                m.ii = _int(summ, "Interval-max") if summ is not None else None
            clock_ns = m.target_clock_ns
            if m.cycles is not None and clock_ns:
                m.latency_ns = m.cycles * clock_ns
                m.throughput = 1e9 / m.latency_ns  # end-to-end designs/sec
        else:
            notes.append("no sibling csynth.xml for latency")

        m.notes = "; ".join(notes)
        return m
