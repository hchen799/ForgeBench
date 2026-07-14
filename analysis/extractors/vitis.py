"""Vitis HLS report extractor.

Consolidates the five duplicated per-domain parsers
(gemm/conv/llm/scale_models/modular_data `parse_*_resource_utlization.py`).

Report locations under each design dir:
  csynth: project_1/solution1/syn/report/csynth.xml          (AreaEstimates)
  impl:   project_1/solution1/impl/report/verilog/export_impl.xml (AreaReport)

csynth gives area + latency. Power, achieved clock / Fmax, timing closure and
throughput are NOT in csynth.xml; they come from implementation/Vivado reports
whose exact structure we cannot confirm without a sample. Those are extracted by
`_parse_impl_ppa`, currently a STUB flagged with TODO — fill in the real tag
paths once a sample impl report bundle is available.
"""
import os
import xml.etree.ElementTree as ET

from analysis.extractors.base import Extractor, register
from analysis.schema import PPAMetrics

CSYNTH_REL = "project_1/solution1/syn/report/csynth.xml"
IMPL_REL = "project_1/solution1/impl/report/verilog/export_impl.xml"


def _int(node, tag):
    el = node.find(tag)
    return int(el.text) if el is not None and el.text is not None else None


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
        perf = root.find("PerformanceEstimates")

        m = PPAMetrics(
            design=design_name, domain=domain, tool="vitis", flow="impl",
            datatype=datatype,
            bram=_int(res, "BRAM_18K") if res is not None else None,
            dsp=_int(res, "DSP") if res is not None else None,
            ff=_int(res, "FF") if res is not None else None,
            lut=_int(res, "LUT") if res is not None else None,
            bram_avail=_int(avail, "BRAM_18K") if avail is not None else None,
            dsp_avail=_int(avail, "DSP") if avail is not None else None,
            ff_avail=_int(avail, "FF") if avail is not None else None,
            lut_avail=_int(avail, "LUT") if avail is not None else None,
        )
        m.cycles, _ = _latency_ns(perf)
        self._parse_impl_ppa(m, design_dir, root)
        return m

    def _parse_impl_ppa(self, m, design_dir, impl_root):
        """STUB (FLAG): fill in power / achieved-clock / Fmax / timing-closure /
        throughput once a real Vitis impl report bundle is available.

        Expected sources (to confirm against a sample):
          - Achieved clock & timing: Vivado post-route timing summary
            (e.g. project_1/solution1/impl/report/verilog/*.rpt or a routed
            timing .xml); fmax_mhz = 1000/achieved_clock_ns; timing_met = WNS>=0.
          - Power: Vivado power report (e.g. *_power_routed.rpt) -> total/dyn/static W.
          - Throughput: derived (designs/sec = 1e9/latency_ns) or II-based; DEFINITION
            to confirm with the user.
        Leaving these None and recording a note so downstream clearly shows the gap.
        """
        m.notes = ("impl PPA (power/Fmax/timing/throughput) not parsed: "
                   "awaiting sample impl report to lock tag paths [TODO]")
        # TODO(sample): set m.achieved_clock_ns, m.fmax_mhz, m.timing_met,
        #               m.power_w, m.power_dyn_w, m.power_static_w, m.throughput
        return m
