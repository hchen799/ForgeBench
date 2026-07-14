"""Extractor interface shared by all HLS-tool backends.

A tool backend implements `discover` (find design result dirs under a results
tree) and `parse` (turn one design's reports into a PPAMetrics). Plots and tables
consume PPAMetrics only, so a new tool = one new Extractor subclass registered in
`_REGISTRY`.
"""
from abc import ABC, abstractmethod

from analysis.schema import PPAMetrics


class Extractor(ABC):
    tool = "base"

    @abstractmethod
    def discover(self, results_dir, flow="csynth"):
        """Yield (design_name, design_dir) for each design under results_dir."""

    @abstractmethod
    def parse(self, design_name, design_dir, flow="csynth", domain="", datatype=""):
        """Parse one design's reports into a PPAMetrics (or None if unavailable)."""


_REGISTRY = {}


def register(cls):
    _REGISTRY[cls.tool] = cls
    return cls


def get_extractor(tool):
    if tool not in _REGISTRY:
        raise ValueError(f"unknown tool {tool!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[tool]()
