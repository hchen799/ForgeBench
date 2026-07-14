"""PPA report extractors. One module per HLS tool; all return PPAMetrics."""
from analysis.extractors.base import Extractor, get_extractor

# Import backends so they self-register in the extractor registry.
from analysis.extractors import vitis as _vitis  # noqa: F401
from analysis.extractors import catapult as _catapult  # noqa: F401

__all__ = ["Extractor", "get_extractor"]
