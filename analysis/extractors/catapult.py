"""Catapult HLS report extractor — STUB (concern E1).

Same Extractor interface as the Vitis backend. Catapult emits its own report
format (typically per-solution `.rpt`/`.xml` under a `*_v/` or `Catapult/`
directory, with area, latency/throughput, and timing). Fill this in when
Catapult report samples are available; nothing else in the analysis package needs
to change because everything consumes `PPAMetrics`.

Run `python3 -m analysis.extractors.catapult` to confirm the slot is wired.
"""
from analysis.extractors.base import Extractor, register


@register
class CatapultExtractor(Extractor):
    tool = "catapult"

    def discover(self, results_dir, flow="csynth"):
        raise NotImplementedError(
            "Catapult extractor is a stub. TODO: walk the Catapult results tree "
            "and yield (design_name, design_dir). Provide a sample report to lock "
            "the directory layout and report filenames."
        )

    def parse(self, design_name, design_dir, flow="csynth", domain="", datatype=""):
        raise NotImplementedError(
            "Catapult extractor is a stub. TODO: parse Catapult area/latency/"
            "throughput/power/timing reports into a PPAMetrics. Map Catapult report "
            "fields to the schema in analysis/schema.py."
        )


if __name__ == "__main__":
    from analysis.extractors.base import get_extractor
    ex = get_extractor("catapult")
    print(f"catapult extractor registered: {type(ex).__name__}")
    try:
        list(ex.discover("."))
    except NotImplementedError as e:
        print("STUB OK ->", e)
