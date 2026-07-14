"""ForgeBench PPA analysis: tool-agnostic extraction, aggregation, and plotting.

Reproducibility / artifact package (reviewer concerns F1-F3), structured so a
Catapult extractor drops in alongside the Vitis one (concern E1). Replaces the
per-domain `parse_*_resource_utlization.py` scripts and the duplicated
`analyze_report_data_new_plot_final.py` plotting copies.

See analysis/README.md for the figure/table -> command mapping.
"""
