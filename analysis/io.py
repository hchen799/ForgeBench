"""CSV IO + normalization for the analysis package.

Two CSV shapes are supported:
  1. The unified metrics.csv emitted by `analysis.collect` (columns from
     schema.CSV_COLUMNS).
  2. The legacy per-domain CSVs already committed as artifact fixtures
     (`plotting/{gemm,conv,llm}_data/*.csv`, `scale_models/…`) with columns
     Project, Function, CYCLE, LATENCY, BRAM, DSP, FF, LUT, *_Utilization.

Both are normalized to a common DataFrame (design, domain, cycles, *_util, …) so
plots/tables don't care which produced the data.
"""
import os

import pandas as pd

# legacy CSV column -> normalized column
_LEGACY_MAP = {
    "Project": "design",
    "CYCLE": "cycles",
    "LATENCY": "latency_str",
    "BRAM": "bram", "DSP": "dsp", "FF": "ff", "LUT": "lut",
    "BRAM_Utilization": "bram_util",
    "DSP_Utilization": "dsp_util",
    "FF_Utilization": "ff_util",
    "LUT_Utilization": "lut_util",
}

_NUMERIC = ["cycles", "bram", "dsp", "ff", "lut",
            "bram_util", "dsp_util", "ff_util", "lut_util"]


def _coerce_numeric(df):
    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # 'undef' -> NaN
    return df


def load_legacy_csv(path, domain=""):
    """Load one legacy per-domain CSV into the normalized schema."""
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in _LEGACY_MAP.items() if k in df.columns})
    if domain:
        df["domain"] = domain
    return _coerce_numeric(df)


def load_legacy_dir(dirpath, domain=""):
    """Concatenate every *.csv in a directory (legacy schema)."""
    frames = []
    for fn in sorted(os.listdir(dirpath)):
        if fn.endswith(".csv"):
            frames.append(load_legacy_csv(os.path.join(dirpath, fn), domain))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_coverage_dirs(dir_domain_map):
    """Build the Fig-4 coverage DataFrame from {dir: domain} of legacy CSV dirs."""
    frames = []
    for dirpath, domain in dir_domain_map.items():
        df = load_legacy_dir(dirpath, domain)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def read_metrics_csv(path):
    """Read a unified metrics.csv (already normalized column names)."""
    return _coerce_numeric(pd.read_csv(path))
