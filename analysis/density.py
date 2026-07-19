"""Shared KDE + density-panel helpers for domain-grouped distribution plots.

Reproduces the overlapping filled-KDE style of the reference distribution
figures: one panel per metric, one filled curve per domain (GEMM/CONV/LLM),
colored cyan/yellow/red. No scipy dependency — Gaussian KDE is implemented in
numpy (Scott's-rule bandwidth), computed in log-space for log-scaled axes.
"""
import numpy as np

# Domain colors matched to the reference figure (cyan / yellow / red).
DOMAIN_STYLE = {
    "gemm": {"color": "#1CC5D6", "label": "GEMM"},   # cyan
    "conv": {"color": "#F2C14E", "label": "CONV"},   # yellow
    "llm":  {"color": "#E8635A", "label": "LLM"},    # red
}


def gaussian_kde_1d(samples, grid, bw=None):
    """1-D Gaussian KDE evaluated on `grid`. Scott's rule bandwidth if bw is None."""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if n == 0:
        return np.zeros_like(grid)
    std = samples.std(ddof=1) if n > 1 else 0.0
    if bw is None:
        bw = std * n ** (-1.0 / 5.0) if std > 0 else 1.0
    if not np.isfinite(bw) or bw <= 0:
        bw = 1.0
    u = (grid[:, None] - samples[None, :]) / bw
    k = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
    return k.sum(axis=1) / (n * bw)


def density_panel(ax, df, value_col, log=False, pad=0.3, n_grid=400,
                  domain_style=None, fill_alpha=0.35, line_width=1.6):
    """Draw one overlapping-KDE density panel (one filled curve per domain).

    Returns the number of domains actually drawn (>=2 samples each). For
    `log=True` axes the KDE is computed on log10(value) and the x-axis is set
    to log scale, matching the reference resource/latency distribution figures.
    """
    domain_style = domain_style or DOMAIN_STYLE
    drawn = 0
    for domain, style in domain_style.items():
        vals = df.loc[df["domain"] == domain, value_col]
        vals = vals.dropna()
        if log:
            vals = vals[vals > 0]
        vals = vals.to_numpy(dtype=float)
        if vals.size < 2:
            continue
        if log:
            lv = np.log10(vals)
            grid = np.linspace(lv.min() - pad, lv.max() + pad, n_grid)
            dens = gaussian_kde_1d(lv, grid)
            x = 10.0 ** grid
        else:
            span = max(vals.std(ddof=1), 1e-9)
            lo = max(0.0, vals.min() - pad * span)
            grid = np.linspace(lo, vals.max() + pad * span, n_grid)
            dens = gaussian_kde_1d(vals, grid)
            x = grid
        ax.fill_between(x, dens, color=style["color"], alpha=fill_alpha,
                        label=style["label"], linewidth=0)
        ax.plot(x, dens, color=style["color"], linewidth=line_width)
        drawn += 1
    if log:
        ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    return drawn
