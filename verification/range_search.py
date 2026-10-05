"""Search for an operator's operating window: the contiguous range of input magnitudes in which it is faithful.

`faithful(r)` is a callable supplied by the engine: True iff every output element of every sampled input set at range r is within the
error bound (and nothing crashed). This module only decides *which r to try*; it knows nothing about Vitis.

Algorithm (see docs/revision_r2 verification notes):
  1. scan a geometric grid upward and record pass/fail at each point;
  2. the window is the largest contiguous run of passing grid points -- scanning stops once a run has ended (an operator with a floor,
     such as a norm dividing by a vanishing rms, fails at the smallest r; one without simply starts passing at r_lo);
  3. bisect (in log space) between the window's last passing point and the first failing point to tighten r_hi;
  4. the engine then confirms r_hi with the full N trials; `confirm_and_step_down` lowers r_hi one grid step at a time until N trials pass.
"""
import math


def geometric_grid(start, stop, factor):
    if not (0 < start < stop and factor > 1):
        raise ValueError("need 0 < start < stop and factor > 1")
    n = int(math.floor(math.log(stop / start) / math.log(factor) + 1e-9))
    return [start * factor ** i for i in range(n + 1)]


def find_window(faithful, grid, bisect_steps=4, stop_after_window=True):
    """-> dict(r_lo, r_hi, window (list of passing grid points), first_fail_above, curve [(r, bool)], status).
    status: 'window' | 'none' (no grid point passed) | 'unbounded' (still passing at the last grid point: r_hi is the grid's end, not a limit)."""
    curve, window, in_window, first_fail = [], [], False, None
    for r in grid:
        ok = bool(faithful(r))
        curve.append((r, ok))
        if ok:
            window.append(r)
            in_window = True
        elif in_window:
            first_fail = r
            if stop_after_window:
                break
    if not window:
        return {"r_lo": None, "r_hi": None, "first_fail_above": None, "curve": curve, "status": "none"}
    r_lo, r_hi = window[0], window[-1]
    status = "window"
    if first_fail is None:
        status = "unbounded" if curve[-1][1] else "window"
    else:
        lo, hi = r_hi, first_fail
        for _ in range(bisect_steps):
            mid = math.sqrt(lo * hi)
            ok = bool(faithful(mid))
            curve.append((mid, ok))
            if ok:
                lo = mid
            else:
                hi = mid
        r_hi, first_fail = lo, hi
    return {"r_lo": r_lo, "r_hi": r_hi, "first_fail_above": first_fail, "curve": sorted(curve), "status": status}


def confirm_and_step_down(confirm, r_hi, step=1.19, max_steps=8):
    """`confirm(r)` runs the full N trials. Returns (validated r, steps taken) or (None, steps) if none passes."""
    r = r_hi
    for k in range(max_steps + 1):
        if confirm(r):
            return r, k
        r /= step
    return None, max_steps + 1
