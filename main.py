# gta6-twitch-ccu-forecast-01 — launch-day Twitch viewer forecaster
# Built for the make.fun competition gta6-twitch-ccu-forecast-01 (Nov 2026).
# Reads one case (/input/<id>.in) and writes /output/<id>.out: for hours 2..24 of
# the release day (UTC), five quantiles (5/25/50/75/95%) of that hour's Twitch
# viewer count. Python 3.13, standard library only, no network, deterministic.
#
# Model (fit at runtime on the case's own comparables — past big launches' hourly
# curves shipped in the case file; nothing is hard-coded from any launch):
#   * Anchor on the given first-hour print p1; work in log-ratios ln(v_h / p1).
#   * Two features describe how launched hour 1 already is:
#       f1   = ln(p1 / pre48) — hour 1 vs the category's median viewers over the
#              48 hours before the release day (a small hour-1 print means the
#              unlock has barely started anywhere, so later hours explode);
#       lnp1 = ln(p1) — absolute launch size (bigger launches run flatter).
#   * Three sub-models per hour, blended in log space (weights 0.34 / 0.33 / 0.33):
#       E-all: ridge regression of log-ratio on [1, f1, lnp1], fit on ALL comparables.
#       E-con: same regression, fit on console-only comparables (no "PC" in platforms)
#              — the launches shaped like GTA VI (console-only, rolling midnight).
#       B-con: median log-ratio curve of the console-only comparables, with spread
#              from its residuals.
#   * Spread: pooled residual variance of the three sub-models plus their
#     disagreement, times 0.9; quantiles are log-normal around the blended median.
#   * Prediction features are clamped to the training range so an unprecedented
#     GTA VI hour-1 print cannot extrapolate the regression wildly.
#   * Fallbacks: if there are too few console comparables, the console sub-models
#     use all comparables; if there is no usable training data at all, the entry
#     falls back to the house persistence baseline (flat at p1, 0.15*sqrt(h-1)).
#
# Validation (dev/analyze.py, dev/practice_eval.py, dev/blend_select.py):
#   * Practice cases (4 past console launches; truth = SullyGnome hourly averages,
#     the same source the kit cites — my pipeline reproduces the kit's published
#     house-baseline practice loss exactly: 0.387, miss 82.8%, coverage 0.783):
#     this model 0.165 pinball, 25.4% typical miss, 100% 90%-coverage.
#     House baseline on the same cases: 0.387. Improvement ~57%.
#   * Leave-one-out over the 8 console-only comparables (train on the rest, predict
#     the held-out launch): this model 0.177 vs baseline 0.323. Improvement ~45%.
#   * The house AI model is not public, so it cannot be compared directly; the
#     margin over the published house baseline is the evidence available.

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Blend weights (E-all, E-console, B-console) and the spread multiplier, chosen on
# the practice cases and on leave-one-out over console-only comparables.
W_ALL, W_CON_E, W_CON_B = 0.34, 0.33, 0.33
# Spread multiplier, tightened 0.9 -> 0.85 after both validations agreed:
# practice pinball 0.1611 at 0.85 vs 0.1653 at 0.9 and 0.1745 at 1.0, and
# leave-one-out over console-only comparables 0.1725 vs 0.1770 vs 0.1867.
# The 90% coverage stays 1.000 on practice, so the tighter spread is honest.
SIGMA_SCALE = 0.85
RIDGE = 1e-6
MIN_CONSOLE = 4  # fewer console comparables than this: console sub-models use all


def round2(x):
    """x rounded to 2 decimals, half up (as JavaScript Math.round(x*100)/100)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def pstdev(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return math.sqrt(sum((v - m) ** 2 for v in vals) / n)


def solve_linear(a, b):
    """Solve a small dense linear system by Gauss-Jordan with partial pivoting."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / m[col][col]
                for cc in range(col, n + 1):
                    m[r][cc] -= f * m[col][cc]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def ridge_fit(rows, ys):
    """Least squares with a tiny ridge on the non-intercept terms. rows: list of [1, f1, lnp1]."""
    n = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) + (RIDGE if i == j and i > 0 else 0.0)
          for j in range(n)] for i in range(n)]
    b = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(n)]
    return solve_linear(a, b)


def pre48_median(hourly, day_start):
    """Median viewers over the 48 hours before the release day (whole-hour rows)."""
    pts = [v for t, v in hourly if day_start - timedelta(hours=48) <= t < day_start]
    if len(pts) < 5:  # sparse sampling: fall back to whatever precedes the release day
        pts = [v for t, v in hourly if t < day_start][-48:]
    if not pts:
        return None
    return median(pts)


def comparable_training(case):
    """Build the training rows from the case's comparables.

    Returns (all_rows, console_rows) where each row is
    {'ratios': [ln(v_h/v1) for h=2..24], 'f1': ..., 'lnp1': ...}.
    A comparable is usable only if every release-day hour 1..24 has a point.
    """
    all_rows, console_rows = [], []
    for comp in case.get("comparables") or []:
        hourly = []
        for row in comp.get("hourly") or []:
            try:
                t = parse_ts(row["from"])
                v = row["viewers"]
            except (KeyError, ValueError, TypeError):
                continue
            if v is None:
                continue
            hourly.append((t, max(1, v)))
        if not hourly:
            continue
        try:
            day_start = parse_ts(comp["day_start"])
        except (KeyError, ValueError, TypeError):
            continue
        by_hour = dict(hourly)
        vals = []
        ok = True
        for h in range(1, 25):
            v = by_hour.get(day_start + timedelta(hours=h - 1))
            if v is None:
                ok = False
                break
            vals.append(v)
        if not ok:
            continue
        base = pre48_median(hourly, day_start)
        if base is None:
            continue
        p1 = vals[0]
        row = {
            "ratios": [math.log(v) - math.log(p1) for v in vals[1:]],
            "f1": math.log(p1) - math.log(base),
            "lnp1": math.log(p1),
        }
        all_rows.append(row)
        platforms = comp.get("platforms") or ""
        if "PC" not in platforms:
            console_rows.append(row)
    return all_rows, console_rows


def fit_median_curve(rows):
    """Per hour: (median log-ratio, residual std) over the training rows."""
    out = []
    for i in range(23):
        rs = [r["ratios"][i] for r in rows]
        med = median(rs)
        resid = [r - med for r in rs]
        out.append((med, max(0.15, pstdev(resid))))
    return out


def fit_regression(rows):
    """Per hour: (predicted log-ratio function, residual std) from ridge on [1, f1, lnp1]."""
    xs = [[1.0, r["f1"], r["lnp1"]] for r in rows]
    lo_f1 = min(r["f1"] for r in rows)
    hi_f1 = max(r["f1"] for r in rows)
    lo_ln = min(r["lnp1"] for r in rows)
    hi_ln = max(r["lnp1"] for r in rows)
    out = []
    for i in range(23):
        ys = [r["ratios"][i] for r in rows]
        beta = ridge_fit(xs, ys)
        resid = [y - sum(b * x for b, x in zip(beta, row)) for row, y in zip(xs, ys)]
        out.append((beta, max(0.15, pstdev(resid))))
    return out, (lo_f1, hi_f1, lo_ln, hi_ln)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    # The case's own pre-release baseline, from whole-hour history rows.
    hourly_hist = []
    for row in case.get("history_hourly") or []:
        try:
            t = parse_ts(row["from"])
            v = row["viewers"]
        except (KeyError, ValueError, TypeError):
            continue
        if v is not None:
            hourly_hist.append((t, max(1, v)))
    day_start = parse_ts(case["day_start"])
    base = pre48_median(hourly_hist, day_start)
    f1 = (math.log(p1) - math.log(base)) if base else 0.0
    lnp1 = math.log(p1)

    all_rows, console_rows = comparable_training(case)
    if len(all_rows) < 4:
        # No usable training data: house persistence baseline.
        return {"hours": [{"hour": row["hour"],
                           "q": [round2(p1 * math.exp(z * 0.15 * math.sqrt(row["hour"] - 1))) for z in Z]}
                          for row in case["forecast"]]}
    if len(console_rows) < MIN_CONSOLE:
        console_rows = all_rows

    reg_all, bounds_all = fit_regression(all_rows)
    reg_con, bounds_con = fit_regression(console_rows)
    med_con = fit_median_curve(console_rows)

    # Clamp the prediction features to each fit's training range.
    f1_a = clamp(f1, bounds_all[0], bounds_all[1])
    ln_a = clamp(lnp1, bounds_all[2], bounds_all[3])
    f1_c = clamp(f1, bounds_con[0], bounds_con[1])
    ln_c = clamp(lnp1, bounds_con[2], bounds_con[3])

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        i = h - 2
        beta_a, sd_a = reg_all[i]
        beta_c, sd_c = reg_con[i]
        med_b, sd_b = med_con[i]
        pred_a = sum(b * x for b, x in zip(beta_a, [1.0, f1_a, ln_a]))
        pred_c = sum(b * x for b, x in zip(beta_c, [1.0, f1_c, ln_c]))
        med = W_ALL * pred_a + W_CON_E * pred_c + W_CON_B * med_b
        var = W_ALL * sd_a ** 2 + W_CON_E * sd_c ** 2 + W_CON_B * sd_b ** 2
        var += (W_ALL * (pred_a - med) ** 2 + W_CON_E * (pred_c - med) ** 2
                + W_CON_B * (med_b - med) ** 2)
        sigma = max(0.12, math.sqrt(var)) * SIGMA_SCALE
        qs = [p1 * math.exp(med + z * sigma) for z in Z]
        qs = [min(1e9, max(1.0, q)) for q in qs]
        # Keep the row non-decreasing after rounding.
        qs = [round2(q) for q in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
