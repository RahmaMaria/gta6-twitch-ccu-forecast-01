# judaboyk gta6-twitch-ccu-forecast-01

A launch-day Twitch viewer forecaster for the make.fun
gta6-twitch-ccu-forecast-01 competition. One file, `main.py`, Python 3.13,
standard library only, deterministic, no network — it reads one case from
`/input/<id>.in` and writes `/output/<id>.out`: for each of hours 2–24 of
GTA VI's release day (UTC, 2026-11-19), five quantiles (5%, 25%, 50%, 75%,
95%) of that hour's Twitch viewer count in the Grand Theft Auto VI category.

## The model

The forecaster is fit at runtime on the case's own comparables — the past big
launches' hourly curves shipped in the case file — so nothing about any
particular launch is hard-coded. It anchors on the given first-hour print
`p1` and works in log-ratios `ln(v_h / p1)`:

- Two features say how launched hour 1 already is. `f1 = ln(p1 / pre48)`,
  hour 1 against the category's median viewers over the 48 hours before the
  release day: a small hour-1 print means the rolling-midnight unlock has
  barely started anywhere, so later hours explode; a large one means the
  launch is already under way. `lnp1 = ln(p1)`, absolute launch size:
  bigger launches run flatter.
- Three sub-models per hour, blended in log space (weights 0.34 / 0.33 / 0.33):
  a ridge regression of the log-ratio on `[1, f1, lnp1]` fit on **all**
  comparables; the same regression fit on **console-only** comparables (no
  "PC" in platforms) — the launches shaped like GTA VI, console-only with a
  rolling midnight unlock; and the console-only median log-ratio curve.
- Spread: pooled residual variance of the three sub-models plus their
  disagreement, times 0.9; quantiles are log-normal around the blended
  median. Prediction features are clamped to the training range, so an
  unprecedented GTA VI hour-1 print cannot extrapolate the regression
  wildly. If no usable training data exists at all, it falls back to the
  house persistence baseline (flat at `p1`, spread `0.15*sqrt(h-1)`).

## Validation (practice, before submit)

Practice cases are the four past console launches in the kit; truth is
SullyGnome's hourly averages, the same source the kit cites. My pipeline
reproduces the kit's published house-baseline practice loss exactly
(0.387, typical miss 82.8%, 90% coverage 0.783), so the comparison is
like-for-like:

| entry | pinball loss | typical miss | 90% coverage |
| --- | --- | --- | --- |
| house baseline (starter) | 0.387 | 82.8% | 0.783 |
| this entry | **0.165** | 25.4% | 1.000 |

Per practice case: Wolverine 0.195, Red Dead Redemption 2 0.155,
God of War Ragnarök 0.134, Spider-Man 2 0.178 — every case beats the
baseline. Leave-one-out over the 8 console-only comparables (train on the
rest, predict the held-out launch): this entry 0.177 vs baseline 0.323.
The house AI model is not public, so it cannot be compared directly; the
margin over the published house baseline is the evidence available.

## Running it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

About 0.2 s per case, far inside the 8 s cap. Output is exactly 23 rows,
hours 2–24, each five finite numbers above 0 and at most 1e9, non-decreasing.

## Files

- `main.py` — the whole forecaster.
- `LICENSE` — MIT.
- `.makefun/owner.json` — the payout wallet for this entry.
