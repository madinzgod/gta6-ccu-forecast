#!/usr/bin/env python3
"""Forecast a game launch day's Twitch viewers, hours 2..24, as quantiles.

Model: a launch's release-day shape, measured as the log ratio of each
hour's viewers to its own hour-1 viewers, is borrowed from past comparable
launches. Every target this contest poses is a console game that unlocks
at midnight local time with no PC version at release, so comparables are
weighted by how much they resemble that: a comparable on PC at launch, one
whose unlock was a single global moment rather than a rolling midnight,
one whose release day was contaminated by an earlier early-access peak,
or a free-to-play / live-service launch each count for less, and a
comparable whose hour-1 audience is far from the target's counts for less
too (a Gaussian kernel on the log scale gap). The forecast for an hour
is the weighted empirical quantiles of the comparables' log ratios at
that hour, mapped back through the case's own hour-1 print, with the
spread around the median scaled by a single global factor, and the
log-quantile curve lightly smoothed across neighbouring hours (launch
days evolve smoothly; the per-hour empirical quantiles, each drawn from
a slightly different subset of comparables, do not).

The weights, the kernel width, the spread factor and the smoothing
were chosen by leave-one-out cross-validation over the comparable
launches themselves: each comparable in turn is forecast from the
others and scored with the contest's exact pinball loss on the log
scale. On the console single-player launches (the target's own kind)
that scores about 0.185 against the persistence baseline's about 0.32
on the same launches, and about 0.39 against about 0.93 over all
comparable launches.
"""

import json
import math
import os
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Chosen on a cross-validation plateau, not at its sharpest point.
SPREAD_MULT = 0.78
SCALE_SIGMA = 1.0
SMOOTH = 0.25
PC_MISMATCH = 0.5
NOT_MIDNIGHT = 0.7
GLOBAL_UNLOCK = 0.6
CONTAMINATED = 0.6
FREE_OR_LIVE = 0.5


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()


def day_curve(comp):
    """Hour (1..24) -> viewers for the comparable's release day."""
    start = parse_ts(comp["day_start"])
    curve = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        off = int(round((parse_ts(row["from"]) - start) / 3600.0))
        if 0 <= off <= 23:
            curve[off + 1] = float(viewers)
    return curve


def has_pc(platforms):
    tokens = [t.strip() for t in (platforms or "").replace("/", ",")
              .split(",")]
    return any(t == "PC" or t.startswith("PC ") for t in tokens)


def features(comp):
    unlock = (comp.get("unlock") or "").lower()
    return {
        "pc": has_pc(comp.get("platforms")),
        "midnight": "midnight local" in unlock,
        "global": "global unlock" in unlock or "simultaneous" in unlock,
        "contaminated": ("contaminat" in unlock or "data peak" in unlock
                         or "data shows" in unlock),
        "free_or_live": ("free-to-play" in unlock or "f2p" in unlock
                         or "live-service" in unlock),
        "not_launch": "not a launch" in unlock,
    }


def ref_weight(feat, log_p, h1):
    """Weight of one comparable for a console midnight-unlock target."""
    w = 1.0
    if feat["pc"]:
        w *= PC_MISMATCH
    if not feat["midnight"]:
        w *= NOT_MIDNIGHT
    if feat["global"]:
        w *= GLOBAL_UNLOCK
    if feat["contaminated"]:
        w *= CONTAMINATED
    if feat["free_or_live"]:
        w *= FREE_OR_LIVE
    gap = math.log(h1) - log_p
    w *= math.exp(-(gap * gap) / (2.0 * SCALE_SIGMA * SCALE_SIGMA))
    return w


def weighted_quantile(pairs, tau):
    """pairs: list of (value, weight). Linear-interpolated quantile."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * total
    acc = 0.0
    for i, (value, w) in enumerate(pairs):
        acc += w
        if acc >= target:
            if i == 0:
                return value
            prev_value = pairs[i - 1][0]
            frac = (target - (acc - w)) / w if w > 0 else 0.0
            return prev_value + frac * (value - prev_value)
    return pairs[-1][0]


def forecast_case(case):
    p = float(case["known"][0]["viewers"])
    if p < 1.0:
        p = 1.0
    log_p = math.log(p)

    refs = []
    for comp in case.get("comparables") or []:
        feat = features(comp)
        if feat["not_launch"]:
            continue
        curve = day_curve(comp)
        if 1 not in curve:
            continue
        base = math.log(curve[1])
        ratios = {h: math.log(v) - base for h, v in curve.items() if h >= 2}
        if ratios:
            refs.append((ratios, ref_weight(feat, log_p, curve[1])))

    # Log-scale offsets from the hour-1 print, one row of five per hour.
    offsets = {}
    for hour in range(2, 25):
        pairs = [(r[hour], w) for r, w in refs if hour in r]
        if pairs:
            med = weighted_quantile(pairs, 0.5)
            row = []
            for tau in LEVELS:
                q = weighted_quantile(pairs, tau)
                row.append(med + SPREAD_MULT * (q - med))
            offsets[hour] = row
        else:
            offsets[hour] = [z * 0.15 * math.sqrt(hour - 1) for z in Z]

    # Light smoothing across neighbouring hours, per quantile level,
    # with the kernel renormalised at the day's edges.
    smoothed = {}
    for hour in range(2, 25):
        acc = [0.0] * len(LEVELS)
        wsum = 0.0
        for dh, kw in ((-1, SMOOTH), (0, 1.0 - 2.0 * SMOOTH), (1, SMOOTH)):
            if hour + dh in offsets:
                wsum += kw
                for k in range(len(LEVELS)):
                    acc[k] += kw * offsets[hour + dh][k]
        smoothed[hour] = [v / wsum for v in acc]

    rows = []
    for hour in range(2, 25):
        qs = [math.exp(log_p + v) for v in smoothed[hour]]
        qs = [min(max(q, 1e-6), 1e9) for q in qs]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"hour": hour, "q": [round(q, 2) for q in qs]})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
