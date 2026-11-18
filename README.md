# GTA VI Twitch CCU Forecast

[![make.fun competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/madinzgod/gta6-ccu-forecast/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/madinzgod/gta6-ccu-forecast/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

An entry for the make.fun competition
[gta6-twitch-ccu-forecast-01](https://make.fun/c/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day, 2026-11-19 (UTC), after the first — five quantiles
(5/25/50/75/95%) for each of hours 2 to 24, scored by mean pinball loss on
the log scale.

## The model

`main.py` (Python 3.13, standard library only, no network) reads the case
file — everything recorded up to the end of hour 1, including hour 1's
viewer count and 42 past big launches ("comparables") with their hourly
viewer curves — and writes one forecast per case.

A launch's release-day *shape* — each hour's viewers as a log ratio to its
own hour-1 viewers — is borrowed from the comparables:

- Every target this contest poses is a console game that unlocks at
  midnight local time with no PC version at release, so comparables are
  weighted by resemblance to that: a comparable that launched on PC, whose
  unlock was one global moment instead of a rolling midnight, whose
  release day was contaminated by an earlier early-access peak, or that is
  free-to-play / live-service counts for less.
- A comparable whose hour-1 audience is far from the target's also counts
  for less, through an exponential kernel on the log scale gap.
- Each hour's forecast is the weighted empirical quantiles of the
  comparables' log ratios at that hour, mapped back through the case's
  hour-1 print, with one global spread factor around the median.

The weights and the spread factor were chosen by leave-one-out
cross-validation over the comparable launches themselves — each comparable
in turn forecast from the others and scored with the contest's exact loss.
On the console single-player launches (the target's own kind) that scores
a mean loss of about **0.20**, against about **0.32** for the house
persistence baseline on the same launches; the model beat the baseline on
every one of the eight.

The program is deterministic (two runs are byte-identical), runs in about
0.2 s per case, and degrades to the house baseline's spread around the
hour-1 print for any hour no comparable covers.

## Layout

- `main.py` — the entry: reads `$MAKEFUN_INPUT_DIR` (else `/input`) for
  `*.in` case files, writes `<id>.out` forecasts to
  `$MAKEFUN_OUTPUT_DIR` (else `/output`).
- `LICENSE` — MIT.
