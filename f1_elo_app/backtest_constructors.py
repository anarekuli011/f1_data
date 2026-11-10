"""Grid-search the constructor K-factors by backtesting.

    py backtest_constructors.py                    # 1990-latest, plus last 5 as a check
    py backtest_constructors.py --last 10          # last 10 seasons only
    py backtest_constructors.py --period 2015-2026 --period 1979-2014
    py backtest_constructors.py --race-k 40 48 56 --quali-k 10 12 14

For every (race K, qualifying K) pair it replays ALL races from 1950 (so the
ratings are properly warmed up), but only scores the races inside each
evaluation period: each one is predicted from the ratings *before* that race,
and the pairwise log loss is reported (lower is better; 0.6931 = coin flip).
See f1elo/backtest.py for details.

The intended workflow: after each season, rerun this with the defaults and
copy the K-factors suggested for the long window (1990 to the latest season)
into f1elo/config.py. The last-5-seasons table is shown as a check: if it
keeps disagreeing season after season (e.g. after a big rule change), that is
a real shift worth acting on; a one-off difference is usually noise.

Writes the full results to output/backtest_constructors.csv.
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from f1elo import build_race_frame, config
from f1elo.backtest import prepare_races, sweep_constructor_k

# Scores within this much of the best are treated as a tie: on a few seasons
# of data, differences this small are noise, not a real preference.
NEAR_BEST = 0.0005

# Default scoring window: every season from this year to the latest. Tested
# season by season (pick K from earlier seasons, score the next one), a long
# window predicted better than the last 5 seasons alone, which is too little
# data and makes the chosen K jump around from year to year.
DEFAULT_SINCE = 1990


def parse_period(text: str) -> tuple[str, tuple[int, int]]:
    """'2015-2026' -> ('2015-2026', (2015, 2026)); '2024' -> one season."""
    m = re.fullmatch(r"\s*(\d{4})\s*(?:-\s*(\d{4}))?\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(
            f"period must look like 2015-2026 or 2024, got {text!r}")
    start = int(m.group(1))
    end = int(m.group(2) or start)
    if end < start:
        raise argparse.ArgumentTypeError(f"period {text!r} ends before it starts")
    return f"{start}-{end}", (start, end)


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest constructor K-factors.")
    p.add_argument("--period", type=parse_period, action="append",
                   help="Seasons to score, e.g. 2015-2026 or 2024. Repeatable.")
    p.add_argument("--last", type=int, default=None,
                   help="Score the last N seasons in the data. Includes the "
                        "current, possibly unfinished, season. With no --period "
                        "or --last, scores 1990-latest and the last 5 seasons.")
    p.add_argument("--race-k", type=float, nargs="+",
                   default=[16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256])
    p.add_argument("--quali-k", type=float, nargs="+",
                   default=[0, 4, 8, 12, 16, 20, 24, 32, 48])
    p.add_argument("--metric", default="pre_logloss",
                   choices=["pre_logloss", "postq_logloss", "pre_brier", "postq_brier"],
                   help="pre_* = predict before the weekend, "
                        "postq_* = predict after qualifying.")
    args = p.parse_args()

    print("Preparing races ...")
    prepared = prepare_races(build_race_frame())
    latest = max(year for _, year, _, _ in prepared)

    periods: dict[str, tuple[int, int]] = dict(args.period or [])
    if args.last is not None:
        start = latest - args.last + 1
        periods[f"last {args.last} ({start}-{latest})"] = (start, latest)
    if not periods:
        # Default: choose K on the long window, and show the last 5 seasons
        # alongside as a check for a genuine recent change.
        periods[f"{DEFAULT_SINCE}-{latest} (use this)"] = (DEFAULT_SINCE, latest)
        periods[f"last 5 ({latest - 4}-{latest}, check)"] = (latest - 4, latest)

    print(f"Scoring periods: {', '.join(periods)}")
    print(f"Running {len(args.race_k) * len(args.quali_k)} backtests ...")
    res = sweep_constructor_k(prepared, args.race_k, args.quali_k, periods,
                              progress=True)

    out = config.OUTPUT_DIR / "backtest_constructors.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"Wrote {out}\n")

    acc_col = args.metric.split("_")[0] + "_acc"
    pd.set_option("display.width", 200)
    for period in periods:
        r = res[res["period"] == period]
        if r.empty:
            print(f"=== {period}: no races in the data ===\n")
            continue
        grid = r.pivot(index="k_race", columns="k_quali", values=args.metric)
        best = r.loc[r[args.metric].idxmin()]
        near = r[r[args.metric] <= best[args.metric] + NEAR_BEST]
        print(f"=== {period}: {args.metric}, {int(best.races)} races, "
              f"{int(best.pairs):,} pairs (rows = race K, columns = quali K) ===")
        print(grid.round(4).to_string())
        print(f"Best: race K {best.k_race:g}, quali K {best.k_quali:g}  "
              f"{args.metric} {best[args.metric]:.4f}  accuracy {best[acc_col]:.3f}")
        print(f"Within {NEAR_BEST} of best (effectively tied): "
              f"race K {near.k_race.min():g}-{near.k_race.max():g}, "
              f"quali K {near.k_quali.min():g}-{near.k_quali.max():g}\n")


if __name__ == "__main__":
    main()
