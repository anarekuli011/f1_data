"""Grid-search the constructor K-factors by backtesting.

    py backtest_constructors.py                    # 1990-latest, plus last 5 as a check
    py backtest_constructors.py --last 10          # last 10 seasons only
    py backtest_constructors.py --period 2015-2026 --period 1979-2014
    py backtest_constructors.py --period 2010-2026 --race-k 40 72 --race-increment 4 --quali-k 0 24 --quali-increment 2

--race-k / --quali-k take MIN MAX (inclusive); --race-increment /
--quali-increment set the step between them. Defaults: race K 16-128 in
steps of 8, quali K 0-32 in steps of 4.

Era-start boost (higher K for the first races after a regulation change):

    py backtest_constructors.py --race-k 48 --quali-k 8 --boost-race-k 56 96 --boost-quali-k 12 36 --boost-quali-increment 12

Every normal K pair is also run without the boost, so you can see whether
the boost actually helps.

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


def k_values(parser: argparse.ArgumentParser, name: str,
             bounds: list[float], step: float) -> list[float]:
    """Turn 'MIN MAX' plus an increment into the list of K values to test.

    MAX is included when the increment lands on it exactly; one value means
    "test just this K".
    """
    if len(bounds) == 1:
        bounds = bounds * 2
    if len(bounds) != 2:
        parser.error(f"{name} takes MIN MAX (or a single value)")
    lo, hi = bounds
    if lo < 0 or hi < lo:
        parser.error(f"{name}: need 0 <= MIN <= MAX, got {lo:g} {hi:g}")
    if step <= 0:
        parser.error(f"{name} increment must be positive, got {step:g}")
    values = []
    i = 0
    while lo + i * step <= hi + 1e-9:
        v = round(lo + i * step, 6)
        values.append(int(v) if v.is_integer() else v)
        i += 1
    return values


def print_boost_summary(period: str, r: pd.DataFrame, metric: str,
                        acc_col: str) -> None:
    """Best combos with an era-start boost, against the best without one."""
    plain, boosted = r[~r["boosted"]], r[r["boosted"]]
    best_plain = plain.loc[plain[metric].idxmin()]
    best = boosted.loc[boosted[metric].idxmin()]
    near = boosted[boosted[metric] <= best[metric] + NEAR_BEST]

    def k(v):
        return "normal" if pd.isna(v) else f"{v:g}"

    def rng(col):
        vals = near[col]
        if vals.isna().all():
            return "normal"
        return f"{vals.min():g}-{vals.max():g}"

    print(f"=== {period}: {metric}, {int(best.races)} races, "
          f"{int(best.pairs):,} pairs ===")
    top = boosted.nsmallest(10, metric)[
        ["k_race", "k_quali", "boost_race", "boost_quali", metric, acc_col]]
    top[["boost_race", "boost_quali"]] = top[["boost_race", "boost_quali"]].map(k)
    print("Top 10 with the era-start boost:")
    print(top.round(4).to_string(index=False))
    print(f"Best without boost: race K {best_plain.k_race:g}, quali K "
          f"{best_plain.k_quali:g}  {metric} {best_plain[metric]:.4f}")
    print(f"Best with boost:    race K {best.k_race:g}, quali K {best.k_quali:g}, "
          f"era-start race K {k(best.boost_race)}, quali K {k(best.boost_quali)}  "
          f"{metric} {best[metric]:.4f}  "
          f"(improvement {best_plain[metric] - best[metric]:+.4f})")
    print(f"Within {NEAR_BEST} of best (effectively tied): race K {rng('k_race')}, "
          f"quali K {rng('k_quali')}, era-start race K {rng('boost_race')}, "
          f"quali K {rng('boost_quali')}")
    print("Note: the boost only changes scores in or after a regulation-change "
          "season, so judge it on periods that include one.\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest constructor K-factors.")
    p.add_argument("--period", type=parse_period, action="append",
                   help="Seasons to score, e.g. 2015-2026 or 2024. Repeatable.")
    p.add_argument("--last", type=int, default=None,
                   help="Score the last N seasons in the data. Includes the "
                        "current, possibly unfinished, season. With no --period "
                        "or --last, scores 1990-latest and the last 5 seasons.")
    p.add_argument("--race-k", type=float, nargs="+", metavar=("MIN", "MAX"),
                   default=[16, 128],
                   help="Race K range to test: MIN MAX (or one value). "
                        "Default 16 128.")
    p.add_argument("--race-increment", type=float, default=8,
                   help="Step between race K values. Default 8.")
    p.add_argument("--quali-k", type=float, nargs="+", metavar=("MIN", "MAX"),
                   default=[0, 32],
                   help="Qualifying K range to test: MIN MAX (or one value). "
                        "Default 0 32.")
    p.add_argument("--quali-increment", type=float, default=4,
                   help="Step between qualifying K values. Default 4.")
    boost = p.add_argument_group(
        "era-start boost",
        "Higher constructor K for the first races of each season in "
        "regulation_change_years (see f1elo/config.py). Give --boost-race-k "
        "and/or --boost-quali-k to tune it; without them, no boost is used.")
    boost.add_argument("--boost-race-k", type=float, nargs="+",
                       metavar=("MIN", "MAX"),
                       help="Era-start race K range: MIN MAX (or one value).")
    boost.add_argument("--boost-race-increment", type=float, default=8,
                       help="Step between era-start race K values. Default 8.")
    boost.add_argument("--boost-quali-k", type=float, nargs="+",
                       metavar=("MIN", "MAX"),
                       help="Era-start quali K range: MIN MAX (or one value).")
    boost.add_argument("--boost-quali-increment", type=float, default=4,
                       help="Step between era-start quali K values. Default 4.")
    boost.add_argument("--boost-races", type=int, default=None,
                       help="How many races at the start of each new era get "
                            "the boost (default: era_start_races in config, 8).")
    p.add_argument("--metric", default="pre_logloss",
                   choices=["pre_logloss", "postq_logloss", "pre_brier", "postq_brier"],
                   help="pre_* = predict before the weekend, "
                        "postq_* = predict after qualifying.")
    args = p.parse_args()
    race_ks = k_values(p, "--race-k", args.race_k, args.race_increment)
    quali_ks = k_values(p, "--quali-k", args.quali_k, args.quali_increment)
    boost_mode = args.boost_race_k is not None or args.boost_quali_k is not None
    boost_race_ks = (k_values(p, "--boost-race-k", args.boost_race_k,
                              args.boost_race_increment)
                     if args.boost_race_k is not None else [None])
    boost_quali_ks = (k_values(p, "--boost-quali-k", args.boost_quali_k,
                               args.boost_quali_increment)
                      if args.boost_quali_k is not None else [None])
    if args.boost_races is not None and args.boost_races < 1:
        p.error("--boost-races must be at least 1")
    print(f"Race K:  {', '.join(f'{k:g}' for k in race_ks)}")
    print(f"Quali K: {', '.join(f'{k:g}' for k in quali_ks)}")
    if boost_mode:
        fmt = lambda ks: ", ".join("normal" if k is None else f"{k:g}" for k in ks)
        print(f"Era-start race K:  {fmt(boost_race_ks)}")
        print(f"Era-start quali K: {fmt(boost_quali_ks)}")
        print(f"Boosted races per new era: "
              f"{args.boost_races or config.DEFAULT_CONFIG.era_start_races}")

    print("Preparing races ...")
    prepared = prepare_races(build_race_frame())
    latest = max(p[1] for p in prepared)

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
    n_runs = len(race_ks) * len(quali_ks)
    if boost_mode:
        # Also run without the boost, as the baseline to beat.
        n_runs *= 1 + len(boost_race_ks) * len(boost_quali_ks)
    print(f"Running {n_runs} backtests (about 1.3 s each) ...")
    res = sweep_constructor_k(prepared, race_ks, quali_ks, periods,
                              progress=True)
    if boost_mode:
        res["boosted"] = False
        boosted = sweep_constructor_k(prepared, race_ks, quali_ks, periods,
                                      progress=True,
                                      boost_race_ks=boost_race_ks,
                                      boost_quali_ks=boost_quali_ks,
                                      boost_races=args.boost_races)
        boosted["boosted"] = True
        res = pd.concat([res, boosted], ignore_index=True)

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
        if boost_mode:
            print_boost_summary(period, r, args.metric, acc_col)
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
