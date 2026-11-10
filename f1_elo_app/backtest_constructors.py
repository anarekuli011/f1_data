"""Grid-search the constructor K-factors by backtesting.

    py backtest_constructors.py
    py backtest_constructors.py --race-k 48 64 80 96 --quali-k 0 12 24

For every (race K, qualifying K) pair it replays all races, predicting each
race from the ratings *before* that race, and reports the pairwise log loss
(lower is better; 0.6931 = coin flip). See f1elo/backtest.py for details.

Writes the full results to output/backtest_constructors.csv.
"""
from __future__ import annotations

import argparse

import pandas as pd

from f1elo import build_race_frame, config
from f1elo.backtest import prepare_races, sweep_constructor_k

PERIODS = {
    "1960-2026": (1960, 2026),   # everything after a 10-season warm-up
    "1979-2014": (1979, 2014),   # the Bell et al. window
    "2015-2026": (2015, 2026),   # recent era
}


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest constructor K-factors.")
    p.add_argument("--race-k", type=float, nargs="+",
                   default=[16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256])
    p.add_argument("--quali-k", type=float, nargs="+",
                   default=[0, 6, 12, 24, 48])
    p.add_argument("--metric", default="pre_logloss",
                   choices=["pre_logloss", "postq_logloss", "pre_brier", "postq_brier"])
    args = p.parse_args()

    print("Preparing races ...")
    prepared = prepare_races(build_race_frame())
    print(f"Running {len(args.race_k) * len(args.quali_k)} backtests ...")
    res = sweep_constructor_k(prepared, args.race_k, args.quali_k, PERIODS,
                              progress=True)

    out = config.OUTPUT_DIR / "backtest_constructors.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"Wrote {out}\n")

    pd.set_option("display.width", 200)
    for period in PERIODS:
        r = res[res["period"] == period]
        grid = r.pivot(index="k_race", columns="k_quali", values=args.metric)
        best = r.loc[r[args.metric].idxmin()]
        print(f"=== {period}: {args.metric} (rows = race K, columns = quali K) ===")
        print(grid.round(4).to_string())
        print(f"Best: race K {best.k_race:g}, quali K {best.k_quali:g}  "
              f"{args.metric} {best[args.metric]:.4f}  "
              f"accuracy {best[args.metric.split('_')[0] + '_acc']:.3f}\n")


if __name__ == "__main__":
    main()
