"""Grid-search the driver settings by backtesting on teammate pairs.

    py backtest_drivers.py --race-k 24 80 --race-increment 8 --exponent 1 5 --exponent-increment 1

(about 3 s per combination; the script tells you how many it will run)

Any setting you don't give stays at its current value in f1elo/config.py, so
you can tune one or two at a time:

    --race-k MIN MAX        veteran (100% confidence) race K
    --quali-k MIN MAX       veteran quali K
    --rookie-ratio MIN MAX  rookie (0% confidence) K = ratio x veteran K,
                            for both race and quali (now 112/56 = 2)
    --exponent MIN MAX      constructor_adjustment_exponent
    --dnf-factor MIN MAX    driver_dnf_penalty_factor
    --confidence-drop MIN MAX
                            era_start_confidence_drop (0.5 = every driver
                            loses 50 points of confidence at a new era)
    --car-weight MIN MAX    driver_car_weight (lam): the car's weight in the
                            driver expectation, D + lam * C
    --car-before-update 0 1 1 = driver passes see the car ratings from before
                            that session's constructor update
    --opponent-floor MIN MAX
                            driver_opponent_confidence_floor: how much a
                            result against a 0%-confidence driver counts
                            (1 = as much as any other, i.e. off)
    --confidence-races MIN MAX
                            driver_confidence_races: races from zero to full
                            confidence (whole numbers; default step 2)

each with a matching --...-increment. The constructor settings are taken
from the config as they are, so tune them first (backtest_constructors.py).

Periods: with no --period or --last, every table is scored on 1990-2022
(tune: pick settings from this one) and 1950-1979 (check: where the
multi-car and part-timer fixes matter). 2023 onwards is the holdout, left out
of everything, --era-seasons included, until you add --holdout once at the
very end. Constants in f1elo/backtest.py (TUNE_PERIOD, CHECK_PERIOD,
HOLDOUT_START).

Besides the usual teammate scores, every table shows the same scores for
NEW teammate pairs (newrace / newquali: the first --new-races races, default
8, after either driver moved team) and ``leak``, the correlation between
driver and car ratings. New pairs are where car strength leaking into driver
ratings hurts most, so check them when tuning --car-weight:

    py backtest_drivers.py --car-weight 0 1 --exponent 0 2.25 --exponent-increment 2.25 --metric newrace_logloss

--era-seasons adds a score over just the regulation-change seasons
(1998, 2009, 2014, ...) pooled together, plus the season after each one --
where an era-start setting such as --confidence-drop actually matters.

Every race from 1950 is replayed with the full engine; before each race in
the scoring period, every pair of TEAMMATES is predicted from the driver
ratings at that moment. Teammates share a car, so this tests whether the
driver ratings capture the driver rather than the car. Lower log loss is
better; 0.6931 = coin flip. See f1elo/backtest.py for details.

Retune order (after the constructor settings; pairs on one line are tuned
together because they trade off against each other):
  1. --car-weight with --exponent      (check the exponent stays at 0)
  2. --race-k with --car-weight         (exponent from step 1)
  3. --quali-k                          (--metric quali_logloss)
  4. --dnf-factor
  5. --confidence-races with --rookie-ratio
  6. --opponent-floor with --rookie-ratio   (watch the 1950-1979 check)
  7. --confidence-drop                  (with --era-seasons)
  8. --race-k, --car-weight and --confidence-races, narrow ranges, to confirm
  9. the final settings once with --holdout.

Writes the full results to output/backtest_drivers.csv.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from f1elo import build_race_frame, config
from f1elo.backtest import (DRIVER_PARAMS, HOLDOUT_START, INTEGER_PARAMS,
                            TUNE_PERIOD, check_whole, default_periods,
                            driver_backtest, driver_config, driver_defaults,
                            holdout_period, parse_period, prepare_races,
                            summarise_drivers, sweep_drivers, value_range)

# Teammate pairs are far fewer than all car pairs, so scores are noisier:
# settings within this much of the best are treated as a tie.
NEAR_BEST = 0.001

# name -> (flag, increment flag, default increment, help)
OPTIONS = {
    "race_k": ("--race-k", "--race-increment", 8, "veteran race K"),
    "quali_k": ("--quali-k", "--quali-increment", 2, "veteran quali K"),
    "rookie_ratio": ("--rookie-ratio", "--rookie-ratio-increment", 0.1,
                     "rookie K / veteran K"),
    "exponent": ("--exponent", "--exponent-increment", 0.5,
                 "constructor adjustment exponent"),
    "dnf_factor": ("--dnf-factor", "--dnf-factor-increment", 0.1,
                   "driver DNF penalty factor"),
    "confidence_drop": ("--confidence-drop", "--confidence-drop-increment", 0.1,
                        "era-start driver confidence drop (0.5 = 50 points)"),
    "car_weight": ("--car-weight", "--car-weight-increment", 0.25,
                   "driver_car_weight (lam: car weight in the driver expectation)"),
    "car_before_update": ("--car-before-update", "--car-before-update-increment", 1,
                          "1 = driver passes use car ratings from before the "
                          "session's constructor update, 0 = after"),
    "opponent_floor": ("--opponent-floor", "--opponent-floor-increment", 0.1,
                       "driver_opponent_confidence_floor (weight of a result "
                       "against a zero-confidence driver; 1 = off)"),
    "confidence_races": ("--confidence-races", "--confidence-races-increment", 2,
                         "driver_confidence_races (races from zero to full "
                         "confidence; whole numbers)"),
}

# Scores shown next to the chosen metric for every combination.
SHOWN = ["race_logloss", "quali_logloss", "newrace_logloss", "newquali_logloss",
         "leak"]


def main() -> None:
    current = driver_defaults()
    p = argparse.ArgumentParser(description="Backtest driver settings on teammate pairs.")
    p.add_argument("--period", action="append",
                   help="Seasons to score, e.g. 2015-2026 or 2024. Repeatable.")
    p.add_argument("--last", type=int, default=None,
                   help="Score the last N seasons. With no --period or --last, "
                        "scores 1990-2022 (tune) and 1950-1979 (check).")
    p.add_argument("--holdout", action="store_true",
                   help="Also score the holdout seasons (2023 to the latest). "
                        "Use once, after tuning is finished.")
    for name, (flag, step_flag, step, desc) in OPTIONS.items():
        p.add_argument(flag, type=float, nargs="+", metavar=("MIN", "MAX"),
                       dest=name, help=f"{desc} range: MIN MAX (or one value). "
                                       f"Default: current value {current[name]:g}.")
        p.add_argument(step_flag, type=float, default=step,
                       dest=f"{name}_step", help=f"Step for {flag}. Default {step:g}.")
    p.add_argument("--era-seasons", action="store_true",
                   help="Also score the regulation-change seasons since 1990 "
                        "pooled together, and the seasons right after them.")
    p.add_argument("--metric", default="race_logloss",
                   choices=[f"{s}_{m}" for m in ("logloss", "brier")
                            for s in ("race", "raceall", "quali", "newrace", "newquali")],
                   help="race = teammates both classified (default); raceall = "
                        "incl. DNFs as losses; quali = teammates by grid; "
                        "newrace / newquali = the same, new teammate pairs only.")
    p.add_argument("--new-races", type=int, default=8,
                   help="A teammate pair counts as new for this many races after "
                        "either driver moved team (default 8).")
    args = p.parse_args()

    periods: dict[str, tuple[int, int]] = {}
    for text in args.period or []:
        try:
            label, rng = parse_period(text)
        except ValueError as e:
            p.error(str(e))
        periods[label] = rng

    grid: dict[str, list] = {}
    for name, (flag, _, _, _) in OPTIONS.items():
        bounds = getattr(args, name)
        if bounds is not None:
            try:
                grid[name] = value_range(bounds, getattr(args, f"{name}_step"))
                if name in INTEGER_PARAMS:
                    check_whole(grid[name])
            except ValueError as e:
                p.error(f"{flag}: {e}")
    varied = [k for k in DRIVER_PARAMS if len(grid.get(k, [None])) > 1]
    for name in DRIVER_PARAMS:
        vals = grid.get(name, [current[name]])
        tag = "" if name in grid else "  (current)"
        print(f"{name:<13} {', '.join(f'{v:g}' for v in vals)}{tag}")

    print("Preparing races ...")
    prepared = prepare_races(build_race_frame())
    latest = max(r[1] for r in prepared)

    if args.last is not None:
        start = latest - args.last + 1
        periods[f"last {args.last} ({start}-{latest})"] = (start, latest)
    if not periods:
        periods.update(default_periods(latest))
    if args.era_seasons:
        # Holdout seasons stay out unless --holdout is given.
        last_year = latest if args.holdout else HOLDOUT_START - 1
        eras = sorted(y for y in config.DEFAULT_CONFIG.regulation_change_years
                      if TUNE_PERIOD[0] <= y <= last_year)
        periods[f"era-start seasons ({', '.join(map(str, eras))})"] = frozenset(eras)
        after = [y + 1 for y in eras if y + 1 <= last_year]
        if after:
            periods[f"seasons after ({', '.join(map(str, after))})"] = frozenset(after)
    if args.holdout:
        periods.update(holdout_period(latest))
    print(f"Scoring periods: {', '.join(periods)}")

    # The current config, as the baseline to beat.
    t0 = time.time()
    base_run = driver_backtest(prepared, driver_config(config.DEFAULT_CONFIG, **current),
                               new_races=args.new_races)
    secs = time.time() - t0
    n_runs = 1
    for name in DRIVER_PARAMS:
        n_runs *= len(grid.get(name, [0]))
    print(f"Running {n_runs} backtests (about {secs:.0f} s each, "
          f"~{n_runs * secs / 60:.0f} min) ...")
    res = sweep_drivers(prepared, grid, periods, progress=True,
                        new_races=args.new_races)

    out = config.OUTPUT_DIR / "backtest_drivers.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"Wrote {out}\n")

    prefix = args.metric.split("_")[0]
    acc_col, pairs_col = f"{prefix}_acc", f"{prefix}_pairs"
    pd.set_option("display.width", 200)
    for period, spec in periods.items():
        r = res[res["period"] == period]
        if r.empty:
            print(f"=== {period}: no races in the data ===\n")
            continue
        best = r.loc[r[args.metric].idxmin()]
        print(f"=== {period}: {args.metric}, {int(best.races)} races, "
              f"{int(best[pairs_col]):,} teammate pairs "
              f"({int(best.newrace_pairs):,} new race / "
              f"{int(best.newquali_pairs):,} new quali) ===")
        if len(varied) == 2:
            grid_tbl = r.pivot(index=varied[0], columns=varied[1], values=args.metric)
            print(f"(rows = {varied[0]}, columns = {varied[1]})")
            print(grid_tbl.round(4).to_string())
        cols = varied or list(DRIVER_PARAMS)
        shown = [args.metric] + [c for c in SHOWN if c != args.metric] + [acc_col]
        print("Top 10 (leak = driver-vs-car rating correlation):")
        print(r.nsmallest(10, args.metric)[cols + shown]
              .round(4).to_string(index=False))

        cur_all = summarise_drivers(base_run, spec)
        cur = cur_all[args.metric]
        print("Current config:  " + "  ".join(
            f"{c} {cur_all[c]:.4f}" for c in shown))
        desc = ", ".join(f"{k} {best[k]:g}" for k in (varied or DRIVER_PARAMS))
        print(f"Best:            {desc}  {args.metric} {best[args.metric]:.4f}  "
              f"accuracy {best[acc_col]:.3f}  (vs current {cur - best[args.metric]:+.4f})")
        near = r[r[args.metric] <= best[args.metric] + NEAR_BEST]
        if varied:
            ranges = ", ".join(f"{k} {near[k].min():g}-{near[k].max():g}" for k in varied)
            print(f"Within {NEAR_BEST} of best (effectively tied): {ranges}")
        print()


if __name__ == "__main__":
    main()
