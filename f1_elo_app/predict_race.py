"""Predict a race: each driver's chance of every finishing position.

    py predict_race.py                       # the next race (latest race's field)
    py predict_race.py --latest              # the latest race with results, vs what happened
    py predict_race.py --year 2026 --round 21
    py predict_race.py --race-id 1170 --positions 10 --check

(about 20-30 s: the engine is replayed up to the race; the simulation
itself takes well under a second)

Each race is simulated --sims times (default 10,000, --seed 0): first every
car retires or not, with its own chance (f1elo/reliability.py), then the
cars that finish are ordered by Plackett-Luce on their pre-race strengths
D + finish_car_weight * C (f1elo/simulate.py). For a past race everything is
pre-race: the ratings before that race and retirement counts from earlier
races only.

Columns: driver, team, D (driver rating), C (car rating), strength, P(DNF)
(the input chance of not finishing), win / podium / points chances, the
simulated DNF share, exp_pos (expected position if classified) and, for a
past race, the actual result. --positions N adds the P1 ... PN columns.
--csv writes the full table (every position) to output/.

--check runs the simulation's sanity checks on the same field: rows sum to
1, position columns against the exact chance that at least k cars finish,
simulated head-to-heads against expected_score, same seed -> same result,
and the time for one run.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from f1elo import build_race_frame, config
from f1elo.backtest import prepare_races
from f1elo.reliability import entry_history
from f1elo.simulate import (DNF, N_SIMS, SEED, predict_next_race, predict_race,
                            sanity_checks, summary)


def main() -> None:
    cfg = config.DEFAULT_CONFIG
    p = argparse.ArgumentParser(description="Simulate a race's finishing positions.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--latest", action="store_true",
                   help="The latest race with results (compared with what happened).")
    g.add_argument("--race-id", type=int, help="A past race by raceId.")
    g.add_argument("--year", type=int, help="A past race by year (with --round).")
    p.add_argument("--round", type=int, help="Round, with --year.")
    p.add_argument("--sims", type=int, default=N_SIMS, help="Races to simulate (10000).")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed (0).")
    p.add_argument("--positions", type=int, default=0,
                   help="Also show the P1 ... PN columns.")
    p.add_argument("--check", action="store_true", help="Run the sanity checks.")
    p.add_argument("--csv", action="store_true", help="Write the full table to output/.")
    args = p.parse_args()
    if (args.year is None) != (args.round is None):
        p.error("--year and --round go together")
    if args.sims < 1:
        p.error("--sims must be at least 1")

    t0 = time.time()
    df = build_race_frame()
    prepared = prepare_races(df, cfg)
    hist = entry_history(df, cfg)
    race_id = args.race_id
    if args.latest:
        race_id = prepared[-1][0]
    elif args.year is not None:
        hit = [r[0] for r in prepared if r[1] == args.year and r[2] == args.round]
        if not hit:
            p.error(f"no results for {args.year} round {args.round}")
        race_id = hit[0]

    if race_id is None:
        table = predict_next_race(df=df, prepared=prepared, hist=hist, cfg=cfg,
                                  n_sims=args.sims, seed=args.seed)
        what = "next race"
    else:
        table = predict_race(race_id, df=df, prepared=prepared, hist=hist, cfg=cfg,
                             n_sims=args.sims, seed=args.seed)
        what = "past race, pre-race ratings"
    info = table.attrs["race"]
    pts = table.attrs["points_positions"]
    print(f"\n{info.get('year')} round {info.get('round')}: {info.get('name')} "
          f"(raceId {info.get('raceId')}; {what}) -- {args.sims:,} simulations, "
          f"seed {args.seed}, points = top {pts}  [{time.time() - t0:.0f} s]")
    print(f"strength = D + {cfg.finish_car_weight:g} C, Plackett-Luce scale "
          f"{cfg.finish_scale:g}; P(DNF) from f1elo/reliability.py")

    show = summary(table, args.positions).rename(columns={
        "driver_elo": "D", "constructor_elo": "C", "p_dnf": "P(DNF)",
        "win": "win %", "podium": "podium %", "points": "points %", DNF: "DNF %"})
    pct = [c for c in show.columns if c.endswith("%") or c.startswith("P")]
    show[pct] = show[pct] * 100
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    print(show.round({"D": 0, "C": 0, "strength": 0, "exp_pos": 1,
                      **{c: 1 for c in pct}}).to_string(index=False))
    n = len(table)
    print(f"\nExpected finishers {(1 - table['p_dnf']).sum():.1f} of {n}"
          + (f"; actually {sum(a != DNF for a in table['actual'])}" if "actual" in table
             else ""))

    if args.csv:
        out = config.OUTPUT_DIR / f"prediction_{info.get('year')}_{info.get('round')}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"Wrote {out}")

    if args.check:
        print("\nSanity checks:")
        for k, v in sanity_checks(table, cfg, args.sims, args.seed).items():
            print(f"  {k:22s} {v:.4g}" if isinstance(v, float) else f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
