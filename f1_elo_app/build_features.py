"""Build the race feature table for the step 6 models, plus its reports.

    py build_features.py                        # all stages, cached; coverage + signal reports
    py build_features.py --stage post_quali     # one stage
    py build_features.py --form-races 5         # recent form over the last 5 races
    py build_features.py --period 1990-2022 --period 2023-2026
                                                # periods for the signal report
    py build_features.py --rebuild              # ignore the cache
    py build_features.py --no-reports           # just the tables

One row per car per race (f1elo/features.py; every column is listed in the
README). Everything in a row was known at that stage of the weekend:
  pre          before the weekend
  post_quali   + that weekend's qualifying and grid
  post_sprint  + the sprint result (sprint weekends only)
It joins to the ELO model's predictions (calibration.cached_predictions)
on raceId + driverId.

Written to output/:
  features_<stage>.parquet (+ .json: cache key, settings, build timings)
  features_coverage_<stage>_by_era.csv / _by_year.csv
      share of rows with a value, per feature
  features_signal_<stage>.csv
      a quick look at each feature's relationship with the result, NO
      fitting (finish_rho, beyond_elo, dnf_auc, dnf_q1/q5; see
      features.signal)

The first build takes a minute or so (longer if the ELO predictions have
to be rebuilt: about 15 s replay + 1 min simulation per stage); a cached
rebuild takes a few seconds. Lap times and pit stops come from
update_laps.py; without them those features are just empty.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from f1elo import config, features
from f1elo.backtest import parse_period
from f1elo.config import DEFAULT_FEATURES
from f1elo.data_loader import load_raw

# Feature groups for the printed coverage summary: (label, column test).
GROUPS = [
    ("ELO model", lambda c: c.startswith("elo_")),
    ("retirement rates", lambda c: c.startswith("dnf_")),
    ("form: results", lambda c: c.split("_")[0] in ("drv", "team")
     and c.split("_")[1] in ("deficit", "laps", "gained", "finished")),
    ("form: fastest lap", lambda c: c.split("_")[0] in ("drv", "team") and "fastlap" in c),
    ("form: lap data", lambda c: c.split("_")[0] in ("drv", "team")
     and any(k in c for k in ("pace", "lap1", "consistency"))),
    ("form: pit stops", lambda c: c.startswith("team_pit")),
    ("circuit", lambda c: c.startswith("circ_") or c == "street"),
    ("qualifying times", lambda c: c.startswith(("q_gap", "q_pos", "q_grid_penalty"))),
    ("grid", lambda c: c.startswith(("q_grid", "q_pit")) and "penalty" not in c),
    ("sprint", lambda c: c.startswith("s_")),
    ("context / driver", lambda c: c in ("era_start", "era_race", "round", "season_progress",
                                         "field_size", "sprint_weekend", "age", "starts",
                                         "team_starts", "home", "team_home")),
]


def coverage_summary(by_era: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for label, test in GROUPS:
        cols = [c for c in by_era.index if test(c)]
        if cols:
            rows[f"{label} ({len(cols)})"] = by_era.loc[cols].mean()
    return pd.DataFrame(rows).T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=list(features.STAGES) + ["all"], default="all")
    p.add_argument("--form-races", type=int, default=DEFAULT_FEATURES.form_races,
                   help=f"races in the recent-form averages (default {DEFAULT_FEATURES.form_races})")
    p.add_argument("--circuit-shrink", type=float, default=DEFAULT_FEATURES.circuit_shrink,
                   help="races' worth of 'average circuit' in the circuit features "
                        f"(default {DEFAULT_FEATURES.circuit_shrink:g})")
    p.add_argument("--period", action="append", metavar="YYYY-YYYY",
                   help="period(s) for the signal report (default 1990-2022)")
    p.add_argument("--rebuild", action="store_true", help="ignore the cache")
    p.add_argument("--no-reports", action="store_true", help="skip the coverage and signal reports")
    p.add_argument("--top", type=int, default=12, help="signals to print per list (default 12)")
    a = p.parse_args()

    fcfg = replace(DEFAULT_FEATURES, form_races=a.form_races, circuit_shrink=a.circuit_shrink)
    periods = [parse_period(x) for x in (a.period or ["1990-2022"])]
    stages = features.STAGES if a.stage == "all" else (a.stage,)
    pd.set_option("display.width", 200, "display.max_columns", 20)

    t0 = time.perf_counter()
    raw = load_raw()
    laps, pits = raw["lap_times"], raw["pit_stops"]
    print(f"Data: {len(raw['results']):,} results, {len(laps):,} lap times "
          f"({laps['raceId'].nunique() if len(laps) else 0} races), {len(pits):,} pit stops "
          f"({pits['raceId'].nunique() if len(pits) else 0} races)")
    if not len(laps):
        print("  No lap times yet: the lap-data features will be empty. See update_laps.py.")

    for stage in stages:
        t = time.perf_counter()
        table = features.cached_features(stage, fcfg, rebuild=a.rebuild, raw=raw, progress=print)
        feats, labels = features.stage_columns(table, stage)
        print(f"\n== {stage}: {len(table):,} rows, {len(feats)} features, {len(labels)} label "
              f"columns ({time.perf_counter() - t:.1f}s) -> output/features_{stage}.parquet")
        if a.no_reports:
            continue
        by_year, by_era = features.coverage(table, stage)
        by_year.to_csv(config.OUTPUT_DIR / f"features_coverage_{stage}_by_year.csv")
        by_era.to_csv(config.OUTPUT_DIR / f"features_coverage_{stage}_by_era.csv")
        print("\nShare of rows with a value (mean over each group's features):")
        print(coverage_summary(by_era).round(2).to_string())

        sig = []
        for name, period in periods:
            s = features.signal(table, stage, period)
            s.insert(0, "period", name)
            sig.append(s)
        sig = pd.concat(sig, ignore_index=True)
        sig.to_csv(config.OUTPUT_DIR / f"features_signal_{stage}.csv", index=False)
        for name, _ in periods:
            s = sig[(sig["period"] == name) & (sig["coverage"] > 0.05)]
            print(f"\nSignal, {name} (no fitting; see features.signal):")
            order = s.dropna(subset=["beyond_elo"])
            order = order.reindex(order["beyond_elo"].abs().sort_values(ascending=False).index)
            print("  Finishing order beyond the ELO model (beyond_elo: within-race rank "
                  "correlation with the\n  finishing place, partial on ELO's expected position; "
                  "positive = a higher value finishes further back):")
            print(order[["feature", "coverage", "finish_rho", "beyond_elo"]].head(a.top)
                  .round(3).to_string(index=False))
            dnf = s.dropna(subset=["dnf_beyond"])
            dnf = dnf.reindex((dnf["dnf_beyond"] - 0.5).abs().sort_values(ascending=False).index)
            print("  Retirements beyond the ELO model's P(DNF) (dnf_beyond: AUC within tenths "
                  "of P(DNF); 0.5 = adds nothing):")
            print(dnf[["feature", "coverage", "dnf_auc", "dnf_beyond", "dnf_q1", "dnf_q5"]]
                  .head(a.top).round(3).to_string(index=False))
            race = s.dropna(subset=["race_dnf_rho"])
            if len(race):
                print("  One value per race (Spearman across races with the race's retirement "
                      "rate, and with how well its grid order held):")
                print(race[["feature", "coverage", "race_dnf_rho", "race_order_rho"]]
                      .round(3).to_string(index=False))
    print(f"\nDone in {time.perf_counter() - t0:.0f}s.")


if __name__ == "__main__":
    main()
