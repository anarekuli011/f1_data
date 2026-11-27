"""Backtest the retirement model: each car's chance of not finishing.

    py backtest_reliability.py
    py backtest_reliability.py --team-races 10 40 --team-races-increment 5 --team-k 10 40 --team-k-increment 5

(about a minute: the default grid is ~130,000 combinations, and the engine
replay for the rating features takes about 15 s; --no-ratings skips the
replay, and narrower ranges are quicker)

Every car in every race is scored on one question: did it retire? A
retirement is counted as the engine counts one, position == \\N. Chances
use only the races before the one scored (f1elo/reliability.py):

    base   = field DNF share over its last B races          (--base-races)
    team   = (team DNFs + k_t * base) / (team entries + k_t),
             over the team's last N_t races, all its cars    (--team-races, --team-k)
    driver = the same over the driver's last N_d races       (--driver-races, --driver-k)
    blend  = (1 - w) * team + w * driver                     (--driver-weight)

then, optionally, a logistic regression on top of the blend:

    logit P = a + b * logit(blend) + c * car_rel + d * driver_rel

with car_rel / driver_rel the pre-race car / driver rating minus the field's
mean, per 100 points (the engine's ratings before that race, as in
f1elo/strength.py). That answers whether the ratings know anything about
retirements that the counts don't.

The script:
  1. scores the base rate alone for every B (the baseline to beat; B is
     part of the grid below too, since the team and driver rates are shrunk
     toward it);
  2. grid-searches B, N_t, k_t, N_d, k_d and w together on the tuning
     period, and shows how flat the optimum is;
  3. fits the logistic coefficients on the tuning period with the best
     blend, and checks them once more fitted on 1990-2009 only and scored
     on 2010-2022 (an out-of-sample test inside the tuning period);
  4. compares the candidates on every period and prints a calibration
     table (predicted vs actual retirement rate, by predicted band).

Options (each range is MIN MAX, or one value; with a matching --...-increment):
    --base-races MIN MAX    B (default 10-60, step 10)
    --team-races MIN MAX    N_t (default 10-60, step 10)
    --team-k MIN MAX        k_t (default 0-80, step 10)
    --driver-races MIN MAX  N_d (default 10-60, step 10)
    --driver-k MIN MAX      k_d (default 0-100, step 20)
    --driver-weight MIN MAX w (default 0-1, step 0.1)
    --period 2015-2022      extra seasons to score (repeatable)
    --check                 also score 1950-1979
    --no-ratings            skip the rating features (and the engine replay)

Periods: settings are chosen on 1990-2022 (tune) only. 2023 onwards
(holdout) and 2022 and 2026 (era starts) are reported alongside. Lower log
loss is better.

Writes the grid to output/backtest_reliability.csv. The chosen settings
belong in the dnf_* values in f1elo/config.py.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from f1elo import build_race_frame, config
from f1elo.backtest import (CHECK_PERIOD, HOLDOUT_START, TUNE_PERIOD,
                            parse_period, period_mask, prepare_races,
                            value_range)
from f1elo.reliability import (attach_ratings, base_rate, blend, entry_history,
                               fit_logistic, log_loss, logit, relative_to_field,
                               shrunk_rate, window_counts)

# ~15,000 entries in the tuning period: differences this small are ties.
NEAR_BEST = 0.0005
SPLIT = (1990, 2009), (2010, 2022)   # fit on the first, score the second


def brier(p, y) -> float:
    return float(((np.asarray(p) - np.asarray(y)) ** 2).mean())


def main() -> None:
    cfg = config.DEFAULT_CONFIG
    p = argparse.ArgumentParser(description="Backtest the chance of not finishing.")

    def rng(name, default, inc, help_):
        p.add_argument(f"--{name}", type=float, nargs="+", metavar=("MIN", "MAX"),
                       default=default, help=help_)
        p.add_argument(f"--{name}-increment", type=float, default=inc)

    rng("base-races", [10, 60], 10, "B: races in the field's base rate (default 10 60).")
    rng("team-races", [10, 60], 10, "N_t: the team's last N races (default 10 60).")
    rng("team-k", [0, 80], 10, "k_t: shrinkage toward the base, in entries (default 0 80).")
    rng("driver-races", [10, 60], 10, "N_d: the driver's last N races (default 10 60).")
    rng("driver-k", [0, 100], 20, "k_d: shrinkage for the driver (default 0 100).")
    rng("driver-weight", [0, 1], 0.1, "w: the driver's share of the blend (default 0 1).")
    p.add_argument("--period", action="append",
                   help="Extra seasons to score, e.g. 2015-2022. Repeatable.")
    p.add_argument("--check", action="store_true", help="Also score 1950-1979.")
    p.add_argument("--no-ratings", action="store_true",
                   help="Skip the rating features (no engine replay).")
    args = p.parse_args()

    try:
        grids = {k: value_range(getattr(args, k), getattr(args, f"{k}_increment"))
                 for k in ("base_races", "team_races", "team_k", "driver_races",
                           "driver_k", "driver_weight")}
    except ValueError as e:
        p.error(str(e))
    for k in ("base_races", "team_races", "driver_races"):
        if min(grids[k]) < 1 or any(not float(v).is_integer() for v in grids[k]):
            p.error(f"--{k.replace('_', '-')} takes whole numbers of at least 1")
        grids[k] = [int(v) for v in grids[k]]
    if max(grids["driver_weight"]) > 1:
        p.error("--driver-weight must be between 0 and 1")

    print("Loading results ...")
    df = build_race_frame()
    hist = entry_history(df, cfg)
    y = hist["dnf"].to_numpy(float)
    years = hist["year"].to_numpy()
    latest = int(years.max())

    tune = f"{TUNE_PERIOD[0]}-{TUNE_PERIOD[1]} (tune)"
    periods: dict = {tune: TUNE_PERIOD}
    if latest >= HOLDOUT_START:
        periods[f"{HOLDOUT_START}-{latest} (holdout)"] = (HOLDOUT_START, latest)
    for yr in (2022, 2026):
        if yr <= latest:
            periods[f"{yr} (era start)"] = (yr, yr)
    if args.check:
        periods[f"{CHECK_PERIOD[0]}-{CHECK_PERIOD[1]} (check)"] = CHECK_PERIOD
    for text in args.period or []:
        try:
            label, spec = parse_period(text)
        except ValueError as e:
            p.error(str(e))
        periods[label] = spec
    masks = {label: period_mask(years, spec) for label, spec in periods.items()}
    tm = masks[tune]
    yt = y[tm]

    # ----- window counts, once per window length -----
    base_counts = {b: window_counts(hist, None, b) for b in grids["base_races"]}
    team_counts = {n: window_counts(hist, "constructorId", n) for n in grids["team_races"]}
    drv_counts = {n: window_counts(hist, "driverId", n) for n in grids["driver_races"]}
    bases = {b: base_rate(*c) for b, c in base_counts.items()}

    # ----- 1. the base rate alone -----
    print(f"\n=== Base rate alone: the field's DNF share over its last B races "
          f"(the baseline) ===")
    allb = base_rate(*window_counts(hist, None, None))
    tbl = [{"B races": "all earlier", **{lab: log_loss(allb[m], y[m])
                                         for lab, m in masks.items()}}]
    for b, base in bases.items():
        tbl.append({"B races": b, **{lab: log_loss(base[m], y[m])
                                     for lab, m in masks.items()}})
    tbl = pd.DataFrame(tbl)
    print(tbl.round(4).to_string(index=False))
    best_base_b = int(tbl.iloc[1:].loc[tbl.iloc[1:][tune].idxmin(), "B races"])
    print(f"Actual DNF share: " + ", ".join(f"{lab} {y[m].mean():.3f}"
                                           for lab, m in masks.items()))

    # ----- 2. the full grid (scored on the tuning period only) -----
    ws = np.array(grids["driver_weight"], float)
    combos = (len(bases) * len(grids["team_races"]) * len(grids["team_k"])
              * len(grids["driver_races"]) * len(grids["driver_k"]) * len(ws))
    print(f"\nGrid: {combos:,} combinations on {tm.sum():,} tuning entries ...")
    t0 = time.time()
    rows = []
    for b, base in bases.items():
        bt = base[tm]
        teams = {(n, k): shrunk_rate(team_counts[n][0][tm], team_counts[n][1][tm], bt, k)
                 for n in grids["team_races"] for k in grids["team_k"]}
        drvs = {(n, k): shrunk_rate(drv_counts[n][0][tm], drv_counts[n][1][tm], bt, k)
                for n in grids["driver_races"] for k in grids["driver_k"]}
        for (nt, kt), T in teams.items():
            for (nd, kd), D in drvs.items():
                P = np.clip((1 - ws)[:, None] * T + ws[:, None] * D, 1e-4, 1 - 1e-4)
                ll = -(yt * np.log(P) + (1 - yt) * np.log(1 - P)).mean(axis=1)
                for w, v in zip(ws, ll):
                    rows.append((b, nt, kt, nd, kd, w, v))
    grid = pd.DataFrame(rows, columns=["base_races", "team_races", "team_k",
                                       "driver_races", "driver_k",
                                       "driver_weight", "logloss"])
    print(f"  done in {time.time() - t0:.0f} s")
    out = config.OUTPUT_DIR / "backtest_reliability.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out, index=False)

    best = grid.loc[grid["logloss"].idxmin()]
    B, NT, KT, ND, KD, W = (int(best["base_races"]), int(best["team_races"]),
                            best["team_k"], int(best["driver_races"]),
                            best["driver_k"], float(best["driver_weight"]))
    near = grid[grid["logloss"] <= best["logloss"] + NEAR_BEST]
    print(f"\n=== Best blend on {tune}: B {B}, team {NT} races / k {KT:g}, "
          f"driver {ND} races / k {KD:g}, driver weight {W:g}  "
          f"(log loss {best['logloss']:.4f}) ===")
    print(f"Within {NEAR_BEST} of it: " + ", ".join(
        f"{c} {near[c].min():g}-{near[c].max():g}" for c in grid.columns[:-1]))

    def marginal(r, c):
        return (grid.groupby([r, c])["logloss"].min().unstack(c).round(4))

    print("\nBest log loss for each team window x team k (the rest at their best):")
    print(marginal("team_races", "team_k").to_string())
    print("\nBest log loss for each driver window x driver k:")
    print(marginal("driver_races", "driver_k").to_string())
    print("\nBest log loss for each driver weight x base window:")
    print(marginal("driver_weight", "base_races").to_string())

    # ----- the candidates on every row -----
    base = bases[B]
    team = shrunk_rate(*team_counts[NT], base, KT)
    drv = shrunk_rate(*drv_counts[ND], base, KD)
    p_blend = blend(team, drv, W)
    cand = {
        f"base rate, last {best_base_b} races": bases[best_base_b],
        f"base rate, last {B} races (the blend's)": base,
        f"team only ({NT} races, k {KT:g})": blend(team, drv, 0.0),
        f"driver only ({ND} races, k {KD:g})": blend(team, drv, 1.0),
        f"blend, driver weight {W:g}": p_blend,
    }
    c = cfg
    cfg_base = base_rate(*window_counts(hist, None, c.dnf_base_races))
    cfg_blend = blend(
        shrunk_rate(*window_counts(hist, "constructorId", c.dnf_team_races), cfg_base,
                    c.dnf_team_shrink),
        shrunk_rate(*window_counts(hist, "driverId", c.dnf_driver_races), cfg_base,
                    c.dnf_driver_shrink),
        c.dnf_driver_weight)
    cand["config blend (before the rating step)"] = cfg_blend

    coef = None
    if not args.no_ratings:
        from f1elo.strength import pre_race_strengths
        print("\nReplaying the engine for pre-race ratings ...")
        t0 = time.time()
        rated = attach_ratings(hist, pre_race_strengths(
            None, prepared=prepare_races(df, cfg), cfg=cfg))
        print(f"  {time.time() - t0:.0f} s")
        race = hist["raceId"].to_numpy()
        car_rel = relative_to_field(rated["constructor_elo"], race)
        drv_rel = relative_to_field(rated["driver_elo"], race)
        ones = np.ones(len(hist))
        designs = {
            "blend, recalibrated (a + b logit blend)":
                np.column_stack([ones, logit(p_blend)]),
            "blend + car rating":
                np.column_stack([ones, logit(p_blend), car_rel]),
            "blend + car + driver rating":
                np.column_stack([ones, logit(p_blend), car_rel, drv_rel]),
            f"base ({B}) + car + driver rating, no counts":
                np.column_stack([ones, logit(base), car_rel, drv_rel]),
        }
        fitted = {}
        for name, X in designs.items():
            b = fit_logistic(X[tm], y[tm])
            fitted[name] = b
            cand[name] = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-4, 1 - 1e-4)
        coef = fitted["blend + car + driver rating"]
        cfg_X = np.column_stack([ones, logit(cfg_blend), car_rel, drv_rel])
        cfg_b = np.array([c.dnf_logit_intercept, c.dnf_logit_blend,
                          c.dnf_logit_car, c.dnf_logit_driver])
        cand["config (blend + rating step)"] = np.clip(
            1 / (1 + np.exp(-(cfg_X @ cfg_b))), 1e-4, 1 - 1e-4)

        print("\n=== Logistic coefficients, fitted on the tuning period ===")
        for name, b in fitted.items():
            print(f"  {name:45s} " + "  ".join(f"{v:+.3f}" for v in b))
        print("  (columns: intercept, logit of the blend or base, car_rel, "
              "driver_rel; per 100 rating points)")

        # The same fits on 1990-2009 only, scored on 2010-2022.
        fit_m = period_mask(years, SPLIT[0])
        test_m = period_mask(years, SPLIT[1])
        print(f"\n=== Out-of-sample inside the tuning period: fitted on "
              f"{SPLIT[0][0]}-{SPLIT[0][1]}, scored on {SPLIT[1][0]}-{SPLIT[1][1]} ===")
        tbl = [{"candidate": "base rate (B {})".format(B),
                "fit ll": log_loss(base[fit_m], y[fit_m]),
                "test ll": log_loss(base[test_m], y[test_m])},
               {"candidate": "blend (grid settings)",
                "fit ll": log_loss(p_blend[fit_m], y[fit_m]),
                "test ll": log_loss(p_blend[test_m], y[test_m])}]
        for name, X in designs.items():
            b = fit_logistic(X[fit_m], y[fit_m])
            pr = 1 / (1 + np.exp(-(X @ b)))
            tbl.append({"candidate": name, "fit ll": log_loss(pr[fit_m], y[fit_m]),
                        "test ll": log_loss(pr[test_m], y[test_m])})
        print(pd.DataFrame(tbl).round(4).to_string(index=False))
        print("(the blend's grid settings were themselves chosen on 1990-2022, so "
              "this tests the rating step, not the counts)")

    print("\n=== Candidates (everything fitted on the tuning period only) ===")
    for label, m in masks.items():
        print(f"\n--- {label}: {m.sum():,} entries, {y[m].mean():.3f} retired ---")
        tbl = [{"candidate": name, "logloss": log_loss(pr[m], y[m]),
                "brier": brier(pr[m], y[m]), "mean P(DNF)": float(pr[m].mean())}
               for name, pr in cand.items()]
        tbl = pd.DataFrame(tbl)
        tbl["vs base"] = tbl["logloss"] - tbl["logloss"].iloc[0]
        print(tbl.round(4).to_string(index=False))

    final_name = ("blend + car + driver rating" if coef is not None
                  else f"blend, driver weight {W:g}")
    final = cand[final_name]
    print(f"\n=== Calibration of '{final_name}': predicted vs actual, by band ===")
    bands = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    for label in (tune, *[l for l in periods if "holdout" in l]):
        m = masks[label]
        cut = pd.cut(final[m], bands)
        t = (pd.DataFrame({"band": cut, "p": final[m], "y": y[m]})
             .groupby("band", observed=True)
             .agg(entries=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")))
        print(f"\n{label}:")
        print(t.round(3).to_string())

    print(f"\nBest on {tune}:")
    print(f"    dnf_base_races = {B}, dnf_team_races = {NT}, dnf_team_shrink = {KT:g}, "
          f"dnf_driver_races = {ND},\n    dnf_driver_shrink = {KD:g}, "
          f"dnf_driver_weight = {W:g}")
    if coef is not None:
        print("    dnf_logit_intercept = {:.3f}, dnf_logit_blend = {:.3f}, "
              "dnf_logit_car = {:.3f}, dnf_logit_driver = {:.3f}".format(*coef))
    print(f"In f1elo/config.py:\n    dnf_base_races = {c.dnf_base_races}, "
          f"dnf_team_races = {c.dnf_team_races}, dnf_team_shrink = {c.dnf_team_shrink:g}, "
          f"dnf_driver_races = {c.dnf_driver_races},\n    dnf_driver_shrink = "
          f"{c.dnf_driver_shrink:g}, dnf_driver_weight = {c.dnf_driver_weight:g}\n"
          f"    dnf_logit_intercept = {c.dnf_logit_intercept:.3f}, dnf_logit_blend = "
          f"{c.dnf_logit_blend:.3f}, dnf_logit_car = {c.dnf_logit_car:.3f}, "
          f"dnf_logit_driver = {c.dnf_logit_driver:.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
