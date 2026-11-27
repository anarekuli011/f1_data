"""Backtest the pre-race strengths on whole-field race results.

    py backtest_strength.py
    py backtest_strength.py --car-weight 1.5 2.25 --car-weight-increment 0.125 --scale 1000 1400 --scale-increment 25

(one engine replay, about 15 s; every candidate is then scored in well under
a second, so wide ranges are cheap)

Each car's strength before a race is S = D + w * C (f1elo/strength.py): D the
driver rating, C the constructor rating, both from the engine as they stood
before that race. P(car A finishes ahead of car B) = 1 / (1 + 10^((S_b -
S_a) / s)). Every pair of different drivers in every race is scored --
teammates AND cross-team pairs, unlike backtest_drivers.py (teammates only)
and backtest_constructors.py (cross-team only) -- with the engine's outcome
rules: a classified car beats everyone classified below it and every DNF;
DNF-vs-DNF pairs are skipped. The tables also split the log loss into
cross-team pairs, teammate pairs and pairs where both cars were classified.

The question: the engine's driver_car_weight (0.875) and elo_scale (400)
were tuned for the rating updates, on teammate pairs. Do they still give the
best whole-field predictions? The script compares

    car only        S = C
    driver only     S = D
    D + w*C         w from --car-weight (default 0-3)
    scale s         from --scale (default 300-1600) instead of 400
    fitted          maximum-likelihood w and s (logistic regression)

and picks the best (w, s) on the tuning period by --metric. Options:

    --car-weight MIN MAX    w range, with --car-weight-increment (0.25)
    --scale MIN MAX         s range, with --scale-increment (100)
    --stage pre|post_quali  ratings before the weekend (default) or after
                            that weekend's qualifying pass
    --period 2015-2022      extra seasons to score (repeatable)
    --check                 also score 1950-1979
    --metric logloss|brier  what "best" means (default logloss)
    --pairs all|classified  all pairs (default), or only pairs where both
                            cars were classified: what a race model that
                            handles retirements separately would need

Periods: 1990-2022 is the tuning period: settings are chosen there only.
2023 onwards (holdout), and 2022 and 2026 on their own (the starts of the
ground-effect and 2026 eras, where the car order was reshuffled) are
reported alongside. Lower log loss is better; 0.6931 = coin flip.

Also prints the share of the within-race strength variance that comes from
the car (w^2 Var C) vs the driver (Var D). Bell et al. (2016) put the team at
about 86% of the variance in results, the driver about 14%: a sanity check,
not a target.

Writes the grid to output/backtest_strength.csv. The chosen setting belongs
in prediction_car_weight / prediction_scale in f1elo/config.py (all pairs:
a one-step view where a retirement loses to every finisher), or with
--pairs classified in finish_car_weight / finish_scale (the finishing order
of the race simulation, which draws retirements separately). The engine's
own settings are left alone.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from f1elo import build_race_frame, config
from f1elo.backtest import (CHECK_PERIOD, HOLDOUT_START, TUNE_PERIOD,
                            fit_scale, fit_strength, parse_period, period_mask,
                            prepare_races, score_strength, strength_pairs,
                            value_range, variance_shares)

# Pairs are many (~300k in the tuning period), so differences this small
# are effectively ties.
NEAR_BEST = 0.0005


def main() -> None:
    cfg = config.DEFAULT_CONFIG
    p = argparse.ArgumentParser(description="Backtest pre-race strengths on all pairs.")
    p.add_argument("--car-weight", type=float, nargs="+", metavar=("MIN", "MAX"),
                   default=[0, 3], help="w range for S = D + w*C (default 0 3).")
    p.add_argument("--car-weight-increment", type=float, default=0.25)
    p.add_argument("--scale", type=float, nargs="+", metavar=("MIN", "MAX"),
                   default=[300, 1600], help="Logistic scale range (default 300 1600).")
    p.add_argument("--scale-increment", type=float, default=100)
    p.add_argument("--stage", choices=("pre", "post_quali"), default="pre",
                   help="Ratings before the weekend (default) or after its "
                        "qualifying pass.")
    p.add_argument("--period", action="append",
                   help="Extra seasons to score, e.g. 2015-2022. Repeatable.")
    p.add_argument("--check", action="store_true", help="Also score 1950-1979.")
    p.add_argument("--metric", choices=("logloss", "brier"), default="logloss")
    p.add_argument("--pairs", choices=("all", "classified"), default="all",
                   help="all = the engine's outcome rules, a DNF losing to every "
                        "classified car (default); classified = only pairs where "
                        "both cars were classified.")
    args = p.parse_args()

    try:
        weights = value_range(args.car_weight, args.car_weight_increment)
        scales = value_range(args.scale, args.scale_increment)
    except ValueError as e:
        p.error(str(e))
    if min(scales) <= 0:
        p.error("--scale must be positive")

    print("Preparing races ...")
    prepared = prepare_races(build_race_frame())
    latest = max(r[1] for r in prepared)
    tune = f"{TUNE_PERIOD[0]}-{TUNE_PERIOD[1]} (tune)"
    periods: dict = {tune: TUNE_PERIOD}
    if latest >= HOLDOUT_START:
        periods[f"{HOLDOUT_START}-{latest} (holdout)"] = (HOLDOUT_START, latest)
    for y in (2022, 2026):
        if y <= latest:
            periods[f"{y} (era start)"] = (y, y)
    if args.check:
        periods[f"{CHECK_PERIOD[0]}-{CHECK_PERIOD[1]} (check)"] = CHECK_PERIOD
    for text in args.period or []:
        try:
            label, rng = parse_period(text)
        except ValueError as e:
            p.error(str(e))
        periods[label] = rng
    first = min(v[0] for v in periods.values())

    print(f"Replaying the engine ({args.stage} ratings) ...")
    t0 = time.time()
    pairs, spread = strength_pairs(prepared, cfg, stage=args.stage, start_year=first)
    print(f"  {len(pairs['dD']):,} pairs in {time.time() - t0:.0f} s")
    if args.pairs == "classified":
        keep = ~pairs["dnf"]
        pairs = {k: v[keep] for k, v in pairs.items()}
        print(f"  scoring only the {keep.sum():,} pairs with both cars classified")
    masks = {label: period_mask(pairs["year"], spec) for label, spec in periods.items()}

    def score(label, a, w, s, which=None):
        m = masks[label]
        if which == "cross":
            m = m & ~pairs["mate"]
        elif which == "mates":
            m = m & pairs["mate"]
        elif which == "classified":
            m = m & ~pairs["dnf"]
        return score_strength(pairs["dD"][m], pairs["dC"][m], a, w, s)

    # ----- the (w, s) grid -----
    rows = []
    for w in weights:
        for s in scales:
            for label in periods:
                rows.append({"car_weight": w, "scale": s, "period": label,
                             **score(label, 1.0, w, s)})
    grid = pd.DataFrame(rows)
    out = config.OUTPUT_DIR / "backtest_strength.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out, index=False)

    metric = args.metric
    pd.set_option("display.width", 220)
    t = grid[grid["period"] == tune]
    print(f"\n=== {tune}: {metric}, all {t['pairs'].iloc[0]:,} pairs; "
          "rows = car weight w, columns = scale s ===")
    print(t.pivot(index="car_weight", columns="scale", values=metric)
          .round(4).to_string())
    best = t.loc[t[metric].idxmin()]
    bw, bs = float(best["car_weight"]), float(best["scale"])
    near = t[t[metric] <= best[metric] + NEAR_BEST]
    print(f"Best on the grid: w {bw:g}, s {bs:g}  ({metric} {best[metric]:.4f}). "
          f"Within {NEAR_BEST} of it: w {near['car_weight'].min():g}-"
          f"{near['car_weight'].max():g}, s {near['scale'].min():g}-{near['scale'].max():g}")

    # ----- candidates, fitted on the tuning period -----
    tm = masks[tune]
    dD, dC = pairs["dD"][tm], pairs["dC"][tm]
    fw, fs = fit_strength(dD, dC)
    lam, sc = cfg.driver_car_weight, cfg.elo_scale
    s_car = fit_scale(dC)
    s_drv = fit_scale(dD)
    s_eng = fit_scale(dD + lam * dC)
    w400 = min(weights, key=lambda w: score(tune, 1.0, w, sc)[metric])
    candidates = [
        (f"car only (C), s {sc:g}", 0.0, 1.0, sc),
        (f"car only (C), fitted s {s_car:.0f}", 0.0, 1.0, s_car),
        (f"driver only (D), s {sc:g}", 1.0, 0.0, sc),
        (f"driver only (D), fitted s {s_drv:.0f}", 1.0, 0.0, s_drv),
        (f"engine: D + {lam:g}C, s {sc:g}", 1.0, lam, sc),
        (f"D + {lam:g}C, fitted s {s_eng:.0f}", 1.0, lam, s_eng),
        (f"D + {w400:g}C (best w at s {sc:g})", 1.0, w400, sc),
        (f"D + {bw:g}C, s {bs:g} (best on grid)", 1.0, bw, bs),
        (f"D + {fw:.3f}C, s {fs:.0f} (max. likelihood)", 1.0, fw, fs),
    ]
    pw, ps = cfg.prediction_car_weight, cfg.prediction_scale
    if (pw, ps) not in {(lam, sc), (bw, bs)}:
        candidates.append((f"config prediction: D + {pw:g}C, s {ps:g}", 1.0, pw, ps))
    fw_, fs_ = cfg.finish_car_weight, cfg.finish_scale
    if (fw_, fs_) not in {(lam, sc), (bw, bs), (pw, ps)}:
        candidates.append((f"config finish: D + {fw_:g}C, s {fs_:g}", 1.0, fw_, fs_))

    print(f"\n=== Candidates (w and s fitted on {tune} only) ===")
    for label, spec in periods.items():
        races = len(spread[period_mask(spread["year"].to_numpy(), spec)])
        n_mates = int((masks[label] & pairs["mate"]).sum())
        print(f"\n--- {label}: {races} races, {int(masks[label].sum()):,} pairs "
              f"({n_mates:,} teammate, "
              f"{int((masks[label] & pairs['dnf']).sum()):,} against a DNF) ---")
        tbl = []
        for name, a, w, s in candidates:
            allp = score(label, a, w, s)
            tbl.append({"candidate": name, "logloss": allp["logloss"],
                        "brier": allp["brier"], "accuracy": allp["acc"],
                        "cross-team ll": score(label, a, w, s, "cross")["logloss"],
                        "teammate ll": score(label, a, w, s, "mates")["logloss"],
                        "classified ll": score(label, a, w, s, "classified")["logloss"]})
        print(pd.DataFrame(tbl).round(4).to_string(index=False))

    # ----- the best setting in each period on its own (stability) -----
    print("\n=== Best setting fitted on each period by itself (stability only; "
          "the choice uses the tuning period) ===")
    tbl = []
    for label in periods:
        m = masks[label]
        w, s = fit_strength(pairs["dD"][m], pairs["dC"][m])
        r = grid[grid["period"] == label]
        g = r.loc[r[metric].idxmin()]
        tbl.append({"period": label, "fitted w": w, "fitted s": s,
                    "grid w": g["car_weight"], "grid s": g["scale"],
                    f"grid best {metric}": g[metric],
                    f"tune choice {metric}": score(label, 1.0, bw, bs)[metric],
                    f"engine {metric}": score(label, 1.0, lam, sc)[metric]})
    # The same fit without the pairs a DNF decided: how much of the flatter
    # scale is retirements (which ratings can barely predict).
    m = masks[tune] & ~pairs["dnf"]
    w, s = fit_strength(pairs["dD"][m], pairs["dC"][m])
    tbl.append({"period": f"{tune}, both classified", "fitted w": w, "fitted s": s})
    print(pd.DataFrame(tbl).round(4).to_string(index=False))

    # ----- car vs driver share of the strength spread -----
    print(f"\n=== Within-race strength variance, car vs driver (w = {bw:g}; "
          "Bell et al. 2016: team ~86%, driver ~14%) ===")
    tbl = []
    for label, spec in periods.items():
        v = variance_shares(spread, spec, bw)
        if v:
            tbl.append({"period": label, **v})
    print(pd.DataFrame(tbl).round(3).to_string(index=False))
    print("car_share = w^2 Var(C) / (Var(D) + w^2 Var(C)); cov_share = share of "
          "Var(S) from the D-C covariance;\ncorr_dc = correlation of D and C "
          "within races.")

    names = (("finish_car_weight", "finish_scale", fw_, fs_)
             if args.pairs == "classified" else
             ("prediction_car_weight", "prediction_scale", pw, ps))
    print(f"\nBest on {tune}: {names[0]} = {bw:g}, {names[1]} = {bs:g}")
    print(f"In f1elo/config.py:    {names[0]} = {names[2]:g}, "
          f"{names[1]} = {names[3]:g}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
