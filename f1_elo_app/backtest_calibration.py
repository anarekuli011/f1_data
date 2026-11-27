"""Backtest the race predictions: how far can the win, podium and points
chances be trusted, and do they beat fair baselines?

    py backtest_calibration.py                       # pre-race: scores, baselines, calibration, experiments
    py backtest_calibration.py --stage post_quali    # after qualifying, vs results by grid slot
    py backtest_calibration.py --stage both --experiments none
    py backtest_calibration.py --experiments finish --top-scale 300 500 --top-scale-increment 25

(The first run replays the engine once per stage, about 15 s, and
simulates every race once, about a minute. Both are cached in output/, so
reruns start in seconds. The experiments re-simulate the scored races for
every setting tried: about 8 minutes for the pre-race stage with the
default ranges (most of it the finish grid), 1 minute after qualifying;
--experiments none, or picking some, or narrower ranges is much quicker.)

What is predicted
-----------------
For every race from 1990 on, the race simulation (f1elo/simulate.py) gives
each car its chance of every finishing position and of a DNF. Everything is
from information available beforehand: the engine's ratings before the race
(--stage pre) or after that weekend's qualifying pass (--stage post_quali),
and retirement counts from earlier races. The engine is replayed ONCE and
each race is simulated from those ratings (f1elo/calibration.py), not one
replay per race.

Scores (lower is better)
------------------------
  win ll        -ln P(the actual winner wins), per race
  podium ll/br  log loss / Brier of "finished in the top 3", per car
  points ll/br  the same for "finished in the season's points places"
  rps           ranked probability score over P1 ... Pn then DNF, per car
  dnf ll        log loss of "did not finish", per car
  fin ll        -ln P(the actual number of finishers), per race
  fin var       actual (finishers - expected)^2 over the predicted variance;
                1 = the spread of finishers per race is right
Simulated chances are floored at 0.5 / sims before taking logs (5e-05 at
10,000 simulations); the win chance is computed exactly for each simulated
set of finishers, so it needs no floor. Confidence intervals are 95%, from
resampling races (2,000 times); a baseline's interval is for its
difference from the model, resampled in pairs.

Baselines (the same information as the model they are compared with)
--------------------------------------------------------------------
Pre-race:
  uniform               every car alike, retiring at the field's base rate
  base DNF + strengths  the model's finishing order, but every car retiring
                        at the base rate (is the retirement model worth it?)
  all pairs 1.875/1175  one Plackett-Luce order over all cars on the
                        all-pairs strengths, no separate retirement model;
                        the cars ranked below the number of finishers
                        (drawn from the base rate) retire
  strength rank         the historical results of the k-th strongest car
                        (1990-2022), the DNF chance shifted to the base rate
After qualifying (compared ONLY with the after-qualifying model):
  grid slot             the historical results from each grid slot
                        (1990-2022), the DNF chance shifted to the base rate
The two rank tables are fitted on 1990-2022 itself, so on that period they
are scored in-sample (a small advantage to them); the holdout is fair.

Experiments (each on top of the settings in f1elo/config.py)
------------------------------------------------------------
  corr      shared race factor for retirements (dnf_race_corr), fitted to
            the variance of the number of finishers (exact distributions,
            no simulation), then the race scores with and without it
  finish    the finishing order, tuned on whole results: the exact
            Plackett-Luce likelihood of the finishing orders (whole order,
            top 3, winner), then simulated win / podium / points / RPS for
            a single scale vs a sharper scale for the first places
            (finish_top_scale / finish_top_places)
  newcomer  extra spread for drivers with low confidence
            (finish_newcomer_spread), judged on rookies (first 20 starts)
  grid-dnf  after qualifying: the grid slot in the retirement chance
            (lap-1 crashes in the midfield, pit-lane starts)
  era       how well the retirement chance is calibrated in the first races
            of a new regulation era

Options
-------
  --stage pre|post_quali|both   (default pre)
  --sims N, --seed N            simulations per race for the scored model
                                (10000, seed 0)
  --grid-sims N                 simulations per race in the experiment grids
                                (2000; every setting uses the same random
                                numbers, so differences are smooth)
  --period 2015-2022            extra seasons to score (repeatable)
  --check                       also score 1950-1979 (no Indianapolis 500)
  --baselines all|none|LIST     comma-separated: uniform,base,allpairs,rank,grid
  --experiments all|none|LIST   comma-separated: corr,finish,newcomer,grid-dnf,era
  --corr MIN MAX                correlation grid (0 0.1, increment 0.01)
  --car-weight MIN MAX          finish car weight grid (config +- 0.125)
  --top-scale MIN MAX           top scale grid (350 450, increment 25)
  --top-places N [N ...]        top places to try (1 2 3 4)
  --single-scale MIN MAX        single-scale reference grid (300 650, increment 50)
  --finish-metric M             what "best" means in the finish experiment:
                                podium_ll (default), win_ll, points_ll or rps
  --spread MIN MAX              newcomer spread grid (0 200, increment 50)
  --dnf-corr X, --finish-car-weight X, --finish-scale X,
  --finish-top-scale X, --finish-top-places N, --newcomer-spread X
                                evaluate the model with these settings
                                instead of the config's
  --boot N                      bootstrap resamples (2000)
  --rebuild                     ignore the caches
  --no-save                     don't write the tables

Periods: settings are chosen on 1990-2022 (tune) only; 2023 onwards
(holdout) is reported alongside. 2023+ has fewer than 100 winners, so lean
on podium and points there.

Writes to output/ (for the app's calibration page):
  race_predictions_<stage>.parquet   the model's table for every race (cache)
  calibration_scores_<stage>.csv     every predictor x period x score, with CIs
  calibration_bands_<stage>.csv      predicted vs actual frequency by band:
                                     win, podium, points (per era), DNF
  calibration_finishers_<stage>.csv  races with k retirements, predicted vs actual
  calibration_experiments_<stage>.csv  the experiment grids
The chosen settings belong in f1elo/config.py (finish_*, dnf_race_corr).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from f1elo import build_race_frame, config
from f1elo import calibration as cal
from f1elo.backtest import (CHECK_PERIOD, HOLDOUT_START, TUNE_PERIOD,
                            parse_period, value_range)
from f1elo.reliability import log_loss

BASELINES = ("uniform", "base", "allpairs", "rank", "grid")
EXPERIMENTS = ("corr", "finish", "newcomer", "grid-dnf", "era")
SHOW = ["win_ll", "podium_ll", "podium_brier", "points_ll", "points_brier", "rps",
        "dnf_ll", "fin_ll", "fin_var_ratio"]
HEAD = {"win_ll": "win ll", "podium_ll": "podium ll", "podium_brier": "podium br",
        "points_ll": "points ll", "points_brier": "points br", "rps": "rps",
        "dnf_ll": "dnf ll", "fin_ll": "fin ll", "fin_var_ratio": "fin var"}
ROOKIE_STARTS = 20


def fmt(t: pd.DataFrame, digits: int = 4) -> str:
    return t.round(digits).to_string(index=False)


def score_table(fields, preds: dict, masks: dict, boot: int, model: str = "model"):
    """Scores of every predictor on every period, with bootstrap CIs (the
    model's own, and each baseline's difference from the model)."""
    rows, per = [], {}
    for label, m in masks.items():
        for name, p in preds.items():
            per[(name, label)] = cal.race_scores(fields, p, races=m)
    for label in masks:
        base = per[(model, label)]
        for name in preds:
            r = per[(name, label)]
            s = cal.summarise_scores(r)
            for k in SHOW:
                if k not in s:
                    continue
                row = {"predictor": name, "period": label, "metric": k, "value": s[k],
                       "races": s["races"], "entries": s["entries"]}
                if boot:
                    row["ci_lo"], row["ci_hi"] = cal.bootstrap(r, k, reps=boot,
                                                               level=0.95)
                    if name != model and k in base:
                        row["diff"] = s[k] - cal.summarise_scores(base)[k]
                        row["diff_lo"], row["diff_hi"] = cal.bootstrap(
                            r, k, base, reps=boot, level=0.95)
                rows.append(row)
    return pd.DataFrame(rows), per


def print_scores(scores: pd.DataFrame, label: str, model: str = "model") -> None:
    t = scores[scores["period"] == label]
    wide = t.pivot(index="predictor", columns="metric", values="value")
    wide = wide[[k for k in SHOW if k in wide]].rename(columns=HEAD)
    wide = wide.loc[[p for p in dict.fromkeys(t["predictor"])]]
    r = t.iloc[0]
    print(f"\n--- {label}: {int(r['races'])} races, {int(r['entries']):,} cars ---")
    print(wide.round(4).to_string())
    if "ci_lo" not in t:
        return
    m = t[t["predictor"] == model].set_index("metric")
    print(f"95% intervals, {model}: " + "; ".join(
        f"{HEAD[k]} {m.loc[k, 'ci_lo']:.4f}-{m.loc[k, 'ci_hi']:.4f}"
        for k in ("win_ll", "podium_ll", "points_ll", "rps") if k in m.index))
    for name in dict.fromkeys(t["predictor"]):
        if name == model:
            continue
        d = t[t["predictor"] == name].set_index("metric")
        bits = []
        for k in ("win_ll", "podium_ll", "points_ll", "rps"):
            if k in d.index and not pd.isna(d.loc[k].get("diff", np.nan)):
                lo, hi = d.loc[k, "diff_lo"], d.loc[k, "diff_hi"]
                mark = " (model better)" if lo > 0 else (" (baseline better)" if hi < 0 else "")
                bits.append(f"{HEAD[k]} {d.loc[k, 'diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]{mark}")
        if bits:
            print(f"  {name} minus model: " + "; ".join(bits))


def band_tables(fields, preds: dict, masks: dict) -> pd.DataFrame:
    """Calibration bands for every predictor / period / outcome; points per
    points-places era (top 6 / 8 / 10)."""
    out = []
    for name, p in preds.items():
        ch = cal.entry_chances(fields, p)
        for label, m in masks.items():
            rows = m[fields["race_no"].to_numpy()]
            c = ch[rows]
            for o in cal.OUTCOMES:
                groups = [("all", np.ones(len(c), bool))]
                if o == "points":
                    groups = [(g, (cal.points_era(c["points_positions"]) == g))
                              for g in sorted(set(cal.points_era(c["points_positions"])),
                                              key=lambda s: int(s.split()[1]))]
                for g, gm in groups:
                    if not gm.any():
                        continue
                    t = cal.calibration_bands(c[o][gm], c[f"y_{o}"][gm])
                    t.insert(0, "group", g)
                    t.insert(0, "outcome", o)
                    t.insert(0, "period", label)
                    t.insert(0, "predictor", name)
                    out.append(t)
    return pd.concat(out, ignore_index=True)


def print_bands(bands: pd.DataFrame, name: str, label: str) -> None:
    t = bands[(bands["predictor"] == name) & (bands["period"] == label)]
    print(f"\n--- {name}, {label}: predicted vs actual frequency by band "
          "(z = (actual - predicted) / se) ---")
    for (o, g), b in t.groupby(["outcome", "group"], sort=False):
        b = b.assign(band=[f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(b["band_lo"], b["band_hi"])])
        tot = (b["entries"] * b["predicted"]).sum(), (b["entries"] * b["actual"]).sum()
        what = o if g == "all" else f"{o} ({g})"
        print(f"{what}: predicted {tot[0]:.0f}, actual {tot[1]:.0f} "
              f"of {int(b['entries'].sum()):,}")
        print("   " + fmt(b[["band", "entries", "predicted", "actual", "z"]], 3)
              .replace("\n", "\n   "))


def main() -> None:
    cfg0 = config.DEFAULT_CONFIG
    p = argparse.ArgumentParser(description="Backtest the race predictions' calibration.")
    p.add_argument("--stage", choices=("pre", "post_quali", "both"), default="pre")
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-sims", type=int, default=2000)
    p.add_argument("--period", action="append", help="Extra seasons, e.g. 2015-2022.")
    p.add_argument("--check", action="store_true", help="Also score 1950-1979.")
    p.add_argument("--baselines", default="all")
    p.add_argument("--experiments", default="all")

    def rng(name, default, inc):
        p.add_argument(f"--{name}", type=float, nargs="+", metavar=("MIN", "MAX"),
                       default=default)
        p.add_argument(f"--{name}-increment", type=float, default=inc)

    rng("corr", [0, 0.1], 0.01)
    rng("car-weight", None, 0.125)
    rng("top-scale", [350, 450], 25)
    rng("single-scale", [300, 650], 50)
    rng("spread", [0, 200], 50)
    p.add_argument("--top-places", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--finish-metric", default="podium_ll",
                   choices=("podium_ll", "win_ll", "points_ll", "rps"))
    p.add_argument("--dnf-corr", type=float)
    p.add_argument("--finish-car-weight", type=float)
    p.add_argument("--finish-scale", type=float)
    p.add_argument("--finish-top-scale", type=float)
    p.add_argument("--finish-top-places", type=int)
    p.add_argument("--newcomer-spread", type=float)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    def pick(text, allowed, what):
        if text == "all":
            return set(allowed)
        if text == "none":
            return set()
        got = {s.strip() for s in text.split(",") if s.strip()}
        bad = got - set(allowed)
        if bad:
            p.error(f"unknown {what}: {', '.join(sorted(bad))} (choose from {', '.join(allowed)})")
        return got

    baselines = pick(args.baselines, BASELINES, "baseline")
    experiments = pick(args.experiments, EXPERIMENTS, "experiment")
    over = {k: v for k, v in {
        "dnf_race_corr": args.dnf_corr, "finish_car_weight": args.finish_car_weight,
        "finish_scale": args.finish_scale, "finish_top_scale": args.finish_top_scale,
        "finish_top_places": args.finish_top_places,
        "finish_newcomer_spread": args.newcomer_spread}.items() if v is not None}
    cfg = replace(cfg0, **over)
    try:
        grids = {
            "corr": value_range(args.corr, args.corr_increment),
            "car_weight": value_range(args.car_weight or [cfg.finish_car_weight - 0.125,
                                                          cfg.finish_car_weight + 0.125],
                                      args.car_weight_increment),
            "top_scale": value_range(args.top_scale, args.top_scale_increment),
            "single_scale": value_range(args.single_scale, args.single_scale_increment),
            "spread": value_range(args.spread, args.spread_increment),
        }
    except ValueError as e:
        p.error(str(e))
    if max(grids["corr"]) >= 1:
        p.error("--corr must be below 1")
    if args.sims < 100 or args.grid_sims < 100:
        p.error("--sims and --grid-sims need at least 100")

    print("Loading results ...")
    df = build_race_frame()
    latest = int(df["year"].max())
    tune = f"{TUNE_PERIOD[0]}-{TUNE_PERIOD[1]} (tune)"
    hold = f"{HOLDOUT_START}-{latest} (holdout)"
    periods = {tune: TUNE_PERIOD}
    if latest >= HOLDOUT_START:
        periods[hold] = (HOLDOUT_START, latest)
    if args.check:
        periods[f"{CHECK_PERIOD[0]}-{CHECK_PERIOD[1]} (check)"] = CHECK_PERIOD
    for text in args.period or []:
        try:
            label, spec = parse_period(text)
        except ValueError as e:
            p.error(str(e))
        periods[label] = spec

    stages = ("pre", "post_quali") if args.stage == "both" else (args.stage,)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    print(f"Model: strength D + {cfg.finish_car_weight:g} C; finishing order "
          f"Plackett-Luce at scale {cfg.finish_scale:g}"
          + (f", the first {cfg.finish_top_places} places at {cfg.finish_top_scale:g}"
             if cfg.finish_top_places else "")
          + f"; shared retirement factor {cfg.dnf_race_corr:g}; newcomer spread "
          f"{cfg.finish_newcomer_spread:g}. {args.sims:,} simulations per race, "
          f"seed {args.seed}; log-loss floor {cal.floor(args.sims):.0e}.")
    for stage in stages:
        run_stage(stage, cfg, df, periods, tune, hold, baselines, experiments,
                  grids, args)


def run_stage(stage, cfg, df, periods, tune, hold, baselines, experiments, grids, args):
    t0 = time.time()
    title = "PRE-RACE" if stage == "pre" else "AFTER QUALIFYING"
    print(f"\n{'=' * 30} {title} (ratings: {stage}) {'=' * 30}")
    fields = cal.build_fields(stage, cfg, df, cache=not args.rebuild, progress=True)
    masks = {label: cal.race_mask(fields, spec) for label, spec in periods.items()}
    scored = np.logical_or.reduce(list(masks.values()))

    # ----- the model, from the cache of every race -----
    table = cal.cached_predictions(stage, cfg, args.sims, args.seed, df, fields,
                                   rebuild=args.rebuild, progress=True)
    p_model = cal.stage_dnf(fields, stage, cfg)
    model = cal.pred_from_table(fields, table, cal.exact_finishers(
        fields, p_model, cfg.dnf_race_corr, scored), args.sims)
    preds = {"model": model}
    tune_m = masks[tune]

    # ----- baselines -----
    base_rate = fields["base_rate"].to_numpy(float)
    if stage == "pre":
        if "uniform" in baselines:
            preds["uniform"] = cal.uniform_pred(fields, scored)
        if "base" in baselines:
            b = cal.model_pred(fields, cfg, args.sims, args.seed, scored, p_dnf=base_rate)
            b.fin = cal.exact_finishers(fields, base_rate, cfg.dnf_race_corr, scored)
            preds["base DNF + strengths"] = b
        if "allpairs" in baselines:
            preds[f"all pairs {cfg.prediction_car_weight:g}/{cfg.prediction_scale:g}"] = \
                cal.all_pairs_pred(fields, cfg, args.sims, args.seed, scored)
        if "rank" in baselines:
            rk = cal.strength_rank(fields, cfg.finish_car_weight)
            preds["strength rank"] = cal.rank_table_pred(
                fields, rk, cal.rank_table(fields, rk, tune_m), scored)
    elif "grid" in baselines:
        gr = cal.grid_rank(fields)
        preds["grid slot"] = cal.rank_table_pred(fields, gr, cal.rank_table(fields, gr, tune_m),
                                                 scored)
    print(f"  predictions ready [{time.time() - t0:.0f} s]")

    print(f"\n=== Model vs baselines ({title.lower()}) ===")
    scores, per = score_table(fields, preds, masks, args.boot)
    for label in masks:
        print_scores(scores, label)

    print(f"\n=== Calibration of the model ({title.lower()}) ===")
    bands = band_tables(fields, preds, masks)
    for label in (tune, hold):
        if label in masks:
            print_bands(bands, "model", label)

    print("\n=== Number of finishers per race ===")
    fins = []
    for label, m in masks.items():
        s = cal.summarise_scores(per[("model", label)])
        r = per[("model", label)]
        print(f"{label}: mean actual {r['finishers'].mean():.2f}, expected "
              f"{r['fin_expected'].mean():.2f}; variance of actual - expected "
              f"{r['fin_sq_err'].mean():.2f} vs predicted {r['fin_var'].mean():.2f} "
              f"(ratio {s['fin_var_ratio']:.2f}); fin ll {s['fin_ll']:.4f}")
        ft = cal.finishers_table(fields, model, m)
        ft.insert(0, "period", label)
        fins.append(ft)
    fins = pd.concat(fins, ignore_index=True)
    print("Races with k retirements, actual vs predicted (tuning period):")
    print("   " + fmt(fins[fins["period"] == tune].drop(columns="period"), 1)
          .replace("\n", "\n   "))

    exp_rows = []
    if stage == "pre":
        if "corr" in experiments:
            exp_rows += experiment_corr(fields, cfg, masks, tune, hold, grids, args)
        if "finish" in experiments:
            exp_rows += experiment_finish(fields, cfg, masks, tune, hold, grids, args)
        if "newcomer" in experiments:
            exp_rows += experiment_newcomer(fields, cfg, masks, tune, hold, grids, args)
        if "era" in experiments:
            experiment_era(fields, cfg, masks)
    elif "grid-dnf" in experiments:
        exp_rows += experiment_grid_dnf(fields, cfg, masks, tune, hold, args)

    if not args.no_save:
        out = config.OUTPUT_DIR
        out.mkdir(parents=True, exist_ok=True)
        scores.insert(0, "stage", stage)
        scores.to_csv(out / f"calibration_scores_{stage}.csv", index=False)
        bands.insert(0, "stage", stage)
        bands.to_csv(out / f"calibration_bands_{stage}.csv", index=False)
        fins.insert(0, "stage", stage)
        fins.to_csv(out / f"calibration_finishers_{stage}.csv", index=False)
        written = [f"calibration_scores_{stage}.csv", f"calibration_bands_{stage}.csv",
                   f"calibration_finishers_{stage}.csv"]
        if exp_rows:
            pd.DataFrame(exp_rows).to_csv(out / f"calibration_experiments_{stage}.csv",
                                          index=False)
            written.append(f"calibration_experiments_{stage}.csv")
        print(f"\nWrote output/{', output/'.join(written)} and the prediction cache "
              f"output/race_predictions_{stage}.parquet")
    print(f"[{stage}: {time.time() - t0:.0f} s]")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def _scores(fields, pred, masks) -> dict:
    return {label: cal.summarise_scores(cal.race_scores(fields, pred, races=m))
            for label, m in masks.items()}


def experiment_corr(fields, cfg, masks, tune, hold, grids, args) -> list:
    print("\n=== Experiment 1: shared race factor for retirements (dnf_race_corr) ===")
    print("Exact distribution of the number of finishers per race for each "
          "correlation; var ratio 1 = the right spread.")
    rows = []
    fits = {}
    for label in (tune, hold):
        if label not in masks:
            continue
        t = cal.finishers_fit(fields, grids["corr"], masks[label])
        fits[label] = t
        print(f"\n{label}:")
        print(fmt(t, 4))
        for r in t.to_dict("records"):
            rows.append({"experiment": "corr", "period": label, **r})
    t = fits[tune]
    v = t["var_ratio"].to_numpy()
    c = t["dnf_race_corr"].to_numpy(float)
    hit = np.flatnonzero((v[:-1] - 1) * (v[1:] - 1) <= 0)
    if len(hit):
        i = hit[0]
        fitted = c[i] + (c[i + 1] - c[i]) * (v[i] - 1) / (v[i] - v[i + 1])
        print(f"\nVariance matched on {tune} at correlation {fitted:.3f}; best fin ll at "
              f"{c[t['fin_ll'].idxmin()]:g}. Config: {cfg.dnf_race_corr:g}.")
    # Full simulations here: the two settings draw different random numbers,
    # so at 2,000 simulations Monte Carlo noise (~0.0005 in podium log loss
    # on the holdout) would be as large as the effect.
    sims = args.sims
    scored = np.logical_or.reduce([masks[l] for l in (tune, hold) if l in masks])
    print(f"\nRace scores with and without it ({sims:,} simulations per race; the "
          "two use different random numbers, so differences under ~0.0005 are noise):")
    out = []
    for corr in dict.fromkeys([0.0, cfg.dnf_race_corr]):
        pr = cal.model_pred(fields, replace(cfg, dnf_race_corr=corr), sims, args.seed, scored)
        pr.fin = cal.exact_finishers(fields, fields["p_dnf"], corr, scored)
        for label, s in _scores(fields, pr, {k: masks[k] for k in (tune, hold)
                                             if k in masks}).items():
            out.append({"dnf_race_corr": corr, "period": label,
                        **{HEAD[k]: s[k] for k in SHOW if k in s}})
    print(fmt(pd.DataFrame(out)))
    return rows


def experiment_finish(fields, cfg, masks, tune, hold, grids, args) -> list:
    print("\n=== Experiment 2: the finishing order, tuned on whole results ===")
    labels = [l for l in (tune, hold) if l in masks]
    rows = []
    ws = grids["car_weight"]
    ss = grids["single_scale"]
    print("Exact Plackett-Luce loss of the classified cars' actual order (mean "
          "-log-likelihood per race), single scale; rows = car weight, columns = scale:")
    for top, what in ((None, "whole classified order"), (3, "top 3 places"),
                      (1, "winner among the finishers")):
        for label in labels:
            g = pd.DataFrame([[cal.pl_order_loglik(fields, w, s, masks[label], top)
                               for s in ss] for w in ws], index=ws, columns=ss)
            i, j = np.unravel_index(np.argmin(g.values), g.shape)
            print(f"\n{what}, {label}: best w {ws[i]:g}, scale {ss[j]:g}")
            print(g.round(4).to_string())
            for w in ws:
                for s in ss:
                    rows.append({"experiment": "finish_pl", "period": label,
                                 "places": top or "all", "car_weight": w, "scale": s,
                                 "pl_loss": g.loc[w, s]})
    # Simulated: a single scale, then a sharper scale for the first places.
    sims = args.grid_sims
    scored = np.logical_or.reduce([masks[l] for l in labels])
    use = {l: masks[l] for l in labels}
    combos = [(w, s, s, 0) for w in ws for s in ss]
    combos += [(w, cfg.finish_scale, ts, k) for w in ws for ts in grids["top_scale"]
               for k in args.top_places]
    cur = (cfg.finish_car_weight, cfg.finish_scale,
           cfg.finish_top_scale if cfg.finish_top_places else cfg.finish_scale,
           cfg.finish_top_places)
    if cur not in combos:
        combos.append(cur)
    print(f"\nSimulated scores, {len(combos)} settings x {sims:,} simulations per race "
          "(rest scale = finish_scale for the two-scale rows) ...", flush=True)
    t0 = time.time()
    sim = []
    for w, s, ts, k in combos:
        c = replace(cfg, finish_car_weight=w, finish_scale=s, finish_top_scale=ts,
                    finish_top_places=k)
        pr = cal.model_pred(fields, c, sims, args.seed, scored)
        for label, sc in _scores(fields, pr, use).items():
            sim.append({"car_weight": w, "scale": s, "top_scale": ts if k else np.nan,
                        "top_places": k, "period": label,
                        **{m: sc[m] for m in ("win_ll", "podium_ll", "points_ll", "rps")}})
    sim = pd.DataFrame(sim)
    print(f"  {time.time() - t0:.0f} s")
    metric = args.finish_metric
    st = sim[sim["period"] == tune]
    single = st[st["top_places"] == 0]
    for m in ("win_ll", "podium_ll", "points_ll", "rps"):
        print(f"\nSingle scale, {HEAD[m]} on {tune}; rows = car weight, columns = scale:")
        print(single.pivot(index="car_weight", columns="scale", values=m)
              .round(4).to_string())
    two = st[st["top_places"] > 0].sort_values(metric)
    print(f"\nSharper scale for the first places (rest at {cfg.finish_scale:g}), best 15 "
          f"by {HEAD[metric]} on {tune}:")
    print(fmt(two.head(15).drop(columns="period")))
    best = st.loc[st[metric].idxmin()]
    key = ["car_weight", "scale", "top_places"]
    print(f"\nBest by {HEAD[metric]} on {tune}: w {best['car_weight']:g}, scale "
          f"{best['scale']:g}" + (f", first {int(best['top_places'])} places at "
                                  f"{best['top_scale']:g}" if best["top_places"] else ""))

    def match(r):
        same = (sim[key] == r[key].values).all(axis=1)
        if r["top_places"]:
            same &= sim["top_scale"] == r["top_scale"]
        return sim[same]

    comp = [("current config", sim[(sim["car_weight"] == cur[0]) & (sim["scale"] == cur[1])
                                    & (sim["top_places"] == cur[3])
                                    & ((sim["top_scale"] == cur[2]) if cur[3]
                                       else sim["top_scale"].isna())]),
            ("step 3 (1.25 / 525, one scale)",
             sim[(sim["car_weight"] == 1.25) & (sim["scale"] == 525)
                 & (sim["top_places"] == 0)]),
            (f"best single scale by {HEAD[metric]}",
             match(single.loc[single[metric].idxmin()])),
            (f"best by {HEAD[metric]}", match(best))]
    out = []
    for name, t in comp:
        for r in t.to_dict("records"):
            out.append({"setting": name, **r})
    print("\nOn every period:")
    print(fmt(pd.DataFrame(out)))
    rows += [{"experiment": "finish_sim", **r} for r in sim.to_dict("records")]
    return rows


def experiment_newcomer(fields, cfg, masks, tune, hold, grids, args) -> list:
    print("\n=== Experiment 3: extra spread for newcomers (finish_newcomer_spread) ===")
    print(f"Spread = setting x (1 - driver confidence) rating points per simulated race. "
          f"Rookies = a driver's first {ROOKIE_STARTS} starts.")
    labels = [l for l in (tune, hold) if l in masks]
    scored = np.logical_or.reduce([masks[l] for l in labels])
    rookie = fields["starts_before"].to_numpy() < ROOKIE_STARTS
    rows = []
    for sp in grids["spread"]:
        pr = cal.model_pred(fields, replace(cfg, finish_newcomer_spread=sp),
                            args.grid_sims, args.seed, scored)
        ch = cal.entry_chances(fields, pr)
        for label in labels:
            m = masks[label]
            allsc = cal.summarise_scores(cal.race_scores(fields, pr, races=m))
            rk = cal.summarise_scores(cal.race_scores(fields, pr, races=m, entries=rookie))
            sel = m[fields["race_no"].to_numpy()] & rookie
            rows.append({"experiment": "newcomer", "spread": sp, "period": label,
                         "win_ll": allsc["win_ll"], "podium_ll": allsc["podium_ll"],
                         "points_ll": allsc["points_ll"], "rps": allsc["rps"],
                         "rookies": int(sel.sum()),
                         "rookie_podium_ll": rk["podium_ll"],
                         "rookie_points_ll": rk["points_ll"], "rookie_rps": rk["rps"],
                         "rookie_win_brier": rk["win_brier"],
                         "rookie_podium_pred": float(ch["podium"][sel].sum()),
                         "rookie_podium_actual": int(ch["y_podium"][sel].sum()),
                         "rookie_points_pred": float(ch["points"][sel].sum()),
                         "rookie_points_actual": int(ch["y_points"][sel].sum())})
    t = pd.DataFrame(rows)
    for label in labels:
        print(f"\n{label}:")
        print(fmt(t[t["period"] == label].drop(columns=["experiment", "period"])))
    return rows


def experiment_grid_dnf(fields, cfg, masks, tune, hold, args) -> list:
    print("\n=== Experiment 4: the grid slot in the retirement chance (after qualifying) ===")
    print("logit P(DNF) = logit(model) + a + b1 slot + b2 slot^2 + b3 pit lane, slot = "
          "(grid - 1) / (n - 1) (0 = pole, 1 = last; grid 0 counts as last); fitted on "
          "the tuning period. Grid 0 also covers cars that withdrew before the start "
          "(28 of 76 since 1990 never did a lap), which isn't fairly known after "
          "qualifying, so the pit-lane flag is shown but not trusted.")
    rows_of = fields["race_no"].to_numpy()
    tm = masks[tune][rows_of]
    y = fields["dnf"].to_numpy()
    no_g0 = fields["grid"].to_numpy() > 0
    variants = {"slot": (True, False, False), "slot + slot^2": (True, True, False),
                "slot + slot^2 + pit lane": (True, True, True)}
    coefs = {name: cal.fit_grid_dnf(fields, tm, terms) for name, terms in variants.items()}
    cfg_coef = np.array([cfg.dnf_grid_a, cfg.dnf_grid_b, cfg.dnf_grid_c, cfg.dnf_grid_pitlane])
    if cfg_coef.any():
        coefs["config"] = cfg_coef
    rows, t = [], []
    for name, coef in coefs.items():
        adj = cal.grid_adjusted_dnf(fields, coef)
        row = {"variant": name, "a": coef[0], "b1": coef[1], "b2": coef[2], "b3": coef[3]}
        for label, m in masks.items():
            r = m[rows_of]
            row[f"{label} ll"] = log_loss(adj[r], y[r])
            row[f"{label} ll, grid > 0"] = log_loss(adj[r & no_g0], y[r & no_g0])
        t.append(row)
        rows.append({"experiment": "grid-dnf", **row})
    base = {"variant": "model (no grid)", "a": 0.0, "b1": 0.0, "b2": 0.0, "b3": 0.0}
    for label, m in masks.items():
        r = m[rows_of]
        base[f"{label} ll"] = log_loss(fields["p_dnf"][r], y[r])
        base[f"{label} ll, grid > 0"] = log_loss(fields["p_dnf"][r & no_g0], y[r & no_g0])
    print("DNF log loss:")
    print(fmt(pd.DataFrame([base] + t)))

    chosen = "config" if "config" in coefs else "slot"
    adj = cal.grid_adjusted_dnf(fields, coefs[chosen])
    slot = cal.grid_features(fields)[:, 0]
    for label, m in masks.items():
        r = m[rows_of]
        bins = pd.cut(slot[r], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
        g = (pd.DataFrame({"slot": bins, "model": fields["p_dnf"].to_numpy()[r],
                           f"with grid ({chosen})": adj[r], "actual": y[r]})
             .groupby("slot", observed=True).mean())
        print(f"\n{label}: mean P(DNF) by grid slot (0 = pole, 1 = last):")
        print("   " + g.round(3).to_string().replace("\n", "\n   "))
    labels = [l for l in (tune, hold) if l in masks]
    scored = np.logical_or.reduce([masks[l] for l in labels])
    out = []
    for name, q in (("model (no grid)", fields["p_dnf"].to_numpy(float)),
                    (f"with grid ({chosen})", adj)):
        pr = cal.model_pred(fields, cfg, args.grid_sims, args.seed, scored, p_dnf=q)
        pr.fin = cal.exact_finishers(fields, q, cfg.dnf_race_corr, scored)
        for label, s in _scores(fields, pr, {l: masks[l] for l in labels}).items():
            out.append({"retirements": name, "period": label,
                        **{HEAD[k]: s[k] for k in SHOW if k in s}})
    print(f"\nRace scores ({args.grid_sims:,} simulations per race):")
    print(fmt(pd.DataFrame(out)))
    return rows


def experiment_era(fields, cfg, masks) -> None:
    print("\n=== Experiment 5: retirements at the start of a regulation era ===")
    k = cfg.era_start_races
    years = [y for y in cfg.regulation_change_years if y >= TUNE_PERIOD[0]]
    f = fields
    first = f["round"] <= k
    era = f["year"].isin(years) & first
    rows = []
    for name, sel in ((f"first {k} races of {', '.join(map(str, years))}", era),
                      (f"first {k} races of other seasons", first & ~f["year"].isin(years)),
                      ("all other races", ~first)):
        sel = sel & (f["year"] >= TUNE_PERIOD[0])
        y = f["dnf"][sel]
        rows.append({"races": name, "cars": int(sel.sum()), "actual": y.mean(),
                     "P(DNF)": f["p_dnf"][sel].mean(), "base rate": f["base_rate"][sel].mean(),
                     "ll model": log_loss(f["p_dnf"][sel], y),
                     "ll base": log_loss(f["base_rate"][sel], y)})
    print(fmt(pd.DataFrame(rows), 4))
    per = []
    for yr in years:
        sel = (f["year"] == yr) & first
        if sel.any():
            y = f["dnf"][sel]
            per.append({"season": yr, "cars": int(sel.sum()), "actual": y.mean(),
                        "P(DNF)": f["p_dnf"][sel].mean(),
                        "base rate": f["base_rate"][sel].mean(),
                        "ll model": log_loss(f["p_dnf"][sel], y),
                        "ll base": log_loss(f["base_rate"][sel], y)})
    print(fmt(pd.DataFrame(per), 4))


if __name__ == "__main__":
    main()
