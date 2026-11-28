"""Train and score the finishing-order models (step 6b) against ELO and the
grid slot.

    py train_rank_models.py                              # both stages, every model (about 25 min)
    py train_rank_models.py --stage post_quali --models logit
    py train_rank_models.py --stage pre --models logit --logit-set form --lam 0.3 1 3
    py train_rank_models.py --stage post_quali --models catboost --depth 2 3 4 --l2 3 10
    py train_rank_models.py --models logit,xgb_pl --no-save --boot 0   # quick look

What it does, per stage ("post_quali" first, then "pre")
--------------------------------------------------------
1. Loads the calibration fields, ELO's prediction table and the step 6a
   feature table (all cached by the earlier steps).
2. For every model and every setting in its grid: a season-by-season
   walk-forward (train on --train-start .. s-1, predict season s, for s
   from --first-oof to the latest season). Trained on classified cars
   only; every car gets a score.
3. Picks each model's setting (and number of trees) by the Plackett-Luce
   loss of the whole classified order on --score seasons (2005-2022),
   with each season's temperature fitted on the seasons before it.
4. Picks how many top places get their own temperature (--top-places,
   default 1 2 3) by --top-metric (default the top-3 loss, the podium; the
   same choice step 4 made for ELO's finish_top_places), then which places
   the main temperature is fitted on (--temp-places), by the simulated
   points log loss on the tuning seasons.
5. Simulates every race from the first fully out-of-sample season with
   the model's strengths and ELO's retirement model (same seeds, same
   retirement draws), and scores it against ELO and (after qualifying)
   the grid-slot table: win / podium / points log loss and Brier, RPS,
   95% bootstrap intervals, calibration bands.
6. Fits the final model on every season up to the latest and saves it.

Options
-------
  --stage pre|post_quali|both       (both)
  --models LIST                     logit,catboost,xgb,xgb_pl (all)
  --logit-set NAME                  feature set (f1elo/rank_models.py LOGIT_SETS;
                                    default form after qualifying, split pre-race)
  --logit-terms a,b,c*d             custom terms instead of a set
  --lam X [X ...]                   L2 penalty grid (1 3 10)
  --clip X                          winsorising quantile (0.02)
  --no-indicators                   no missing-value indicators
  --cat-loss YetiRank|PairLogit     (YetiRank; PairLogit is ~4x slower)
  Tree settings (defaults per model in TREE_DEFAULTS below; an option given
  here applies to every tree model in the run):
  --depth N [N ...]                 tree depth grid (catboost 3, xgb 2, xgb_pl 2)
  --lr X                            learning rate (0.05, 0.03, 0.02)
  --l2 X [X ...]                    leaf L2 grid (10)
  --min-child X                     XGBoost min_child_weight (1, 1, 100)
  --iterations N                    most trees (800, 1200, 600); predictions are made at
  --checkpoints N [N ...]           these counts and the best count is chosen
  --offset elo|none                 start from ELO's strength (elo, none, elo)
  --tree-features all|LIST          (all: every stage feature, relative versions)
  --top-places N [N ...]            (1 2 3)
  --top-metric order|top3|win       (top3)
  --temp-places N [N ...]           places the main temperature is fitted on (5 6 8 0;
                                    0 = the whole order), chosen by the simulated
                                    points log loss on the tuning seasons
  --select-sims N                   simulations per race for that choice (2000)
  --elo-temp-places N               the same for the "elo + temperature" reference (5)
  --train-start Y                   (1990)
  --first-oof Y                     first walk-forward season (2000); the
                                    tables start 3 seasons later (temperatures)
  --score Y1-Y2                     tuning seasons that settings are chosen on (2005-2022)
  --sims N, --seed N                simulations per race (10000), seed (0)
  --boot N                          bootstrap resamples (2000; 0 = none)
  --no-save                         don't write anything
  --threads N                       CPU threads for the trees (all)

Writes (output/, gitignored)
----------------------------
  race_predictions_<model>_<stage>.parquet (+ .json)  the model's table, the
                        columns of race_predictions_<stage>.parquet
  rank_models_tuning_<stage>.csv     every setting tried: walk-forward losses
  rank_models_scores_<stage>.csv     model x period x metric, CIs, diff vs the
                                     reference (ELO, and the recommended model)
  rank_models_bands_<stage>.csv      calibration bands (win / podium / points)
  rank_models_importance_<stage>.csv logit coefficients; mean |SHAP| for trees
  rank_models_temperatures_<stage>.csv  each season's temperatures
  models/<model>_<stage>.json (+ .cbm / .ubj)  the final models
"""
from __future__ import annotations

import argparse
import itertools
import time
import warnings

import numpy as np
import pandas as pd

from f1elo import calibration as cal
from f1elo import config
from f1elo import rank_models as rm
from f1elo.backtest import parse_period

MODELS = ("logit", "catboost", "xgb", "xgb_pl")
# Per-model tree defaults (each --option given on the command line wins),
# from walk-forward runs after qualifying and pre-race (README):
#   catboost  best starting from ELO's strength; still improving at 600 trees (lr 0.03)
#   xgb       best from scratch at depth 2 with a slow learning rate
#   xgb_pl    overfits within ~100-300 trees at lr 0.03-0.1, so slower and larger leaves
TREE_DEFAULTS = {
    "catboost": {"depth": [3], "lr": 0.05, "l2": [10.0], "iterations": 800,
                 "checkpoints": [200, 400, 600, 800], "offset": "elo", "min_child": 1.0},
    "xgb": {"depth": [2], "lr": 0.03, "l2": [10.0], "iterations": 1200,
            "checkpoints": [300, 600, 900, 1200], "offset": "none", "min_child": 1.0},
    "xgb_pl": {"depth": [2], "lr": 0.02, "l2": [10.0], "iterations": 600,
               "checkpoints": [100, 200, 300, 450, 600], "offset": "elo", "min_child": 100.0},
}
METRIC = {"order": "order_nll", "top3": "top3_nll", "win": "win_nll"}
SHOW = ["win_ll", "podium_ll", "podium_brier", "points_ll", "points_brier", "rps"]
HEAD = {"win_ll": "win ll", "podium_ll": "podium ll", "podium_brier": "podium br",
        "points_ll": "points ll", "points_brier": "points br", "rps": "rps"}


def candidates(kind: str, data: rm.StageData, args) -> list[tuple[dict, callable]]:
    """(settings, model factory) for every point of a model's grid."""
    out = []
    if kind == "logit":
        terms = (args.logit_terms.split(",") if args.logit_terms
                 else rm.LOGIT_SETS[data.stage][args.logit_set or rm.DEFAULT_SET[data.stage]])
        for lam in args.lam:
            s = {"terms": ",".join(terms), "lam": lam, "clip": args.clip,
                 "indicators": not args.no_indicators}
            out.append((s, lambda lam=lam: rm.LogitModel(terms, lam, not args.no_indicators,
                                                        args.clip)))
        return out
    cols = (rm.feature_columns(data.X, data.stage) if args.tree_features == "all"
            else args.tree_features.split(","))
    o = {k: (getattr(args, k) if getattr(args, k) is not None else v)
         for k, v in TREE_DEFAULTS[kind].items()}
    offset = None if o["offset"] == "none" else o["offset"]
    loss = args.cat_loss if kind == "catboost" else kind
    for depth, l2 in itertools.product(o["depth"], o["l2"]):
        s = {"loss": loss, "depth": depth, "l2": l2, "lr": o["lr"], "offset": o["offset"],
             "min_child": o["min_child"], "features": len(cols)}
        out.append((s, lambda depth=depth, l2=l2: rm.TreeModel(
            kind, cols, o["iterations"], depth, o["lr"], l2, args.cat_loss, offset,
            o["checkpoints"], min_child=o["min_child"], threads=args.threads)))
    return out


def evaluate(data, score, seasons, top_places, period) -> tuple[dict, dict]:
    temps = rm.season_temperatures(data, score, seasons, top_places)
    xm, xt = rm.scaled(data, score, temps)
    return rm.oof_metrics(data, xm, xt, top_places, period), temps


def tune(kind, data, args, say) -> dict:
    """Walk-forward every setting; the best (setting, trees) by the
    whole-order loss at 3 top places."""
    rows, best = [], None
    for settings, make in candidates(kind, data, args):
        t0 = time.perf_counter()
        wf = rm.walk_forward(data, make, args.first_oof, None, args.train_start)
        for k, score in wf.scores.items():
            m, _ = evaluate(data, score, wf.seasons, 3, args.score)
            row = {"model": kind, **settings, "iterations": k, **m,
                   "seconds": round(wf.seconds, 1)}
            rows.append(row)
            if best is None or m["order_nll"] < best["metrics"]["order_nll"]:
                best = {"settings": settings, "make": make, "n_iter": k, "wf": wf,
                        "metrics": m}
        say(f"    {kind} {settings_text(settings)}: best order loss "
            f"{min(r['order_nll'] for r in rows[-len(wf.scores):]):.4f} "
            f"[{time.perf_counter() - t0:.0f} s]")
    best["tuning"] = rows
    return best


def settings_text(s: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in s.items() if k not in ("terms", "features"))


def choose_top_places(data, score, seasons, args) -> tuple[int, list]:
    rows = []
    for tp in args.top_places:
        m, _ = evaluate(data, score, seasons, tp, args.score)
        rows.append({"top_places": tp, **m})
    key = METRIC[args.top_metric]
    best = min(rows, key=lambda r: r[key])["top_places"]
    return best, rows


def choose_temp_places(data, score, seasons, top_places, args) -> tuple[int, list]:
    """The places the main temperature is fitted on, by the simulated
    points log loss on the tuning seasons (--select-sims per race)."""
    rows_t = data.rows(*args.score)
    f = rm.subset(data.fields, rows_t)
    out = []
    for k in args.temp_places:
        temps = rm.season_temperatures(data, score, seasons, top_places, max_places=k or None)
        xm, _ = rm.scaled(data, score, temps)
        _, pred = rm.simulate_model(data, xm, temps, rows_t, args.select_sims, args.seed)
        s = cal.summarise_scores(cal.race_scores(f, pred))
        t = temps[max(t for t in temps if t <= args.score[1])]
        out.append({"temp_places": k, "points_ll": s["points_ll"], "rps": s["rps"],
                    "t": t.t, "t_top": t.t_top})
    best = min(out, key=lambda r: r["points_ll"])["temp_places"]
    return best, out


def print_scores(scores: pd.DataFrame, label: str, ref: str) -> None:
    t = scores[scores["period"] == label]
    if t.empty:
        return
    races = int(t["races"].iloc[0])
    print(f"\n{label}: {races} races (lower is better; [diff vs {ref}, 95% CI])")
    names = list(dict.fromkeys(t["predictor"]))
    w = max(len(n) for n in names) + 2
    print(" " * w + "".join(f"{HEAD[m]:>26}" for m in SHOW))
    for n in names:
        line = f"{n:<{w}}"
        for m in SHOW:
            r = t[(t["predictor"] == n) & (t["metric"] == m)]
            if r.empty:
                line += " " * 26
                continue
            r = r.iloc[0]
            cell = f"{r['value']:.4f}"
            if "diff" in r and pd.notna(r.get("diff")):
                cell += f" [{r['diff']:+.4f} {r['diff_lo']:+.3f}..{r['diff_hi']:+.3f}]"
            line += f"{cell:>26}"
        print(line)


def verdict(scores: pd.DataFrame, models: list, tune_label: str, hold_label: str,
            stage: str) -> tuple[str | None, list]:
    """A model qualifies if it beats ELO on every score on the tuning
    period, beats the grid slot on winners there (after qualifying), and
    is no worse than ELO on the holdout's podium, points and RPS scores.
    The best qualifier by tuning podium log loss is recommended."""
    def val(p, label, m):
        r = scores[(scores["predictor"] == p) & (scores["period"] == label)
                   & (scores["metric"] == m)]
        return float(r["value"].iloc[0]) if len(r) else np.nan

    lines, ok = [], []
    for p in models:
        why = []
        worse = [m for m in SHOW if val(p, tune_label, m) >= val("elo", tune_label, m)]
        if worse:
            why.append(f"not better than ELO on {', '.join(worse)} ({tune_label})")
        if stage == "post_quali" and val(p, tune_label, "win_ll") >= val("grid slot", tune_label,
                                                                          "win_ll"):
            why.append(f"not better than the grid slot on winners ({tune_label})")
        hw = [m for m in ("podium_ll", "points_ll", "rps")
              if val(p, hold_label, m) > val("elo", hold_label, m)]
        if hw:
            why.append(f"worse than ELO on {', '.join(hw)} ({hold_label})")
        if why:
            lines.append(f"  {p}: no - {'; '.join(why)}")
        else:
            ok.append(p)
            lines.append(f"  {p}: qualifies (tune podium ll {val(p, tune_label, 'podium_ll'):.4f}, "
                         f"win ll {val(p, tune_label, 'win_ll'):.4f}; holdout win ll "
                         f"{val(p, hold_label, 'win_ll'):.4f} vs ELO "
                         f"{val('elo', hold_label, 'win_ll'):.4f})")
    best = min(ok, key=lambda p: val(p, tune_label, "podium_ll")) if ok else None
    return best, lines


def run_stage(stage, args) -> None:
    t0 = time.time()
    say = print
    print(f"\n{'=' * 28} {stage.upper()} {'=' * 28}")
    data = rm.load_stage(stage, progress=say)
    latest = data.latest_year
    first_table = None
    tuning, temps_rows, imp_rows, chosen = [], [], [], {}
    for kind in args.models:
        print(f"\n--- {kind}: walk-forward {args.first_oof}-{latest}, trained from "
              f"{args.train_start}, settings chosen on {args.score[0]}-{args.score[1]} ---")
        best = tune(kind, data, args, say)
        tuning += best["tuning"]
        wf = best["wf"]
        score = wf.scores[best["n_iter"]]
        tp, tp_rows = choose_top_places(data, score, wf.seasons, args)
        print(f"  chosen: {settings_text(best['settings'])}"
              + (f", {best['n_iter']} trees" if best["n_iter"] else ""))
        print("  top places with their own temperature (walk-forward losses per race, "
              f"{args.score[0]}-{args.score[1]}):")
        for r in tp_rows:
            mark = "  <- chosen" if r["top_places"] == tp else ""
            print(f"    {r['top_places']}: order {r['order_nll']:.4f}, top 3 {r['top3_nll']:.4f}, "
                  f"winner {r['win_nll']:.4f}{mark}")
        kp, kp_rows = choose_temp_places(data, score, wf.seasons, tp, args)
        print("  places the main temperature is fitted on (simulated, "
              f"{args.select_sims:,} per race, {args.score[0]}-{args.score[1]}):")
        for r in kp_rows:
            mark = "  <- chosen" if r["temp_places"] == kp else ""
            what = f"first {r['temp_places']}" if r["temp_places"] else "whole order"
            print(f"    {what}: points ll {r['points_ll']:.4f}, rps {r['rps']:.4f} "
                  f"(t {r['t']:.3f}, t_top {r['t_top']:.3f} in {args.score[1]}){mark}")
        temps = rm.season_temperatures(data, score, wf.seasons, tp, max_places=kp or None)
        for s, t in temps.items():
            temps_rows.append({"model": kind, "season": s, "t": t.t, "t_top": t.t_top,
                               "top_places": tp, "fit_places": kp})
        first = min(temps)
        first_table = first if first_table is None else max(first_table, first)
        chosen[kind] = {"best": best, "score": score, "temps": temps, "top_places": tp,
                        "temp_places": kp}

    # ----- simulate, score -----
    rows = data.rows(first_table, latest)
    f = rm.subset(data.fields, rows)
    tune_label = f"{args.score[0]}-{args.score[1]} (walk-forward)"
    hold_label = f"{rm.HOLDOUT_START}-{latest} (holdout)"
    masks = {tune_label: cal.race_mask(f, args.score),
             hold_label: cal.race_mask(f, (rm.HOLDOUT_START, latest))}
    preds = {"elo": rm.elo_pred(data, rows, args.sims)}
    if stage == "post_quali":
        preds["grid slot"] = rm.grid_pred(data, rows)
        preds["grid slot (walk-forward)"] = rm.grid_pred(data, rows, True)
    # ELO's own strengths with a fitted temperature: how much is rescaling alone?
    elo_x = rm.elo_offset(data)
    seasons = sorted(set(int(y) for y in data.year if y >= args.first_oof))
    elo_t = rm.season_temperatures(data, elo_x, seasons, 3, max_places=args.elo_temp_places or None)
    print(f"\n  simulating {int(f['race_no'].nunique())} races ({first_table}-{latest}) "
          f"x {args.sims:,} per model ...")
    _, preds["elo + temperature"] = rm.simulate_model(data, rm.scaled(data, elo_x, elo_t)[0],
                                                      elo_t, rows, args.sims, args.seed)
    tables = {}
    for kind, c in chosen.items():
        xm, _ = rm.scaled(data, c["score"], c["temps"])
        tables[kind], preds[kind] = rm.simulate_model(data, xm, c["temps"], rows, args.sims,
                                                      args.seed)
    scores = rm.compare(f, preds, masks, "elo", args.boot)
    print(f"\n=== {stage}: models vs ELO "
          + ("and the grid slot " if stage == "post_quali" else "") + "===")
    print("('elo + temperature' = ELO's own strengths rescaled like the models; the grid "
          "slot table is fitted on 1990-2022, in-sample on the tuning period)")
    for label in masks:
        print_scores(scores, label, "elo")
    best, lines = verdict(scores, list(chosen), tune_label, hold_label, stage)
    print("\nVerdict (beats ELO on every score on the tuning period"
          + (", beats the grid slot on winners" if stage == "post_quali" else "")
          + ", no worse than ELO on the holdout's podium / points / RPS):")
    print("\n".join(lines))
    print(f"  -> recommended for the app ({stage}): {best or 'none (keep ELO)'}")
    scores.insert(0, "reference", "elo")
    # The models against each other: every predictor vs the recommended
    # model (or, with none, the best model on tuning podium log loss).
    ref = best or min(chosen, key=lambda p: float(scores[
        (scores["predictor"] == p) & (scores["period"] == tune_label)
        & (scores["metric"] == "podium_ll")]["value"].iloc[0]))
    vs = rm.compare(f, preds, masks, ref, args.boot)
    vs.insert(0, "reference", ref)
    print(f"\n=== {stage}: every predictor vs {ref} ===")
    for label in masks:
        print_scores(vs, label, ref)
    scores = pd.concat([scores, vs], ignore_index=True)
    bnd = rm.bands(f, preds, masks)

    # ----- importance and the final models -----
    for kind, c in chosen.items():
        best_k = c["best"]
        m = best_k["make"]()
        if not isinstance(m, rm.LogitModel):
            m.iterations, m.checkpoints = best_k["n_iter"], [best_k["n_iter"]]
        m.fit(data, (data.year >= args.train_start) & (data.year <= latest))
        g = rm.Groups(data.race_no, data.finish, np.isin(data.year, best_k["wf"].seasons))
        final_t = rm.fit_temperature(c["score"][g.rows], g, c["top_places"],
                                     c["temp_places"] or None)
        if isinstance(m, rm.LogitModel):
            co = m.coefficients()
            co["rating_points_per_unit"] = co["per_unit"] * final_t.t * data.cfg.finish_scale / rm.LN10
            co.insert(0, "model", kind)
            imp_rows.append(co)
            print(f"\n--- {kind} ({stage}) coefficients, final model "
                  f"({args.train_start}-{latest}); t {final_t.t:.3f}, t_top {final_t.t_top:.3f} "
                  f"for the first {c['top_places']} places ---")
            print(co.drop(columns="model").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        else:
            im = m.importance(data, data.rows(args.score[0], latest))
            im.insert(0, "model", kind)
            imp_rows.append(im)
            print(f"\n--- {kind} ({stage}) top features by mean |SHAP| (final model, "
                  f"{best_k['n_iter']} trees) ---")
            print(im.drop(columns="model").head(15).to_string(index=False,
                                                              float_format=lambda v: f"{v:.4g}"))
        c["final"], c["final_t"] = m, final_t

    if not args.no_save:
        out = config.OUTPUT_DIR
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tuning).assign(stage=stage).to_csv(out / f"rank_models_tuning_{stage}.csv",
                                                        index=False)
        scores.insert(0, "stage", stage)
        scores.to_csv(out / f"rank_models_scores_{stage}.csv", index=False)
        bnd.insert(0, "stage", stage)
        bnd.to_csv(out / f"rank_models_bands_{stage}.csv", index=False)
        pd.concat(imp_rows, ignore_index=True).assign(stage=stage).to_csv(
            out / f"rank_models_importance_{stage}.csv", index=False)
        pd.DataFrame(temps_rows).assign(stage=stage).to_csv(
            out / f"rank_models_temperatures_{stage}.csv", index=False)
        saved = []
        for kind, c in chosen.items():
            info = {"stage": stage, "settings": c["best"]["settings"],
                    "iterations": c["best"]["n_iter"], "top_places": c["top_places"],
                    "temp_places": c["temp_places"],
                    "train": [args.train_start, latest], "first_oof": args.first_oof,
                    "score_period": list(args.score), "sims": args.sims, "seed": args.seed,
                    "data": data.fingerprint, "first_table_season": first_table,
                    "recommended": kind == best}
            rm.write_table(tables[kind], kind, stage, info)
            saved += rm.save_model(c["final"], kind, stage, c["final_t"], info)
        print(f"\nWrote output/rank_models_{{tuning,scores,bands,importance,temperatures}}_"
              f"{stage}.csv, output/race_predictions_<model>_{stage}.parquet and "
              f"{len(saved)} model files in output/models/")
    print(f"[{stage}: {time.time() - t0:.0f} s]")


def main() -> None:
    p = argparse.ArgumentParser(description="Train and score the finishing-order models.")
    p.add_argument("--stage", choices=("pre", "post_quali", "both"), default="both")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--logit-set")
    p.add_argument("--logit-terms")
    p.add_argument("--lam", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    p.add_argument("--clip", type=float, default=0.02)
    p.add_argument("--no-indicators", action="store_true")
    p.add_argument("--cat-loss", default="YetiRank", choices=("YetiRank", "PairLogit"))
    p.add_argument("--depth", type=int, nargs="+")
    p.add_argument("--lr", type=float)
    p.add_argument("--l2", type=float, nargs="+")
    p.add_argument("--min-child", type=float, dest="min_child")
    p.add_argument("--iterations", type=int)
    p.add_argument("--checkpoints", type=int, nargs="+")
    p.add_argument("--offset", choices=("elo", "none"))
    p.add_argument("--tree-features", default="all")
    p.add_argument("--top-places", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--top-metric", default="top3", choices=tuple(METRIC))
    p.add_argument("--temp-places", type=int, nargs="+", default=[5, 6, 8, 0])
    p.add_argument("--select-sims", type=int, default=2000)
    p.add_argument("--elo-temp-places", type=int, default=5)
    p.add_argument("--train-start", type=int, default=rm.TRAIN_START)
    p.add_argument("--first-oof", type=int, default=rm.FIRST_OOF)
    p.add_argument("--score", default=f"{rm.SCORE_PERIOD[0]}-{rm.SCORE_PERIOD[1]}")
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--threads", type=int, default=-1)
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in args.models if m not in MODELS]
    if bad:
        p.error(f"unknown models {bad}; choose from {MODELS}")
    args.score = parse_period(args.score)[1]
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    stages = ("post_quali", "pre") if args.stage == "both" else (args.stage,)
    for s in stages:
        run_stage(s, args)


if __name__ == "__main__":
    main()
