"""Backtests: how well do the ratings predict results they haven't seen yet?

Constructor backtest
--------------------
Constructor ratings don't depend on drivers at all, so the constructor half of
the engine can be run on its own (fast) and scored race by race:

  1. Before a race is processed, take every pair of cars from *different*
     constructors and predict P(car A finishes ahead of car B) with the
     ratings as they stand at that moment (``expected_score``).
  2. Compare against what happened, using the same rules as the engine's race
     pass: a classified car beats every car classified below it and every DNF;
     DNF-vs-DNF pairs are skipped (no information).
  3. Then run the engine's own qualifying + race constructor passes for that
     race, and move on.

Each race is scored twice:
  * ``pre``   -- ratings before the weekend (what a pre-weekend prediction
                 would use).
  * ``post_q`` -- ratings after the qualifying pass of that weekend (what a
                 prediction made after qualifying would use).

Scores, all averaged over pairs:
  * log loss  -- -ln(p of what actually happened). Lower is better.
                 Guessing 50/50 every time gives ln 2 = 0.6931.
  * Brier     -- (1 - p)^2. Lower is better; 50/50 gives 0.25.
  * accuracy  -- share of pairs where the favourite won (p = 0.5 counts half).

Only the ratio K / elo_scale matters for these predictions (everyone starts
equal), so tune K with the scale held fixed.
"""
from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd

from .config import DEFAULT_CONFIG, EloConfig
from .elo_engine import EloEngine, expected_score


def prepare_races(df: pd.DataFrame, cfg: EloConfig = DEFAULT_CONFIG) -> list[tuple]:
    """Pre-sort every race once so a parameter sweep doesn't redo it.

    Returns a list of ``(raceId, year, quali_order, race_order)`` in
    chronological order.
    """
    prepared = []
    for race_id in df["raceId"].unique():
        race = df[df["raceId"] == race_id]
        quali = EloEngine._order_by_grid(race, cfg.no_grid_sentinel)
        result = EloEngine._order_by_position(race)
        prepared.append((
            race_id,
            int(race["year"].iloc[0]),
            quali[["constructorId", "grid"]].reset_index(drop=True),
            result[["constructorId", "position"]].reset_index(drop=True),
        ))
    return prepared


def _score_race(ratings: dict, ids: list, vals: list, sentinel, scale: float):
    """Score one race's pairwise outcomes against the current ratings."""
    n_pairs = 0
    logloss = brier = correct = 0.0
    n = len(ids)
    for i in range(n):
        a, va = ids[i], vals[i]
        for j in range(i + 1, n):
            b, vb = ids[j], vals[j]
            if va == sentinel and vb == sentinel:
                continue
            if a == b:
                continue
            # i finished ahead of j, so p is the probability of what happened.
            p = expected_score(ratings[a], ratings[b], scale)
            p = min(max(p, 1e-12), 1 - 1e-12)
            logloss -= math.log(p)
            brier += (1.0 - p) ** 2
            correct += 1.0 if p > 0.5 else (0.5 if p == 0.5 else 0.0)
            n_pairs += 1
    return n_pairs, logloss, brier, correct


def constructor_backtest(prepared: list[tuple],
                         cfg: EloConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Run the constructor passes over every race, scoring each one first.

    Returns one row per race with pair counts and summed scores for the
    ``pre`` and ``post_q`` predictions (sums, so periods can be pooled).
    """
    engine = EloEngine(cfg)
    init = cfg.initial_elo
    sentinel = cfg.dnf_sentinel
    rows = []
    for race_id, year, quali, result in prepared:
        for cid in result["constructorId"].unique():
            engine.constructor_elo.setdefault(cid, init)
            engine.highest_constructor.setdefault(cid, init)

        ids = result["constructorId"].tolist()
        vals = result["position"].tolist()

        pre = _score_race(engine.constructor_elo, ids, vals, sentinel, cfg.elo_scale)
        engine._constructor_pass(quali, "grid", cfg.no_grid_sentinel, is_race=False)
        post_q = _score_race(engine.constructor_elo, ids, vals, sentinel, cfg.elo_scale)
        engine._constructor_pass(result, "position", sentinel, is_race=True)

        rows.append({
            "raceId": race_id, "year": year, "pairs": pre[0],
            "pre_logloss": pre[1], "pre_brier": pre[2], "pre_correct": pre[3],
            "postq_logloss": post_q[1], "postq_brier": post_q[2],
            "postq_correct": post_q[3],
        })
    return pd.DataFrame(rows)


def summarise(per_race: pd.DataFrame, start: int, end: int) -> dict:
    """Pool per-race sums into mean scores for races in [start, end]."""
    r = per_race[(per_race["year"] >= start) & (per_race["year"] <= end)]
    n = r["pairs"].sum()
    if n == 0:
        return {}
    out = {"races": len(r), "pairs": int(n)}
    for when in ("pre", "postq"):
        out[f"{when}_logloss"] = r[f"{when}_logloss"].sum() / n
        out[f"{when}_brier"] = r[f"{when}_brier"].sum() / n
        out[f"{when}_acc"] = r[f"{when}_correct"].sum() / n
    return out


def sweep_constructor_k(prepared: list[tuple], race_ks, quali_ks,
                        periods: dict[str, tuple[int, int]],
                        base: EloConfig = DEFAULT_CONFIG,
                        progress: bool = False) -> pd.DataFrame:
    """Grid-search constructor K-factors; one row per (K pair, period)."""
    rows = []
    combos = [(kr, kq) for kr in race_ks for kq in quali_ks]
    for n, (kr, kq) in enumerate(combos, 1):
        cfg = replace(base, k_constructor_race=float(kr),
                      k_constructor_quali=float(kq))
        per_race = constructor_backtest(prepared, cfg)
        for label, (start, end) in periods.items():
            s = summarise(per_race, start, end)
            if s:
                rows.append({"k_race": kr, "k_quali": kq, "period": label, **s})
        if progress:
            print(f"  {n}/{len(combos)}  K race={kr:<5} K quali={kq:<5}", flush=True)
    return pd.DataFrame(rows)
