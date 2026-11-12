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

Driver backtest
---------------
Runs the full engine (constructors and drivers) and, before each race, scores
the driver ratings on TEAMMATE pairs only. Teammates share a car, so who
beats whom is mostly down to the drivers; that tests whether the driver
ratings measure skill rather than car strength. Three scores per race:
  * ``race``    -- teammates both classified (the main score: who was faster,
                   without mechanical DNFs adding noise).
  * ``raceall`` -- all teammate pairs, a DNF counting as a loss.
  * ``quali``   -- teammates by qualifying position (both with a time): the
                   real qualifying result from 1994 on, the grid before
                   (as in the engine's qualifying pass).

The race and quali scores are also kept for NEW teammate pairs only
(``newrace`` / ``newquali``): races in the first ``new_races`` (default 8)
after either driver moved from another team. Those pairings have never been
compared head to head, so their ratings were earned against other teammates
in other cars; car strength leaked into a driver rating shows up here.
(A renamed team, e.g. Toro Rosso -> AlphaTauri, counts as a move.)

``leak`` is the correlation between driver rating and car (constructor)
rating over every entry in the period, before each race. Part of it is real
(top teams hire top drivers), part is car strength leaking in; lower at equal
teammate scores means less leak.

Before 1980 a make often ran more than two cars (works and privateer cars
share one constructorId). With ``driver_shared_make_weighting`` on, as in the
engine, each pair within such a make is weighted 1 / (cars - 1) in these
scores, so pair counts can be fractional.

Periods
-------
Both scripts tune on TUNE_PERIOD (1990-2022) and report CHECK_PERIOD
(1950-1979) alongside, where the multi-car and part-timer fixes matter. The
seasons from HOLDOUT_START (2023) on are the holdout: left out of every
default table, and scored only with --holdout, once, after tuning is done.
Settings that still hold up there are real, not fitted to noise.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace

import pandas as pd

from .config import DEFAULT_CONFIG, EloConfig
from .elo_engine import EloEngine, expected_score


# ---------------------------------------------------------------------------
# Small helpers shared by the backtest scripts
# ---------------------------------------------------------------------------
# Default scoring periods (see "Periods" above).
TUNE_PERIOD = (1990, 2022)
CHECK_PERIOD = (1950, 1979)
HOLDOUT_START = 2023


def default_periods(latest: int) -> dict[str, tuple[int, int]]:
    """The tuning period, plus the pre-1980 check period."""
    return {
        f"{TUNE_PERIOD[0]}-{TUNE_PERIOD[1]} (tune)": TUNE_PERIOD,
        f"{CHECK_PERIOD[0]}-{CHECK_PERIOD[1]} (check)": CHECK_PERIOD,
    }


def holdout_period(latest: int) -> dict[str, tuple[int, int]]:
    """The holdout seasons: score these once, after tuning is finished."""
    return {f"{HOLDOUT_START}-{latest} (holdout)": (HOLDOUT_START, latest)}


def check_whole(values: list) -> None:
    """Raise ValueError unless every value is a whole number."""
    bad = [v for v in values if not float(v).is_integer()]
    if bad:
        raise ValueError(f"takes whole numbers, got {', '.join(f'{v:g}' for v in bad)}")


def parse_period(text: str) -> tuple[str, tuple[int, int]]:
    """'2015-2026' -> ('2015-2026', (2015, 2026)); '2024' -> one season."""
    m = re.fullmatch(r"\s*(\d{4})\s*(?:-\s*(\d{4}))?\s*", text)
    if not m:
        raise ValueError(f"period must look like 2015-2026 or 2024, got {text!r}")
    start = int(m.group(1))
    end = int(m.group(2) or start)
    if end < start:
        raise ValueError(f"period {text!r} ends before it starts")
    return f"{start}-{end}", (start, end)


def value_range(bounds: list[float], step: float) -> list[float]:
    """MIN MAX plus a step -> the values to test (MAX included when hit).

    One value means "just this value". Whole numbers come back as ints so
    tables print cleanly.
    """
    if len(bounds) == 1:
        bounds = bounds * 2
    if len(bounds) != 2:
        raise ValueError("takes MIN MAX (or a single value)")
    lo, hi = bounds
    if lo < 0 or hi < lo:
        raise ValueError(f"need 0 <= MIN <= MAX, got {lo:g} {hi:g}")
    if step <= 0:
        raise ValueError(f"increment must be positive, got {step:g}")
    values, i = [], 0
    while lo + i * step <= hi + 1e-9:
        v = round(lo + i * step, 6)
        values.append(int(v) if float(v).is_integer() else v)
        i += 1
    return values


def prepare_races(df: pd.DataFrame, cfg: EloConfig = DEFAULT_CONFIG) -> list[tuple]:
    """Pre-sort every race once so a parameter sweep doesn't redo it.

    Returns a list of ``(raceId, year, round, quali_order, race_order)`` in
    chronological order. The qualifying order follows ``cfg`` (real
    qualifying result or grid; see ``EloEngine._order_by_quali``).
    """
    prepared = []
    for race_id in df["raceId"].unique():
        race = df[df["raceId"] == race_id]
        quali = EloEngine._order_by_quali(race, cfg)
        result = EloEngine._order_by_position(race)
        prepared.append((
            race_id,
            int(race["year"].iloc[0]),
            int(race["round"].iloc[0]),
            quali[["driverId", "constructorId", "quali_rank"]].reset_index(drop=True),
            result[["driverId", "constructorId", "position"]].reset_index(drop=True),
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
    for race_id, year, rnd, quali, result in prepared:
        for cid in result["constructorId"].unique():
            engine.constructor_elo.setdefault(cid, init)
            engine.highest_constructor.setdefault(cid, init)
        engine.era_conf = engine.era_confidence(year, rnd)

        ids = result["constructorId"].tolist()
        vals = result["position"].tolist()

        pre = _score_race(engine.constructor_elo, ids, vals, sentinel, cfg.elo_scale)
        engine._constructor_pass(quali, "quali_rank", cfg.no_grid_sentinel, is_race=False)
        post_q = _score_race(engine.constructor_elo, ids, vals, sentinel, cfg.elo_scale)
        engine._constructor_pass(result, "position", sentinel, is_race=True)

        rows.append({
            "raceId": race_id, "year": year, "pairs": pre[0],
            "pre_logloss": pre[1], "pre_brier": pre[2], "pre_correct": pre[3],
            "postq_logloss": post_q[1], "postq_brier": post_q[2],
            "postq_correct": post_q[3],
        })
    return pd.DataFrame(rows)


def summarise(per_race: pd.DataFrame, period) -> dict:
    """Pool per-race sums into mean scores for races in a period: a
    (start, end) pair of seasons, or a set of seasons."""
    r = select_period(per_race, period)
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
                        periods: dict,
                        base: EloConfig = DEFAULT_CONFIG,
                        progress: bool = False,
                        boost_race_ks=(None,), boost_quali_ks=(None,),
                        boost_races: int | None = None) -> pd.DataFrame:
    """Grid-search constructor K-factors; one row per (K combo, period).

    ``boost_race_ks`` / ``boost_quali_ks`` are the era-start K values to try
    (None = no boost for that pass); ``boost_races`` overrides
    ``era_start_races``.
    """
    rows = []
    combos = [(kr, kq, br, bq) for kr in race_ks for kq in quali_ks
              for br in boost_race_ks for bq in boost_quali_ks]
    for n, (kr, kq, br, bq) in enumerate(combos, 1):
        cfg = replace(
            base, k_constructor_race=float(kr), k_constructor_quali=float(kq),
            k_constructor_race_era_start=None if br is None else float(br),
            k_constructor_quali_era_start=None if bq is None else float(bq),
        )
        if boost_races is not None:
            cfg = replace(cfg, era_start_races=boost_races)
        per_race = constructor_backtest(prepared, cfg)
        for label, period in periods.items():
            s = summarise(per_race, period)
            if s:
                rows.append({"k_race": kr, "k_quali": kq,
                             "boost_race": br, "boost_quali": bq,
                             "period": label, **s})
        if progress:
            boost = "" if br is None and bq is None else f"  boost {br}/{bq}"
            print(f"  {n}/{len(combos)}  K race={kr:<5} K quali={kq:<5}{boost}",
                  flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver backtest
# ---------------------------------------------------------------------------
# The driver settings a sweep can vary, in the order they're reported.
DRIVER_PARAMS = ("race_k", "quali_k", "rookie_ratio", "exponent", "dnf_factor",
                 "confidence_drop", "car_weight", "car_before_update",
                 "opponent_floor", "confidence_races")

# Sweep parameters that must be whole numbers.
INTEGER_PARAMS = ("car_before_update", "confidence_races")

# Teammate comparisons scored by driver_backtest.
DRIVER_SCORES = ("race", "raceall", "quali", "newrace", "newquali")


def driver_defaults(cfg: EloConfig = DEFAULT_CONFIG) -> dict[str, float]:
    """The current config expressed as the sweep's parameters."""
    return {
        "race_k": cfg.k_driver_race_veteran,
        "quali_k": cfg.k_driver_quali_veteran,
        "rookie_ratio": round(cfg.k_driver_race_rookie / cfg.k_driver_race_veteran, 4),
        "exponent": cfg.constructor_adjustment_exponent,
        "dnf_factor": cfg.driver_dnf_penalty_factor,
        "confidence_drop": cfg.era_start_confidence_drop,
        "car_weight": cfg.driver_car_weight,
        "car_before_update": int(cfg.driver_car_before_update),
        "opponent_floor": cfg.driver_opponent_confidence_floor,
        "confidence_races": cfg.driver_confidence_races,
    }


def driver_config(base: EloConfig, race_k: float, quali_k: float,
                  rookie_ratio: float, exponent: float,
                  dnf_factor: float, confidence_drop: float,
                  car_weight: float, car_before_update: float,
                  opponent_floor: float, confidence_races: int) -> EloConfig:
    """Build a config from sweep parameters.

    race_k / quali_k are the VETERAN (100% confidence) K-factors; the rookie
    (0% confidence) K-factors are rookie_ratio times those.
    """
    return replace(
        base,
        k_driver_race_veteran=float(race_k),
        k_driver_race_rookie=float(race_k) * rookie_ratio,
        k_driver_quali_veteran=float(quali_k),
        k_driver_quali_rookie=float(quali_k) * rookie_ratio,
        constructor_adjustment_exponent=float(exponent),
        driver_dnf_penalty_factor=float(dnf_factor),
        era_start_confidence_drop=float(confidence_drop),
        driver_car_weight=float(car_weight),
        driver_car_before_update=bool(car_before_update),
        driver_opponent_confidence_floor=float(opponent_floor),
        driver_confidence_races=int(round(confidence_races)),
    )


def _score_teammates(ratings: dict, drivers: list, teams: list, vals: list,
                     skip, scale: float, only=None, shared_make: bool = False):
    """Score teammate pairs (i is ahead of j) against the current ratings.

    ``only``: if given, a set of drivers; a pair counts only if at least one
    of the two is in it.
    ``shared_make``: a make with k > 2 cars in the session (works and
    privateer cars sharing one constructorId) has each of its pairs weighted
    1 / (k - 1), as in the engine, so they add up to one teammate pair per
    driver. Pair counts and score sums are then weighted.
    """
    cars = Counter(teams) if shared_make else None
    n_pairs = 0.0
    logloss = brier = correct = 0.0
    n = len(drivers)
    for i in range(n):
        a, ta, va = drivers[i], teams[i], vals[i]
        for j in range(i + 1, n):
            b, tb, vb = drivers[j], teams[j], vals[j]
            if ta != tb or a == b or skip(va, vb):
                continue
            if only is not None and a not in only and b not in only:
                continue
            w = 1.0 / (cars[ta] - 1) if cars is not None and cars[ta] > 2 else 1.0
            p = expected_score(ratings[a], ratings[b], scale)
            p = min(max(p, 1e-12), 1 - 1e-12)
            logloss -= w * math.log(p)
            brier += w * (1.0 - p) ** 2
            correct += w * (1.0 if p > 0.5 else (0.5 if p == 0.5 else 0.0))
            n_pairs += w
    return n_pairs, logloss, brier, correct


def driver_backtest(prepared: list[tuple], cfg: EloConfig = DEFAULT_CONFIG,
                    new_races: int = 8) -> pd.DataFrame:
    """Run the full engine over every race, scoring teammate pairs first.

    Returns one row per race with pair counts and summed scores for each
    comparison in DRIVER_SCORES, plus sums for the driver-vs-car rating
    correlation (``leak``). ``new_races``: for how many races after a team
    move a teammate pair counts as new.
    """
    engine = EloEngine(cfg)
    init = cfg.initial_elo
    dnf, no_grid, scale = cfg.dnf_sentinel, cfg.no_grid_sentinel, cfg.elo_scale
    classified = lambda va, vb: va == dnf or vb == dnf
    on_grid = lambda va, vb: va == no_grid or vb == no_grid
    scores = (  # name, frame, column, skip rule, new teammate pairs only
        ("race", "result", "position", classified, False),
        ("raceall", "result", "position", lambda va, vb: va == dnf and vb == dnf, False),
        ("quali", "quali", "quali_rank", on_grid, False),
        ("newrace", "result", "position", classified, True),
        ("newquali", "quali", "quali_rank", on_grid, True),
    )
    # Per driver: current team, races with it, and whether they ever moved.
    team: dict = {}
    stint: dict = {}
    moved: dict = {}
    rows = []
    for race_id, year, rnd, quali, result in prepared:
        entries = list(zip(result["driverId"], result["constructorId"]))
        for d, c in entries:
            engine.driver_elo.setdefault(d, init)
            engine.highest_driver.setdefault(d, init)
            engine.constructor_elo.setdefault(c, init)
            engine.highest_constructor.setdefault(c, init)

        # Drivers in their first new_races with a team they moved to.
        new = {d for d, c in entries
               if (d in team and team[d] != c)
               or (moved.get(d) and stint[d] < new_races)}

        row = {"raceId": race_id, "year": year}
        frames = {"result": result, "quali": quali}
        for name, frame, col, skip, only_new in scores:
            f = frames[frame]
            n, ll, br, co = _score_teammates(
                engine.driver_elo, f["driverId"].tolist(),
                f["constructorId"].tolist(), f[col].tolist(),
                skip, scale, only=new if only_new else None,
                shared_make=cfg.driver_shared_make_weighting)
            row.update({f"{name}_pairs": n, f"{name}_logloss": ll,
                        f"{name}_brier": br, f"{name}_correct": co})

        # Driver vs car rating before the race (sums, for a pooled corr).
        ds = [engine.driver_elo[d] for d, _ in entries]
        cs = [engine.constructor_elo[c] for _, c in entries]
        row.update({
            "leak_n": len(ds), "leak_d": sum(ds), "leak_c": sum(cs),
            "leak_dd": sum(x * x for x in ds), "leak_cc": sum(x * x for x in cs),
            "leak_dc": sum(x * y for x, y in zip(ds, cs)),
        })
        rows.append(row)

        engine.process_ordered(year, rnd, quali, result)

        seen = set()
        for d, c in entries:
            if d in seen:      # 1950s shared drives: count the race once
                continue
            seen.add(d)
            if d in team and team[d] != c:
                moved[d] = True
                stint[d] = 0
            team[d] = c
            stint[d] = stint.get(d, 0) + 1
    return pd.DataFrame(rows)


def select_period(per_race: pd.DataFrame, period) -> pd.DataFrame:
    """Rows in a period: a (start, end) pair of seasons, or a set of seasons."""
    if isinstance(period, (set, frozenset)):
        return per_race[per_race["year"].isin(period)]
    start, end = period
    return per_race[(per_race["year"] >= start) & (per_race["year"] <= end)]


def summarise_drivers(per_race: pd.DataFrame, period) -> dict:
    """Pool per-race sums into mean scores for races in a period
    (see ``select_period``)."""
    r = select_period(per_race, period)
    if r.empty:
        return {}
    out = {"races": len(r)}
    for name in DRIVER_SCORES:
        n = r[f"{name}_pairs"].sum()
        out[f"{name}_pairs"] = int(n)
        for stat in ("logloss", "brier"):
            out[f"{name}_{stat}"] = r[f"{name}_{stat}"].sum() / n if n else float("nan")
        out[f"{name}_acc"] = r[f"{name}_correct"].sum() / n if n else float("nan")
    out["leak"] = _pooled_corr(r)
    return out


def _pooled_corr(r: pd.DataFrame) -> float:
    """Driver-vs-car rating correlation from the per-race leak_* sums."""
    n = r["leak_n"].sum()
    md, mc = r["leak_d"].sum() / n, r["leak_c"].sum() / n
    cov = r["leak_dc"].sum() / n - md * mc
    var_d = r["leak_dd"].sum() / n - md * md
    var_c = r["leak_cc"].sum() / n - mc * mc
    if var_d <= 0 or var_c <= 0:
        return float("nan")
    return cov / math.sqrt(var_d * var_c)


def sweep_drivers(prepared: list[tuple], grid: dict[str, list],
                  periods: dict,
                  base: EloConfig = DEFAULT_CONFIG,
                  progress: bool = False, new_races: int = 8) -> pd.DataFrame:
    """Grid-search driver settings; one row per (combo, period).

    ``grid`` maps each name in DRIVER_PARAMS to the values to try; missing
    names stay at the current config value. ``periods`` maps a label to a
    (start, end) pair or a set of seasons.
    """
    values = {**{k: [v] for k, v in driver_defaults(base).items()}, **grid}
    combos = [{}]
    for name in DRIVER_PARAMS:
        combos = [{**c, name: v} for c in combos for v in values[name]]
    rows = []
    for n, combo in enumerate(combos, 1):
        per_race = driver_backtest(prepared, driver_config(base, **combo),
                                   new_races=new_races)
        for label, period in periods.items():
            s = summarise_drivers(per_race, period)
            if s:
                rows.append({**combo, "period": label, **s})
        if progress:
            desc = "  ".join(f"{k}={combo[k]:g}" for k in DRIVER_PARAMS
                             if len(values[k]) > 1)
            print(f"  {n}/{len(combos)}  {desc}", flush=True)
    return pd.DataFrame(rows)
