"""Race simulation: each driver's chance of every finishing position.

A race is simulated in two parts, because retirements and pace are
different things (step 2 found 43% of whole-field pairs decided by a
retirement, which ratings barely predict):

  1. Retirements (f1elo/reliability.py): every car retires with its own
     chance P(DNF), drawn independently.
  2. Finishing order among the cars that finish: Plackett-Luce on the
     pre-race strengths S = D + finish_car_weight * C, with weights
     10^(S / finish_scale). Positions are filled one at a time, each by a
     remaining car with probability proportional to its weight. For two
     cars that both finish, P(A ahead of B) = expected_score(S_a, S_b,
     finish_scale) exactly -- the same pairwise rule the strength backtest
     scores (--pairs classified).
  3. Retired cars get the "DNF" outcome instead of a position.

Sampling uses the Gumbel-max trick, which is exactly Plackett-Luce: sort
the cars by S * ln(10) / finish_scale + Gumbel noise. All races are drawn at
once with numpy (10,000 races of a 22-car field take a few hundredths of a
second); a seed makes the result repeatable.

The output has one row per car: P1 ... Pn and DNF (summing to 1), win,
podium, points (finishing inside that season's points positions) and
exp_pos (the expected position if classified), plus the inputs: D, C,
strength and P(DNF).

Points positions follow each season's real scoring: the position most
races of that season paid points down to (results.csv): top 5 in 1950-59,
6 in 1960-2002, 8 in 2003-09, 10 since 2010. A fastest-lap point outside
the scoring places, or a shared drive, doesn't change it.

Possible later improvements, not built:
  * the strength frames carry each driver's ``driver_confidence``; a
    newcomer's rating is a guess, so their strength could get extra spread.
  * retirements are independent per car, so the number of finishers varies
    less between simulated races than between real ones (variance 4.1 vs
    5.9 on 1990-2022; see f1elo/reliability.py). A shared race-level factor
    (a first-lap pile-up, rain) would widen it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .backtest import prepare_races
from .config import DEFAULT_CONFIG, EloConfig
from .data_loader import build_race_frame
from .reliability import (add_dnf_chance, entry_history, next_race_dnf_chances,
                          race_dnf_chances)
from .strength import add_names, next_race_strengths, pre_race_strengths

N_SIMS = 10_000
SEED = 0
DNF = "DNF"
KEYS = ["raceId", "driverId", "constructorId"]


# ---------------------------------------------------------------------------
# The simulation itself
# ---------------------------------------------------------------------------
def simulate_positions(strength, p_dnf, n_sims: int = N_SIMS, seed: int | None = SEED,
                       scale: float = DEFAULT_CONFIG.finish_scale) -> np.ndarray:
    """Simulate ``n_sims`` races; returns an (n_sims, n_cars) int array of
    finishing positions (0 = P1, 1 = P2, ...), with n_cars meaning DNF.

    ``strength``: each car's S; ``p_dnf``: each car's chance of not
    finishing. Same seed, same inputs -> the same races.
    """
    s = np.asarray(strength, float)
    q = np.asarray(p_dnf, float)
    n = len(s)
    rng = np.random.default_rng(seed)
    out = rng.random((n_sims, n)) < q
    keys = s * (math.log(10.0) / scale) + rng.gumbel(size=(n_sims, n))
    keys[out] = -np.inf                      # retired cars sort last
    order = np.argsort(-keys, axis=1, kind="stable")
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.arange(n)[None, :].repeat(n_sims, 0), axis=1)
    pos[out] = n
    return pos


def position_probabilities(pos: np.ndarray) -> np.ndarray:
    """(n_cars, n_cars + 1) array from ``simulate_positions``: row i is car
    i's share of races in P1 ... Pn, then DNF."""
    n_sims, n = pos.shape
    idx = (np.arange(n)[None, :] * (n + 1) + pos).ravel()
    return np.bincount(idx, minlength=n * (n + 1)).reshape(n, n + 1) / n_sims


def simulate_field(field: pd.DataFrame, cfg: EloConfig = DEFAULT_CONFIG,
                   n_sims: int = N_SIMS, seed: int | None = SEED,
                   points_positions: int = 10) -> pd.DataFrame:
    """Add the simulated chances to a field (one row per car, with
    ``strength`` and ``p_dnf``): P1 ... Pn, DNF, win, podium, points and
    exp_pos (expected position if classified, NaN if it never finishes)."""
    pos = simulate_positions(field["strength"], field["p_dnf"], n_sims, seed,
                             cfg.finish_scale)
    probs = position_probabilities(pos)
    n = len(field)
    cols = [f"P{i + 1}" for i in range(n)] + [DNF]
    table = pd.DataFrame(probs, columns=cols, index=field.index)
    fin = probs[:, :n]
    out = pd.concat([field, table], axis=1)
    out["win"] = fin[:, 0]
    out["podium"] = fin[:, :3].sum(axis=1)
    out["points"] = fin[:, :points_positions].sum(axis=1)
    finished = fin.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["exp_pos"] = (fin * np.arange(1, n + 1)).sum(axis=1) / finished
    out.attrs = dict(field.attrs)
    out.attrs.update({"n_sims": n_sims, "seed": seed,
                      "points_positions": points_positions})
    return out


# ---------------------------------------------------------------------------
# Points positions by season
# ---------------------------------------------------------------------------
def season_points_positions(df: pd.DataFrame,
                            cfg: EloConfig = DEFAULT_CONFIG) -> dict[int, int]:
    """{year: points positions}: the lowest classified position that scored
    in each race, the most common value over the season's races."""
    fin = df[df["position"].astype(str) != cfg.dnf_sentinel]
    fin = fin[fin["points"] > 0]
    low = (fin.assign(pos=fin["position"].astype(int))
           .groupby(["year", "raceId"])["pos"].max())
    return {int(y): int(s.mode().iloc[0]) for y, s in low.groupby(level=0)}


def points_positions_for(year, table: dict[int, int], default: int = 10) -> int:
    """The season's points positions; a year not in the data (the next
    season) takes the latest earlier season's."""
    if not table:
        return default
    if year in table:
        return table[year]
    earlier = [y for y in table if year is None or y <= year]
    return table[max(earlier)] if earlier else default


# ---------------------------------------------------------------------------
# Past and next races
# ---------------------------------------------------------------------------
def _merge_dnf(strengths: pd.DataFrame, rel: pd.DataFrame,
               cfg: EloConfig) -> pd.DataFrame:
    """The strength frame (in its own row order) plus each car's
    retirement counts and P(DNF)."""
    cols = ["base_rate", "team_dnfs", "team_entries", "driver_dnfs",
            "driver_entries", "team_rate", "driver_rate", "p_blend"]
    left = strengths.copy()
    right = rel[KEYS + cols].copy()
    left["_n"] = left.groupby(KEYS, dropna=False).cumcount()
    right["_n"] = right.groupby(KEYS, dropna=False).cumcount()
    out = left.merge(right, on=KEYS + ["_n"], how="left").drop(columns="_n")
    out = add_dnf_chance(out, cfg)
    out.attrs = dict(strengths.attrs)
    return out


def _finish(table: pd.DataFrame, names: bool, raw: dict | None) -> pd.DataFrame:
    attrs = dict(table.attrs)
    table = table.sort_values("strength", ascending=False, ignore_index=True)
    if names:
        table = add_names(table, raw)
    table.attrs = attrs
    return table


def predict_race(race_id, *, df: pd.DataFrame | None = None,
                 prepared: list | None = None, hist: pd.DataFrame | None = None,
                 cfg: EloConfig = DEFAULT_CONFIG, n_sims: int = N_SIMS,
                 seed: int | None = SEED, names: bool = True,
                 raw: dict | None = None) -> pd.DataFrame:
    """Simulated chances for a past race, from pre-race data only.

    Ratings are the engine's before that race; retirement counts use only
    earlier races. One row per car, strongest first, with the actual
    result alongside (``position`` as in the data, ``actual`` = the
    position as a number or "DNF"). ``df`` / ``prepared`` / ``hist`` (from
    build_race_frame / prepare_races / reliability.entry_history) can be
    passed in to save rebuilding them for many races.
    """
    df = build_race_frame() if df is None else df
    prepared = prepare_races(df, cfg) if prepared is None else prepared
    idx = next((i for i, r in enumerate(prepared) if r[0] == race_id), None)
    if idx is None:
        raise ValueError(f"raceId {race_id!r} is not in the data")
    strengths = pre_race_strengths([race_id], prepared=prepared[:idx + 1], cfg=cfg,
                                   car_weight=cfg.finish_car_weight)
    rel = race_dnf_chances(race_id, hist=hist, df=df, cfg=cfg)
    field = _merge_dnf(strengths, rel, cfg)
    _, year, rnd, _, _ = prepared[idx]
    pts = points_positions_for(year, season_points_positions(df, cfg))
    info = {"raceId": race_id, "year": year, "round": rnd}
    name = df.loc[df["raceId"] == race_id, "name"]
    info["name"] = name.iloc[0] if len(name) else None
    field.attrs["race"] = info
    table = simulate_field(field, cfg, n_sims, seed, pts)
    table["actual"] = [DNF if str(p) == cfg.dnf_sentinel else int(p)
                       for p in table["position"]]
    return _finish(table, names, raw)


def predict_next_race(entries: pd.DataFrame | None = None, *,
                      df: pd.DataFrame | None = None,
                      prepared: list | None = None,
                      hist: pd.DataFrame | None = None,
                      cfg: EloConfig = DEFAULT_CONFIG, n_sims: int = N_SIMS,
                      seed: int | None = SEED, names: bool = True,
                      races: pd.DataFrame | None = None,
                      raw: dict | None = None) -> pd.DataFrame:
    """Simulated chances for the next, not-yet-run race.

    The field is ``entries`` (driverId, constructorId) if given, else the
    latest race's entry list; ratings and retirement counts are as they
    stand after the latest race in the data. ``.attrs["race"]`` has the
    race (from races.csv).
    """
    df = build_race_frame() if df is None else df
    prepared = prepare_races(df, cfg) if prepared is None else prepared
    strengths = next_race_strengths(entries, prepared=prepared, cfg=cfg,
                                    car_weight=cfg.finish_car_weight, races=races)
    info = strengths.attrs["race"]
    rel = next_race_dnf_chances(strengths[["driverId", "constructorId"]],
                                hist=hist, df=df, cfg=cfg, race_id=info["raceId"],
                                year=info["year"], rnd=info["round"])
    field = _merge_dnf(strengths, rel, cfg)
    pts = points_positions_for(info["year"], season_points_positions(df, cfg))
    table = simulate_field(field, cfg, n_sims, seed, pts)
    return _finish(table, names, raw)


def finishers_distribution(p_dnf) -> np.ndarray:
    """Exact P(exactly m cars finish), m = 0 ... n, for independent
    retirements (a Poisson-binomial distribution)."""
    dist = np.array([1.0])
    for q in np.asarray(p_dnf, float):
        dist = np.convolve(dist, [q, 1.0 - q])
    return dist


def sanity_checks(field: pd.DataFrame, cfg: EloConfig = DEFAULT_CONFIG,
                  n_sims: int = N_SIMS, seed: int | None = SEED) -> dict:
    """Checks that the simulation does what it claims, on one field (with
    ``strength`` and ``p_dnf``). Monte Carlo errors are reported as z
    (difference / standard error); |z| up to about 3-4 is expected for the
    largest of a few hundred.

      rows_sum_max_error   largest |sum of a car's row - 1| (exactly 0)
      column_max_z         each position column's sum against the exact
                           P(at least k cars finish), for the columns with
                           10+ expected races either way
      column_max_abs_error the same, largest absolute difference (all columns)
      finishers            simulated vs exact expected number of finishers
      h2h_max_z / h2h_pairs  for every pair, the simulated share of races
                           (both finished) with A ahead of B against
                           expected_score(S_a, S_b, finish_scale); pairs
                           with under 10 expected races either way skipped
      h2h_share_over_3     share of pairs with |z| > 3 (about 0.003 expected)
      same_seed_identical  two runs with the same seed give identical tables
      other_seed_differs   a different seed gives a different table
      seconds              time for one simulate_positions call
    """
    import time

    from .elo_engine import expected_score

    s = field["strength"].to_numpy(float)
    q = field["p_dnf"].to_numpy(float)
    n = len(s)
    t0 = time.perf_counter()
    pos = simulate_positions(s, q, n_sims, seed, cfg.finish_scale)
    seconds = time.perf_counter() - t0
    probs = position_probabilities(pos)

    # Columns: sum over cars of P(car in Pk) = P(at least k cars finish).
    dist = finishers_distribution(q)
    at_least = dist[::-1].cumsum()[::-1][1:]          # P(m >= k), k = 1..n
    col = probs[:, :n].sum(axis=0)
    # z only where the normal approximation holds (10+ expected races on
    # each side); near-certain columns (P1 with 22 cars) are left to the
    # absolute difference.
    ok = n_sims * np.minimum(at_least, 1 - at_least) >= 10
    se = np.sqrt(np.maximum(at_least * (1 - at_least), 1e-12) / n_sims)
    col_z = ((col - at_least) / se)[ok]

    # Head to head, both finished.
    zs = []
    fin = pos < n
    for i in range(n):
        for j in range(i + 1, n):
            both = fin[:, i] & fin[:, j]
            m = int(both.sum())
            p = expected_score(s[i], s[j], cfg.finish_scale)
            if m * min(p, 1 - p) < 10:   # too lopsided for the normal approx.
                continue
            share = float((pos[both, i] < pos[both, j]).mean())
            zs.append((share - p) / math.sqrt(p * (1 - p) / m))
    zs = np.abs(np.array(zs))

    again = position_probabilities(simulate_positions(s, q, n_sims, seed, cfg.finish_scale))
    other = position_probabilities(simulate_positions(
        s, q, n_sims, (seed or 0) + 1, cfg.finish_scale))
    return {
        "cars": n, "n_sims": n_sims,
        "rows_sum_max_error": float(np.abs(probs.sum(axis=1) - 1).max()),
        "column_max_z": float(np.abs(col_z).max()) if len(col_z) else float("nan"),
        "column_max_abs_error": float(np.abs(col - at_least).max()),
        "finishers_simulated": float(col.sum()),
        "finishers_exact": float((1 - q).sum()),
        "h2h_pairs": int(len(zs)),
        "h2h_max_z": float(zs.max()) if len(zs) else float("nan"),
        "h2h_share_over_3": float((zs > 3).mean()) if len(zs) else float("nan"),
        "same_seed_identical": bool(np.array_equal(probs, again)),
        "other_seed_differs": bool(not np.array_equal(probs, other)),
        "seconds": seconds,
    }


def summary(table: pd.DataFrame, positions: int = 0) -> pd.DataFrame:
    """The readable part of a prediction table: who, the inputs, win /
    podium / points / DNF chances, expected position, the first
    ``positions`` position columns and (past races) the actual result."""
    lead = [c for c in ("driver", "team") if c in table]
    cols = lead + ["driver_elo", "constructor_elo", "strength", "p_dnf",
                   "win", "podium", "points", DNF, "exp_pos"]
    cols += [f"P{i + 1}" for i in range(positions) if f"P{i + 1}" in table]
    if "actual" in table:
        cols.append("actual")
    return table[cols]
