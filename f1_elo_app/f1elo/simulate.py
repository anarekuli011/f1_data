"""Race simulation: each driver's chance of every finishing position.

A race is simulated in two parts, because retirements and pace are
different things (step 2 found 43% of whole-field pairs decided by a
retirement, which ratings barely predict):

  1. Retirements (f1elo/reliability.py): every car retires with its own
     chance P(DNF). The cars share a race factor (dnf_race_corr, a
     Gaussian copula): each keeps exactly its own chance, but in some
     races many retire together (a first-lap pile-up, rain), so the number
     of finishers varies as much as in real races.
  2. Finishing order among the cars that finish: Plackett-Luce on the
     pre-race strengths S = D + finish_car_weight * C. Places are filled
     one at a time, each by a remaining car with probability proportional
     to its weight: 10^(S / finish_top_scale) for the first
     finish_top_places places (the front of a race is more predictable
     than the midfield), 10^(S / finish_scale) for the rest. With
     finish_top_places 0, P(A ahead of B | both finish) =
     expected_score(S_a, S_b, finish_scale) exactly -- the pairwise rule
     the strength backtest scores (--pairs classified).
  3. Retired cars get the "DNF" outcome instead of a position.

Sampling uses the Gumbel-max trick, which is exactly Plackett-Luce: sort
the cars by S * ln(10) / scale + Gumbel noise (the top places one at a
time, each with fresh noise). All races are drawn at once with numpy
(10,000 races of a 22-car field take a few hundredths of a second). Each
race has its own seed (``race_seed``: the seed and the raceId) and is
simulated with its cars in pre-race order (strongest first), so the result
is repeatable and never depends on how the rows were sorted.

The output has one row per car: P1 ... Pn and DNF (summing to 1), win,
podium, points (finishing inside that season's points positions) and
exp_pos (the expected position if classified), plus the inputs: D, C,
strength and P(DNF). ``win`` is the exact chance given each simulated set
of finishers (less Monte Carlo noise than counting wins); P1 is the share
of simulated wins.

Points positions follow each season's real scoring: the position most
races of that season paid points down to (results.csv): top 5 in 1950-59,
6 in 1960-2002, 8 in 2003-09, 10 since 2010. A fastest-lap point outside
the scoring places, or a shared drive, doesn't change it.

How far the chances can be trusted, against fair baselines and with
calibration tables, is measured by backtest_calibration.py
(f1elo/calibration.py). Extra spread for newcomers
(finish_newcomer_spread, via ``driver_confidence``) is supported but off:
it made every score worse.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

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
def draw_retirements(rng: np.random.Generator, p_dnf, n_sims: int,
                     corr: float = 0.0) -> np.ndarray:
    """(n_sims, n_cars) bool array: True = that car retired in that race.

    ``corr`` 0: every car independently, with its own chance. ``corr`` > 0:
    a shared race factor (a Gaussian copula). Car i retires when
    sqrt(corr) * Z + sqrt(1 - corr) * e_i < Phi^-1(p_i), with Z drawn once
    per race and e_i once per car. Each car still retires with exactly its
    own chance p_i, but in a race with a low Z (a first-lap pile-up, rain)
    many cars retire together, so the number of finishers varies more.
    ``corr`` is the correlation of the cars' hidden "trouble" scores.
    """
    q = np.asarray(p_dnf, float)
    n = len(q)
    if corr <= 0:
        return rng.random((n_sims, n)) < q
    z = (math.sqrt(corr) * rng.standard_normal((n_sims, 1))
         + math.sqrt(1.0 - corr) * rng.standard_normal((n_sims, n)))
    return z < ndtri(q)


def simulate_race(strength, p_dnf, n_sims: int = N_SIMS, seed: int | None = SEED,
                  scale: float = DEFAULT_CONFIG.finish_scale, dnf_corr: float = 0.0,
                  spread=None, top_scale: float | None = None,
                  top_places: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``n_sims`` races. Returns ``(pos, win)``:

      pos  (n_sims, n_cars) int array of finishing positions (0 = P1,
           1 = P2, ...), with n_cars meaning DNF;
      win  each car's chance of winning, computed exactly for each
           simulated set of finishers (its Plackett-Luce weight over the
           sum of the finishers' weights) and averaged over the races. Far
           less Monte Carlo noise than counting P1s, and never exactly 0
           for a car that finishes at least once.

    ``strength``: each car's S; ``p_dnf``: each car's chance of not
    finishing; ``dnf_corr``: the shared race factor (``draw_retirements``);
    ``spread``: optional per-car standard deviation (rating points) of
    extra noise added to S in each simulated race -- a rating that is
    only a guess (a newcomer) gets a wider spread of results.

    ``top_scale`` / ``top_places``: the first ``top_places`` places are
    filled one at a time, each by a remaining finisher with probability
    proportional to 10^(S / top_scale); the rest are ordered with
    ``scale`` as usual. A top_scale below scale makes the front of the
    race more predictable than the midfield, which is what real results
    show (see backtest_calibration.py). Off by default (top_places 0).

    Same seed, same inputs -> the same races.
    """
    s = np.asarray(strength, float)
    n = len(s)
    rng = np.random.default_rng(seed)
    out = draw_retirements(rng, p_dnf, n_sims, dnf_corr)
    if spread is not None and np.any(np.asarray(spread) > 0):
        sv = s[None, :] + np.asarray(spread, float)[None, :] * rng.standard_normal((n_sims, n))
    else:
        sv = np.broadcast_to(s, (n_sims, n))
    x = sv * (math.log(10.0) / scale)
    keys = x + rng.gumbel(size=(n_sims, n))
    keys[out] = -np.inf                      # retired cars sort last
    rows = np.arange(n_sims)
    k_top = min(int(top_places), n) if top_scale else 0
    if k_top:
        # The front, place by place, with the top scale (fresh Gumbel noise
        # per place = Plackett-Luce at that scale over the cars left).
        xt = sv * (math.log(10.0) / top_scale)
        left = ~out
        taken = np.zeros(n_sims, int)
        pos = np.full((n_sims, n), n)
        for k in range(k_top):
            kk = np.where(left, xt + rng.gumbel(size=(n_sims, n)), -np.inf)
            pick = kk.argmax(axis=1)
            ok = left[rows, pick]
            pos[rows[ok], pick[ok]] = k
            left[rows[ok], pick[ok]] = False
            taken += ok
        keys[~left] = -np.inf                # placed or retired
        order = np.argsort(-keys, axis=1, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(n_sims, 0), axis=1)
        pos = np.where(left, taken[:, None] + rank, pos)
        x = xt                               # the winner is picked at the top scale
    else:
        order = np.argsort(-keys, axis=1, kind="stable")
        pos = np.empty_like(order)
        np.put_along_axis(pos, order, np.arange(n)[None, :].repeat(n_sims, 0), axis=1)
        pos[out] = n

    # Exact win chance given each race's finishers.
    xw = np.where(out, -np.inf, x)
    top = xw.max(axis=1, keepdims=True)
    top[~np.isfinite(top)] = 0.0             # nobody finished: no winner
    e = np.exp(xw - top)
    tot = e.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(tot > 0, e / tot, 0.0)
    return pos, share.mean(axis=0)


def simulate_positions(strength, p_dnf, n_sims: int = N_SIMS, seed: int | None = SEED,
                       scale: float = DEFAULT_CONFIG.finish_scale,
                       dnf_corr: float = 0.0, spread=None,
                       top_scale: float | None = None, top_places: int = 0) -> np.ndarray:
    """Simulate ``n_sims`` races; returns an (n_sims, n_cars) int array of
    finishing positions (0 = P1, 1 = P2, ...), with n_cars meaning DNF.

    ``strength``: each car's S; ``p_dnf``: each car's chance of not
    finishing; the rest as in ``simulate_race``. Same seed, same inputs ->
    the same races.
    """
    return simulate_race(strength, p_dnf, n_sims, seed, scale, dnf_corr, spread,
                         top_scale, top_places)[0]


def race_seed(seed: int | None, race_id=None):
    """The random seed for one race: ``seed`` combined with the raceId, so
    different races get independent draws (``seed`` alone if there is no
    raceId, e.g. the next race before races.csv lists it)."""
    if seed is None:
        return None
    try:
        rid = int(race_id)
    except (TypeError, ValueError):
        return seed
    return [int(seed), rid]


def pre_race_order(strength, driver_ids=None) -> np.ndarray:
    """Row order in which a field is simulated: strongest first (ties by
    driverId). The draws then never depend on how the rows happened to be
    sorted -- a past race's rows are often in finishing order, and giving
    "the row that won" the same random numbers in every race would make the
    Monte Carlo error line up with the result."""
    s = np.asarray(strength, float)
    tie = (np.zeros(len(s)) if driver_ids is None
           else pd.to_numeric(pd.Series(driver_ids), errors="coerce").fillna(0).to_numpy())
    return np.lexsort((tie, -s))


def strength_spread(confidence, cfg: EloConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Per-car extra spread for the simulation: finish_newcomer_spread *
    (1 - driver confidence). 0 for a driver the engine is sure of."""
    c = np.clip(np.nan_to_num(np.asarray(confidence, float), nan=0.0), 0.0, 1.0)
    return cfg.finish_newcomer_spread * (1.0 - c)


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
    exp_pos (expected position if classified, NaN if it never finishes).

    ``win`` is the exact chance given each simulated set of finishers
    (``simulate_race``); ``P1`` is the share of simulated races won. They
    agree within Monte Carlo error. The shared retirement factor
    (``dnf_race_corr``) and the newcomer spread (``finish_newcomer_spread``,
    if the field has ``driver_confidence``) come from ``cfg``.
    """
    spread = (strength_spread(field["driver_confidence"], cfg)
              if "driver_confidence" in field and cfg.finish_newcomer_spread > 0
              else None)
    # Simulate in pre-race order, then put the rows back.
    order = pre_race_order(field["strength"], field.get("driverId"))
    back = np.argsort(order)
    pos, win = simulate_race(field["strength"].to_numpy(float)[order],
                             field["p_dnf"].to_numpy(float)[order], n_sims, seed,
                             cfg.finish_scale, cfg.dnf_race_corr,
                             None if spread is None else spread[order],
                             cfg.finish_top_scale, cfg.finish_top_places)
    probs = position_probabilities(pos)[back]
    win = win[back]
    n = len(field)
    cols = [f"P{i + 1}" for i in range(n)] + [DNF]
    table = pd.DataFrame(probs, columns=cols, index=field.index)
    fin = probs[:, :n]
    out = pd.concat([field, table], axis=1)
    out["win"] = win
    out["podium"] = fin[:, :3].sum(axis=1)
    out["points"] = fin[:, :points_positions].sum(axis=1)
    finished = fin.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["exp_pos"] = (fin * np.arange(1, n + 1)).sum(axis=1) / finished
    out.attrs = dict(field.attrs)
    out.attrs.update({"n_sims": n_sims,
                      "seed": seed[0] if isinstance(seed, list) else seed,
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
    table = simulate_field(field, cfg, n_sims, race_seed(seed, race_id), pts)
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
    table = simulate_field(field, cfg, n_sims, race_seed(seed, info["raceId"]), pts)
    return _finish(table, names, raw)


def _poisson_binomial(p_dnf) -> np.ndarray:
    dist = np.array([1.0])
    for q in np.asarray(p_dnf, float):
        dist = np.convolve(dist, [q, 1.0 - q])
    return dist


def finishers_distribution(p_dnf, corr: float = 0.0, nodes: int = 40) -> np.ndarray:
    """Exact P(exactly m cars finish), m = 0 ... n.

    Independent retirements (``corr`` 0): a Poisson-binomial distribution.
    With the shared race factor: given the race's Z, the cars are
    independent with chances Phi((Phi^-1(p_i) - sqrt(corr) Z) /
    sqrt(1 - corr)), so the distribution is a mixture of Poisson-binomials
    over Z, integrated with Gauss-Hermite quadrature (``nodes`` points).
    """
    q = np.asarray(p_dnf, float)
    if corr <= 0:
        return _poisson_binomial(q)
    z, w = np.polynomial.hermite_e.hermegauss(nodes)   # weight exp(-z^2 / 2)
    w = w / w.sum()
    t = ndtri(q)
    dist = np.zeros(len(q) + 1)
    for zk, wk in zip(z, w):
        qk = ndtr((t - math.sqrt(corr) * zk) / math.sqrt(1.0 - corr))
        dist += wk * _poisson_binomial(qk)
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
      finishers            simulated vs exact expected number of finishers,
                           and their variance (finishers_distribution)
      h2h_max_z / h2h_pairs  for every pair, the simulated share of races
                           (both finished) with A ahead of B against
                           expected_score(S_a, S_b, finish_scale); pairs
                           with under 10 expected races either way skipped.
                           With a top scale (finish_top_places > 0) this is
                           checked on a separate run with one scale, the
                           core sampler; the top scale is checked by
                           win_vs_p1 (the winner is picked at that scale)
      h2h_share_over_3     share of pairs with |z| > 3 (about 0.003 expected)
      win_vs_p1_max_z      the exact win chance against the share of
                           simulated wins (cars with 10+ expected wins)
      same_seed_identical  two runs with the same seed give identical tables
      other_seed_differs   a different seed gives a different table
      seconds              time for one simulate_race call

    The shared retirement factor (cfg.dnf_race_corr) and the top scale
    are included; the newcomer spread is left out, since it makes the
    head-to-heads deliberately flatter than expected_score.
    """
    import time

    from .elo_engine import expected_score

    s = field["strength"].to_numpy(float)
    q = field["p_dnf"].to_numpy(float)
    n = len(s)
    t0 = time.perf_counter()
    corr = cfg.dnf_race_corr
    top = (cfg.finish_top_scale, cfg.finish_top_places)
    pos, win = simulate_race(s, q, n_sims, seed, cfg.finish_scale, corr, None, *top)
    seconds = time.perf_counter() - t0
    one_scale = (pos if not cfg.finish_top_places
                 else simulate_positions(s, q, n_sims, seed, cfg.finish_scale, corr))
    probs = position_probabilities(pos)

    # Columns: sum over cars of P(car in Pk) = P(at least k cars finish).
    dist = finishers_distribution(q, corr)
    at_least = dist[::-1].cumsum()[::-1][1:]          # P(m >= k), k = 1..n
    col = probs[:, :n].sum(axis=0)
    # z only where the normal approximation holds (10+ expected races on
    # each side); near-certain columns (P1 with 22 cars) are left to the
    # absolute difference.
    ok = n_sims * np.minimum(at_least, 1 - at_least) >= 10
    se = np.sqrt(np.maximum(at_least * (1 - at_least), 1e-12) / n_sims)
    col_z = ((col - at_least) / se)[ok]

    # Head to head, both finished (one scale).
    zs = []
    fin = one_scale < n
    for i in range(n):
        for j in range(i + 1, n):
            both = fin[:, i] & fin[:, j]
            m = int(both.sum())
            p = expected_score(s[i], s[j], cfg.finish_scale)
            if m * min(p, 1 - p) < 10:   # too lopsided for the normal approx.
                continue
            share = float((one_scale[both, i] < one_scale[both, j]).mean())
            zs.append((share - p) / math.sqrt(p * (1 - p) / m))
    zs = np.abs(np.array(zs))

    # The exact win chance against the share of simulated wins.
    p1 = probs[:, 0]
    okw = n_sims * np.minimum(win, 1 - win) >= 10
    wz = (p1 - win) / np.sqrt(np.maximum(win * (1 - win), 1e-12) / n_sims)

    again = position_probabilities(simulate_positions(s, q, n_sims, seed,
                                                      cfg.finish_scale, corr, None, *top))
    other = position_probabilities(simulate_positions(
        s, q, n_sims, (seed or 0) + 1, cfg.finish_scale, corr, None, *top))
    return {
        "cars": n, "n_sims": n_sims,
        "rows_sum_max_error": float(np.abs(probs.sum(axis=1) - 1).max()),
        "column_max_z": float(np.abs(col_z).max()) if len(col_z) else float("nan"),
        "column_max_abs_error": float(np.abs(col - at_least).max()),
        "finishers_simulated": float(col.sum()),
        "finishers_exact": float((1 - q).sum()),
        "finishers_var_simulated": float((n - (pos == n).sum(axis=1)).var()),
        "finishers_var_exact": float((dist * (np.arange(n + 1) - (1 - q).sum()) ** 2).sum()),
        "h2h_pairs": int(len(zs)),
        "h2h_max_z": float(zs.max()) if len(zs) else float("nan"),
        "h2h_share_over_3": float((zs > 3).mean()) if len(zs) else float("nan"),
        "win_vs_p1_max_z": float(np.abs(wz[okw]).max()) if okw.any() else float("nan"),
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
