"""How far can the race predictions be trusted? Scores, baselines, calibration.

Everything here works on one table of every entry in every race (the
"fields"), built with ONE engine replay instead of one replay per race:

  1. ``strength.replay`` over all races, recording each car's D, C and the
     driver's confidence before the race (``stage="pre"``) or after that
     weekend's qualifying pass (``stage="post_quali"``);
  2. the retirement counts and chances for all races at once
     (``reliability.features`` / ``add_dnf_chance``), from earlier races
     only;
  3. ``simulate.simulate_race`` per race.

The fields are cached in output/ (``calibration_fields_<stage>.parquet``),
and so are the model's per-race prediction tables
(``race_predictions_<stage>.parquet``, one row per car, P1 ... Pn, DNF, win,
podium, points), keyed on the settings and the data, so reruns and the app
are fast. ``race_table`` gives one race back in the shape of
``simulate.predict_race``.

A prediction (``Pred``) is, per entry, the chance of every finishing
position and of a DNF, plus the win chance; and, per race, the chance of
every number of finishers if the predictor has one. It is scored
(``race_scores``) as:

  win_ll       -ln P(the actual winner wins), per race
  podium_*     log loss / Brier score of "finished in the top 3", per entry
  points_*     the same for "finished in the season's points positions"
  dnf_*        the same for "did not finish"
  rps          ranked probability score over P1 ... Pn then DNF, per entry
               (sum of squared differences of the cumulative predicted and
               actual distributions, divided by n): 0 is perfect
  fin_ll       -ln P(the actual number of finishers), per race
  fin_var      (actual finishers - expected)^2 against the predicted
               variance, pooled over races (ratio 1 = right spread)

Probabilities from simulation counts can be exactly 0 (a backmarker never
on the podium in 10,000 races); they are floored at ``floor(n_sims)`` =
0.5 / n_sims before taking logs. The win chance is computed exactly for
each simulated set of finishers, so it never needs the floor in practice.

The actual result is scored the way the engine counts it: a DNF is
position \\N (did not start included); a classified car's place is its rank
among the classified cars (in 3 races of the 2023-24 data a disqualified
car was given a classified position at the back; the rank keeps the
places contiguous). The 1950s Indianapolis 500 (a separate field) is
left out.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom

from . import config
from .backtest import prepare_races
from .config import DEFAULT_CONFIG, EloConfig
from .data_loader import build_race_frame
from .reliability import add_dnf_chance, entry_history, features, logit
from .simulate import (DNF, KEYS, N_SIMS, SEED, finishers_distribution,
                       points_positions_for, pre_race_order, race_seed,
                       season_points_positions, simulate_race, strength_spread)
from .strength import STAGES, replay

CACHE_VERSION = 2

# Settings that only the simulation reads: changing them doesn't need a
# new engine replay.
SIM_SETTINGS = ("finish_car_weight", "finish_scale", "finish_top_scale",
                "finish_top_places", "finish_newcomer_spread",
                "dnf_race_corr", "dnf_grid_a", "dnf_grid_b", "dnf_grid_c",
                "dnf_grid_pitlane")

OUTCOMES = ("win", "podium", "points", "dnf")
# Calibration bands for the chances (10 bins; narrow at the bottom, where
# most win and podium chances are).
BANDS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)

FIELD_COLUMNS = [
    "race_no", "raceId", "year", "round", "name", "n", "points_positions",
    "driverId", "constructorId", "driver_elo", "constructor_elo",
    "driver_confidence", "starts_before", "grid", "quali_rank", "position",
    "finish", "dnf", "base_rate", "p_blend", "car_rel", "driver_rel", "p_dnf"]


def floor(n_sims: int) -> float:
    """Smallest chance used in a log loss for simulated counts."""
    return 0.5 / n_sims


def is_indy(names) -> np.ndarray:
    """The 1950-60 Indianapolis 500s (a separate field), by race name."""
    return pd.Series(names).astype(str).str.startswith("Indianapolis").to_numpy()


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def data_fingerprint(df: pd.DataFrame) -> str:
    """Changes whenever results are added or changed."""
    h = hashlib.sha1()
    h.update(str(len(df)).encode())
    for c in ("raceId", "driverId", "constructorId", "position", "grid"):
        h.update(pd.util.hash_pandas_object(df[c].astype(str), index=False).values.tobytes())
    return h.hexdigest()[:16]


def _settings_key(cfg: EloConfig, skip=(), **extra) -> str:
    d = {k: v for k, v in asdict(cfg).items() if k not in skip}
    d.update(extra, version=CACHE_VERSION)
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _read_cache(path, key: str) -> pd.DataFrame | None:
    meta = path.with_suffix(".json")
    if not (path.exists() and meta.exists()):
        return None
    try:
        if json.loads(meta.read_text()).get("key") != key:
            return None
        return pd.read_parquet(path)
    except Exception:          # a broken cache is just rebuilt
        return None


def _write_cache(frame: pd.DataFrame, path, key: str, info: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({"key": key, **(info or {})},
                                                    indent=1, default=str))


# ---------------------------------------------------------------------------
# The fields: every entry of every race, with everything a prediction needs
# ---------------------------------------------------------------------------
def build_fields(stage: str = "pre", cfg: EloConfig = DEFAULT_CONFIG,
                 df: pd.DataFrame | None = None, cache: bool = True,
                 progress: bool = False) -> pd.DataFrame:
    """One row per car per race (race order, then finishing order), from
    ONE engine replay. Columns: FIELD_COLUMNS.

    ``stage``: "pre" = ratings before the weekend, "post_quali" = after
    that weekend's qualifying pass (the retirement chance's rating step
    uses the same ratings). ``finish`` = rank among the classified cars
    (0 = DNF); ``n`` = cars in the race; ``points_positions`` = that
    season's scoring places; ``starts_before`` = the driver's earlier
    entries. Cached in output/ unless ``cache`` is False.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    df = build_race_frame() if df is None else df
    key = _settings_key(cfg, SIM_SETTINGS, stage=stage, data=data_fingerprint(df))
    path = config.OUTPUT_DIR / f"calibration_fields_{stage}.parquet"
    if cache:
        hit = _read_cache(path, key)
        if hit is not None:
            return hit
    if progress:
        print(f"  replaying the engine ({stage}) ...", flush=True)

    prepared = prepare_races(df, cfg)
    strengths, _ = replay(prepared, cfg, None, stage, cfg.finish_car_weight)
    hist = entry_history(df, cfg)
    rel = features(hist, cfg)
    rel["starts_before"] = hist.groupby("driverId").cumcount()
    cols = ["base_rate", "p_blend", "starts_before"]
    left = strengths.copy()
    right = rel[KEYS + cols].copy()
    left["_n"] = left.groupby(KEYS, dropna=False).cumcount()
    right["_n"] = right.groupby(KEYS, dropna=False).cumcount()
    out = left.merge(right, on=KEYS + ["_n"], how="left")
    grid = df[KEYS + ["grid"]].copy()
    grid["_n"] = grid.groupby(KEYS, dropna=False).cumcount()
    out = out.merge(grid, on=KEYS + ["_n"], how="left").drop(columns="_n")
    out = add_dnf_chance(out, cfg)

    names = df.groupby("raceId")["name"].first()
    out["name"] = out["raceId"].map(names)
    out = out[~is_indy(out["name"])].reset_index(drop=True)
    out["dnf"] = (out["position"].astype(str) == cfg.dnf_sentinel).astype(np.int8)
    race = out["raceId"].to_numpy()
    out["race_no"] = np.r_[0, np.cumsum(race[1:] != race[:-1])]
    out["n"] = out.groupby("race_no")["raceId"].transform("size")
    # Rows are in finishing order with DNFs last, so the classified rank
    # is a running count.
    fin = 1 - out["dnf"]
    out["finish"] = (fin.groupby(out["race_no"]).cumsum() * fin).astype(int)
    pts = season_points_positions(df, cfg)
    out["points_positions"] = [points_positions_for(y, pts) for y in out["year"]]
    out["grid"] = out["grid"].fillna(0).astype(int)
    out = out[FIELD_COLUMNS]
    if cache:
        _write_cache(out, path, key, {"stage": stage, "rows": len(out)})
    return out


def race_bounds(fields: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Start and end row of every race (race_no 0, 1, ...)."""
    r = fields["race_no"].to_numpy()
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    ends = np.r_[starts[1:], len(r)]
    return starts, ends


def race_frame(fields: pd.DataFrame) -> pd.DataFrame:
    """One row per race: race_no, raceId, year, round, name, n,
    points_positions, finishers (actual)."""
    g = fields.groupby("race_no", sort=True)
    out = g[["raceId", "year", "round", "name", "n", "points_positions"]].first()
    out["finishers"] = g["finish"].apply(lambda s: int((s > 0).sum()))
    return out.reset_index()


def race_mask(fields: pd.DataFrame, period) -> np.ndarray:
    """Which races (by race_no) fall in a (start, end) period."""
    years = race_frame(fields)["year"].to_numpy()
    return (years >= period[0]) & (years <= period[1])


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
@dataclass
class Pred:
    """A predictor's chances for every entry (rows of the fields).

    P    (entries, M) chance of each classified place P1 ... PM (0 past
         the race's field size)
    dnf  (entries,) chance of not finishing
    win  (entries,) chance of winning (P[:, 0] unless computed exactly)
    fin  (races, M + 1) chance of exactly m finishers, or None
    sims number of simulated races behind P (0 = exact), for the floor
    """
    P: np.ndarray
    dnf: np.ndarray
    win: np.ndarray
    fin: np.ndarray | None = None
    sims: int = 0

    def podium(self) -> np.ndarray:
        return self.P[:, :3].sum(axis=1)

    def points(self, positions) -> np.ndarray:
        cum = np.cumsum(self.P, axis=1)
        k = np.clip(np.asarray(positions, int), 1, self.P.shape[1]) - 1
        return cum[np.arange(len(cum)), k]


def _empty(fields: pd.DataFrame, with_fin: bool = True) -> Pred:
    e, m = len(fields), int(fields["n"].max())
    r = int(fields["race_no"].max()) + 1
    return Pred(np.zeros((e, m)), np.full(e, np.nan), np.full(e, np.nan),
                np.zeros((r, m + 1)) if with_fin else None)


def _races_to_do(fields: pd.DataFrame, races) -> np.ndarray:
    r = int(fields["race_no"].max()) + 1
    if races is None:
        return np.arange(r)
    races = np.asarray(races)
    return np.flatnonzero(races) if races.dtype == bool else races


def strength_of(fields: pd.DataFrame, car_weight: float) -> np.ndarray:
    return (fields["driver_elo"] + car_weight * fields["constructor_elo"]).to_numpy(float)


def simulate_all(fields: pd.DataFrame, strength, p_dnf, scale: float,
                 n_sims: int = N_SIMS, seed: int | None = SEED,
                 corr: float = 0.0, spread=None, races=None,
                 top_scale: float | None = None, top_places: int = 0) -> Pred:
    """Simulate every race (or the ``races`` asked for: a mask or a list
    of race_no) with ``simulate_race``. As in ``simulate.simulate_field``,
    each race is simulated in pre-race order (strongest first) with its own
    seed (``race_seed(seed, raceId)``), so a race's table is exactly what
    ``simulate.predict_race`` gives for the same inputs, and the random
    numbers never depend on the finishing order the fields are sorted in."""
    s = np.asarray(strength, float)
    q = np.asarray(p_dnf, float)
    sp = None if spread is None else np.asarray(spread, float)
    ids = fields["raceId"].to_numpy()
    drv = fields["driverId"].to_numpy()
    out = _empty(fields)
    out.sims = n_sims
    starts, ends = race_bounds(fields)
    for r in _races_to_do(fields, races):
        a, b = starts[r], ends[r]
        n = b - a
        rows = a + pre_race_order(s[a:b], drv[a:b])
        pos, win = simulate_race(s[rows], q[rows], n_sims, race_seed(seed, ids[a]),
                                 scale, corr, None if sp is None else sp[rows],
                                 top_scale, top_places)
        idx = (np.arange(n)[None, :] * (n + 1) + pos).ravel()
        probs = np.bincount(idx, minlength=n * (n + 1)).reshape(n, n + 1) / n_sims
        out.P[rows, :n] = probs[:, :n]
        out.dnf[rows] = probs[:, n]
        out.win[rows] = win
        out.fin[r, :n + 1] = np.bincount((pos < n).sum(axis=1), minlength=n + 1) / n_sims
    return out


def model_pred(fields: pd.DataFrame, cfg: EloConfig = DEFAULT_CONFIG,
               n_sims: int = N_SIMS, seed: int | None = SEED, races=None,
               p_dnf=None) -> Pred:
    """The race simulation with ``cfg``'s settings (finish weight and
    scale, shared retirement factor, newcomer spread). ``p_dnf`` replaces
    the fields' chance of not finishing (e.g. adjusted for the grid)."""
    spread = (strength_spread(fields["driver_confidence"], cfg)
              if cfg.finish_newcomer_spread > 0 else None)
    return simulate_all(fields, strength_of(fields, cfg.finish_car_weight),
                        fields["p_dnf"] if p_dnf is None else p_dnf,
                        cfg.finish_scale, n_sims, seed, cfg.dnf_race_corr, spread, races,
                        cfg.finish_top_scale, cfg.finish_top_places)


def exact_finishers(fields: pd.DataFrame, p_dnf, corr: float = 0.0, races=None,
                    m: int | None = None) -> np.ndarray:
    """(races, M + 1): the exact chance of every number of finishers per
    race (``simulate.finishers_distribution``), for the races asked for
    (zeros elsewhere)."""
    q = np.asarray(p_dnf, float)
    m = int(fields["n"].max()) if m is None else m
    out = np.zeros((int(fields["race_no"].max()) + 1, m + 1))
    starts, ends = race_bounds(fields)
    for r in _races_to_do(fields, races):
        a, b = starts[r], ends[r]
        out[r, :b - a + 1] = finishers_distribution(q[a:b], corr)
    return out


def pred_from_table(fields: pd.DataFrame, table: pd.DataFrame, fin=None,
                    sims: int = N_SIMS) -> Pred:
    """A ``Pred`` from a ``prediction_table`` built on the same fields
    (``sims`` = the simulations behind it, for the log-loss floor)."""
    if (len(table) != len(fields)
            or not np.array_equal(table["raceId"].to_numpy(), fields["raceId"].to_numpy())
            or not np.array_equal(table["driverId"].to_numpy(), fields["driverId"].to_numpy())):
        raise ValueError("the prediction table doesn't match the fields")
    m = int(fields["n"].max())
    P = table[[f"P{i + 1}" for i in range(m)]].to_numpy(float)
    return Pred(P, table[DNF].to_numpy(float), table["win"].to_numpy(float), fin, sims)


# ----- baselines -----------------------------------------------------------
def uniform_pred(fields: pd.DataFrame, races=None) -> Pred:
    """Every car the same, retiring at the field's base rate (its DNF share
    over the last dnf_base_races races): exact, no simulation."""
    out = _empty(fields)
    starts, ends = race_bounds(fields)
    base = fields["base_rate"].to_numpy(float)
    for r in _races_to_do(fields, races):
        a, b = starts[r], ends[r]
        n, q = b - a, base[a]
        dist = binom.pmf(np.arange(n + 1), n, 1.0 - q)        # finishers
        at_least = dist[::-1].cumsum()[::-1]                  # P(M >= k)
        out.P[a:b, :n] = at_least[1:] / n
        out.dnf[a:b] = q
        out.win[a:b] = at_least[1] / n
        out.fin[r, :n + 1] = dist
    return out


def all_pairs_pred(fields: pd.DataFrame, cfg: EloConfig = DEFAULT_CONFIG,
                   n_sims: int = N_SIMS, seed: int | None = SEED, races=None) -> Pred:
    """The single all-pairs model, no separate retirement model: one
    Plackett-Luce order over ALL cars on D + prediction_car_weight * C at
    prediction_scale (1.875 / 1175), a retirement simply losing to every
    finisher. To say who retires it needs the number of finishers: drawn
    from the base rate (binomial), the cars ranked below it retire."""
    s = strength_of(fields, cfg.prediction_car_weight)
    base = fields["base_rate"].to_numpy(float)
    a_ = math.log(10.0) / cfg.prediction_scale
    out = _empty(fields)
    out.sims = n_sims
    starts, ends = race_bounds(fields)
    ids = fields["raceId"].to_numpy()
    drv = fields["driverId"].to_numpy()
    for r in _races_to_do(fields, races):
        a, b = starts[r], ends[r]
        n, q = b - a, base[a]
        rows = a + pre_race_order(s[a:b], drv[a:b])
        rng = np.random.default_rng(race_seed(seed, ids[a]))
        keys = s[rows] * a_ + rng.gumbel(size=(n_sims, n))
        order = np.argsort(-keys, axis=1, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(n_sims, 0), axis=1)
        m = rng.binomial(n, 1.0 - q, size=n_sims)
        pos = np.where(rank >= m[:, None], n, rank)
        idx = (np.arange(n)[None, :] * (n + 1) + pos).ravel()
        probs = np.bincount(idx, minlength=n * (n + 1)).reshape(n, n + 1) / n_sims
        out.P[rows, :n] = probs[:, :n]
        out.dnf[rows] = probs[:, n]
        w = np.exp(s[a:b] * a_ - (s[a:b] * a_).max())
        out.win[a:b] = w / w.sum() * (1.0 - q ** n)       # exact
        out.fin[r, :n + 1] = binom.pmf(np.arange(n + 1), n, 1.0 - q)
    return out


def ranks_by(fields: pd.DataFrame, key) -> np.ndarray:
    """1, 2, ... within each race by ``key`` (lower = better; ties keep
    the fields' row order)."""
    tmp = pd.DataFrame({"r": fields["race_no"].to_numpy(),
                        "k": np.asarray(key, float), "i": np.arange(len(fields))})
    return (tmp.sort_values(["r", "k", "i"]).groupby("r").cumcount()
            .sort_index().to_numpy() + 1)


def strength_rank(fields: pd.DataFrame, car_weight: float) -> np.ndarray:
    """1 = the strongest car in the race (D + car_weight * C)."""
    return ranks_by(fields, -strength_of(fields, car_weight))


def grid_rank(fields: pd.DataFrame) -> np.ndarray:
    """1 = pole; pit-lane starts (grid 0) at the back."""
    g = fields["grid"].to_numpy(float)
    return ranks_by(fields, np.where(g > 0, g, 10_000))


def rank_table(fields: pd.DataFrame, rank: np.ndarray, fit_races) -> np.ndarray:
    """Counts of every outcome by rank over the fitting races: (ranks,
    M + 1), columns P1 ... PM then DNF."""
    m = int(fields["n"].max())
    fit = np.asarray(fit_races)[fields["race_no"].to_numpy()]
    fin = fields["finish"].to_numpy()
    col = np.where(fin > 0, fin - 1, m)
    counts = np.zeros((m, m + 1))
    np.add.at(counts, (rank[fit] - 1, col[fit]), 1.0)
    return counts


def rank_table_pred(fields: pd.DataFrame, rank: np.ndarray, counts: np.ndarray,
                    races=None, alpha: float = 0.5) -> Pred:
    """Chances from the historical results of each rank (the k-th
    strongest car, or the k-th grid slot): counts + ``alpha`` per
    possible outcome, normalised. The DNF chances are then shifted (one
    logit shift per race) so the field's mean equals its base rate, the
    same era information every other predictor has: the pooled 1990-2022
    retirement rate (~30%) would be far off in the 2020s (~15%)."""
    m = counts.shape[1] - 1
    seen = np.flatnonzero(counts.sum(axis=1) > 0)
    out = _empty(fields, with_fin=False)
    starts, ends = race_bounds(fields)
    base = fields["base_rate"].to_numpy(float)
    for r in _races_to_do(fields, races):
        a, b = starts[r], ends[r]
        n = b - a
        rows = np.minimum(rank[a:b] - 1, seen.max())
        c = counts[rows][:, list(range(n)) + [m]] + alpha
        p = c / c.sum(axis=1, keepdims=True)
        d = np.clip(p[:, n], 1e-6, 1 - 1e-6)
        target = base[a]
        shift = brentq(lambda t: (1 / (1 + np.exp(-(logit(d) + t)))).mean() - target,
                       -20, 20)
        d2 = 1 / (1 + np.exp(-(logit(d) + shift)))
        out.P[a:b, :n] = p[:, :n] * ((1 - d2) / (1 - d))[:, None]
        out.dnf[a:b] = d2
        out.win[a:b] = out.P[a:b, 0]
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _ll(p, y, eps):
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def entry_chances(fields: pd.DataFrame, pred: Pred) -> pd.DataFrame:
    """Per entry: the predicted win / podium / points / dnf chances and
    what happened (y_*), for calibration tables."""
    fin = fields["finish"].to_numpy()
    pts = fields["points_positions"].to_numpy()
    return pd.DataFrame({
        "race_no": fields["race_no"].to_numpy(), "year": fields["year"].to_numpy(),
        "points_positions": pts, "starts_before": fields["starts_before"].to_numpy(),
        "win": pred.win, "podium": pred.podium(), "points": pred.points(pts),
        "dnf": pred.dnf,
        "y_win": (fin == 1).astype(int), "y_podium": ((fin >= 1) & (fin <= 3)).astype(int),
        "y_points": ((fin >= 1) & (fin <= pts)).astype(int),
        "y_dnf": (fin == 0).astype(int)})


def race_scores(fields: pd.DataFrame, pred: Pred, races=None,
                entries=None) -> pd.DataFrame:
    """Per-race sums of every score (so periods and bootstrap samples can
    be pooled): one row per race asked for. ``entries`` (bool per row)
    restricts the entry-level scores to some entries (e.g. rookies)."""
    eps = floor(pred.sims) if pred.sims else 1e-12
    ch = entry_chances(fields, pred)
    n = fields["n"].to_numpy()
    fin = fields["finish"].to_numpy()
    # Ranked probability score: categories P1 ... Pn, DNF.
    m = pred.P.shape[1]
    k = np.arange(1, m + 1)[None, :]
    cum = np.cumsum(pred.P, axis=1)
    obs = ((fin[:, None] >= 1) & (k >= fin[:, None])).astype(float)
    rps = (((cum - obs) ** 2) * (k <= n[:, None])).sum(axis=1) / n

    e = pd.DataFrame({"race_no": ch["race_no"], "entries": 1.0, "rps": rps})
    for o in ("podium", "points", "dnf"):
        p, y = ch[o].to_numpy(), ch[f"y_{o}"].to_numpy()
        e[f"{o}_ll"] = _ll(p, y, eps)
        e[f"{o}_brier"] = (p - y) ** 2
    # Win: log loss of the winner, per race; also a per-entry Brier.
    e["win_ll"] = np.where(ch["y_win"] == 1,
                           -np.log(np.clip(ch["win"].to_numpy(), 1e-12, 1.0)), 0.0)
    e["win_brier"] = (ch["win"] - ch["y_win"]) ** 2
    if entries is not None:
        keep = np.asarray(entries, bool)
        e.loc[~keep, [c for c in e.columns if c not in ("race_no", "win_ll")]] = 0.0
    per = e.groupby("race_no").sum()
    per["races"] = 1.0

    rf = race_frame(fields).set_index("race_no")
    per = per.join(rf[["raceId", "year", "n", "finishers"]])
    if pred.fin is not None:
        dist = pred.fin[per.index.to_numpy()]
        mm = np.arange(dist.shape[1])[None, :]
        ex = (dist * mm).sum(axis=1)
        var = (dist * (mm - ex[:, None]) ** 2).sum(axis=1)
        act = per["finishers"].to_numpy()
        per["fin_expected"] = ex
        per["fin_var"] = var
        per["fin_sq_err"] = (act - ex) ** 2
        per["fin_err"] = act - ex
        pa = dist[np.arange(len(dist)), act]
        per["fin_ll"] = -np.log(np.clip(pa, eps, 1.0))
    if races is not None:
        want = _races_to_do(fields, races)
        per = per.loc[per.index.intersection(want)]
    return per


# metric -> (numerator column, denominator column)
METRICS = {
    "win_ll": ("win_ll", "races"),
    "podium_ll": ("podium_ll", "entries"), "podium_brier": ("podium_brier", "entries"),
    "points_ll": ("points_ll", "entries"), "points_brier": ("points_brier", "entries"),
    "win_brier": ("win_brier", "entries"), "rps": ("rps", "entries"),
    "dnf_ll": ("dnf_ll", "entries"), "dnf_brier": ("dnf_brier", "entries"),
    "fin_ll": ("fin_ll", "races"),
    "fin_var_ratio": ("fin_sq_err", "fin_var"),
}


def summarise_scores(per: pd.DataFrame) -> dict:
    """Pooled scores over the races in ``per`` (from ``race_scores``)."""
    out = {"races": int(per["races"].sum()), "entries": int(per["entries"].sum())}
    for k, (num, den) in METRICS.items():
        if num in per and den in per and per[den].sum() > 0:
            out[k] = float(per[num].sum() / per[den].sum())
    if "fin_err" in per:
        out["fin_bias"] = float(per["fin_err"].mean())
    return out


def bootstrap(per: pd.DataFrame, metric: str, other: pd.DataFrame | None = None,
              reps: int = 2000, seed: int = 1, level: float = 0.9) -> tuple[float, float]:
    """Confidence interval of a pooled score by resampling races (with
    replacement). With ``other`` (the same races, another predictor), the
    interval of the DIFFERENCE per - other, resampled in pairs."""
    num, den = METRICS[metric]
    a_n, a_d = per[num].to_numpy(float), per[den].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(reps, len(per)))
    stat = a_n[idx].sum(axis=1) / a_d[idx].sum(axis=1)
    if other is not None:
        o = other.loc[per.index]
        stat = stat - o[num].to_numpy(float)[idx].sum(axis=1) / o[den].to_numpy(float)[idx].sum(axis=1)
    lo, hi = np.quantile(stat, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def calibration_bands(p, y, edges=BANDS) -> pd.DataFrame:
    """Predicted vs actual frequency by band of predicted chance: entries,
    mean predicted, actual share, its standard error (from the predicted
    chances: sqrt(sum p(1-p)) / entries) and z = (actual - predicted) / se."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    cut = pd.cut(p, edges, include_lowest=True)
    t = (pd.DataFrame({"band": cut, "p": p, "y": y, "v": p * (1 - p)})
         .groupby("band", observed=True)
         .agg(entries=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"),
              v=("v", "sum")))
    t["se"] = np.sqrt(t["v"]) / t["entries"]
    t["z"] = (t["actual"] - t["predicted"]) / t["se"].replace(0, np.nan)
    t = t.drop(columns="v").reset_index()
    t["band_lo"] = [b.left if b.left > 0 else 0.0 for b in t["band"]]
    t["band_hi"] = [b.right for b in t["band"]]
    return t.drop(columns="band")[["band_lo", "band_hi", "entries", "predicted",
                                   "actual", "se", "z"]]


def points_era(positions) -> np.ndarray:
    """A label per entry: "top 6" etc. (points calibration is per era)."""
    return np.array([f"top {int(p)}" for p in positions])


def finishers_table(fields: pd.DataFrame, pred: Pred, races) -> pd.DataFrame:
    """Number of RETIREMENTS per race (comparable across field sizes):
    actual count of races vs the predicted count (sum over races of the
    chance of that many retirements)."""
    want = _races_to_do(fields, races)
    rf = race_frame(fields).set_index("race_no").loc[want]
    m = pred.fin.shape[1] - 1
    pred_counts = np.zeros(m + 1)
    for r, n in zip(want, rf["n"]):
        d = pred.fin[r, :n + 1]
        pred_counts[:n + 1] += d[::-1]                 # retirements = n - finishers
    act = np.bincount((rf["n"] - rf["finishers"]).to_numpy(), minlength=m + 1)
    out = pd.DataFrame({"retirements": np.arange(m + 1), "actual_races": act,
                        "predicted_races": pred_counts})
    last = int(max(np.flatnonzero(act).max(), np.flatnonzero(pred_counts > 0.5).max()))
    return out.iloc[:last + 1]


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------
def finishers_fit(fields: pd.DataFrame, corrs, races) -> pd.DataFrame:
    """For each shared-factor correlation: the exact finisher-count
    distribution per race (no simulation), scored on the races asked for:
    variance ratio (actual squared error / predicted variance), the mean
    predicted variance and actual squared error, and mean -ln P(actual
    finishers)."""
    starts, ends = race_bounds(fields)
    q = fields["p_dnf"].to_numpy(float)
    act = race_frame(fields)["finishers"].to_numpy()
    want = _races_to_do(fields, races)
    rows = []
    for c in corrs:
        se = var = ll = 0.0
        for r in want:
            a, b = starts[r], ends[r]
            d = finishers_distribution(q[a:b], c)
            mm = np.arange(len(d))
            ex = (d * mm).sum()
            var += (d * (mm - ex) ** 2).sum()
            se += (act[r] - ex) ** 2
            ll -= math.log(max(d[act[r]], 1e-12))
        k = len(want)
        rows.append({"dnf_race_corr": c, "var_ratio": se / var, "pred_var": var / k,
                     "actual_sq_err": se / k, "fin_ll": ll / k})
    return pd.DataFrame(rows)


def pl_order_loglik(fields: pd.DataFrame, car_weight: float, scale: float,
                    races, top: int | None = None) -> float:
    """Exact Plackett-Luce log-likelihood of the classified cars' actual
    order (per race, averaged), with strengths D + w*C at ``scale``.
    ``top`` = only the first ``top`` places (who won, then who was second
    of the rest, ...); None = the whole classified order. Returned as a
    mean loss per race (lower is better)."""
    s = strength_of(fields, car_weight) * math.log(10.0) / scale
    fin = fields["finish"].to_numpy()
    starts, ends = race_bounds(fields)
    tot = 0.0
    want = _races_to_do(fields, races)
    for r in want:
        a, b = starts[r], ends[r]
        x = s[a:b][fin[a:b] > 0]                   # finishing order already
        k = len(x) if top is None else min(top, len(x))
        # log P = sum_j x_j - logsumexp(x_j ... x_m)
        rev = np.logaddexp.accumulate(x[::-1])[::-1]
        tot -= float((x[:k] - rev[:k]).sum())
    return tot / len(want)


def grid_features(fields: pd.DataFrame) -> np.ndarray:
    """(slot, slot^2, pit lane) for the grid-slot retirement step: slot =
    (grid - 1) / (n - 1), 0 = pole, 1 = last; pit-lane starts (grid 0)
    count as last, with their own flag."""
    g = fields["grid"].to_numpy(float)
    n = fields["n"].to_numpy(float)
    pit = (g <= 0).astype(float)
    slot = np.where(pit > 0, 1.0, (np.clip(g, 1, n) - 1) / np.maximum(n - 1, 1))
    return np.column_stack([slot, slot ** 2, pit])


def grid_adjusted_dnf(fields: pd.DataFrame, coef) -> np.ndarray:
    """logit P' = logit P(DNF) + b1 slot + b2 slot^2 + b3 pit lane + a."""
    a, b1, b2, b3 = coef
    x = grid_features(fields)
    z = logit(fields["p_dnf"].to_numpy(float)) + a + x @ np.array([b1, b2, b3])
    return np.clip(1 / (1 + np.exp(-z)), 1e-4, 1 - 1e-4)


def fit_grid_dnf(fields: pd.DataFrame, fit_rows, terms=(True, True, True)) -> np.ndarray:
    """(a, b1, b2, b3) by logistic regression with logit P(DNF) as an
    offset (fitted with its coefficient fixed at 1). ``terms``: which of
    slot, slot^2 and pit lane to fit (the others stay 0)."""
    use = np.asarray(terms, bool)
    x = grid_features(fields)[fit_rows][:, use]
    off = logit(fields["p_dnf"].to_numpy(float))[fit_rows]
    y = fields["dnf"].to_numpy(float)[fit_rows]
    X = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ b + off)))
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    out = np.zeros(4)
    out[0] = b[0]
    out[1:][use] = b[1:]
    return out


# ---------------------------------------------------------------------------
# Cached prediction tables (for reruns and the app)
# ---------------------------------------------------------------------------
def prediction_table(fields: pd.DataFrame, pred: Pred,
                     car_weight: float = DEFAULT_CONFIG.finish_car_weight) -> pd.DataFrame:
    """Long table, one row per car: the fields' identifying columns, D, C,
    strength (D + car_weight * C), P(DNF), P1 ... PM, DNF, win, podium,
    points, exp_pos and the actual result ("DNF" or the classified place)."""
    keep = ["race_no", "raceId", "year", "round", "name", "n", "points_positions",
            "driverId", "constructorId", "driver_elo", "constructor_elo",
            "driver_confidence", "grid", "quali_rank", "position", "p_dnf"]
    out = fields[keep].copy().reset_index(drop=True)
    out.insert(out.columns.get_loc("driver_confidence"), "strength",
               strength_of(fields, car_weight))
    m = pred.P.shape[1]
    pcols = pd.DataFrame(pred.P, columns=[f"P{i + 1}" for i in range(m)])
    out = pd.concat([out, pcols], axis=1)
    out[DNF] = pred.dnf
    out["win"] = pred.win
    out["podium"] = pred.podium()
    out["points"] = pred.points(out["points_positions"])
    fin = pred.P.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["exp_pos"] = (pred.P * np.arange(1, m + 1)).sum(axis=1) / fin
    f = fields["finish"].to_numpy()
    out["actual"] = np.where(f > 0, f.astype(str), DNF)
    return out


def cached_predictions(stage: str = "pre", cfg: EloConfig = DEFAULT_CONFIG,
                       n_sims: int = N_SIMS, seed: int | None = SEED,
                       df: pd.DataFrame | None = None, fields: pd.DataFrame | None = None,
                       rebuild: bool = False, progress: bool = False) -> pd.DataFrame:
    """The model's prediction table for every race (output/
    race_predictions_<stage>.parquet), rebuilt when the settings or the
    data change. With stage "post_quali" and grid coefficients in ``cfg``
    (dnf_grid_*), the retirement chance includes the grid slot."""
    df = build_race_frame() if df is None else df
    key = _settings_key(cfg, (), stage=stage, data=data_fingerprint(df),
                        n_sims=n_sims, seed=seed)
    path = config.OUTPUT_DIR / f"race_predictions_{stage}.parquet"
    if not rebuild:
        hit = _read_cache(path, key)
        if hit is not None:
            return hit
    if fields is None:
        fields = build_fields(stage, cfg, df, cache=not rebuild, progress=progress)
    if progress:
        print(f"  simulating {fields['race_no'].nunique():,} races ({stage}) ...",
              flush=True)
    p_dnf = stage_dnf(fields, stage, cfg)
    table = prediction_table(fields, model_pred(fields, cfg, n_sims, seed, p_dnf=p_dnf),
                             cfg.finish_car_weight)
    table["p_dnf"] = p_dnf
    _write_cache(table, path, key, {"stage": stage, "n_sims": n_sims, "seed": seed,
                                    "settings": {k: getattr(cfg, k) for k in SIM_SETTINGS
                                                 if hasattr(cfg, k)}})
    return table


def stage_dnf(fields: pd.DataFrame, stage: str, cfg: EloConfig = DEFAULT_CONFIG) -> np.ndarray:
    """The retirement chance the model uses at a stage: after qualifying,
    with the grid-slot step if cfg has one (dnf_grid_*), else unchanged."""
    coef = [getattr(cfg, k, 0.0) for k in ("dnf_grid_a", "dnf_grid_b",
                                            "dnf_grid_c", "dnf_grid_pitlane")]
    if stage == "post_quali" and any(coef):
        return grid_adjusted_dnf(fields, coef)
    return fields["p_dnf"].to_numpy(float)


def race_table(predictions: pd.DataFrame, race_id) -> pd.DataFrame:
    """One race from ``cached_predictions``, strongest first, with only its
    own P1 ... Pn columns (like simulate.predict_race)."""
    t = predictions[predictions["raceId"] == race_id]
    if t.empty:
        raise ValueError(f"raceId {race_id!r} is not in the predictions")
    n = int(t["n"].iloc[0])
    drop = [c for c in t.columns if c.startswith("P") and c[1:].isdigit() and int(c[1:]) > n]
    return (t.drop(columns=drop).sort_values("strength", ascending=False)
            .reset_index(drop=True))
