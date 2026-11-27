"""Retirement model: each car's chance of NOT finishing a race.

The race simulation (f1elo/simulate.py) handles retirements separately from
the finishing order: first each car retires or not, then the cars that
finish are put in order by their strengths. A non-finish is counted exactly
as the engine counts one: ``position`` == ``\\N`` (dnf_sentinel). A car
classified after stopping (completed 90% of the distance) has a position, so
it counts as a finish.

Why not just the ratings? Retirements are mostly down to the car, and the
rate moves hugely with the era (about 45-50% of entries before 2000, 12-18%
since 2010), which a rating doesn't know. So the chance is built from recent
retirement counts, shrunk toward the era's rate (empirical Bayes), and the
ratings only adjust it:

    base   = share of all entries that retired, over the field's last
             dnf_base_races races (so it follows the era)
    team   = (team DNFs + k_t * base) / (team entries + k_t)
             over the team's last dnf_team_races races, all its cars
    driver = (driver DNFs + k_d * base) / (driver entries + k_d)
             over the driver's last dnf_driver_races races
    blend  = (1 - dnf_driver_weight) * team + dnf_driver_weight * driver
    logit P(DNF) = a + b * logit(blend) + c * car_rel + d * driver_rel

k is how many entries' worth of the base rate a count is blended with: a
team or driver with no history gets the base rate, one with a long history
mostly its own. car_rel / driver_rel are the pre-race car and driver
ratings minus the field's mean, per 100 points; a, b, c, d are the
dnf_logit_* settings (0, 1, 0, 0 = the blend unchanged). Only races BEFORE
the one predicted are counted, and the ratings are the pre-race ones.

The settings were tuned by log loss on "did this car retire", 1990-2022,
with 2023 onwards held out (backtest_reliability.py; numbers in config.py).
Log loss falls from 0.5665 (base rate alone) to 0.5560 (blend) to 0.5534
(with the rating step) on 1990-2022, and from 0.3808 to 0.3744 on 2023-2026.

On average the number of finishers comes out right (1990-2022: 0.003 cars
off per race), but races vary more than independent draws allow -- a
first-lap crash or a wet race takes out several cars at once. The variance
of finishers per race around the expected number is 5.9 vs 4.1 for
independent draws on 1990-2022 (2.9 vs 2.2 on 2023-2026). The simulation
therefore draws retirements with a shared race factor (dnf_race_corr, see
f1elo/simulate.py), which keeps every car's chance here unchanged.

After qualifying, the grid slot can shift the chance a little
(dnf_grid_*; the front of the grid retires less), applied by
f1elo/calibration.py for the post-qualifying predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, EloConfig
from .data_loader import build_race_frame

# Base rate used before there is any history at all (the very first race in
# the data); irrelevant from the second race on.
PRIOR_RATE = 0.4

HISTORY_COLUMNS = ["race_idx", "raceId", "year", "round", "driverId",
                   "constructorId", "dnf"]


def entry_history(df: pd.DataFrame | None = None,
                  cfg: EloConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """One row per entry (result row), in race order, with ``dnf`` 0/1.

    ``df`` is ``build_race_frame()`` (built from the data folder if None).
    ``race_idx`` numbers the races 0, 1, 2, ... in chronological order.
    """
    df = build_race_frame() if df is None else df
    df = df.sort_values(["year", "round"], kind="stable")
    out = df[["raceId", "year", "round", "driverId", "constructorId"]].copy()
    out["dnf"] = (df["position"].astype(str) == cfg.dnf_sentinel).astype(np.int8)
    order = {r: i for i, r in enumerate(out["raceId"].unique())}
    out.insert(0, "race_idx", out["raceId"].map(order).astype(np.int64))
    return out.reset_index(drop=True)[HISTORY_COLUMNS]


def with_next_race(hist: pd.DataFrame, entries: pd.DataFrame,
                   race_id=None, year=None, rnd=None) -> pd.DataFrame:
    """``hist`` plus a not-yet-run race with the given entries (driverId,
    constructorId) at the end, so window counts can be read for it. The
    new rows have ``dnf`` 0, which never counts (windows look backwards)."""
    nxt = entries[["driverId", "constructorId"]].copy()
    nxt.insert(0, "race_idx", int(hist["race_idx"].max()) + 1 if len(hist) else 0)
    nxt.insert(1, "raceId", race_id)
    nxt.insert(2, "year", year)
    nxt.insert(3, "round", rnd)
    nxt["dnf"] = np.int8(0)
    return pd.concat([hist, nxt[HISTORY_COLUMNS]], ignore_index=True)


def window_counts(hist: pd.DataFrame, key: str | None,
                  n_races: int | None) -> tuple[np.ndarray, np.ndarray]:
    """(DNFs, entries) before each row, over the last ``n_races`` races.

    ``key`` = "constructorId" (the team's last n races, all its cars),
    "driverId" (the driver's last n races) or None (the whole field's last
    n races). ``n_races`` None = all earlier races. The race of the row
    itself is never included. Returns two float arrays aligned with
    ``hist``'s rows.
    """
    keys = ["race_idx"] if key is None else [key, "race_idx"]
    agg = (hist.groupby(keys, sort=True)["dnf"].agg(["sum", "size"])
           .astype(float))
    cs = agg.groupby(level=0).cumsum() if key is not None else agg.cumsum()

    def shifted(k):
        if key is None:
            return cs.shift(k)
        return cs.groupby(level=0).shift(k)

    before = shifted(1).fillna(0.0)
    if n_races is not None:
        before = before - shifted(n_races + 1).fillna(0.0)
    idx = pd.MultiIndex.from_frame(hist[keys]) if key is not None \
        else pd.Index(hist["race_idx"])
    got = before.reindex(idx)
    return got["sum"].to_numpy(), got["size"].to_numpy()


def base_rate(dnfs: np.ndarray, entries: np.ndarray) -> np.ndarray:
    """Field rate from window counts; PRIOR_RATE where there is no history."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = dnfs / entries
    return np.where(entries > 0, rate, PRIOR_RATE)


def shrunk_rate(dnfs, entries, base, k: float) -> np.ndarray:
    """(dnfs + k * base) / (entries + k): the base rate where there are no
    entries (also for k = 0)."""
    dnfs, entries, base = (np.asarray(x, float) for x in (dnfs, entries, base))
    den = entries + k
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (dnfs + k * base) / den
    return np.where(den > 0, rate, base)


def blend(team, driver, driver_weight: float) -> np.ndarray:
    """(1 - w) * team + w * driver, kept inside (0, 1) for the log loss."""
    p = (1.0 - driver_weight) * np.asarray(team) + driver_weight * np.asarray(driver)
    return np.clip(p, 1e-4, 1.0 - 1e-4)


def logit(p) -> np.ndarray:
    p = np.asarray(p, float)
    return np.log(p / (1.0 - p))


def relative_to_field(values, race_keys) -> np.ndarray:
    """Each value minus its race's mean, in units of 100 rating points."""
    s = pd.Series(np.asarray(values, float))
    return ((s - s.groupby(np.asarray(race_keys)).transform("mean")) / 100.0).to_numpy()


def rating_adjusted(p_blend, car_rel, driver_rel,
                    cfg: EloConfig = DEFAULT_CONFIG) -> np.ndarray:
    """The blended rate, adjusted by the ratings (a logistic regression):

        logit P = a + b * logit(blend) + c * car_rel + d * driver_rel

    car_rel / driver_rel: the car's and driver's pre-race rating minus the
    field's mean, per 100 points (``relative_to_field``). a, b, c, d are
    dnf_logit_intercept / _blend / _car / _driver; 0, 1, 0, 0 gives the
    blend back unchanged.
    """
    z = (cfg.dnf_logit_intercept + cfg.dnf_logit_blend * logit(p_blend)
         + cfg.dnf_logit_car * np.asarray(car_rel, float)
         + cfg.dnf_logit_driver * np.asarray(driver_rel, float))
    return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-4, 1.0 - 1e-4)


def fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Maximum-likelihood coefficients of P(y = 1) = 1 / (1 + exp(-X b))
    (Newton's method). Put a column of ones in X for an intercept."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        step = np.linalg.solve((X * (p * (1.0 - p))[:, None]).T @ X, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def log_loss(p, y) -> float:
    """Mean log loss of chances ``p`` of the outcome y = 1."""
    p = np.clip(np.asarray(p, float), 1e-4, 1.0 - 1e-4)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def features(hist: pd.DataFrame, cfg: EloConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """``hist`` with the pre-race counts and rates added, for every row:
    base_rate, team_dnfs / team_entries, driver_dnfs / driver_entries,
    team_rate, driver_rate and p_blend (the blend, before the rating
    adjustment, which needs the ratings: see ``rating_adjusted``)."""
    out = hist.copy()
    bd, bn = window_counts(hist, None, cfg.dnf_base_races)
    td, tn = window_counts(hist, "constructorId", cfg.dnf_team_races)
    dd, dn = window_counts(hist, "driverId", cfg.dnf_driver_races)
    base = base_rate(bd, bn)
    out["base_rate"] = base
    out["team_dnfs"], out["team_entries"] = td, tn
    out["driver_dnfs"], out["driver_entries"] = dd, dn
    out["team_rate"] = shrunk_rate(td, tn, base, cfg.dnf_team_shrink)
    out["driver_rate"] = shrunk_rate(dd, dn, base, cfg.dnf_driver_shrink)
    out["p_blend"] = blend(out["team_rate"], out["driver_rate"], cfg.dnf_driver_weight)
    return out


def add_dnf_chance(frame: pd.DataFrame, cfg: EloConfig = DEFAULT_CONFIG,
                   race_key: str = "raceId") -> pd.DataFrame:
    """Add car_rel, driver_rel and p_dnf (the final chance of not finishing)
    to a frame with p_blend, driver_elo and constructor_elo (one row per
    car; races told apart by ``race_key``)."""
    out = frame.copy()
    keys = out[race_key].fillna(-1).to_numpy() if race_key in out else np.zeros(len(out))
    out["car_rel"] = relative_to_field(out["constructor_elo"], keys)
    out["driver_rel"] = relative_to_field(out["driver_elo"], keys)
    out["p_dnf"] = rating_adjusted(out["p_blend"], out["car_rel"], out["driver_rel"], cfg)
    return out


def attach_ratings(frame: pd.DataFrame, strengths: pd.DataFrame) -> pd.DataFrame:
    """Add driver_elo, constructor_elo (and driver_confidence) from a
    strength frame (f1elo/strength.py) to a frame of entries, matched on
    raceId, driverId and constructorId. A driver in two cars of one make in
    the same race (1950s) is matched car by car; both have the same ratings.
    """
    keys = ["raceId", "driverId", "constructorId"]
    cols = [c for c in ("driver_elo", "constructor_elo", "driver_confidence")
            if c in strengths]
    left = frame.drop(columns=[c for c in cols if c in frame]).copy()
    right = strengths[keys + cols].copy()
    left["_n"] = left.groupby(keys).cumcount()
    right["_n"] = right.groupby(keys).cumcount()
    return left.merge(right, on=keys + ["_n"], how="left").drop(columns="_n")


def race_dnf_chances(race_ids=None, *, hist: pd.DataFrame | None = None,
                     df: pd.DataFrame | None = None,
                     cfg: EloConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Every entry's chance of not finishing, from the races before it.

    ``race_ids``: one raceId, a list, or None for every race. One row per
    entry with raceId, year, round, driverId, constructorId, the counts and
    rates of ``features`` and ``dnf`` (what actually happened).
    """
    hist = entry_history(df, cfg) if hist is None else hist
    out = features(hist, cfg)
    if race_ids is not None:
        if not isinstance(race_ids, (list, tuple, set, frozenset)):
            race_ids = [race_ids]
        out = out[out["raceId"].isin(list(race_ids))]
    return out.reset_index(drop=True)


def next_race_dnf_chances(entries: pd.DataFrame, *,
                          hist: pd.DataFrame | None = None,
                          df: pd.DataFrame | None = None,
                          cfg: EloConfig = DEFAULT_CONFIG,
                          race_id=None, year=None, rnd=None) -> pd.DataFrame:
    """Chances of not finishing for a race that hasn't been run: ``entries``
    (driverId, constructorId), counted over every race in the data."""
    hist = entry_history(df, cfg) if hist is None else hist
    full = with_next_race(hist, entries, race_id, year, rnd)
    out = features(full, cfg)
    return out.iloc[len(hist):].drop(columns="dnf").reset_index(drop=True)
