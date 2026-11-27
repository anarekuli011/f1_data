"""Race features for the step 6 models: one row per car per race.

Every value in a row is something that was known at that stage of the race
weekend, built from earlier races only (plus, after qualifying, that
weekend's qualifying and grid, and after a sprint, the sprint result):

  stage "pre"          before the weekend
  stage "post_quali"   + that weekend's qualifying and starting grid
  stage "post_sprint"  + the sprint result (sprint weekends, 2021 on)

The table joins to ``calibration.cached_predictions`` on raceId + driverId
(unique from 1965; for the 1950s shared drives use raceId, driverId,
constructorId and ``entry``, the car's number among that driver's cars of
that make in the race). Rows are in the same order as the predictions: race
order, then finishing order. Values that weren't recorded in an era (lap
times before 1996, pit stops before 2011, qualifying times before 1994) are
missing (NaN), never 0. Columns starting ``y_`` are what happened in the
race (the labels for step 6b and 6c), never features.

Feature groups (column prefixes; the README lists every column):

  elo_*     the ELO model's inputs and outputs for this stage
            (calibration.cached_predictions; after a sprint the same as
            after qualifying, since the engine has no sprint pass)
  dnf_*     the retirement model's pre-race rates (reliability.features)
  q_*       qualifying and grid (post_quali on)
  s_*       sprint result (post_sprint)
  drv_* / team_*
            recent form over the driver's / team's last ``form_races``
            races: gap to the winner, places gained, finishing rate, race
            pace, lap-1 gains, consistency, fastest-lap gap, pit stops
  circ_*    the circuit's history, relative to the field at the time
  context   era, races into the era, round, field size, sprint weekend
  driver    age, starts, starts with this team, home race

Relative versions: ``<feature>_rel`` is the feature minus the mean of the
cars in that race that have it (the levels drift with the era; the gaps
between cars are what a ranking model needs).

Clean laps (race pace and consistency), per race:
  1. lap 1 is dropped, and so are each driver's pit in-laps and out-laps
     (the lap of the stop and the next, from pit_stops; 2011 on);
  2. each lap's field median is taken over the remaining laps; a lap whose
     field median is more than ``neutralised_lap`` (5%) slower than the
     race's median lap is neutralised (safety car, VSC, a shower) and
     dropped for everyone;
  3. a driver's lap more than ``slow_lap`` (10%) slower, or faster, than
     that lap's field median is dropped (pit laps before 2011, spins,
     damage; timing glitches);
  4. pace = the median of the driver's remaining laps relative to each
     lap's field median, in % (negative = faster than the median car);
     consistency = 1.4826 x the median absolute deviation of the same, in
     % (a robust standard deviation). At least ``min_clean_laps`` laps.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import config
from .config import DEFAULT_CONFIG, DEFAULT_FEATURES, EloConfig, FeatureConfig
from .data_loader import load_raw
from .reliability import HISTORY_COLUMNS, features as dnf_rates

STAGES = ("pre", "post_quali", "post_sprint")
FEATURE_VERSION = 1
KEYS = ["raceId", "driverId", "constructorId"]
# A qualifying time more than this far from its session's median is a
# timing glitch (the 99.9th percentile of best times is 1.22 x the race
# median; the few beyond 1.3 are times like 16:42.640).
QUALI_OUTLIER = 0.25
ID_COLUMNS = ["raceId", "year", "round", "name", "circuitId", "driverId",
              "constructorId", "entry", "n"]

# ELO columns taken from calibration.cached_predictions, renamed.
ELO_COLUMNS = {"driver_elo": "elo_driver", "constructor_elo": "elo_car",
               "strength": "elo_strength", "driver_confidence": "elo_confidence",
               "p_dnf": "elo_p_dnf", "win": "elo_win", "podium": "elo_podium",
               "points": "elo_points", "exp_pos": "elo_exp_pos"}

# Per-race outcomes that the form features average (the race's own value is
# never used for that race): name -> description.
FORM_VALUES = {
    "deficit": "gap to the winner, % of the race (lead lap: time; lapped: laps down / race laps)",
    "laps_down": "laps behind the winner (classified cars)",
    "gained": "places gained from grid to finish (classified cars)",
    "finished": "finishing rate",
    "pace": "race pace: median clean lap vs the field median, % (1996 on)",
    "lap1_gain": "places gained on lap 1 (1996 on)",
    "consistency": "robust spread of clean laps, % (1996 on)",
    "fastlap": "fastest lap vs the race's fastest, % (2004 on)",
}
CIRCUIT_VALUES = {
    "grid_corr": "grid-to-finish rank correlation of classified cars (1 = order holds)",
    "lap1_out": "share of starters out on lap 1",
    "dnf": "retirement rate",
    "pole_win": "share of races won from pole",
}

# Features that also get a relative version (minus the race mean).
REL_FEATURES = (
    ["elo_driver", "elo_car", "elo_strength", "elo_confidence", "elo_p_dnf",
     "dnf_team", "dnf_driver", "age", "starts", "team_starts",
     "q_gap_pole", "q_gap_tm"]
    + [f"{who}_{v}" for who in ("drv", "team") for v in FORM_VALUES]
    + ["team_pit"])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def is_indy(names) -> np.ndarray:
    """The 1950-60 Indianapolis 500s (a separate field), by race name."""
    return pd.Series(names).astype(str).str.startswith("Indianapolis").to_numpy()


def laptime_ms(text: pd.Series) -> pd.Series:
    """'1:32.456' or '92.456' -> milliseconds (NaN when missing or malformed)."""
    m = text.astype("string").str.extract(r"^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*$")
    mins = pd.to_numeric(m[0], errors="coerce").fillna(0.0)
    secs = pd.to_numeric(m[1], errors="coerce")
    return ((mins * 60.0 + secs) * 1000.0).astype(float)


def prior_mean(frame: pd.DataFrame, key: str, cols: list[str],
               n_races: int | None) -> tuple[pd.DataFrame, np.ndarray]:
    """For every row: the mean of each column over the ``key``'s last
    ``n_races`` races BEFORE the row's race (``race_idx``), skipping
    missing values (NaN where there are none), and the number of the
    key's races in that window."""
    keys = [key, "race_idx"]
    vals = frame[cols].astype(float)
    parts = pd.concat([frame[keys].reset_index(drop=True),
                       vals.fillna(0.0).add_suffix("__s").reset_index(drop=True),
                       vals.notna().astype(float).add_suffix("__n").reset_index(drop=True)],
                      axis=1)
    agg = parts.groupby(keys, sort=True).sum()
    agg["__races"] = 1.0                  # one row per race of the key
    cs = agg.groupby(level=0).cumsum()
    before = cs.groupby(level=0).shift(1).fillna(0.0)
    if n_races is not None:
        before = before - cs.groupby(level=0).shift(n_races + 1).fillna(0.0)
    got = before.reindex(pd.MultiIndex.from_frame(frame[keys]))
    out = pd.DataFrame(index=frame.index)
    for c in cols:
        s, n = got[f"{c}__s"].to_numpy(), got[f"{c}__n"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
    return out, np.round(got["__races"].to_numpy())


def relative(values: pd.Series, race: pd.Series) -> pd.Series:
    """Each value minus the mean of its race's non-missing values."""
    return values - values.groupby(race).transform("mean")


def dnf_cause(status: pd.Series, dnf: pd.Series) -> pd.Series:
    """'finished', 'incident', 'mechanical', 'other' or 'unknown'."""
    s = status.fillna("")
    out = np.where(s.isin(config.DNF_INCIDENT), "incident",
                   np.where(s.isin(config.DNF_OTHER), "other",
                            np.where(s.isin(config.DNF_UNKNOWN) | (s == ""), "unknown",
                                     "mechanical")))
    return pd.Series(np.where(dnf.to_numpy(bool), out, "finished"), index=status.index)


# ---------------------------------------------------------------------------
# Entries and what happened in each race
# ---------------------------------------------------------------------------
def entries(raw: dict) -> pd.DataFrame:
    """Every result row with its race, in race order then finishing order
    (DNFs last), plus race_idx, entry, n, dnf, finish, grid_rank and the
    per-race outcomes the form features average. Indy 500s included (they
    count in the retirement rates); ``indy`` marks them."""
    races = raw["races"].copy()
    races["date"] = pd.to_datetime(races["date"], errors="coerce")
    rcols = ["raceId", "year", "round", "circuitId", "name", "date", "sprint_date"]
    res = raw["results"]
    e = res.merge(races[rcols], on="raceId", how="left")
    # Finishing order as the engine has it: classified cars by position,
    # then the DNFs (a few races list an unclassified car's positionOrder
    # among the classified ones).
    e["_cls_pos"] = e["position"].fillna(10_000)
    e = (e.sort_values(["year", "round", "_cls_pos", "positionOrder"], kind="stable")
         .drop(columns="_cls_pos").reset_index(drop=True))
    order = {r: i for i, r in enumerate(e["raceId"].unique())}
    e["race_idx"] = e["raceId"].map(order).astype(np.int64)
    e["entry"] = e.groupby(KEYS).cumcount()
    e["indy"] = is_indy(e["name"])
    e["n"] = e.groupby("raceId")["raceId"].transform("size")
    status = raw["status"].set_index("statusId")["status"] if len(raw["status"]) else pd.Series(dtype=str)
    e["status"] = e["statusId"].map(status)
    e["dnf"] = e["position"].isna().astype(np.int8)
    cls = e["dnf"] == 0
    e["finish"] = 0
    e.loc[cls, "finish"] = e[cls].groupby("raceId").cumcount() + 1   # rows are in order
    g = e["grid"].fillna(0)
    e["grid_rank"] = (pd.DataFrame({"r": e["raceId"], "g": np.where(g > 0, g, 10_000),
                                    "o": e["positionOrder"]})
                      .sort_values(["r", "g", "o"]).groupby("r").cumcount()
                      .reindex(e.index) + 1)

    # Gap to the winner.
    win = e[e["finish"] == 1].drop_duplicates("raceId").set_index("raceId")
    w_laps = e["raceId"].map(win["laps"])
    w_ms = e["raceId"].map(win["milliseconds"])
    lead = cls & (e["laps"] >= w_laps)
    e["laps_down"] = np.where(cls, (w_laps - e["laps"]).clip(lower=0), np.nan)
    e["deficit"] = np.where(lead & e["milliseconds"].notna() & w_ms.notna(),
                            (e["milliseconds"] - w_ms) / w_ms * 100.0,
                            np.where(cls & ~lead, e["laps_down"] / w_laps * 100.0, np.nan))
    e["gained"] = np.where(cls, e["grid_rank"] - e["finish"], np.nan)
    e["finished"] = 1.0 - e["dnf"]
    fl = laptime_ms(e["fastestLapTime"]) if "fastestLapTime" in e else pd.Series(np.nan, e.index)
    best = fl.groupby(e["raceId"]).transform("min")
    e["fastlap"] = (fl - best) / best * 100.0
    e["race_laps"] = w_laps
    e["date"] = pd.to_datetime(e["date"])
    return e


def lap_metrics(e: pd.DataFrame, laps: pd.DataFrame, pits: pd.DataFrame,
                fcfg: FeatureConfig) -> pd.DataFrame:
    """Per (raceId, driverId): pace, consistency and clean laps (see the
    module docstring) and lap1_pos, the position after lap 1."""
    cols = ["raceId", "driverId", "pace", "consistency", "clean_laps", "lap1_pos"]
    if laps is None or not len(laps):
        return pd.DataFrame(columns=cols)
    lt = laps[laps["raceId"].isin(e["raceId"])].dropna(subset=["milliseconds", "lap"])
    lt = lt[["raceId", "driverId", "lap", "position", "milliseconds"]].copy()
    lap1 = (lt[lt["lap"] == 1].groupby(["raceId", "driverId"])["position"].min()
            .rename("lap1_pos"))
    if pits is not None and len(pits):
        p = pits[["raceId", "driverId", "lap"]].dropna()
        pit_laps = pd.concat([p, p.assign(lap=p["lap"] + 1)]).drop_duplicates()
        pit_laps["pit"] = True
        lt = lt.merge(pit_laps, on=["raceId", "driverId", "lap"], how="left")
        lt = lt[lt["pit"].isna()].drop(columns="pit")
    lt = lt[lt["lap"] >= 2]
    lap_med = lt.groupby(["raceId", "lap"])["milliseconds"].median().rename("lap_med")
    race_med = lap_med.groupby(level=0).median().rename("race_med")
    lt = lt.join(lap_med, on=["raceId", "lap"]).join(race_med, on="raceId")
    lt = lt[lt["lap_med"] <= (1.0 + fcfg.neutralised_lap) * lt["race_med"]]
    lt["rel"] = lt["milliseconds"] / lt["lap_med"] - 1.0
    lt = lt[lt["rel"].abs() <= fcfg.slow_lap]
    g = lt.groupby(["raceId", "driverId"])["rel"]
    med = g.median()
    mad = (lt["rel"] - lt.join(med.rename("m"), on=["raceId", "driverId"])["m"]).abs()
    out = pd.DataFrame({"pace": med * 100.0,
                        "consistency": mad.groupby([lt["raceId"], lt["driverId"]]).median()
                        * 1.4826 * 100.0,
                        "clean_laps": g.size()})
    few = out["clean_laps"] < fcfg.min_clean_laps
    out.loc[few, ["pace", "consistency"]] = np.nan
    out = out.join(lap1, how="outer").reset_index()
    return out[cols]


def pit_metrics(e: pd.DataFrame, pits: pd.DataFrame, fcfg: FeatureConfig) -> pd.DataFrame:
    """Per (raceId, constructorId): ``pit`` = the team's median stop that
    race minus the race's median stop, in seconds (stops over
    ``pit_outlier`` x the race median, and over 2 minutes, left out)."""
    if pits is None or not len(pits):
        return pd.DataFrame(columns=["raceId", "constructorId", "pit"])
    p = pits[pits["raceId"].isin(e["raceId"])].dropna(subset=["milliseconds"])
    p = p[p["milliseconds"] <= 120_000]
    med = p.groupby("raceId")["milliseconds"].transform("median")
    p = p[p["milliseconds"] <= fcfg.pit_outlier * med].copy()
    p["delta"] = (p["milliseconds"] - p.groupby("raceId")["milliseconds"].transform("median")) / 1000.0
    team = e.drop_duplicates(["raceId", "driverId"]).set_index(["raceId", "driverId"])["constructorId"]
    p["constructorId"] = team.reindex(pd.MultiIndex.from_frame(p[["raceId", "driverId"]])).to_numpy()
    p = p.dropna(subset=["constructorId"])
    return (p.groupby(["raceId", "constructorId"])["delta"].median().rename("pit")
            .reset_index())


def race_metrics(e: pd.DataFrame) -> pd.DataFrame:
    """Per race (Indy 500s left out), for the circuit features: grid_corr,
    lap1_out, dnf, pole_win, with race_idx and circuitId."""
    x = e[~e["indy"]]
    cls = x[(x["finish"] > 0) & (x["grid"] > 0)]

    def spearman(g):
        if len(g) < 5:
            return np.nan
        a = g["grid_rank"].rank().to_numpy()
        b = g["finish"].rank().to_numpy()
        return float(np.corrcoef(a, b)[0, 1])

    corr = cls.groupby("raceId")[["grid_rank", "finish"]].apply(spearman)
    started = x[~x["status"].isin(config.DID_NOT_START)]
    lap1 = (started.assign(o=(started["dnf"] == 1) & (started["laps"] == 0))
            .groupby("raceId")["o"].mean())
    winners = x[x["finish"] == 1].drop_duplicates("raceId").set_index("raceId")
    out = x.groupby("raceId").agg(race_idx=("race_idx", "first"),
                                  circuitId=("circuitId", "first"),
                                  dnf=("dnf", "mean"))
    out["grid_corr"] = corr
    out["lap1_out"] = lap1
    out["pole_win"] = (winners["grid"] == 1).astype(float)
    return out.sort_values("race_idx").reset_index()


def circuit_features(e: pd.DataFrame, rm: pd.DataFrame, fcfg: FeatureConfig) -> pd.DataFrame:
    """Per race: for each circuit value, ``circ_<v>`` = the field's level
    over its last circuit_base_races races + this circuit's typical
    difference from the field at the time, over its last circuit_visits
    EARLIER races, shrunk toward 0 by circuit_shrink races;
    ``circ_<v>_rel`` = that difference alone; ``circ_races`` = earlier
    races at the circuit (all of them)."""
    out = pd.DataFrame({"raceId": rm["raceId"]})
    b, k, w = fcfg.circuit_base_races, fcfg.circuit_shrink, fcfg.circuit_visits
    circ = rm["circuitId"]

    def last_visits(x: pd.Series) -> pd.Series:
        """Sum of x over the circuit's last w races before each race."""
        cs = x.groupby(circ).cumsum()
        before = cs.groupby(circ).shift(1).fillna(0.0)
        return before - cs.groupby(circ).shift(w + 1).fillna(0.0)

    for v in CIRCUIT_VALUES:
        m = rm[v]
        ref_incl = m.rolling(b, min_periods=1).mean()     # field level up to and incl. the race
        ref_before = m.shift(1).rolling(b, min_periods=1).mean()
        dev = m - ref_incl
        eff = last_visits(dev.fillna(0.0)) / (last_visits(dev.notna().astype(float)) + k)
        out[f"circ_{v}"] = ref_before + eff
        out[f"circ_{v}_rel"] = eff
    out["circ_races"] = rm.groupby("circuitId").cumcount()
    return out


# ---------------------------------------------------------------------------
# Qualifying and sprint (this weekend)
# ---------------------------------------------------------------------------
def qualifying_features(e: pd.DataFrame, quali: pd.DataFrame) -> pd.DataFrame:
    """Per (raceId, driverId): q_pos (qualifying position), q_gap_pole (best
    session time vs the fastest, %), q_gap_tm (vs the team's fastest other
    car, in the deepest session both set a time in, %)."""
    cols = ["raceId", "driverId", "q_pos", "q_gap_pole", "q_gap_tm"]
    if quali is None or not len(quali):
        return pd.DataFrame(columns=cols)
    q = quali[quali["raceId"].isin(e["raceId"])].copy()
    for s in ("q1", "q2", "q3"):
        q[s] = laptime_ms(q[s])
        # Timing glitches ("16:42.640"): more than quali_outlier off the
        # session's median time.
        med = q.groupby("raceId")[s].transform("median")
        q.loc[(q[s] - med).abs() > QUALI_OUTLIER * med, s] = np.nan
    q = q.groupby(["raceId", "driverId"], as_index=False).agg(
        constructorId=("constructorId", "first"), q_pos=("position", "min"),
        q1=("q1", "min"), q2=("q2", "min"), q3=("q3", "min"))
    q["best"] = q[["q1", "q2", "q3"]].min(axis=1)
    pole = q.groupby("raceId")["best"].transform("min")
    q["q_gap_pole"] = (q["best"] - pole) / pole * 100.0
    mates = q.merge(q[["raceId", "constructorId", "driverId", "best", "q1", "q2", "q3"]],
                    on=["raceId", "constructorId"], suffixes=("", "_tm"))
    mates = mates[mates["driverId"] != mates["driverId_tm"]].dropna(subset=["best_tm"])
    mates = mates.sort_values("best_tm").drop_duplicates(["raceId", "driverId"])
    gap = pd.Series(np.nan, index=mates.index)
    for s in ("q1", "q2", "q3"):           # deepest session last, so it wins
        both = mates[s].notna() & mates[f"{s}_tm"].notna()
        gap[both] = (mates.loc[both, s] - mates.loc[both, f"{s}_tm"]) / mates.loc[both, f"{s}_tm"] * 100.0
    mates["q_gap_tm"] = gap
    q = q.merge(mates[["raceId", "driverId", "q_gap_tm"]], on=["raceId", "driverId"], how="left")
    return q[cols]


def sprint_features(e: pd.DataFrame, sprint: pd.DataFrame) -> pd.DataFrame:
    """Per (raceId, driverId): s_finish (rank among classified, NaN = out),
    s_dnf, s_gained (sprint grid rank - sprint finish), s_grid."""
    cols = ["raceId", "driverId", "s_finish", "s_dnf", "s_gained", "s_grid"]
    if sprint is None or not len(sprint):
        return pd.DataFrame(columns=cols)
    s = sprint[sprint["raceId"].isin(e["raceId"])].copy()
    s["s_dnf"] = s["position"].isna().astype(float)
    s["s_finish"] = s[s["s_dnf"] == 0].groupby("raceId")["positionOrder"].rank(method="first")
    g = s["grid"].fillna(0)
    s["s_grid"] = (s.assign(gg=np.where(g > 0, g, 10_000)).groupby("raceId")["gg"]
                   .rank(method="first"))
    s["s_gained"] = s["s_grid"] - s["s_finish"]
    return s.drop_duplicates(["raceId", "driverId"])[cols]


# ---------------------------------------------------------------------------
# The whole table
# ---------------------------------------------------------------------------
def elo_table(stage: str, cfg: EloConfig = DEFAULT_CONFIG, df=None) -> pd.DataFrame:
    """The ELO model's columns for a stage (calibration.cached_predictions,
    built and cached if needed), keyed on KEYS + entry, plus
    points_positions."""
    from .calibration import cached_predictions
    p = cached_predictions("post_quali" if stage == "post_sprint" else stage, cfg, df=df)
    out = p[KEYS + ["points_positions"] + list(ELO_COLUMNS)].rename(columns=ELO_COLUMNS)
    out.insert(3, "entry", p.groupby(KEYS).cumcount().to_numpy())
    out["_pred_row"] = np.arange(len(out))      # to keep the predictions' row order
    return out


def build_all(raw: dict, fcfg: FeatureConfig = DEFAULT_FEATURES,
              cfg: EloConfig = DEFAULT_CONFIG, elo: pd.DataFrame | None = None,
              timings: dict | None = None) -> pd.DataFrame:
    """Every feature of every stage (no stage selection), one row per car
    per race, Indy 500s left out. ``elo``: the ELO columns (``elo_table``)
    to join; None leaves them out."""
    t = time.perf_counter()
    lap = lambda name: timings.__setitem__(name, round(time.perf_counter() - t, 2)) \
        if timings is not None else None
    e = entries(raw)
    lap("entries")

    # Retirement-model rates, over every entry (as the ELO model counts them).
    hist = e[["race_idx", "raceId", "year", "round", "driverId", "constructorId", "dnf"]]
    rates = dnf_rates(hist[HISTORY_COLUMNS], cfg)
    e["dnf_base"] = rates["base_rate"].to_numpy()
    e["dnf_team"] = rates["team_rate"].to_numpy()
    e["dnf_driver"] = rates["driver_rate"].to_numpy()

    # Lap and pit data, then the per-race values the form features average.
    lm = lap_metrics(e, raw.get("lap_times"), raw.get("pit_stops"), fcfg)
    e = e.merge(lm, on=["raceId", "driverId"], how="left")
    e["lap1_gain"] = e["grid_rank"] - e["lap1_pos"]
    pm = pit_metrics(e, raw.get("pit_stops"), fcfg)
    e = e.merge(pm, on=["raceId", "constructorId"], how="left")
    lap("laps")

    form = e[~e["indy"]].copy()
    vals = list(FORM_VALUES)
    for who, key in (("drv", "driverId"), ("team", "constructorId")):
        m, races = prior_mean(form, key, vals + (["pit"] if who == "team" else []),
                              fcfg.form_races)
        for c in m.columns:
            form[f"{who}_{c}"] = m[c].to_numpy()
        form[f"{who}_races"] = races
    form = form.drop(columns=["drv_pit"], errors="ignore")
    lap("form")

    # Circuit history.
    rm = race_metrics(e)
    cf = circuit_features(e, rm, fcfg)
    out = form.merge(cf, on="raceId", how="left")
    circuits = raw["circuits"].set_index("circuitId") if len(raw["circuits"]) else None
    ref = out["circuitId"].map(circuits["circuitRef"]) if circuits is not None else None
    out["street"] = (ref.isin(config.STREET_CIRCUITS).astype(float) if ref is not None
                     else np.nan)
    lap("circuit")

    # Context.
    eras = sorted(cfg.regulation_change_years)
    out["era_start"] = [max([y0 for y0 in eras if y0 <= y], default=1950) for y in out["year"]]
    first_of_era = out.drop_duplicates("raceId")[["raceId", "race_idx", "era_start"]].copy()
    first_of_era["era_race"] = first_of_era.groupby("era_start").cumcount()
    out["era_race"] = out["raceId"].map(first_of_era.set_index("raceId")["era_race"])
    cal = raw["races"]
    rounds = cal.groupby("year")["round"].max()
    out["season_progress"] = out["round"] / out["year"].map(rounds)
    out["field_size"] = out["n"]
    out["sprint_weekend"] = out["sprint_date"].notna().astype(float)

    # Driver.
    drv = raw["drivers"].set_index("driverId")
    dob = pd.to_datetime(out["driverId"].map(drv["dob"]), errors="coerce")
    out["age"] = (out["date"] - dob).dt.days / 365.25
    out["starts"] = out.groupby("driverId").cumcount()
    out["team_starts"] = out.groupby(["driverId", "constructorId"]).cumcount()
    if circuits is not None:
        country = out["circuitId"].map(circuits["country"])
        home = out["driverId"].map(drv["nationality"]).map(config.NATIONALITY_COUNTRY)
        out["home"] = np.where(home.isna() | country.isna(), np.nan, (home == country).astype(float))
        cons = raw["constructors"].set_index("constructorId")
        thome = out["constructorId"].map(cons["nationality"]).map(config.NATIONALITY_COUNTRY)
        out["team_home"] = np.where(thome.isna() | country.isna(), np.nan,
                                    (thome == country).astype(float))
    lap("context")

    # This weekend: qualifying and grid, sprint.
    qf = qualifying_features(e, raw.get("qualifying"))
    qf[qf.columns[2:]] = qf[qf.columns[2:]].astype(float)
    out = out.merge(qf, on=["raceId", "driverId"], how="left")
    out["q_grid"] = out["grid"].where(out["grid"] > 0)
    out["q_grid_rank"] = out["grid_rank"]
    out["q_grid_slot"] = (out["grid_rank"] - 1) / (out["n"] - 1).clip(lower=1)
    out["q_pit_lane"] = (out["grid"].fillna(0) <= 0).astype(float)
    out["q_grid_penalty"] = out["q_grid"] - out["q_pos"]
    sf = sprint_features(e, raw.get("sprint_results"))
    sf[sf.columns[2:]] = sf[sf.columns[2:]].astype(float)
    out = out.merge(sf, on=["raceId", "driverId"], how="left")
    lap("weekend")

    # Labels: what happened in this race.
    out["y_finish"] = out["finish"]
    out["y_dnf"] = out["dnf"].astype(int)
    out["y_laps"] = out["laps"]
    out["y_laps_share"] = out["laps"] / out["race_laps"]
    out["y_dnf_cause"] = dnf_cause(out["status"], out["dnf"])
    out["y_status"] = out["status"]
    out["y_deficit"] = out["deficit"]
    out["y_gained"] = out["gained"]
    out["y_pace"] = out["pace"]
    out["y_lap1_gain"] = out["lap1_gain"]
    out["y_grid_order"] = out["raceId"].map(rm.set_index("raceId")["grid_corr"])

    if elo is not None:
        out = out.merge(elo, on=KEYS + ["entry"], how="left")
    if "points_positions" in out:
        pts = out["points_positions"]
        out["y_points"] = ((out["finish"] >= 1) & (out["finish"] <= pts)).astype(int)
    out["y_win"] = (out["finish"] == 1).astype(int)
    out["y_podium"] = ((out["finish"] >= 1) & (out["finish"] <= 3)).astype(int)

    extra = {f"{f}_rel": relative(out[f], out["raceId"]) for f in REL_FEATURES if f in out}
    if "elo_strength" in out:
        extra["elo_rank"] = out.groupby("raceId")["elo_strength"].rank(ascending=False,
                                                                      method="first")
    out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    lap("labels")
    order = ["race_idx", "_pred_row"] if "_pred_row" in out else ["race_idx"]
    out = (out.sort_values(order, kind="stable").drop(columns="_pred_row", errors="ignore")
           .reset_index(drop=True))
    return out


def stage_columns(frame: pd.DataFrame, stage: str) -> tuple[list[str], list[str]]:
    """(feature columns, label columns) of a stage, in table order."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    feats = []
    elo = [c for c in frame.columns if c.startswith("elo_")]
    feats += elo + [c for c in frame.columns if c.startswith("dnf_")]
    driver_cols = ("team_starts", "team_starts_rel", "team_home")
    for who in ("drv", "team"):
        feats += [c for c in frame.columns if c.startswith(f"{who}_") and c not in driver_cols]
    feats += [c for c in frame.columns if c.startswith("circ_")] + ["street"]
    feats += ["era_start", "era_race", "round", "season_progress", "field_size",
              "sprint_weekend", "age", "age_rel", "starts", "starts_rel", "team_starts",
              "team_starts_rel", "home", "team_home"]
    if stage in ("post_quali", "post_sprint"):
        feats += [c for c in frame.columns if c.startswith("q_")]
    if stage == "post_sprint":
        feats += [c for c in frame.columns if c.startswith("s_")]
    feats = [c for c in dict.fromkeys(feats) if c in frame.columns]
    labels = [c for c in frame.columns if c.startswith("y_")] + (
        ["points_positions"] if "points_positions" in frame else [])
    return feats, labels


def select_stage(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    """The table of one stage: ids, the stage's features, then the labels."""
    feats, labels = stage_columns(frame, stage)
    out = frame[list(dict.fromkeys(ID_COLUMNS + feats + labels))].copy()
    if stage == "post_sprint":
        out = out[frame["sprint_weekend"].to_numpy() > 0]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def data_fingerprint(raw: dict) -> str:
    """Changes whenever any input table changes."""
    h = hashlib.sha1()
    for name in sorted(raw):
        t = raw[name]
        h.update(f"{name}:{len(t)}:{','.join(map(str, t.columns))}".encode())
        if len(t):
            h.update(pd.util.hash_pandas_object(t, index=False).values.tobytes())
    return h.hexdigest()[:16]


def _key(stage: str, fcfg: FeatureConfig, cfg: EloConfig, fingerprint: str) -> str:
    d = {"stage": stage, "features": asdict(fcfg), "elo": asdict(cfg),
         "version": FEATURE_VERSION, "data": fingerprint}
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_features(stage: str = "post_quali", fcfg: FeatureConfig = DEFAULT_FEATURES,
                    cfg: EloConfig = DEFAULT_CONFIG, rebuild: bool = False,
                    raw: dict | None = None, progress=None) -> pd.DataFrame:
    """The feature table of a stage (output/features_<stage>.parquet),
    rebuilt when the data or any setting changes."""
    say = progress or (lambda *_: None)
    raw = load_raw() if raw is None else raw
    key = _key(stage, fcfg, cfg, data_fingerprint(raw))
    path = config.OUTPUT_DIR / f"features_{stage}.parquet"
    meta = path.with_suffix(".json")
    if not rebuild and path.exists() and meta.exists():
        try:
            if json.loads(meta.read_text()).get("key") == key:
                return pd.read_parquet(path)
        except Exception:                 # a broken cache is just rebuilt
            pass
    say(f"  building the {stage} features ...")
    timings: dict = {}
    t0 = time.perf_counter()
    elo = elo_table(stage, cfg)
    timings["elo"] = round(time.perf_counter() - t0, 2)
    table = select_stage(build_all(raw, fcfg, cfg, elo, timings), stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    meta.write_text(json.dumps({"key": key, "stage": stage, "rows": len(table),
                                "features": asdict(fcfg), "timings_s": timings},
                               indent=1, default=str))
    return table


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
ERAS = [(1950, 1979), (1980, 1989), (1990, 1995), (1996, 2003), (2004, 2010),
        (2011, 2017), (2018, 2022), (2023, 2100)]


def coverage(table: pd.DataFrame, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(by year, by era): the share of rows with a value, per feature."""
    feats, labels = stage_columns(table, stage)
    cols = feats + [c for c in labels if table[c].dtype != object]
    nn = table[cols].notna()
    by_year = nn.groupby(table["year"]).mean().T
    era = pd.cut(table["year"], [a - 1 for a, _ in ERAS] + [2100],
                 labels=[f"{a}-{b}" if b < 2100 else f"{a}+" for a, b in ERAS])
    by_era = nn.groupby(era, observed=True).mean().T
    by_year.index.name = by_era.index.name = "feature"
    return by_year, by_era


def _race_rhos(race: np.ndarray, *cols: np.ndarray, min_n: int = 5) -> list[pd.Series]:
    """Per race, the Spearman correlation of every pair of ``cols`` (over
    the rows where all of them have a value; races with at least ``min_n``
    such rows and some spread in each), in the order (0,1), (0,2), (1,2) ...
    """
    d = pd.DataFrame({"r": race, **{f"c{i}": c for i, c in enumerate(cols)}}).dropna()
    k = len(cols)
    if d.empty:
        return [pd.Series(dtype=float) for _ in range(k * (k - 1) // 2)]
    g = d.groupby("r")
    for i in range(k):
        d[f"k{i}"] = g[f"c{i}"].rank()
    n = g.size()
    sums = {}
    for i in range(k):
        d[f"s{i}{i}"] = d[f"k{i}"] ** 2
    for i in range(k):
        for j in range(i + 1, k):
            d[f"s{i}{j}"] = d[f"k{i}"] * d[f"k{j}"]
    t = d.groupby("r")[[c for c in d.columns if c[0] in "ks" and c != "r"]].sum()
    var = {i: t[f"s{i}{i}"] - t[f"k{i}"] ** 2 / n for i in range(k)}
    ok = n >= min_n
    for i in range(k):
        ok &= var[i] > 1e-9
    out = []
    for i in range(k):
        for j in range(i + 1, k):
            cov = t[f"s{i}{j}"] - t[f"k{i}"] * t[f"k{j}"] / n
            out.append((cov / np.sqrt(var[i] * var[j]))[ok])
    return out


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """P(x of a y=1 row > x of a y=0 row), ties half (Mann-Whitney)."""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc_within(x: np.ndarray, y: np.ndarray, strata: np.ndarray) -> float:
    """The AUC within each stratum, averaged with weights n1 * n0 (the
    pairs compared): how well x separates y among rows alike in the
    stratum."""
    num = den = 0.0
    for s in np.unique(strata[~pd.isna(strata)]):
        m = strata == s
        a = _auc(x[m], y[m])
        w = float((y[m][~np.isnan(x[m])] == 1).sum() * (y[m][~np.isnan(x[m])] == 0).sum())
        if np.isfinite(a) and w:
            num += a * w
            den += w
    return num / den if den else np.nan


def signal(table: pd.DataFrame, stage: str, period=(1990, 2022)) -> pd.DataFrame:
    """A quick look at each feature over ``period``. Nothing is fitted:
    these are rank correlations and AUCs.

    Features that differ between the cars in a race:
      finish_rho   mean within-race Spearman correlation with the finishing
                   place among the cars that finished (positive = a higher
                   value finishes further back)
      beyond_elo   the same, partial on the ELO model's expected order
                   (within each race, the rank correlation left once the
                   ELO model's expected position is accounted for): what
                   the feature says about the order that the ratings don't.
                   0 for the ELO model's own expected position.
      dnf_auc      how well the feature separates retirements, pooled
                   (0.5 = not at all; below 0.5 = higher value, fewer)
      dnf_beyond   the same within tenths of the ELO model's P(DNF): what
                   the feature adds to the retirement model
      dnf_q1 / q5  retirement rate in the lowest / highest fifth of values
    Features with one value per race (circuit, era, field):
      race_dnf_rho    Spearman correlation, across races, with the race's
                      retirement rate
      race_order_rho  ... with how well the grid order held in the race
                      (grid-to-finish rank correlation of the finishers)
    """
    feats, _ = stage_columns(table, stage)
    t = table[(table["year"] >= period[0]) & (table["year"] <= period[1])]
    fin = t["y_finish"].where(t["y_finish"] > 0).to_numpy(float)
    race = t["raceId"].to_numpy()
    n_races = len(np.unique(race))
    elo = (t["elo_exp_pos"].where(t["y_finish"] > 0).to_numpy(float) if "elo_exp_pos" in t
           else np.full(len(t), np.nan))
    y = t["y_dnf"].to_numpy()
    strata = (pd.qcut(t["elo_p_dnf"].rank(method="first"), 10, labels=False).to_numpy()
              if "elo_p_dnf" in t else np.zeros(len(t)))
    # Per race: retirement rate, grid-order correlation.
    per_race = pd.DataFrame({"dnf": t.groupby("raceId")["y_dnf"].mean(),
                             "order": t.groupby("raceId")["y_grid_order"].first()})
    constant = t[feats].groupby(t["raceId"]).nunique(dropna=True).max() <= 1
    rows = []
    for f in feats:
        x = t[f].to_numpy(float)
        row = {"feature": f, "coverage": float(np.isfinite(x).mean())}
        if constant[f]:
            v = t.groupby("raceId")[f].first()
            pr = per_race.join(v.rename("x")).dropna(subset=["x"])
            row["race_dnf_rho"] = float(pr["x"].corr(pr["dnf"], method="spearman"))                 if pr["x"].nunique() > 1 else np.nan
            o = pr.dropna(subset=["order"])
            row["race_order_rho"] = float(o["x"].corr(o["order"], method="spearman"))                 if len(o) and o["x"].nunique() > 1 else np.nan
        else:
            (rho,) = _race_rhos(race, x, fin)
            rxy, rxz, ryz = _race_rhos(race, x, fin, elo)
            with np.errstate(invalid="ignore", divide="ignore"):
                part = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
            part = part.replace([np.inf, -np.inf], np.nan).dropna()
            q = (pd.qcut(pd.Series(x).rank(method="first"), 5, labels=False).to_numpy()
                 if np.isfinite(x).sum() > 50 else None)
            row.update({
                "races": float(len(rho) / n_races) if n_races else 0.0,
                "finish_rho": float(rho.mean()) if len(rho) else np.nan,
                "beyond_elo": float(part.mean()) if len(part) else np.nan,
                "dnf_auc": _auc(x, y), "dnf_beyond": _auc_within(x, y, strata),
                "dnf_q1": float(y[q == 0].mean()) if q is not None else np.nan,
                "dnf_q5": float(y[q == 4].mean()) if q is not None else np.nan})
        rows.append(row)
    cols = ["feature", "coverage", "races", "finish_rho", "beyond_elo", "dnf_auc",
            "dnf_beyond", "dnf_q1", "dnf_q5", "race_dnf_rho", "race_order_rho"]
    return pd.DataFrame(rows).reindex(columns=cols)
