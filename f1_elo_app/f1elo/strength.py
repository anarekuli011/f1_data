"""Pre-race strengths: what the race predictions are built from.

Every entry in a race gets a single strength number,

    strength = D + car_weight * C

with D the driver rating and C the constructor (car) rating as they stood
BEFORE the race. That is the same quantity the engine already uses for its
driver expectation -- driver A vs driver B is predicted as
``expected_score(S_a, S_b, scale)`` -- so a prediction and a rating update
agree on who was expected to beat whom.

``car_weight`` and ``scale`` default to ``prediction_car_weight`` and
``prediction_scale`` in the config. They are kept apart from the engine's own
``driver_car_weight`` / ``elo_scale``: those were tuned for the rating
updates (on teammate pairs), these for predicting whole-field results
(backtest_strength.py), a retirement losing to every finisher. The race
simulation (f1elo/simulate.py) draws retirements separately and orders the
finishers with ``finish_car_weight`` / ``finish_scale`` instead (tuned on
pairs where both cars finished); it passes that car weight in here.

What carries over from the engine, and what doesn't
---------------------------------------------------
The engine's driver expectation is exactly ``expected_score(D_a + lam*C_a,
D_b + lam*C_b, elo_scale)``. Everything else in its driver pass changes how
far a result MOVES a rating, not what was expected:
  * grid-size correction (``reference_field_size``): divides the summed
    deltas; the expectation of each pair is untouched.
  * opponent-confidence weight: scales each side's K by the other driver's
    confidence; again only the size of the update.
  * multi-car averaging (shared makes before 1980): weights same-make pairs
    in the update. In a prediction every car is its own entry with its own
    strength, so there is nothing to average.
  * the DNF factor: softens the update for a classified-vs-DNF pair. The
    prediction still counts a DNF as finishing behind every classified car,
    as the engine's outcome rule does.
So the strength is D + w*C and nothing more. What the engine does know, and
a point strength leaves out, is how sure it is of each rating: the frames
here carry each driver's ``driver_confidence`` (0-1) so the race simulation
can widen the spread for newcomers if that turns out to help.

Ratings are taken from the engine exactly the way the backtests do it: every
race is replayed with ``EloEngine.process_ordered`` and the ratings are read
before that race is processed (or, with ``stage="post_quali"``, between its
qualifying and race passes). Nothing from the race itself leaks in.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .backtest import prepare_races
from .config import DEFAULT_CONFIG, EloConfig
from .data_loader import build_race_frame, load_data
from .elo_engine import EloEngine, expected_score

STAGES = ("pre", "post_quali")

# Columns of every strength frame, in order.
COLUMNS = ["raceId", "year", "round", "driverId", "constructorId",
           "driver_elo", "constructor_elo", "strength", "driver_confidence",
           "position", "quali_rank"]


def strength(driver_elo, constructor_elo, car_weight: float):
    """D + car_weight * C (works on numbers, arrays or Series alike)."""
    return driver_elo + car_weight * constructor_elo


def win_probability(strength_a: float, strength_b: float,
                    cfg: EloConfig = DEFAULT_CONFIG,
                    scale: float | None = None) -> float:
    """P(entry A finishes ahead of entry B) from their strengths."""
    return expected_score(strength_a, strength_b,
                          cfg.prediction_scale if scale is None else scale)


def _prepared(df: pd.DataFrame | None, prepared: list | None,
              cfg: EloConfig) -> list:
    if prepared is not None:
        return prepared
    return prepare_races(build_race_frame() if df is None else df, cfg)


def field_strengths(engine: EloEngine, entries: pd.DataFrame,
                    car_weight: float | None = None) -> pd.DataFrame:
    """Strengths of a field from an engine's CURRENT ratings.

    ``entries`` needs driverId and constructorId (one row per car). A driver
    or team the engine has never seen gets the initial rating, and a driver
    with no races 0 confidence. Returns ``entries`` with driver_elo,
    constructor_elo, strength and driver_confidence added.
    """
    cfg = engine.cfg
    w = cfg.prediction_car_weight if car_weight is None else car_weight
    init = cfg.initial_elo
    out = entries.copy()
    out["driver_elo"] = [engine.driver_elo.get(d, init) for d in out["driverId"]]
    out["constructor_elo"] = [engine.constructor_elo.get(c, init)
                              for c in out["constructorId"]]
    out["strength"] = strength(out["driver_elo"], out["constructor_elo"], w)
    out["driver_confidence"] = [engine.confidence_pct(d) for d in out["driverId"]]
    return out


def _snapshot(engine: EloEngine, race_id, year: int, rnd: int,
              quali: pd.DataFrame, result: pd.DataFrame,
              car_weight: float) -> pd.DataFrame:
    """One race's entries, in finishing order, with the engine's ratings now."""
    # The same entries in qualifying order; look the rank up per car (a
    # 1950s driver sharing two cars keeps the better rank on both).
    rank: dict = {}
    for d, c, q in zip(quali["driverId"], quali["constructorId"], quali["quali_rank"]):
        rank.setdefault((d, c), q)
    entries = result[["driverId", "constructorId", "position"]].copy()
    entries["quali_rank"] = [rank.get(k) for k in
                             zip(entries["driverId"], entries["constructorId"])]
    entries.insert(0, "round", rnd)
    entries.insert(0, "year", year)
    entries.insert(0, "raceId", race_id)
    return field_strengths(engine, entries, car_weight)[COLUMNS]


def replay(prepared: list, cfg: EloConfig = DEFAULT_CONFIG,
           race_ids=None, stage: str = "pre",
           car_weight: float | None = None) -> tuple[pd.DataFrame, EloEngine]:
    """Replay every race, recording ratings before the races asked for.

    ``prepared`` is ``backtest.prepare_races(df)``. ``race_ids``: the races
    to record (None = all). ``stage``: "pre" = before the weekend,
    "post_quali" = after that weekend's qualifying pass. Returns the
    recorded entries (one row per car per race, finishing order within a
    race) and the engine after the LAST race, ready for the next one.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    w = cfg.prediction_car_weight if car_weight is None else car_weight
    keep = None if race_ids is None else set(race_ids)
    engine = EloEngine(cfg)
    init = cfg.initial_elo
    frames = []
    for race_id, year, rnd, quali, result in prepared:
        # Seed newcomers as the backtests do (the engine's run() seeds
        # everyone up front; the ratings come out the same).
        for d, c in zip(result["driverId"], result["constructorId"]):
            engine.driver_elo.setdefault(d, init)
            engine.highest_driver.setdefault(d, init)
            engine.constructor_elo.setdefault(c, init)
            engine.highest_constructor.setdefault(c, init)

        record = keep is None or race_id in keep
        take = (lambda: frames.append(_snapshot(engine, race_id, year, rnd,
                                                quali, result, w)))
        if record and stage == "pre":
            take()
        engine.process_ordered(year, rnd, quali, result,
                               after_quali=take if record and stage == "post_quali"
                               else None)
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=COLUMNS))
    return out, engine


def pre_race_strengths(race_ids=None, *, df: pd.DataFrame | None = None,
                       prepared: list | None = None,
                       cfg: EloConfig = DEFAULT_CONFIG, stage: str = "pre",
                       car_weight: float | None = None) -> pd.DataFrame:
    """Every entry's D, C and strength as they stood before each race.

    ``race_ids``: one raceId, a list of them, or None for every race. The
    race frame is built from the data folder unless ``df`` (from
    ``build_race_frame``) or ``prepared`` (from ``prepare_races``) is given.
    One row per car, in finishing order within each race; ``position`` and
    ``quali_rank`` are what actually happened, for scoring.
    """
    if race_ids is not None and not isinstance(race_ids, (list, tuple, set, frozenset)):
        race_ids = [race_ids]
    frame, _ = replay(_prepared(df, prepared, cfg), cfg, race_ids, stage, car_weight)
    return frame


def ratings_now(*, df: pd.DataFrame | None = None, prepared: list | None = None,
                cfg: EloConfig = DEFAULT_CONFIG) -> tuple[EloEngine, list]:
    """The engine after every race in the data, plus the prepared races."""
    prepared = _prepared(df, prepared, cfg)
    _, engine = replay(prepared, cfg, race_ids=())
    return engine, prepared


def next_race(prepared: list, races: pd.DataFrame | None = None) -> dict:
    """The first race on the calendar after the latest one with results:
    raceId, year, round, name, date. Year and round only (name None) if
    races.csv lists nothing later, e.g. at the end of a season."""
    _, year, rnd, _, _ = prepared[-1]
    if races is None:
        path = Path(config.DATA_DIR) / config.RACES_CSV
        races = pd.read_csv(path) if path.exists() else None
    if races is not None:
        later = races[(races["year"] > year)
                      | ((races["year"] == year) & (races["round"] > rnd))]
        if not later.empty:
            r = later.sort_values(["year", "round"]).iloc[0]
            return {"raceId": int(r["raceId"]), "year": int(r["year"]),
                    "round": int(r["round"]), "name": r.get("name"),
                    "date": r.get("date")}
    return {"raceId": None, "year": year, "round": rnd + 1, "name": None,
            "date": None}


def next_race_strengths(entries: pd.DataFrame | None = None, *,
                        df: pd.DataFrame | None = None,
                        prepared: list | None = None,
                        cfg: EloConfig = DEFAULT_CONFIG,
                        car_weight: float | None = None,
                        races: pd.DataFrame | None = None) -> pd.DataFrame:
    """Strengths for the next, not-yet-run race, strongest first.

    The field is ``entries`` (driverId, constructorId) if given, else the
    entry list of the latest race in the data. Ratings are the ones after
    that race; anyone without a rating gets the initial rating. The frame
    carries the next race's raceId / year / round (from races.csv).
    """
    engine, prepared = ratings_now(df=df, prepared=prepared, cfg=cfg)
    if entries is None:
        entries = (prepared[-1][4][["driverId", "constructorId"]]
                   .drop_duplicates("driverId"))
    info = next_race(prepared, races)
    field = entries[["driverId", "constructorId"]].reset_index(drop=True)
    field.insert(0, "round", info["round"])
    field.insert(0, "year", info["year"])
    field.insert(0, "raceId", info["raceId"])
    out = field_strengths(engine, field, car_weight)
    out = out.sort_values("strength", ascending=False, ignore_index=True)
    out.attrs["race"] = info
    return out


def add_names(frame: pd.DataFrame, raw: dict | None = None) -> pd.DataFrame:
    """Add ``driver`` (full name) and ``team`` columns to a strength frame."""
    raw = load_data() if raw is None else raw
    drivers = raw["drivers"].set_index("driverId")
    names = (drivers["forename"] + " " + drivers["surname"]).to_dict()
    teams = raw["constructors"].set_index("constructorId")["name"].to_dict()
    out = frame.copy()
    out.insert(out.columns.get_loc("driverId"), "driver", out["driverId"].map(names))
    out.insert(out.columns.get_loc("constructorId"), "team",
               out["constructorId"].map(teams))
    out.attrs = frame.attrs
    return out
