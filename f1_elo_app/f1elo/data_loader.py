"""Load and merge the raw Kaggle F1 CSV files into a single tidy DataFrame.

The original notebook merged results + races (sometimes twice, which is where
all those `_x` / `_y` suffix renames came from). This does it once, cleanly,
and returns a frame sorted into chronological race order.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n"
            f"Point config.DATA_DIR at the folder containing the Kaggle F1 CSVs."
        )
    return pd.read_csv(path, low_memory=False)


def load_data(data_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load the four raw CSVs.

    Returns a dict with keys: 'races', 'results', 'constructors', 'drivers'.
    """
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    return {
        "races": _read(data_dir / config.RACES_CSV),
        "results": _read(data_dir / config.RESULTS_CSV),
        "constructors": _read(data_dir / config.CONSTRUCTORS_CSV),
        "drivers": _read(data_dir / config.DRIVERS_CSV),
    }


# Tables for the race features, with the columns they need. Optional files
# that are missing come back as empty frames with these columns.
RAW_TABLES = {
    "races": (config.RACES_CSV, None, True),
    "results": (config.RESULTS_CSV, None, True),
    "drivers": (config.DRIVERS_CSV, ["driverId", "driverRef", "dob", "nationality"], True),
    "constructors": (config.CONSTRUCTORS_CSV, ["constructorId", "constructorRef",
                                               "nationality"], True),
    "qualifying": (config.QUALIFYING_CSV, ["raceId", "driverId", "constructorId",
                                           "position", "q1", "q2", "q3"], False),
    "sprint_results": (config.SPRINT_RESULTS_CSV, ["raceId", "driverId", "constructorId",
                                                   "grid", "position", "positionOrder",
                                                   "laps", "statusId"], False),
    "status": (config.STATUS_CSV, ["statusId", "status"], False),
    "circuits": (config.CIRCUITS_CSV, ["circuitId", "circuitRef", "country"], False),
    "lap_times": (config.LAP_TIMES_CSV, ["raceId", "driverId", "lap", "position",
                                         "milliseconds"], False),
    "pit_stops": (config.PIT_STOPS_CSV, ["raceId", "driverId", "stop", "lap",
                                         "milliseconds"], False),
}


def load_lap_times(data_dir: Path | str | None = None) -> pd.DataFrame:
    """lap_times.csv (raceId, driverId, lap, position, milliseconds), 1996
    onwards; empty if the file isn't there (see update_laps.py)."""
    return load_raw(data_dir, ["lap_times"])["lap_times"]


def load_pit_stops(data_dir: Path | str | None = None) -> pd.DataFrame:
    """pit_stops.csv (raceId, driverId, stop, lap, milliseconds), 2011
    onwards; empty if the file isn't there (see update_laps.py)."""
    return load_raw(data_dir, ["pit_stops"])["pit_stops"]


def load_raw(data_dir: Path | str | None = None, tables=None) -> dict[str, pd.DataFrame]:
    """The raw CSVs the race features use (``RAW_TABLES``), ``\\N`` read as
    missing. ``tables``: only these. Required files must exist; optional
    ones that don't come back empty."""
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    out = {}
    for name in (tables or RAW_TABLES):
        fname, cols, required = RAW_TABLES[name]
        path = data_dir / fname
        if not path.exists():
            if required:
                _read(path)                       # raises the friendly error
            out[name] = pd.DataFrame(columns=cols)
            continue
        frame = pd.read_csv(path, na_values=["\\N"], keep_default_na=False,
                            usecols=cols, low_memory=False)
        out[name] = frame
    return out


def build_race_frame(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Merge results + races into one chronologically sorted DataFrame.

    The returned frame has (at least) these columns:
        raceId, driverId, constructorId, grid, position, points,
        year, round, name (race name), date (datetime),
        quali_position (real qualifying position; NaN if unknown)
    plus an empty ``EloRating`` / ``DriverEloRating`` column ready to be filled
    by the engine.
    """
    raw = load_data(data_dir)
    results, races = raw["results"], raw["races"]

    # races.csv carries: raceId, year, round, circuitId, name, date, time, ...
    # Keep only what we need to avoid column-name collisions on merge.
    race_cols = ["raceId", "year", "round", "name", "date"]
    race_cols = [c for c in race_cols if c in races.columns]

    df = results.merge(races[race_cols], on="raceId", how="left")

    # Real qualifying position (qualifying.csv, 1994 onwards), as
    # ``quali_position``; NaN where there is none. The engine's qualifying
    # pass uses it instead of the penalty-affected grid when a race has it.
    qpath = Path(data_dir if data_dir is not None else config.DATA_DIR) / config.QUALIFYING_CSV
    if qpath.exists():
        q = pd.read_csv(qpath, usecols=["raceId", "driverId", "position"])
        q = q.groupby(["raceId", "driverId"], as_index=False)["position"].min()
        df = df.merge(q.rename(columns={"position": "quali_position"}),
                      on=["raceId", "driverId"], how="left")
    else:
        df["quali_position"] = float("nan")

    # Chronological order: year, then round within the year.
    sort_cols = [c for c in ("year", "round") if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # Parse the race date if present (Kaggle uses YYYY-MM-DD).
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Initialise the rating columns the engine will populate.
    df["EloRating"] = config.DEFAULT_CONFIG.initial_elo
    df["DriverEloRating"] = config.DEFAULT_CONFIG.initial_elo

    return df
