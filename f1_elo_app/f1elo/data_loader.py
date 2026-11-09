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


def build_race_frame(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Merge results + races into one chronologically sorted DataFrame.

    The returned frame has (at least) these columns:
        raceId, driverId, constructorId, grid, position, points,
        year, round, name (race name), date (datetime)
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
