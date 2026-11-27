"""Import lap times and pit stops from the Kaggle "Formula 1 World Championship"
dataset into the database, for the history the Jolpica API is slow to give
(lap times are ~7,000 requests, about 15 hours at the free rate limit).

The Kaggle files number races and drivers their own way, while this database
gives every race and driver its own id the first time the API returns it. So
rows are matched on what both share:

    race    Kaggle races.csv (year, round)  ->  races (year, round)
    driver  Kaggle drivers.csv driverRef    ->  drivers driverRef

Only races with NO rows in the table yet are filled, so data already fetched
from Jolpica is never overwritten. Every imported race is recorded in
fetch_log (source "kaggle"). Rows whose race or driver can't be matched are
counted and reported, never guessed.

    from f1data.kaggle_import import import_kaggle
    report = import_kaggle(db, "C:/Users/me/Downloads/archive")
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .db import Database

KAGGLE_NULL = "\\N"
# dataset key -> (Kaggle file, table, columns, first season)
FILES = {
    "laps": ("lap_times.csv", "lap_times",
             ["raceId", "driverId", "lap", "position", "time", "milliseconds"], 1996),
    "pitstops": ("pit_stops.csv", "pit_stops",
                 ["raceId", "driverId", "stop", "lap", "time", "duration", "milliseconds"],
                 2011),
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=[KAGGLE_NULL], keep_default_na=False,
                       low_memory=False)


def _maps(db: Database, folder: Path) -> tuple[dict, dict]:
    """Kaggle raceId -> our raceId, Kaggle driverId -> our driverId."""
    k_races = _read(folder / "races.csv")[["raceId", "year", "round"]]
    k_drivers = _read(folder / "drivers.csv")[["driverId", "driverRef"]]
    ours_r = {(y, r): rid for rid, y, r in
              db.conn.execute("SELECT raceId, year, round FROM races")}
    ours_d = {ref: did for did, ref in
              db.conn.execute("SELECT driverId, driverRef FROM drivers")}
    race_map = {int(k): ours_r.get((int(y), int(r)))
                for k, y, r in k_races.itertuples(index=False)}
    driver_map = {int(k): ours_d.get(ref) for k, ref in k_drivers.itertuples(index=False)}
    return ({k: v for k, v in race_map.items() if v is not None},
            {k: v for k, v in driver_map.items() if v is not None})


def find_files(folder: Path | str) -> Path:
    """The folder holding the Kaggle CSVs (the folder itself, or one level
    down, as unzipping often adds a folder)."""
    folder = Path(folder)
    if (folder / "races.csv").exists():
        return folder
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []:
        if (sub / "races.csv").exists():
            return sub
    raise FileNotFoundError(
        f"No Kaggle races.csv in {folder} (or a folder inside it). Unzip the "
        "Kaggle 'Formula 1 World Championship' download there first.")


def import_kaggle(db: Database, folder: Path | str, datasets=("pitstops", "laps"),
                  progress=print) -> dict:
    """Fill races that have no lap times / pit stops from the Kaggle files.

    Returns {dataset: {"races": imported races, "rows": rows,
    "skipped_existing": races that already had data, "unmatched_races":
    Kaggle races not in the database, "unmatched_rows": rows dropped because
    their driver isn't in the database, "last_race": "year R round"}}.
    """
    folder = find_files(folder)
    race_map, driver_map = _maps(db, folder)
    report = {}
    for key in datasets:
        fname, table, cols, _since = FILES[key]
        path = folder / fname
        if not path.exists():
            progress(f"{fname} isn't in {folder}; skipping {key}.")
            continue
        progress(f"Reading {fname} ...")
        k = _read(path)
        missing_cols = [c for c in cols if c not in k.columns]
        if missing_cols:
            raise ValueError(f"{fname} has no column(s) {', '.join(missing_cols)}")
        k = k[cols]
        unmatched_races = int(k.loc[~k["raceId"].isin(race_map), "raceId"].nunique())
        k = k[k["raceId"].isin(race_map)].copy()
        k["raceId"] = k["raceId"].map(race_map).astype(int)
        bad = ~k["driverId"].isin(driver_map)
        unmatched_rows = int(bad.sum())
        k = k[~bad].copy()
        k["driverId"] = k["driverId"].map(driver_map).astype(int)

        have = {rid for (rid,) in db.conn.execute(f"SELECT DISTINCT raceId FROM {table}")}
        todo = sorted(set(k["raceId"]) - have)
        skipped = len(set(k["raceId"]) & have)
        k = k[k["raceId"].isin(todo)]
        rows = k.astype(object).where(k.notna(), None)
        db.conn.executemany(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            rows.itertuples(index=False, name=None))
        counts = k.groupby("raceId").size()
        for rid, n in counts.items():
            db.log_fetch(int(rid), key, int(n), "kaggle")
        db.commit()
        last = None
        if todo:
            y, r = db.conn.execute("SELECT year, round FROM races WHERE raceId=?",
                                   [int(max(todo, key=lambda x: _order(db, x)))]).fetchone()
            last = f"{y} R{r}"
        report[key] = {"races": len(todo), "rows": int(len(k)), "skipped_existing": skipped,
                       "unmatched_races": unmatched_races, "unmatched_rows": unmatched_rows,
                       "last_race": last}
        progress(f"{key}: imported {len(todo)} races ({len(k):,} rows); "
                 f"{skipped} races already had data; {unmatched_races} Kaggle races and "
                 f"{unmatched_rows} rows couldn't be matched.")
    return report


def _order(db: Database, race_id: int) -> tuple[int, int]:
    return db.conn.execute("SELECT year, round FROM races WHERE raceId=?",
                           [race_id]).fetchone()
