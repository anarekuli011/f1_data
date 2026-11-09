"""SQLite storage using the same table and column names as the Kaggle dataset.

The API identifies drivers/teams/circuits by text refs ("hamilton",
"red_bull"); this assigns each a stable integer id the first time it is seen,
so the ELO engine keeps working with integer ids.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

RESULT_COLS = """
    resultId INTEGER PRIMARY KEY AUTOINCREMENT,
    raceId INTEGER, driverId INTEGER, constructorId INTEGER, number TEXT,
    grid INTEGER, position INTEGER, positionText TEXT, positionOrder INTEGER,
    points REAL, laps INTEGER, time TEXT, milliseconds INTEGER, fastestLap INTEGER,
    rank INTEGER, fastestLapTime TEXT, fastestLapSpeed TEXT, statusId INTEGER
"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS seasons (year INTEGER PRIMARY KEY, url TEXT);
CREATE TABLE IF NOT EXISTS circuits (
    circuitId INTEGER PRIMARY KEY AUTOINCREMENT, circuitRef TEXT UNIQUE, name TEXT,
    location TEXT, country TEXT, lat REAL, lng REAL, url TEXT);
CREATE TABLE IF NOT EXISTS drivers (
    driverId INTEGER PRIMARY KEY AUTOINCREMENT, driverRef TEXT UNIQUE, number TEXT,
    code TEXT, forename TEXT, surname TEXT, dob TEXT, nationality TEXT, url TEXT);
CREATE TABLE IF NOT EXISTS constructors (
    constructorId INTEGER PRIMARY KEY AUTOINCREMENT, constructorRef TEXT UNIQUE,
    name TEXT, nationality TEXT, url TEXT);
CREATE TABLE IF NOT EXISTS status (
    statusId INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS races (
    raceId INTEGER PRIMARY KEY AUTOINCREMENT, year INTEGER, round INTEGER,
    circuitId INTEGER, name TEXT, date TEXT, time TEXT, url TEXT,
    fp1_date TEXT, fp1_time TEXT, fp2_date TEXT, fp2_time TEXT, fp3_date TEXT,
    fp3_time TEXT, quali_date TEXT, quali_time TEXT, sprint_date TEXT,
    sprint_time TEXT, sprint_quali_date TEXT, sprint_quali_time TEXT,
    UNIQUE (year, round));
CREATE TABLE IF NOT EXISTS results ({RESULT_COLS});
CREATE TABLE IF NOT EXISTS sprint_results ({RESULT_COLS});
CREATE TABLE IF NOT EXISTS qualifying (
    qualifyId INTEGER PRIMARY KEY AUTOINCREMENT, raceId INTEGER, driverId INTEGER,
    constructorId INTEGER, number TEXT, position INTEGER, q1 TEXT, q2 TEXT, q3 TEXT);
CREATE TABLE IF NOT EXISTS pit_stops (
    raceId INTEGER, driverId INTEGER, stop INTEGER, lap INTEGER, time TEXT,
    duration TEXT, milliseconds INTEGER);
CREATE TABLE IF NOT EXISTS lap_times (
    raceId INTEGER, driverId INTEGER, lap INTEGER, position INTEGER, time TEXT,
    milliseconds INTEGER);
CREATE TABLE IF NOT EXISTS driver_standings (
    driverStandingsId INTEGER PRIMARY KEY AUTOINCREMENT, raceId INTEGER,
    driverId INTEGER, points REAL, position INTEGER, positionText TEXT, wins INTEGER);
CREATE TABLE IF NOT EXISTS constructor_standings (
    constructorStandingsId INTEGER PRIMARY KEY AUTOINCREMENT, raceId INTEGER,
    constructorId INTEGER, points REAL, position INTEGER, positionText TEXT, wins INTEGER);
CREATE INDEX IF NOT EXISTS ix_results_race ON results(raceId);
CREATE INDEX IF NOT EXISTS ix_quali_race ON qualifying(raceId);
CREATE INDEX IF NOT EXISTS ix_laps_race ON lap_times(raceId);
"""

# Child tables that are fully replaced whenever a race is re-fetched.
RACE_CHILD_TABLES = ["results", "sprint_results", "qualifying", "pit_stops",
                     "lap_times", "driver_standings", "constructor_standings"]

TABLES = ["seasons", "circuits", "drivers", "constructors", "status", "races",
          *RACE_CHILD_TABLES]

UNKNOWN_CONSTRUCTOR = {"constructorId": "unknown", "name": "Unknown"}


class Database:
    def __init__(self, path: Path | str = "f1.sqlite"):
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)
        self._ids: dict[tuple, int] = {}

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- generic upsert by natural key -------------------------------------
    def _upsert(self, table: str, id_col: str, key_col: str, key, fields: dict) -> int:
        cache_key = (table, key)
        cols = [key_col, *fields]
        placeholders = ", ".join("?" * len(cols))
        updates = ", ".join(f"{c}=COALESCE(excluded.{c}, {c})" for c in fields) or \
            f"{key_col}=excluded.{key_col}"
        self.conn.execute(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT({key_col}) DO UPDATE SET {updates}",
            [key, *fields.values()])
        if cache_key not in self._ids:
            row = self.conn.execute(
                f"SELECT {id_col} FROM {table} WHERE {key_col}=?", [key]).fetchone()
            self._ids[cache_key] = row[0]
        return self._ids[cache_key]

    # -- entities from API JSON -------------------------------------------
    def driver(self, d: dict) -> int:
        return self._upsert("drivers", "driverId", "driverRef", d["driverId"], {
            "number": d.get("permanentNumber"), "code": d.get("code"),
            "forename": d.get("givenName"), "surname": d.get("familyName"),
            "dob": d.get("dateOfBirth"), "nationality": d.get("nationality"),
            "url": d.get("url")})

    def driver_by_ref(self, ref: str) -> int:
        return self.driver({"driverId": ref})

    def constructor(self, c: dict | None) -> int | None:
        if not c:  # a few old results have no team; Kaggle never had blanks here
            c = UNKNOWN_CONSTRUCTOR
        return self._upsert("constructors", "constructorId", "constructorRef",
                            c["constructorId"], {"name": c.get("name"),
                                                 "nationality": c.get("nationality"),
                                                 "url": c.get("url")})

    def circuit(self, c: dict) -> int:
        loc = c.get("Location", {})
        return self._upsert("circuits", "circuitId", "circuitRef", c["circuitId"], {
            "name": c.get("circuitName"), "location": loc.get("locality"),
            "country": loc.get("country"), "lat": _num(loc.get("lat")),
            "lng": _num(loc.get("long")), "url": c.get("url")})

    def status(self, text: str | None) -> int | None:
        if not text:
            return None
        return self._upsert("status", "statusId", "status", text, {})

    def season(self, s: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO seasons (year, url) VALUES (?, ?)",
                          [int(s["season"]), s.get("url")])

    def race(self, r: dict) -> int:
        """Upsert a race (from any RaceTable response) and return its raceId."""
        def sess(name):
            s = r.get(name) or {}
            return s.get("date"), s.get("time")

        sq = r.get("SprintQualifying") or r.get("SprintShootout") or {}
        fields = {
            "circuitId": self.circuit(r["Circuit"]) if r.get("Circuit") else None,
            "name": r.get("raceName"), "date": r.get("date"), "time": r.get("time"),
            "url": r.get("url"),
        }
        for prefix, key in [("fp1", "FirstPractice"), ("fp2", "SecondPractice"),
                            ("fp3", "ThirdPractice"), ("quali", "Qualifying"),
                            ("sprint", "Sprint")]:
            fields[f"{prefix}_date"], fields[f"{prefix}_time"] = sess(key)
        fields["sprint_quali_date"], fields["sprint_quali_time"] = sq.get("date"), sq.get("time")

        key = (int(r["season"]), int(r["round"]))
        cache_key = ("races", key)
        cols = ["year", "round", *fields]
        updates = ", ".join(f"{c}=COALESCE(excluded.{c}, {c})" for c in fields)
        self.conn.execute(
            f"INSERT INTO races ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
            f"ON CONFLICT(year, round) DO UPDATE SET {updates}",
            [*key, *fields.values()])
        if cache_key not in self._ids:
            self._ids[cache_key] = self.conn.execute(
                "SELECT raceId FROM races WHERE year=? AND round=?", key).fetchone()[0]
        return self._ids[cache_key]

    # -- replacing a race's child rows ------------------------------------
    def replace_rows(self, table: str, race_id: int, rows: list[dict]) -> None:
        self.conn.execute(f"DELETE FROM {table} WHERE raceId=?", [race_id])
        if not rows:
            return
        cols = list(rows[0])
        self.conn.executemany(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [[r[c] for c in cols] for r in rows])

    def repair(self) -> None:
        """Fill blanks left by builds made before the placeholder team existed."""
        for table in ("results", "sprint_results", "qualifying"):
            if self.conn.execute(f"SELECT 1 FROM {table} WHERE constructorId IS NULL "
                                 "LIMIT 1").fetchone():
                cid = self.constructor(UNKNOWN_CONSTRUCTOR)
                self.conn.execute(f"UPDATE {table} SET constructorId=? "
                                  "WHERE constructorId IS NULL", [cid])
        self.conn.commit()
        
    def counts(self) -> dict[str, int]:
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in TABLES}

    def races_for_season(self, year: int) -> list[tuple[int, int]]:
        """(round, raceId) for races in a season that already have results."""
        return self.conn.execute(
            "SELECT r.round, r.raceId FROM races r WHERE r.year=? AND EXISTS "
            "(SELECT 1 FROM results x WHERE x.raceId=r.raceId) ORDER BY r.round",
            [year]).fetchall()


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None