"""Turn API responses into database rows, one dataset at a time."""
from __future__ import annotations

import datetime as dt
import logging

from .catalog import BY_KEY
from .client import JolpicaClient
from .db import Database

log = logging.getLogger("f1data")

CURRENT_YEAR = dt.date.today().year
FRESH_TTL = 6 * 3600  # how long current-season responses stay cached


def ttl_for(season: int | None, refresh: bool) -> float | None:
    """Past seasons never change, so cache them forever."""
    if refresh:
        return 0
    if season is not None and season < CURRENT_YEAR:
        return None
    return FRESH_TTL


# -- small parsers ----------------------------------------------------------
def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_millis(text: str | None) -> int | None:
    """'1:32.456' or '22.316' -> milliseconds."""
    if not text:
        return None
    try:
        secs = 0.0
        for part in text.split(":"):
            secs = secs * 60 + float(part)
        return round(secs * 1000)
    except ValueError:
        return None


# -- global datasets --------------------------------------------------------
def ingest_global(client: JolpicaClient, db: Database, key: str, refresh=False) -> None:
    ttl = 0 if refresh else FRESH_TTL
    if key == "seasons":
        for s in client.get_table("seasons", "SeasonTable", "Seasons", ttl):
            db.season(s)
    elif key == "circuits":
        for c in client.get_table("circuits", "CircuitTable", "Circuits", ttl):
            db.circuit(c)
    elif key == "drivers":
        for d in client.get_table("drivers", "DriverTable", "Drivers", ttl):
            db.driver(d)
    elif key == "constructors":
        for c in client.get_table("constructors", "ConstructorTable", "Constructors", ttl):
            db.constructor(c)
    elif key == "races":
        for r in client.get_races("races", None, ttl):
            db.race(r)
    elif key == "status":
        for s in client.get_table("status", "StatusTable", "Status", ttl):
            db.status(s.get("status"))
    db.commit()


# -- per-season datasets ----------------------------------------------------
def _result_row(race_id: int, r: dict, db: Database) -> dict:
    t, fl = r.get("Time") or {}, r.get("FastestLap") or {}
    pos_text = r.get("positionText")
    return {
        "raceId": race_id,
        "driverId": db.driver(r["Driver"]),
        "constructorId": db.constructor(r.get("Constructor")),
        "number": r.get("number"),
        "grid": _int(r.get("grid")),
        # Kaggle convention: position is empty when the car wasn't classified.
        "position": _int(r.get("position")) if (pos_text or "").isdigit() else None,
        "positionText": pos_text,
        "positionOrder": _int(r.get("position")),
        "points": _float(r.get("points")),
        "laps": _int(r.get("laps")),
        "time": t.get("time"),
        "milliseconds": _int(t.get("millis")),
        "fastestLap": _int(fl.get("lap")),
        "rank": _int(fl.get("rank")),
        "fastestLapTime": (fl.get("Time") or {}).get("time"),
        "fastestLapSpeed": (fl.get("AverageSpeed") or {}).get("speed"),
        "statusId": db.status(r.get("status")),
    }


def _quali_row(race_id: int, q: dict, db: Database) -> dict:
    return {
        "raceId": race_id, "driverId": db.driver(q["Driver"]),
        "constructorId": db.constructor(q.get("Constructor")),
        "number": q.get("number"), "position": _int(q.get("position")),
        "q1": q.get("Q1"), "q2": q.get("Q2"), "q3": q.get("Q3"),
    }


SEASON_DATASETS = {
    # key: (path suffix, child list key, target table, row builder)
    "results": ("results", "Results", "results", _result_row),
    "sprint": ("sprint", "SprintResults", "sprint_results", _result_row),
    "qualifying": ("qualifying", "QualifyingResults", "qualifying", _quali_row),
}


def ingest_season(client: JolpicaClient, db: Database, key: str, season: int,
                  refresh=False) -> int:
    suffix, child, table, build = SEASON_DATASETS[key]
    races = client.get_races(f"{season}/{suffix}", child, ttl_for(season, refresh))
    for race in races:
        race_id = db.race(race)
        db.replace_rows(table, race_id, [build(race_id, row, db) for row in race[child]])
    db.commit()
    return len(races)


# -- per-race datasets ------------------------------------------------------
def ingest_race_level(client: JolpicaClient, db: Database, key: str, season: int,
                      refresh=False) -> int:
    ttl = ttl_for(season, refresh)
    done = 0
    for rnd, race_id in db.races_for_season(season):
        if key in ("driverstandings", "constructorstandings"):
            kind = "Driver" if key == "driverstandings" else "Constructor"
            rows = []
            for mr in client.get_all_pages(f"{season}/{rnd}/{key}", ttl):
                for lst in mr["StandingsTable"]["StandingsLists"]:
                    for s in lst[f"{kind}Standings"]:
                        ent_id = (db.driver(s["Driver"]) if kind == "Driver"
                                  else db.constructor(s["Constructor"]))
                        rows.append({"raceId": race_id, f"{kind.lower()}Id": ent_id,
                                     "points": _float(s.get("points")),
                                     "position": _int(s.get("position")),
                                     "positionText": s.get("positionText"),
                                     "wins": _int(s.get("wins"))})
            db.replace_rows(f"{kind.lower()}_standings", race_id, rows)
        elif key == "pitstops":
            rows = []
            for race in client.get_races(f"{season}/{rnd}/pitstops", "PitStops", ttl):
                for p in race["PitStops"]:
                    rows.append({"raceId": race_id, "driverId": db.driver_by_ref(p["driverId"]),
                                 "stop": _int(p.get("stop")), "lap": _int(p.get("lap")),
                                 "time": p.get("time"), "duration": p.get("duration"),
                                 "milliseconds": to_millis(p.get("duration"))})
            db.replace_rows("pit_stops", race_id, rows)
        elif key == "laps":
            rows = []
            for race in client.get_races(f"{season}/{rnd}/laps", "Laps", ttl):
                for lap in race["Laps"]:
                    for t in lap["Timings"]:
                        rows.append({"raceId": race_id,
                                     "driverId": db.driver_by_ref(t["driverId"]),
                                     "lap": _int(lap.get("number")),
                                     "position": _int(t.get("position")),
                                     "time": t.get("time"),
                                     "milliseconds": to_millis(t.get("time"))})
            db.replace_rows("lap_times", race_id, rows)
        done += 1
        db.commit()
    return done


# -- orchestration ----------------------------------------------------------
GLOBAL = ["seasons", "circuits", "drivers", "constructors", "races", "status"]
SEASONAL = list(SEASON_DATASETS)
RACE_LEVEL = ["driverstandings", "constructorstandings", "pitstops", "laps"]


def run(client: JolpicaClient, db: Database, datasets: list[str],
        first: int, last: int, refresh=False) -> None:
    unknown = set(datasets) - set(BY_KEY)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(sorted(unknown))}")

    for key in [k for k in GLOBAL if k in datasets]:
        log.info("Fetching %s ...", key)
        ingest_global(client, db, key, refresh)

    # Results first: race-level datasets loop over races that have results.
    seasonal = [k for k in SEASONAL if k in datasets]
    race_level = [k for k in RACE_LEVEL if k in datasets]
    if race_level and "results" not in seasonal:
        seasonal.insert(0, "results")

    for season in range(first, last + 1):
        for key in seasonal:
            if season < BY_KEY[key].since:
                continue
            n = ingest_season(client, db, key, season, refresh)
            log.info("%d %-12s %3d races   (network calls so far: %d)",
                     season, key, n, client.network_calls)
        for key in race_level:
            if season < BY_KEY[key].since:
                continue
            n = ingest_race_level(client, db, key, season, refresh)
            log.info("%d %-20s %3d races   (network calls so far: %d)",
                     season, key, n, client.network_calls)