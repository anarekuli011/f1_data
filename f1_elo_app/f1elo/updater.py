"""Fetch the latest races from Jolpica, re-export, recompute ratings, report changes.

Uses the f1_data project next to this one (Elo Engine/f1_data). Set the
F1_DATA_DIR environment variable if it lives somewhere else.

    from f1elo import updater
    report = updater.update()
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

from . import config
from .data_loader import build_race_frame
from .elo_engine import EloEngine

def _find_f1_data() -> Path:
    """f1_data sits either next to f1_elo_app, or around it (app nested inside)."""
    if "F1_DATA_DIR" in os.environ:
        return Path(os.environ["F1_DATA_DIR"])
    parent = config.PROJECT_ROOT.parent
    for candidate in (parent / "f1_data", parent):
        if (candidate / "f1data").is_dir():
            return candidate
    return parent / "f1_data"


F1_DATA_DIR = _find_f1_data()
DATASETS = ["results", "qualifying", "sprint"]
TRACKED = {"results": "race result", "qualifying": "qualifying", "sprint_results": "sprint"}


class UpdateError(Exception):
    """Something the user can fix (missing folder, database, network)."""


# -- snapshots of the database, to diff before/after -------------------------
def _latest_results_year(conn) -> int:
    row = conn.execute("SELECT MAX(r.year) FROM races r JOIN results x "
                       "ON x.raceId = r.raceId").fetchone()
    return row[0] or dt.date.today().year


def _snapshot(conn, first_year: int) -> dict:
    """Content of each tracked table, per race, for seasons >= first_year."""
    races = {rid: (year, rnd, name) for rid, year, rnd, name in conn.execute(
        "SELECT raceId, year, round, name FROM races WHERE year >= ?", [first_year])}
    tables = {}
    for table in TRACKED:
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})")][1:]  # skip PK
        per_race: dict[int, list] = {}
        for row in conn.execute(
                f"SELECT {', '.join(cols)} FROM {table} WHERE raceId IN "
                f"(SELECT raceId FROM races WHERE year >= ?)", [first_year]):
            per_race.setdefault(row[0], []).append(row)
        tables[table] = {rid: sorted(rows, key=repr) for rid, rows in per_race.items()}
    drivers = {ref: f"{fn} {sn}" for ref, fn, sn in
               conn.execute("SELECT driverRef, forename, surname FROM drivers")}
    teams = {ref: name for ref, name in
             conn.execute("SELECT constructorRef, name FROM constructors")}
    return {"races": races, "tables": tables, "drivers": drivers, "teams": teams}


def _winner(conn, race_id: int) -> str | None:
    row = conn.execute("SELECT d.forename || ' ' || d.surname FROM results x "
                       "JOIN drivers d ON d.driverId = x.driverId "
                       "WHERE x.raceId = ? AND x.position = 1", [race_id]).fetchone()
    return row[0] if row else None


def _diff(conn, before: dict, after: dict) -> dict:
    races = after["races"]
    label = lambda rid: f"{races[rid][0]} · R{races[rid][1]} · {races[rid][2]}"
    order = sorted(races, key=lambda rid: races[rid][:2])
    new, corrected = [], []
    for rid in order:
        added, changed = [], []
        for table, name in TRACKED.items():
            old = before["tables"][table].get(rid)
            cur = after["tables"][table].get(rid)
            if cur and not old:
                added.append(name)
            elif cur and old != cur:
                changed.append(name)
        if added:
            entry = {"race": label(rid), "sessions": added}
            if "race result" in added:
                entry["winner"] = _winner(conn, rid)
                entry["entries"] = len(after["tables"]["results"][rid])
            new.append(entry)
        if changed:
            corrected.append({"race": label(rid), "sessions": changed})
    return {
        "new": new,
        "corrected": corrected,
        "new_drivers": sorted(v for k, v in after["drivers"].items() if k not in before["drivers"]),
        "new_teams": sorted(v for k, v in after["teams"].items() if k not in before["teams"]),
    }


def _app_files_stale(conn) -> bool:
    """True if the app's CSVs or ratings don't match the database yet
    (e.g. the database was updated from the command line but never exported)."""
    results_csv = config.DATA_DIR / "results.csv"
    ratings = config.OUTPUT_DIR / "results_with_elo.csv"
    if not results_csv.exists() or not ratings.exists():
        return True
    with results_csv.open(encoding="utf-8") as f:
        exported = sum(1 for _ in f) - 1
    in_db = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    return exported != in_db or ratings.stat().st_mtime < results_csv.stat().st_mtime


def latest_race() -> str | None:
    """Label of the most recent race with results in the database (for the UI)."""
    db_path = F1_DATA_DIR / "f1.sqlite"
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT r.year, r.round, r.name FROM races r WHERE EXISTS "
                           "(SELECT 1 FROM results x WHERE x.raceId = r.raceId) "
                           "ORDER BY r.year DESC, r.round DESC LIMIT 1").fetchone()
    return f"{row[0]} · R{row[1]} · {row[2]}" if row else None


# -- the update ----------------------------------------------------------------
def update(progress=print, session=None) -> dict:
    """Refresh the latest season(s), export CSVs and recompute ratings.

    `progress` receives short status messages. Returns a report dict.
    """
    db_path, cache = F1_DATA_DIR / "f1.sqlite", F1_DATA_DIR / "cache"
    if not (F1_DATA_DIR / "f1data").is_dir():
        raise UpdateError(f"Couldn't find the f1_data project at {F1_DATA_DIR}. "
                          "It should sit next to f1_elo_app.")
    if not db_path.exists():
        raise UpdateError(f"No database at {db_path}. Run `py cli.py build` in f1_data first.")

    if str(F1_DATA_DIR) not in sys.path:
        sys.path.insert(0, str(F1_DATA_DIR))
    import requests  # noqa: E402
    from f1data.client import BASE_URL, USER_AGENT, JolpicaClient, RateLimiter  # noqa: E402
    from f1data.db import Database  # noqa: E402
    from f1data.export import export_csv  # noqa: E402
    from f1data.ingest import run  # noqa: E402

    # Fail fast when offline; the client's long back-offs are meant for throttling.
    session = session or requests.Session()
    try:
        session.get(f"{BASE_URL}/seasons.json?limit=1&offset=0",
                    headers={"User-Agent": USER_AGENT}, timeout=15)
    except requests.RequestException as exc:
        raise UpdateError("Couldn't reach the Jolpica API. Check your internet "
                          "connection and try again.") from exc

    # Show the client's retry warnings (e.g. throttling) as progress messages.
    class _Forward(logging.Handler):
        def emit(self, record):
            progress(record.getMessage())
    forward = _Forward(level=logging.WARNING)
    logging.getLogger("f1data").addHandler(forward)

    started = time.monotonic()
    db = Database(db_path)
    try:
        # From the last season we have results for, up to this year, so a gap
        # across New Year (or a long break between updates) is still covered.
        first = min(_latest_results_year(db.conn), dt.date.today().year)
        last = dt.date.today().year
        before = _snapshot(db.conn, first)

        seasons = str(first) if first == last else f"{first}–{last}"
        progress(f"Fetching {seasons} results, qualifying and sprints from Jolpica…")
        client = JolpicaClient(cache, session=session, max_retries=4,
                               limiter=RateLimiter(per_second=1, per_hour=480))
        try:
            run(client, db, DATASETS, first, last, refresh=True)
        except Exception as exc:  # network trouble, throttling, API changes
            raise UpdateError(f"Couldn't fetch from the API: {exc}") from exc

        report = _diff(db.conn, before, _snapshot(db.conn, first))
        report["requests"] = client.network_calls + 1  # + the connection check
        report["seasons"] = seasons

        changed = any(report[k] for k in ("new", "corrected", "new_drivers", "new_teams"))
        report["caught_up"] = not changed and _app_files_stale(db.conn)
        changed = changed or report["caught_up"]
        if changed:
            progress("Exporting CSVs…")
            export_csv(db, config.DATA_DIR)
    finally:
        db.close()
        logging.getLogger("f1data").removeHandler(forward)

    if changed:
        progress("Recomputing ELO ratings for every race (this takes a moment)…")
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df = EloEngine().run(build_race_frame(config.DATA_DIR))
        df.to_csv(config.OUTPUT_DIR / "results_with_elo.csv", index=False)

    report["ratings_recomputed"] = changed
    report["latest"] = latest_race()
    report["seconds"] = round(time.monotonic() - started)
    return report


def describe(report: dict) -> list[str]:
    """The report as plain lines, for the app and the command line."""
    lines = []
    for e in report["new"]:
        extra = []
        if e.get("entries"):
            extra.append(f"{e['entries']} entries")
        if e.get("winner"):
            extra.append(f"won by {e['winner']}")
        lines.append(f"New: {e['race']}: {', '.join(e['sessions'])}"
                     + (f" ({', '.join(extra)})" if extra else ""))
    for e in report["corrected"]:
        lines.append(f"Corrected: {e['race']}: {', '.join(e['sessions'])}")
    if report["new_drivers"]:
        lines.append("New drivers: " + ", ".join(report["new_drivers"]))
    if report["new_teams"]:
        lines.append("New teams: " + ", ".join(report["new_teams"]))
    if report.get("caught_up"):
        lines.append("Nothing new from the API, but the app's files were behind the "
                     "database, so they've been re-exported and ratings recomputed.")
    return lines or [f"Already up to date. Latest race: {report['latest']}"]


if __name__ == "__main__":
    try:
        r = update()
    except UpdateError as exc:
        sys.exit(str(exc))
    print("\n".join(describe(r)))
    print(f"({r['requests']} API requests, {r['seconds']}s)")