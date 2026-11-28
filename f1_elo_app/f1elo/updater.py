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
def _f1data():
    """Import the f1_data project's package (it sits next to this app)."""
    if not (F1_DATA_DIR / "f1data").is_dir():
        raise UpdateError(f"Couldn't find the f1_data project at {F1_DATA_DIR}. "
                          "It should sit next to f1_elo_app.")
    if not (F1_DATA_DIR / "f1.sqlite").exists():
        raise UpdateError(f"No database at {F1_DATA_DIR / 'f1.sqlite'}. Run "
                          "`py cli.py build` in f1_data first.")
    if str(F1_DATA_DIR) not in sys.path:
        sys.path.insert(0, str(F1_DATA_DIR))


def _check_api(session, base_url: str, user_agent: str) -> None:
    """Fail fast when offline or when Jolpica is down; the client's long
    back-offs are meant for throttling. Tell the two apart: a server that
    accepts the connection but never answers is Jolpica's problem, not the
    user's connection."""
    import requests
    try:
        check = session.get(f"{base_url}/seasons.json?limit=1&offset=0",
                            headers={"User-Agent": user_agent}, timeout=15)
    except requests.ReadTimeout as exc:
        raise UpdateError("The Jolpica API isn't responding: it accepted the "
                          "connection but sent no reply within 15 seconds. It's "
                          "probably busy or down for a while (common right after "
                          "a race weekend). Your connection is fine; try again "
                          "later.") from exc
    except requests.ConnectionError as exc:
        raise UpdateError("Couldn't connect to the Jolpica API. Check your "
                          "internet connection and try again.") from exc
    except requests.RequestException as exc:
        raise UpdateError(f"The Jolpica API check failed ({type(exc).__name__}). "
                          "Try again later.") from exc
    if check.status_code >= 500:
        raise UpdateError(f"The Jolpica API is having problems (HTTP "
                          f"{check.status_code}). Try again later.")


def update(progress=print, session=None) -> dict:
    """Refresh the latest season(s), export CSVs and recompute ratings.

    `progress` receives short status messages. Returns a report dict.
    """
    db_path, cache = F1_DATA_DIR / "f1.sqlite", F1_DATA_DIR / "cache"
    _f1data()
    import requests  # noqa: E402
    from f1data.client import BASE_URL, USER_AGENT, JolpicaClient, RateLimiter  # noqa: E402
    from f1data.db import TABLES, Database  # noqa: E402
    from f1data.export import LAP_TABLES, export_csv  # noqa: E402
    from f1data.ingest import run  # noqa: E402

    session = session or requests.Session()
    _check_api(session, BASE_URL, USER_AGENT)

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
            # Lap times and pit stops have their own update (update_laps), so
            # the quick update doesn't rewrite the big lap file every time.
            export_csv(db, config.DATA_DIR,
                       tables=[t for t in TABLES if t not in LAP_TABLES])
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


# -- lap times and pit stops (optional, kept apart from the quick update) -----
# dataset -> (table, first season the API has)
LAP_DATA = {"pitstops": ("pit_stops", 2011), "laps": ("lap_times", 1996)}
# A race that came back empty is asked for again only while it is this
# recent (Jolpica can take a day or two to add a race's laps).
RETRY_EMPTY_DAYS = 14
# Default hourly budget for this update: below Jolpica's ~500 an hour, with
# room left for a quick "Check for new races" meanwhile.
LAPS_PER_HOUR = 400


def missing_lap_races(conn, dataset: str, first: int | None = None,
                      last: int | None = None) -> list[dict]:
    """Races with results but no rows yet for ``dataset`` ("laps" or
    "pitstops"), oldest first, each with an estimate of the requests it
    needs (lap times: one page per 100 laps run, from results.laps). A race
    fetched before that came back empty counts as missing only while it is
    less than RETRY_EMPTY_DAYS old."""
    table, since = LAP_DATA[dataset]
    first = max(since, first or since)
    last = last or dt.date.today().year
    have = {rid for (rid,) in conn.execute(f"SELECT DISTINCT raceId FROM {table}")}
    logged = {rid: rows for rid, rows in conn.execute(
        "SELECT raceId, rows FROM fetch_log WHERE dataset=?", [dataset])}
    cutoff = (dt.date.today() - dt.timedelta(days=RETRY_EMPTY_DAYS)).isoformat()
    out = []
    for rid, year, rnd, name, date, laps_run, cars in conn.execute(
            "SELECT r.raceId, r.year, r.round, r.name, r.date, SUM(x.laps), COUNT(*) "
            "FROM races r JOIN results x ON x.raceId = r.raceId "
            "WHERE r.year BETWEEN ? AND ? GROUP BY r.raceId ORDER BY r.year, r.round",
            [first, last]):
        if rid in have:
            continue
        if rid in logged and (date or "") < cutoff:
            continue                      # fetched, nothing there, not recent
        pages = (max(1, -(-int(laps_run or 0) // 100)) if dataset == "laps"
                 else max(1, -(-cars * 3 // 100)))
        out.append({"raceId": rid, "year": year, "round": rnd, "name": name,
                    "date": date, "requests": pages})
    return out


def update_laps(progress=print, datasets=("pitstops", "laps"), first: int | None = None,
                last: int | None = None, max_requests: int | None = None,
                per_hour: int = LAPS_PER_HOUR, session=None, export: bool = True,
                plan_only: bool = False) -> dict:
    """Fetch lap times and pit stops from Jolpica for the races that don't
    have them yet, then export lap_times.csv / pit_stops.csv.

    Separate from ``update`` (the quick "Check for new races"): a race's
    laps take ~10-15 requests and Jolpica allows ~500 an hour, so the whole
    history is many hours (``import_kaggle_laps`` is the quick way to get
    it). Each race is saved as soon as it is fetched: stopping (Ctrl+C,
    ``max_requests``, an error) loses nothing, and the next run carries on
    where this one stopped. ``plan_only`` = just report what would be
    fetched.
    """
    db_path, cache = F1_DATA_DIR / "f1.sqlite", F1_DATA_DIR / "cache"
    _f1data()
    import requests  # noqa: E402
    from f1data.client import BASE_URL, USER_AGENT, JolpicaClient, RateLimiter  # noqa: E402
    from f1data.db import Database  # noqa: E402
    from f1data.export import LAP_TABLES, export_csv  # noqa: E402
    from f1data.ingest import ingest_race, ttl_for  # noqa: E402

    started = time.monotonic()
    report = {"fetched": {}, "rows": {}, "empty": [], "remaining": {}, "requests": 0,
              "stopped": None}
    db = Database(db_path)
    try:
        todo = {k: missing_lap_races(db.conn, k, first, last) for k in datasets}
        total = sum(r["requests"] for v in todo.values() for r in v)
        for k, races in todo.items():
            progress(f"{k}: {len(races)} races missing "
                     f"(~{sum(r['requests'] for r in races):,} requests)")
        report["remaining"] = {k: len(v) for k, v in todo.items()}
        if not total:
            progress("Nothing to fetch: every race in range has its data.")
            return report
        hours = total / per_hour
        progress(f"About {total:,} requests in all: "
                 + ("a few minutes." if total <= per_hour else
                    f"roughly {hours:.1f} hours at {per_hour} an hour (it pauses "
                    "whenever the hourly budget is used up).")
                 + (f" Stopping after {max_requests:,} requests." if max_requests else ""))
        if plan_only:
            return report

        session = session or requests.Session()
        _check_api(session, BASE_URL, USER_AGENT)

        class _Forward(logging.Handler):
            def emit(self, record):
                progress(record.getMessage())
        forward = _Forward(level=logging.WARNING)
        logging.getLogger("f1data").addHandler(forward)
        client = JolpicaClient(cache, session=session, max_retries=4,
                               limiter=RateLimiter(per_second=1, per_hour=per_hour))
        recent = (dt.date.today() - dt.timedelta(days=RETRY_EMPTY_DAYS)).isoformat()
        planned_done = 0
        try:
            for k, races in todo.items():
                report["fetched"][k] = report["rows"][k] = 0
                for i, race in enumerate(races, 1):
                    if max_requests and client.network_calls + race["requests"] > max_requests:
                        report["stopped"] = (f"request budget of {max_requests:,} reached; "
                                             "run again to continue")
                        break
                    # A recent race may have been cached empty: ask again.
                    ttl = 0 if (race["date"] or "") >= recent else ttl_for(race["year"], False)
                    rows = ingest_race(client, db, k, race["year"], race["round"],
                                       race["raceId"], ttl)
                    planned_done += race["requests"]
                    report["fetched"][k] += 1
                    report["rows"][k] += rows
                    label = f"{race['year']} R{race['round']} {race['name']}"
                    if not rows:
                        report["empty"].append(f"{k}: {label}")
                    progress(f"{k} {i}/{len(races)}: {label}: {rows:,} rows "
                             f"({client.network_calls:,} requests so far, "
                             f"~{max(total - planned_done, 0):,} to go)")
                if report["stopped"]:
                    break
        except KeyboardInterrupt:
            report["stopped"] = "interrupted; run again to continue"
        except Exception as exc:  # network trouble, throttling, API changes
            report["stopped"] = f"stopped by an error ({exc}); run again to continue"
        finally:
            logging.getLogger("f1data").removeHandler(forward)
            report["requests"] = client.network_calls + 1    # + the connection check
        report["remaining"] = {k: len(missing_lap_races(db.conn, k, first, last))
                               for k in datasets}
        if export and any(report["fetched"].values()):
            progress("Exporting lap_times.csv and pit_stops.csv…")
            export_csv(db, config.DATA_DIR, tables=LAP_TABLES)
    finally:
        db.close()
    report["seconds"] = round(time.monotonic() - started)
    return report


def import_kaggle_laps(folder, progress=print, datasets=("pitstops", "laps"),
                       export: bool = True) -> dict:
    """Fill the races that have no lap times / pit stops from the Kaggle
    dataset's lap_times.csv and pit_stops.csv (unzipped in ``folder``),
    matching races on (year, round) and drivers on driverRef, then export
    the two CSVs. Races that already have data are left alone."""
    _f1data()
    from f1data.db import Database  # noqa: E402
    from f1data.export import LAP_TABLES, export_csv  # noqa: E402
    from f1data.kaggle_import import import_kaggle  # noqa: E402

    db = Database(F1_DATA_DIR / "f1.sqlite")
    try:
        try:
            report = import_kaggle(db, folder, datasets, progress)
        except (FileNotFoundError, ValueError) as exc:
            raise UpdateError(str(exc)) from exc
        if export and any(v["races"] for v in report.values()):
            progress("Exporting lap_times.csv and pit_stops.csv…")
            export_csv(db, config.DATA_DIR, tables=LAP_TABLES)
        report = {"imported": report,
                  "remaining": {k: len(missing_lap_races(db.conn, k)) for k in datasets}}
    finally:
        db.close()
    return report


def describe_laps(report: dict) -> list[str]:
    """The lap update's report as plain lines."""
    lines = []
    for k, n in report.get("fetched", {}).items():
        lines.append(f"{k}: fetched {n} races ({report['rows'].get(k, 0):,} rows)")
    for e in report.get("empty", []):
        lines.append(f"No data from Jolpica yet: {e}")
    if report.get("stopped"):
        lines.append(f"Stopped: {report['stopped']}")
    for k, n in report.get("remaining", {}).items():
        lines.append(f"{k}: {n} races still missing" if n else f"{k}: complete")
    return lines


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