"""Command line for rebuilding the F1 dataset from Jolpica-F1.

    python cli.py catalog                      # what the API offers
    python cli.py probe results                # live sample of one endpoint's fields
    python cli.py build                        # core datasets, 1950 -> now
    python cli.py build --with pitstops,driverstandings --from 2011
    python cli.py update                       # refresh the current season
    python cli.py export --out ../f1_elo_app/data
    python cli.py stats
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

from f1data import catalog
from f1data.client import JolpicaClient
from f1data.db import Database
from f1data.export import export_csv
from f1data.ingest import run

THIS_YEAR = dt.date.today().year

# Small, cheap sample requests used by `probe` (one request each).
PROBE_PATHS = {
    "seasons": "seasons", "circuits": "circuits", "drivers": "drivers",
    "constructors": "constructors", "races": "2024/races", "results": "2024/1/results",
    "qualifying": "2024/1/qualifying", "sprint": "2024/5/sprint",
    "driverstandings": "2024/1/driverstandings",
    "constructorstandings": "2024/1/constructorstandings",
    "pitstops": "2024/1/pitstops", "laps": "2024/1/laps", "status": "status",
}

# Rough requests per season, for the up-front estimate.
COST_PER_SEASON = {"results": 4, "qualifying": 2, "sprint": 1, "driverstandings": 18,
                   "constructorstandings": 18, "pitstops": 20, "laps": 230}
GLOBAL_COST = {"seasons": 1, "circuits": 1, "drivers": 9, "constructors": 3,
               "races": 12, "status": 2}


def flatten(obj, prefix=""):
    """Yield (path, sample value) for every leaf in a JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        if obj:
            yield from flatten(obj[0], f"{prefix}[]")
    else:
        yield prefix, obj


def cmd_probe(args, client):
    keys = args.endpoints or list(PROBE_PATHS)
    for key in keys:
        path = PROBE_PATHS[key]
        mr = client.get(path, limit=1, ttl=0)["MRData"]
        print(f"\n=== {key}  (/{path}/, total rows: {mr.get('total')}) ===")
        for field, value in flatten(mr):
            if field.split(".")[0] in ("xmlns", "series", "url", "limit", "offset", "total"):
                continue
            print(f"  {field:<60} {str(value)[:40]}")


def estimate(datasets, first, last):
    seasons = range(first, last + 1)
    total = sum(GLOBAL_COST.get(k, 0) for k in datasets)
    for k in datasets:
        if k in COST_PER_SEASON:
            since = catalog.BY_KEY[k].since
            total += COST_PER_SEASON[k] * sum(1 for s in seasons if s >= since)
    return total


def resolve_datasets(args):
    if args.only:
        return [k.strip() for k in args.only.split(",")]
    extra = []
    if args.with_:
        extra = catalog.OPTIONAL if args.with_ == "all" else \
            [k.strip() for k in args.with_.split(",")]
    return catalog.CORE + [k for k in extra if k not in catalog.CORE]


def main():
    p = argparse.ArgumentParser(description="Rebuild the F1 dataset from Jolpica-F1.")
    p.add_argument("--db", default="f1.sqlite", help="SQLite file (default f1.sqlite)")
    p.add_argument("--cache", default="cache", help="raw response cache folder")
    p.add_argument("--per-hour", type=int, default=480,
                   help="hourly request budget (API limit is 500)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("catalog", help="list every endpoint and what it is good for")

    pr = sub.add_parser("probe", help="fetch one sample row per endpoint and show its fields")
    pr.add_argument("endpoints", nargs="*", choices=list(PROBE_PATHS), metavar="endpoint")

    b = sub.add_parser("build", help="download datasets into the database")
    b.add_argument("--from", dest="first", type=int, default=1950)
    b.add_argument("--to", dest="last", type=int, default=THIS_YEAR)
    b.add_argument("--with", dest="with_", default="",
                   help="extra datasets, comma separated, or 'all'")
    b.add_argument("--only", default="", help="exactly these datasets, comma separated")
    b.add_argument("--refresh", action="store_true", help="ignore the cache")
    b.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    u = sub.add_parser("update", help="refresh the current season for datasets you have")
    u.add_argument("--season", type=int, default=THIS_YEAR)

    e = sub.add_parser("export", help="write Kaggle-style CSVs")
    e.add_argument("--out", default="export")

    sub.add_parser("stats", help="row counts per table")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.cmd == "catalog":
        catalog.print_catalog()
        return

    client = JolpicaClient(args.cache, per_hour=args.per_hour)

    if args.cmd == "probe":
        cmd_probe(args, client)
        return

    db = Database(args.db)
    try:
        if args.cmd == "build":
            datasets = resolve_datasets(args)
            n = estimate(datasets, args.first, args.last)
            print(f"Datasets: {', '.join(datasets)}")
            print(f"Seasons:  {args.first}-{args.last}")
            print(f"Estimate: ~{n:,} requests (~{n / args.per_hour:.1f} h at "
                  f"{args.per_hour}/h; cached responses are free)")
            if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
                return
            run(client, db, datasets, args.first, args.last, args.refresh)

        elif args.cmd == "update":
            have = {t for t, c in db.counts().items() if c}
            table_for = {"sprint": "sprint_results", "pitstops": "pit_stops",
                         "laps": "lap_times", "driverstandings": "driver_standings",
                         "constructorstandings": "constructor_standings"}
            # Races, drivers and teams come along inside the results responses.
            datasets = ["results", "qualifying", "sprint"] + [
                k for k in catalog.OPTIONAL
                if k != "status" and table_for.get(k, k) in have]
            run(client, db, datasets, args.season, args.season, refresh=True)

        elif args.cmd == "export":
            for path in export_csv(db, args.out):
                print(f"wrote {path}")

        elif args.cmd == "stats":
            for table, count in db.counts().items():
                print(f"{table:<24}{count:>10,}")

        print(f"Network requests this run: {client.network_calls}")
    finally:
        db.close()


if __name__ == "__main__":
    main()