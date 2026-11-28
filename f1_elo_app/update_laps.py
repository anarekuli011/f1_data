"""Get lap times (1996 on) and pit stops (2011 on) for the race features.

Kept apart from "Check for new races", which stays quick: a race's lap
times take 10-15 API requests and Jolpica allows about 500 an hour.

    py update_laps.py --plan                  # what's missing, and how long it would take
    py update_laps.py                         # fetch every missing race from Jolpica
    py update_laps.py --from 2025             # only 2025 onwards
    py update_laps.py --only pitstops         # pit stops only (~1 request per race)
    py update_laps.py --max-requests 400      # stop after about an hour's budget
    py update_laps.py --kaggle "C:\\Users\\me\\Downloads\\archive"
                                              # history from the Kaggle download instead

The whole lap history from Jolpica is about 7,000 requests (15+ hours). The
quick way is the Kaggle "Formula 1 World Championship (1950 - 20xx)"
dataset: unzip it anywhere and pass the folder to --kaggle. Its races and
drivers are matched to this project's by (year, round) and driverRef, and
only races without data are filled. Then run this script without --kaggle
to fetch the races since the Kaggle file was made.

Every race is saved as soon as it is fetched, so stopping (Ctrl+C) loses
nothing: run it again to carry on. Afterwards rebuild the features:
    py build_features.py
"""
from __future__ import annotations

import argparse
import sys

from f1elo import updater


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["laps", "pitstops"], help="one dataset only")
    p.add_argument("--from", dest="first", type=int, help="first season (default: all)")
    p.add_argument("--to", dest="last", type=int, help="last season (default: this year)")
    p.add_argument("--max-requests", type=int, help="stop after about this many requests")
    p.add_argument("--per-hour", type=int, default=updater.LAPS_PER_HOUR,
                   help=f"hourly request budget (default {updater.LAPS_PER_HOUR}; "
                        "Jolpica allows ~500)")
    p.add_argument("--plan", action="store_true", help="only show what is missing")
    p.add_argument("--kaggle", metavar="FOLDER",
                   help="import from the unzipped Kaggle dataset instead of the API")
    a = p.parse_args()
    datasets = (a.only,) if a.only else ("pitstops", "laps")
    try:
        if a.kaggle:
            r = updater.import_kaggle_laps(a.kaggle, datasets=datasets)
            for k, n in r["remaining"].items():
                print(f"{k}: {n} races still missing" if n else f"{k}: complete")
            if any(r["remaining"].values()):
                print("Run `py update_laps.py` (without --kaggle) to fetch the rest from Jolpica.")
            return
        r = updater.update_laps(datasets=datasets, first=a.first, last=a.last,
                                max_requests=a.max_requests, per_hour=a.per_hour,
                                plan_only=a.plan)
    except updater.UpdateError as exc:
        sys.exit(str(exc))
    if not a.plan:
        print("\n".join(updater.describe_laps(r)))
        print(f"({r['requests']} API requests, {r.get('seconds', 0)}s)")


if __name__ == "__main__":
    main()
