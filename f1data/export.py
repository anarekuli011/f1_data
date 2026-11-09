"""Export the database as Kaggle-style CSVs (drop-in for the ELO app's data/ folder)."""
from __future__ import annotations

import csv
from pathlib import Path

from .db import TABLES, Database

KAGGLE_NULL = "\\N"  # the Kaggle files write missing values as \N
FILE_NAMES = {"pit_stops": "pit_stops.csv", "lap_times": "lap_times.csv"}

# Columns where Kaggle used 0 rather than \N for "unknown" (the ELO engine
# relies on grid being numeric, with 0 meaning no grid position).
ZERO_IF_MISSING = {"results": {"grid"}, "sprint_results": {"grid"}}


def export_csv(db: Database, out_dir: Path | str, skip_empty: bool = True) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db.repair()
    written = []
    for table in TABLES:
        cur = db.conn.execute(f"SELECT * FROM {table}")
        rows = cur.fetchall()
        if skip_empty and not rows:
            continue
        path = out_dir / FILE_NAMES.get(table, f"{table}.csv")
        cols = [c[0] for c in cur.description]
        zero_cols = ZERO_IF_MISSING.get(table, set())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for row in rows:
                w.writerow([(0 if col in zero_cols else KAGGLE_NULL) if v is None else v
                            for col, v in zip(cols, row)])
        written.append(path)
    return written