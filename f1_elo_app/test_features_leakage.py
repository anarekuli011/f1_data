"""Leakage test for the race features: a race's features must not change when
that race's own results, and everything after it, are taken away.

    py test_features_leakage.py              # the default races, with the ELO check
    py test_features_leakage.py --no-elo     # features only (much quicker)
    py test_features_leakage.py --race 2022-5 --race 1998-1

For each race R tested, a copy of the data folder is written with:
  * every race after R removed (results, qualifying, sprints, lap times,
    pit stops; the calendar, races.csv, is kept: it is known in advance);
  * R's own result scrambled: finishing order, laps, times, statuses and
    fastest laps shuffled between the cars, and R's lap times and pit stops
    deleted;
  * for the "pre" check, R's qualifying deleted, its grid and sprint
    scrambled too; for "post_quali", R's qualifying and grid kept and its
    sprint scrambled; for "post_sprint", R's sprint kept as well.
The features are rebuilt from that copy and R's rows must equal the ones
built from the full data (to 1e-9), column by column. With the ELO check,
the engine is replayed on the same copy and R's pre-race (or
after-qualifying) ratings, confidence and retirement chance must be equal
too; the ELO win / podium / points chances are a fixed function of those
(simulate.simulate_race, seeded per race).

Exit code 0 = no leakage found.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from f1elo import config, features
from f1elo.calibration import build_fields
from f1elo.data_loader import build_race_frame, load_raw

# (year, round): across eras, an era start, a sprint weekend (2021-22 the
# sprint set the grid), races with lap times / pit stops if present.
DEFAULT_RACES = ["1965-3", "1983-1", "1994-12", "1998-1", "2005-7", "2012-10",
                 "2021-10", "2022-5", "2022-21", "2024-6", "2026-18"]
NA = "\\N"
RESULT_OUTCOMES = ["position", "positionText", "positionOrder", "points", "laps", "time",
                   "milliseconds", "fastestLap", "rank", "fastestLapTime",
                   "fastestLapSpeed", "statusId"]
ELO_CHECK = ["driver_elo", "constructor_elo", "driver_confidence", "base_rate",
             "p_blend", "p_dnf"]


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def scramble(frame: pd.DataFrame, rows: pd.Series, cols: list[str], seed: int) -> pd.DataFrame:
    """Shuffle ``cols`` between the selected rows (one permutation for all
    the columns, so each car gets another car's whole result)."""
    out = frame.copy()
    idx = np.flatnonzero(rows.to_numpy())
    if len(idx) > 1:
        perm = np.random.default_rng(seed).permutation(idx)
        if (perm == idx).all():
            perm = np.roll(idx, 1)
        cols = [c for c in cols if c in out]
        out.loc[out.index[idx], cols] = out.loc[out.index[perm], cols].to_numpy()
    return out


def truncated_copy(src: Path, dst: Path, race_id: int, year: int, rnd: int, stage: str,
                   seed: int = 1) -> None:
    """The data folder as it would have looked at ``stage`` of race R, with
    R's later information scrambled or removed (see the module docstring)."""
    dst.mkdir(parents=True, exist_ok=True)
    races = read(src / "races.csv")
    later = races[(races["year"].astype(int) > year)
                  | ((races["year"].astype(int) == year) & (races["round"].astype(int) > rnd))]
    drop = set(later["raceId"].astype(int))
    for f in src.glob("*.csv"):
        shutil.copy(f, dst / f.name)
    for name in ("results.csv", "qualifying.csv", "sprint_results.csv", "lap_times.csv",
                 "pit_stops.csv"):
        path = src / name
        if not path.exists():
            continue
        t = read(path)
        rid = t["raceId"].astype(int)
        t = t[~rid.isin(drop)]
        this = t["raceId"].astype(int) == race_id
        if name in ("lap_times.csv", "pit_stops.csv"):
            t = t[~this]
        elif name == "results.csv":
            cols = RESULT_OUTCOMES + (["grid"] if stage == "pre" else [])
            t = scramble(t, this, cols, seed)
        elif name == "qualifying.csv" and stage == "pre":
            t = t[~this]
        elif name == "sprint_results.csv" and stage != "post_sprint":
            t = scramble(t, this, ["grid"] + RESULT_OUTCOMES, seed + 1)
        write(t, dst / name)


def compare(full: pd.DataFrame, test: pd.DataFrame, cols: list[str], keys: list[str]) -> list[str]:
    """Columns whose values differ between the two (aligned on ``keys``)."""
    a = full.set_index(keys)[cols].sort_index()
    b = test.set_index(keys)[cols].reindex(a.index)
    bad = []
    for c in cols:
        x, y = a[c], b[c]
        if x.dtype == object or y.dtype == object:
            same = (x.astype(str) == y.astype(str))
        else:
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            same = (np.isnan(xv) & np.isnan(yv)) | (np.abs(xv - yv) <= 1e-9)
        if not np.all(same):
            i = int(np.flatnonzero(~np.asarray(same))[0])
            bad.append(f"{c} (e.g. {a.index[i]}: {x.iloc[i]} vs {y.iloc[i]})")
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--race", action="append", metavar="YEAR-ROUND",
                   help="race(s) to test (default: %s)" % ", ".join(DEFAULT_RACES))
    p.add_argument("--no-elo", action="store_true", help="skip the ELO replay check")
    a = p.parse_args()

    t0 = time.perf_counter()
    src = config.DATA_DIR
    raw = load_raw(src)
    races = raw["races"]
    have = set(raw["results"]["raceId"])
    full = features.build_all(raw)
    feat_cols = {s: features.stage_columns(full, s)[0] for s in features.STAGES}
    keys = ["raceId", "driverId", "constructorId", "entry"]
    full_fields = {}
    if not a.no_elo:
        df = build_race_frame(src)
        for s in ("pre", "post_quali"):
            full_fields[s] = build_fields(s, df=df, cache=True)

    failures = 0
    for spec in a.race or DEFAULT_RACES:
        year, rnd = (int(x) for x in spec.split("-"))
        row = races[(races["year"] == year) & (races["round"] == rnd)]
        if row.empty or int(row["raceId"].iloc[0]) not in have:
            print(f"{spec}: no such race with results, skipped")
            continue
        race_id = int(row["raceId"].iloc[0])
        label = f"{year} R{rnd} {row['name'].iloc[0]}"
        sprint = bool(full.loc[full["raceId"] == race_id, "sprint_weekend"].max() > 0)
        mine = full[full["raceId"] == race_id]
        lap_rows = int((raw["lap_times"]["raceId"] == race_id).sum()) if len(raw["lap_times"]) else 0
        stages = ["pre", "post_quali"] + (["post_sprint"] if sprint else [])
        for stage in stages:
            with tempfile.TemporaryDirectory() as tmp:
                dst = Path(tmp)
                truncated_copy(src, dst, race_id, year, rnd, stage)
                traw = load_raw(dst)
                test = features.build_all(traw)
                test = test[test["raceId"] == race_id]
                cols = feat_cols[stage]
                bad = compare(mine, test, cols, keys)
                elo_bad = []
                if not a.no_elo and stage in full_fields:
                    fields = build_fields(stage, df=build_race_frame(dst), cache=False)
                    fk = ["raceId", "driverId", "constructorId"]
                    f_full = full_fields[stage]
                    f_full = f_full[f_full["raceId"] == race_id].copy()
                    f_test = fields[fields["raceId"] == race_id].copy()
                    for f in (f_full, f_test):
                        f["entry"] = f.groupby(fk).cumcount()
                    elo_bad = compare(f_full, f_test, ELO_CHECK, fk + ["entry"])
            n_set = int(mine[cols].notna().any().sum())
            status = "OK" if not bad and not elo_bad else "LEAK"
            failures += status != "OK"
            extra = f", {lap_rows:,} lap rows" if lap_rows else ""
            print(f"{status:4} {label} [{stage}]: {len(cols)} features ({n_set} with values"
                  f"{extra})" + ("" if a.no_elo or stage == "post_sprint" else ", ELO inputs checked"))
            for b in bad + [f"ELO {x}" for x in elo_bad]:
                print(f"       differs: {b}")
    print(f"\n{'No leakage found' if not failures else f'{failures} check(s) FAILED'} "
          f"({time.perf_counter() - t0:.0f}s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
