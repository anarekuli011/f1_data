"""Generate small synthetic CSVs matching the Kaggle F1 schema.

Useful for trying the pipeline before you have the real ``archive2`` data.
Writes races.csv / results.csv / constructors.csv / drivers.csv into ./data.

    python make_sample_data.py
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

random.seed(42)

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

N_CONSTRUCTORS = 6
DRIVERS_PER_CONSTRUCTOR = 2
N_SEASONS = 4
RACES_PER_SEASON = 8

# True "strength" so ratings have something real to converge toward.
constructor_strength = {c: random.uniform(-1, 1) for c in range(1, N_CONSTRUCTORS + 1)}
driver_strength = {d: random.uniform(-1, 1)
                   for d in range(1, N_CONSTRUCTORS * DRIVERS_PER_CONSTRUCTOR + 1)}

# --- constructors.csv ---
constructors = pd.DataFrame({
    "constructorId": list(range(1, N_CONSTRUCTORS + 1)),
    "constructorRef": [f"team{c}" for c in range(1, N_CONSTRUCTORS + 1)],
    "name": [f"Team {c}" for c in range(1, N_CONSTRUCTORS + 1)],
    "nationality": ["British"] * N_CONSTRUCTORS,
    "url": [""] * N_CONSTRUCTORS,
})
constructors.to_csv(OUT / "constructors.csv", index=False)

# --- drivers.csv ---
n_drivers = N_CONSTRUCTORS * DRIVERS_PER_CONSTRUCTOR
drivers = pd.DataFrame({
    "driverId": list(range(1, n_drivers + 1)),
    "driverRef": [f"driver{d}" for d in range(1, n_drivers + 1)],
    "number": list(range(1, n_drivers + 1)),
    "code": [f"D{d:02d}" for d in range(1, n_drivers + 1)],
    "forename": [f"Driver{d}" for d in range(1, n_drivers + 1)],
    "surname": [f"Surname{d}" for d in range(1, n_drivers + 1)],
    "dob": ["1990-01-01"] * n_drivers,
    "nationality": ["British"] * n_drivers,
    "url": [""] * n_drivers,
})
drivers.to_csv(OUT / "drivers.csv", index=False)

# Fixed driver -> constructor lineup for the whole period.
lineup = {}
did = 1
for c in range(1, N_CONSTRUCTORS + 1):
    for _ in range(DRIVERS_PER_CONSTRUCTOR):
        lineup[did] = c
        did += 1

# --- races.csv + results.csv ---
race_rows = []
result_rows = []
race_id = 1
result_id = 1
for season in range(N_SEASONS):
    year = 2021 + season
    for rnd in range(1, RACES_PER_SEASON + 1):
        race_rows.append({
            "raceId": race_id, "year": year, "round": rnd, "circuitId": rnd,
            "name": f"Grand Prix {rnd}", "date": f"{year}-{rnd:02d}-15",
            "time": "13:00:00", "url": "",
        })

        # Score each car = constructor strength + driver strength + noise.
        scored = []
        for d, c in lineup.items():
            s = constructor_strength[c] + driver_strength[d] + random.gauss(0, 0.4)
            scored.append((s, d, c))
        scored.sort(reverse=True)  # best score finishes first

        for finish_pos, (_, d, c) in enumerate(scored, start=1):
            # Grid is roughly correlated with finishing order, with shuffle.
            grid = max(1, finish_pos + random.choice([-1, 0, 0, 1]))
            # ~10% DNF rate.
            dnf = random.random() < 0.10
            result_rows.append({
                "resultId": result_id, "raceId": race_id, "driverId": d,
                "constructorId": c, "grid": grid,
                "position": ("\\N" if dnf else finish_pos),
                "positionText": ("R" if dnf else str(finish_pos)),
                "positionOrder": finish_pos,
                "points": max(0, 26 - finish_pos * 2),
                "time": "", "milliseconds": "", "statusId": 1,
            })
            result_id += 1
        race_id += 1

pd.DataFrame(race_rows).to_csv(OUT / "races.csv", index=False)
pd.DataFrame(result_rows).to_csv(OUT / "results.csv", index=False)

print(f"Wrote sample data to {OUT}")
print(f"  {len(race_rows)} races, {len(result_rows)} results, "
      f"{n_drivers} drivers, {N_CONSTRUCTORS} constructors")
