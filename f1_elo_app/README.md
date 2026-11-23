# F1 ELO

An ELO rating system for Formula 1 drivers and constructors, built on the
Kaggle "Formula 1 World Championship" dataset. This is the algorithm from the
original `F1_ELO.ipynb` notebook, reorganised into a clean Python project you
can open in VS Code, run from the terminal, and later wrap with a UI.

## Project layout

```
f1_elo_app/
├── f1elo/                  # the package (importable)
│   ├── config.py           # paths + all tunable model constants
│   ├── data_loader.py      # load + merge the raw CSVs
│   ├── elo_engine.py       # the core pairwise ELO algorithm  <-- the heart
│   ├── analysis.py         # leaderboards, histories, percentiles
│   └── predictions.py      # finishing-position probabilities
├── run_elo.py              # CLI: load -> compute -> save -> print summaries
├── make_sample_data.py     # generates fake CSVs so you can test immediately
├── requirements.txt
└── data/                   # <-- put the Kaggle CSVs here (see below)
```

## Setup (VS Code)

1. Open the `f1_elo_app` folder in VS Code.
2. Create and select a virtual environment, then install deps:
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Data

Put these four files from the Kaggle F1 dataset into the `data/` folder
(this is your old `archive2` folder):

```
data/races.csv
data/results.csv
data/constructors.csv
data/drivers.csv
```

No data yet? Generate a small synthetic set to test the pipeline:

```bash
python make_sample_data.py
```

## Run

```bash
python run_elo.py                 # uses ./data
python run_elo.py --data ./archive2 --out ./output
```

This writes `output/results_with_elo.csv` (every result row tagged with the
constructor and driver ELO *after* that race) and prints the top constructors
and drivers by peak rating.

## Using it as a library

The same functions a UI will call:

```python
from f1elo import build_race_frame, EloEngine, load_data, analysis
from f1elo.predictions import position_probabilities

raw = load_data()                 # {'races','results','constructors','drivers'}
df  = build_race_frame()          # merged, chronological
df  = EloEngine().run(df)         # fills EloRating / DriverEloRating

analysis.top_drivers_by_peak(df, raw["drivers"], n=20)
analysis.driver_rating_history(df, raw["drivers"], ["Lewis Hamilton", "Max Verstappen"])
```

## How the model works

For each race, in chronological order:

1. **Qualifying pass** (low K-factor), entries ordered by the real qualifying
   result from `qualifying.csv` (1994 onwards), otherwise by the starting grid
   (grid 0 = last). The grid includes engine and gearbox penalties, so it's
   only the fallback.
2. **Race pass** (high K-factor), entries ordered by finishing position
   (`\N` = DNF = last).

Each pass does a full pairwise comparison — every entry "beats" everyone
classified below it. Deltas are summed per entity, averaged over its n − 1
opponents and scaled by 19/20 (`reference_field_size` = 20), so a race moves a
rating by the same amount whatever the grid size, and exactly as much as the
original divide-by-n on a 20-car grid. For drivers, the expected result of a pair includes the
car: driver A vs driver B is predicted from `D_a + lam*C_a` vs `D_b + lam*C_b`
(driver plus weighted constructor rating, `driver_car_weight` = 0.875). A win
the car explains earns little, so car strength doesn't leak into driver
ratings; teammates share a car, so for them only the drivers count. Each
side of a driver pair is also weighted by the other driver's confidence
(0.6 against a newcomer, rising to 1 against a 100%-confidence driver), so
results against part-timers with unsettled ratings count less, for wins and
losses alike.

Before about 1980 a make often ran many cars (works and privateer Maseratis
share one constructor id, up to 15 in a race). A make's constructor update is
the average over its cars (scaled to two cars) rather than the sum, and
driver pairs within one make are weighted 1 / (cars − 1), so they add up to
one teammate's worth. Each driver's race is averaged over the weight of
comparisons they had, so it still moves them as far as anyone's race.
Two-car teams are unaffected. All K-factors and the car
weight live in `config.py`.

## Notes vs. the original notebook

- Same math, restructured: the global dicts/functions are now an `EloEngine`
  that holds its own state, so the UI can create one, run it, and query it.
- Data is merged once (the old `_x`/`_y` column renames are gone).
- The repeated "write CSV, read it back, write another CSV" steps are replaced
  by keeping everything in memory; `run_elo.py` saves one enriched results file
  at the end.
