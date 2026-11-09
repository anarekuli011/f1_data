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

1. **Qualifying pass** (low K-factor), entries ordered by grid (grid 0 = last).
2. **Race pass** (high K-factor), entries ordered by finishing position
   (`\N` = DNF = last).

Each pass does a full pairwise comparison — every entry "beats" everyone
classified below it. Deltas are summed per entity and divided by the field size
before being applied. Driver updates are scaled by a constructor-strength
adjustment, so a strong car earns its driver less for a win and costs more for
a loss. All K-factors and the adjustment exponent live in `config.py`.

## Notes vs. the original notebook

- Same math, restructured: the global dicts/functions are now an `EloEngine`
  that holds its own state, so the UI can create one, run it, and query it.
- Data is merged once (the old `_x`/`_y` column renames are gone).
- The repeated "write CSV, read it back, write another CSV" steps are replaced
  by keeping everything in memory; `run_elo.py` saves one enriched results file
  at the end.
