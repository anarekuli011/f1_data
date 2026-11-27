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
│   ├── strength.py         # pre-race strengths D + w*C, for race predictions
│   ├── reliability.py      # each car's chance of not finishing
│   └── simulate.py         # race simulation: chance of every finishing position
├── run_elo.py              # CLI: load -> compute -> save -> print summaries
├── predict_race.py         # CLI: a race's prediction table (next or past race)
├── backtest_*.py           # backtests that choose the settings in config.py
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
from f1elo.simulate import predict_race, predict_next_race

raw = load_data()                 # {'races','results','constructors','drivers'}
df  = build_race_frame()          # merged, chronological
df  = EloEngine().run(df)         # fills EloRating / DriverEloRating

analysis.top_drivers_by_peak(df, raw["drivers"], n=20)
analysis.driver_rating_history(df, raw["drivers"], ["Lewis Hamilton", "Max Verstappen"])

predict_next_race()               # chance of every position, next race
predict_race(1170)                # the same for a past race, pre-race data only
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

## Pre-race strengths (for race predictions)

`f1elo/strength.py` gives every car one number before a race,
`strength = D + w*C` (driver rating plus weighted car rating, as they stood
before that race), and P(A finishes ahead of B) =
`expected_score(S_a, S_b, s)`:

```python
from f1elo.strength import pre_race_strengths, next_race_strengths, add_names

past = pre_race_strengths(1170)      # one past race (or a list, or None = all)
nxt = add_names(next_race_strengths())  # the next race, latest race's field
```

`w` and `s` are `prediction_car_weight` (1.875) and `prediction_scale`
(1175), tuned on every pair of cars in 1990-2022 races with
`backtest_strength.py`. They differ from the engine's 0.875 / 400 (tuned for
the rating updates, on teammates) mainly because 43% of whole-field pairs
are decided by a retirement: those are hard to predict (a flatter scale)
and mostly down to the car (a larger car weight). On pairs where both cars
finished, the best is 1.25 / 525.

```bash
py backtest_strength.py                       # wide grid, candidates, variance shares
py backtest_strength.py --pairs classified    # only pairs where both finished
py backtest_strength.py --stage post_quali    # ratings after qualifying
```

Two settings, for two uses:

| setting | value | tuned on | use it for |
|---|---|---|---|
| `prediction_car_weight` / `prediction_scale` | 1.875 / 1175 | all pairs, a retirement losing to every finisher | a one-step view of who finishes ahead of whom, retirements included |
| `finish_car_weight` / `finish_scale` | 1.25 / 525 | pairs where both cars finished (`--pairs classified`) | the finishing order of the race simulation, which draws retirements separately |

## Race predictions (simulation)

`f1elo/simulate.py` turns the strengths into each driver's chance of every
finishing position. Each simulated race has three steps:

1. **Retirements**: every car retires with its own chance P(DNF)
   (`f1elo/reliability.py`, below), independently.
2. **Order of the finishers**: Plackett-Luce on `S = D + 1.25*C` with weights
   `10^(S/525)`. Positions are filled one by one, each by a remaining car
   with probability proportional to its weight. For two cars that both
   finish, P(A ahead of B) = `expected_score(S_a, S_b, 525)`. It is sampled
   with the Gumbel-max trick (sort `S*ln(10)/525 + Gumbel noise`), which is
   exactly Plackett-Luce and fully vectorised.
3. **Retired cars** get the `DNF` outcome instead of a position.

By default it runs 10,000 races with seed 0, which takes about 0.02 s for a
22-car field.

```python
from f1elo.simulate import predict_race, predict_next_race, summary

t = predict_next_race()           # next race on the calendar, latest race's field
t = predict_race(1170)            # a past race: pre-race ratings and counts only
summary(t, positions=10)          # the readable columns
```

The table has one row per car, strongest first, with these columns:
- **Positions**: `P1` ... `Pn` and `DNF`. Each row sums to 1.
- **Summaries**: `win`, `podium`, `points` and `exp_pos`, the expected
  position if classified.
- **Inputs**: `driver_elo` (D), `constructor_elo` (C), `strength` and
  `p_dnf`. The retirement counts behind `p_dnf` are included too.
- **Past races only**: `actual`, the real result.

**Points** means finishing inside the season's points positions. These
come from `results.csv`: the lowest classified position that scored, using
the most common value over the season's races. That is the top 5 in
1950-59, 6 in 1960-2002, 8 in 2003-09 and 10 from 2010. Sprint points
aren't counted.

```bash
py predict_race.py                  # the next race
py predict_race.py --latest         # the latest race, next to its actual result
py predict_race.py --year 2026 --round 21 --positions 10 --csv
py predict_race.py --check          # sanity checks on the same field
```

`--check` runs these checks:
- Every row sums to 1.
- Each position column sums to the exact chance that at least k cars finish.
- Every pair's simulated head-to-head, when both finish, matches
  `expected_score` within Monte Carlo error.
- The same seed gives an identical table.
- It reports the time one run takes.

### Retirement model

`f1elo/reliability.py` gives each car a chance of not finishing, counting a
retirement the way the engine does (`position` = `\N`). It uses only races
before the one predicted:

```
base   = field DNF share over its last 40 races (follows the era: ~50% in the 1980s, ~12% now)
team   = team DNF share over its last 20 races, all its cars (shrinkage k_t = 0)
driver = (driver DNFs + 60 * base) / (driver entries + 60), last 50 races
blend  = 0.4 * team + 0.6 * driver
logit P(DNF) = -0.116 + 0.939 * logit(blend) - 0.020 * car_rel - 0.086 * driver_rel
```

`car_rel` and `driver_rel` are the pre-race ratings minus the field's mean,
per 100 points. Better drivers retire a little less than their record
alone suggests.

The settings were tuned by log loss on "did this car retire", on 1990-2022,
with 2023 onwards held out:

| model | 1990-2022 | 2023-2026 (holdout) |
|---|---|---|
| base rate alone | 0.5665 | 0.3808 |
| blend of team and driver | 0.5560 | 0.3762 |
| blend + rating step (chosen) | 0.5534 | 0.3744 |

Fitting the rating step on 1990-2009 only and scoring it on 2010-2022 still
beats the blend (0.4491 vs 0.4494, base rate 0.4537). The grid is flat:
many nearby settings are within 0.0005.

```bash
py backtest_reliability.py          # base-rate baseline, full grid, rating step, calibration
py backtest_reliability.py --team-races 10 40 --team-races-increment 5 --no-ratings
```

Known limits, possible later improvements:
- Retirements are drawn independently, so the number of finishers varies
  less than in real races. A first-lap crash or rain takes out several cars
  at once. The variance of finishers per race is 4.1 in the model vs 5.9 in
  reality on 1990-2022.
- Newcomers' strengths could get more spread, using `driver_confidence`,
  which the strength frames carry.

## Notes vs. the original notebook

- Same math, restructured: the global dicts/functions are now an `EloEngine`
  that holds its own state, so the UI can create one, run it, and query it.
- Data is merged once (the old `_x`/`_y` column renames are gone).
- The repeated "write CSV, read it back, write another CSV" steps are replaced
  by keeping everything in memory; `run_elo.py` saves one enriched results file
  at the end.
