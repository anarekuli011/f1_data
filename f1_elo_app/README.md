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
│   ├── simulate.py         # race simulation: chance of every finishing position
│   └── calibration.py      # scores, baselines and calibration of the race predictions
├── app.py                  # Streamlit app: ratings, race predictions, calibration
├── predictions_app.py      # the app's Predictions and Calibration views
├── app_common.py           # helpers shared by the app's views
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

## The app

```bash
py -m streamlit run app.py        # after py run_elo.py
```

The sidebar's View picks one of four pages. "Check for new races" (under
Data) pulls new results from the Jolpica API and recomputes everything; every
page's caches are keyed on the data files, so they refresh after an update.

- **Drivers / Constructors**: ratings over time, with a summary table.
- **Predictions**: one race's chances from the race simulation below.
  - Race: the next race (default) or any past race by season and round.
    Stage: before or after qualifying. The next race is before qualifying
    only, as there's no qualifying result for it yet.
  - Main table: win, podium, points and DNF chances, and the expected
    position if classified (retirements left out). The points column names
    the season's scoring places, e.g. "Points (top 6)" before 2003. Tiny
    chances show as "<0.1%", not 0.0%. Win is the exact chance per simulated
    set of finishers (the `win` column). For past races, the actual result
    is highlighted.
  - Inputs (collapsed): D, C, strength, P(DNF), the grid slot after
    qualifying, and the retirement counts.
  - Heatmap of every driver × P1 ... Pn plus DNF, with the actual results
    outlined. Below it, one driver's chance of each position, and why their
    P(DNF) is what it is: the team's and driver's recent retirements next to
    the era rate.
  - Next race: the entry list is copied from the latest race. "Change a
    seat" puts another driver from the data in a car.
  - Past races come from the calibration cache,
    `output/race_predictions_<stage>.parquet`. Loading it takes under a
    second; after new data it is rebuilt once, in about 75 s. The next race
    is simulated on demand, in a few seconds. The numbers are identical to
    `py predict_race.py` for the same race.
- **Calibration**: how far to trust the predictions, read from the CSVs
  `backtest_calibration.py` writes. It has plain-language trust notes,
  predicted-vs-actual diagrams for win, podium, points and DNF, the model
  vs baselines with 95% intervals, and the number of finishers per race.
  Pick the stage, the period (1990-2022 tuning or 2023-2026 holdout) and
  the predictor. If the CSVs are missing, the page says how to create them
  (`py backtest_calibration.py`, plus `--stage post_quali`).

The Predictions view reads every race through one function,
`predictions_app.load_predictions`, which returns a standard table: one row
per car with the inputs, P1 ... Pn, DNF, win, podium, points, exp_pos and
actual. The model behind it is looked up in `predictions_app.SOURCES`, so
another model that produces the same table needs only an entry there and a
"Model" option.

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
   (`f1elo/reliability.py`, below). The cars share a race factor
   (`dnf_race_corr` = 0.04, a Gaussian copula): each car keeps exactly its
   own chance, but in some races many retire together (a first-lap
   pile-up, rain), so the number of finishers varies as much as in real
   races.
2. **Order of the finishers**: Plackett-Luce on `S = D + 1.25*C`. Places
   are filled one by one, each by a remaining car with probability
   proportional to its weight: `10^(S/400)` for the first 3 places
   (`finish_top_scale` / `finish_top_places`), `10^(S/525)` for the rest
   (`finish_scale`). The front of a race is more predictable than the
   midfield, and one scale can't fit both (see the calibration section).
   It is sampled with the Gumbel-max trick (sort `S*ln(10)/scale + Gumbel
   noise`, the top places one at a time), which is exactly Plackett-Luce
   and fully vectorised.
3. **Retired cars** get the `DNF` outcome instead of a position.

By default it runs 10,000 races, which takes a few hundredths of a second
for a 22-car field. Each race has its own seed (seed 0 combined with the
raceId), and its cars are simulated strongest first, so results are
repeatable and don't depend on how the rows were sorted. `win` is the exact
chance given each simulated set of finishers, which has less Monte Carlo
noise than counting wins; `P1` is the share of simulated wins.

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

## How far to trust the race predictions (calibration)

`backtest_calibration.py` scores every race from 1990 on, from information
available beforehand, against fair baselines. Settings are chosen on
1990-2022; 2023-2026 is the holdout. The engine is replayed once, not per
race (`f1elo/calibration.py`), and the per-race prediction tables are cached
in `output/race_predictions_<stage>.parquet`. A cached table is identical to
what `predict_race` gives.

```bash
py backtest_calibration.py                      # pre-race: scores, baselines, calibration, experiments (~9 min)
py backtest_calibration.py --experiments none   # just the scores and calibration (~2 min first time, then seconds)
py backtest_calibration.py --stage post_quali   # after qualifying, vs results by grid slot
py backtest_calibration.py --stage both --no-save
py backtest_calibration.py --experiments finish --top-scale 300 500 --top-scale-increment 25
```

```python
from f1elo.calibration import cached_predictions, race_table
preds = cached_predictions("pre")     # every race; "post_quali" for after qualifying
race_table(preds, 1170)               # one race, like predict_race
```

It writes, for the app's calibration page, `output/calibration_scores_<stage>.csv`
(every predictor × period × score, with intervals), `calibration_bands_<stage>.csv`,
`calibration_finishers_<stage>.csv` (races with k retirements, predicted vs actual)
and `calibration_experiments_<stage>.csv` (the experiment grids).

**Scores** (lower is better):
- winner log loss (per race)
- podium and points log loss and Brier score (per car). Points means the
  season's scoring places.
- ranked probability score (RPS) over P1 ... Pn then DNF
- DNF log loss
- log loss of the number of finishers

Chances from simulation counts are floored at 0.5 / sims (5e-05) before
taking logs; the exact win chance needs no floor. Intervals are 95%, from
resampling races; a baseline's interval is for its difference from the
model.

**Pre-race, model vs baselines** (log loss unless marked):

| predictor | win 1990-2022 | podium | points | RPS | win 2023-26 | podium | points | RPS |
|---|---|---|---|---|---|---|---|---|
| **model** | **1.728** | **0.2481** | **0.4720** | **0.1606** | 1.581 | **0.2597** | 0.4948 | **0.1291** |
| uniform (all cars alike, base DNF rate) | 3.083 | 0.3984 | 0.6378 | 0.1970 | 3.017 | 0.4169 | 0.6921 | 0.1781 |
| base DNF rate + the model's order | 1.751 | 0.2496 | 0.4778 | 0.1639 | 1.598 | 0.2597 | **0.4943** | 0.1292 |
| all pairs 1.875 / 1175, no retirement model | 2.123 | 0.2806 | 0.4861 | 0.1697 | 2.048 | 0.2971 | 0.5079 | 0.1355 |
| strength rank (results of the k-th strongest car) | 1.747 | 0.2519 | 0.4840 | 0.1636 | **1.537** | 0.2677 | 0.5290 | 0.1351 |

- In 1990-2022 the model beats every baseline on points and RPS, with
  intervals clear of zero. It also beats every baseline on podium; only
  the gap to "base DNF rate + order" (+0.0015) has an interval touching
  zero.
- On the holdout it beats uniform, all pairs and strength rank on podium,
  points and RPS. It ties with "base DNF rate + order", which differs only
  in the retirement chance. That model's gain is real in 1990-2022
  (points +0.006, RPS +0.003) and in DNF log loss on both periods (0.3744
  vs 0.3807). It is too small to show in the race scores of 91 holdout
  races.
- The strength-rank table is fitted on 1990-2022, so it is scored
  in-sample there. On holdout winners it is 0.044 better than the model,
  but the interval is -0.121 to +0.039: no clear difference. It is clearly
  worse on podium, points and RPS.
- Splitting retirements from the order is clearly better than the single
  all-pairs model on every score.
- Holdout intervals for the model: winner 1.43-1.73, podium 0.242-0.278,
  points 0.462-0.529.

**After qualifying** (ratings after that weekend's qualifying pass, and the
grid slot in the retirement chance), against the historical results from
each grid slot:

| predictor | win 1990-2022 | podium | points | RPS | win 2023-26 | podium | points | RPS |
|---|---|---|---|---|---|---|---|---|
| model | 1.675 | 0.2408 | **0.4625** | **0.1589** | 1.523 | 0.2498 | **0.4805** | **0.1254** |
| grid slot | **1.567** | 0.2408 | 0.4850 | 0.1638 | **1.265** | **0.2364** | 0.5198 | 0.1295 |

Qualifying helps the model: points 0.4720 → 0.4625 before vs after. But the
grid slot predicts the winner better (1990-2022 by 0.107, interval 0.016 to
0.197; holdout by 0.26, interval crosses zero). Pole position carries race
pace that the ratings don't fully absorb. On points and RPS the model is
clearly better.

**Calibration** (`output/calibration_bands_<stage>.csv`): predicted vs actual
frequency in 10 bands, for win, podium, points (per points era: top 6 /
8 / 10) and DNF.
- **DNF and points**: well calibrated, mostly within 2 standard errors. The
  exception is the safest points finishers in the top-10 era (predicted
  88%, actual 84%): a front-runner's bad day is a little more likely than
  the model says.
- **Win**: good from 2% up, apart from a 30-40% band that won a little
  more often (42%). Long shots below 2% win less often than predicted
  (0.5% vs 0.2%). On the holdout, a dominant car is under-rated: chances of
  50-60% won 11 of 12 times.
- **Podium**: the least well calibrated. Long shots are a little too high
  and strong favourites too confident: podium chances above 80% came off
  73% of the time in 1990-2022, 85% on the 39 holdout cases.
- **Number of finishers**: mean right; spread right with the shared factor
  (variance ratio 0.97 tune, 1.01 holdout).

**How far to trust them**: points chances most, then podium, then win.
Points are well calibrated in every era and clearly beat every baseline. On
podium the model is the best pre-race predictor, but favourites are a bit
overconfident. Win chances are sound in the 5-50% range, but 2023-2026 has
only 91 winners, so holdout differences of 0.1 in winner log loss are
within noise. After qualifying, the grid alone is as good or better for
who wins.

**Experiments**, each kept only if it improved 1990-2022 and didn't hurt
2023-2026:

1. **Shared race factor for retirements: kept** (`dnf_race_corr` 0.04).
   Fitted to the variance of finishers per race (matched at 0.037).
   - Finisher-count log loss 2.302 → 2.261 / 1.939 → 1.914 (tuning /
     holdout).
   - Race scores unchanged within Monte Carlo noise.
2. **Finishing order tuned on whole results: kept** (first 3 places at
   400, the rest at 525, car weight 1.25).
   - The exact Plackett-Luce likelihood of real finishing orders wants a
     scale of about 300 for the winner, 400 for the top 3 and 650+ for the
     whole order. The pairwise 525 left win chances too flat.
   - Tuning / holdout: winner 1.802 → 1.728 / 1.700 → 1.582; podium
     0.2524 → 0.2481 / 0.2648 → 0.2607.
   - Points and RPS unchanged or slightly better.
   - One sharper scale for the whole order would cost points (0.4733).
3. **Extra spread for newcomers: not kept** (`finish_newcomer_spread` 0).
   - Every amount made the scores worse, rookies' own included (first 20
     starts).
   - Rookies do beat their ratings slightly: 88 podiums vs 78 predicted.
     That is a level bias, not missing spread.
4. **Grid slot in the retirement chance (after qualifying): kept**
   (`dnf_grid_a` / `dnf_grid_b`).
   - The front of the grid retires a little less. DNF log loss 0.5531 →
     0.5528 / 0.3737 → 0.3726, and every race score improves slightly.
   - A pit-lane flag was dropped: grid 0 also marks cars that withdrew
     before the start, which isn't known after qualifying.
5. **Era starts: not changed.**
   - The first 7 races of a new regulation era have more retirements than
     predicted: 25.5% vs 20.3% (1998-2026), 23% vs 13% in 2026.
   - A fitted shift didn't help when tested on each tuning-period era start
     in turn (2009 went the other way), so the retirement chance is
     probably too low at the next era start.

Data quirks: "did not start" counts as a retirement, as in the engine.
Sprints aren't modelled. A classified car's place is its rank among the
classified cars (3 races in the 2023-24 data give a disqualified car a
classified place at the back). The 1950s Indianapolis 500 is left out.

## Notes vs. the original notebook

- Same math, restructured: the global dicts/functions are now an `EloEngine`
  that holds its own state, so the UI can create one, run it, and query it.
- Data is merged once (the old `_x`/`_y` column renames are gone).
- The repeated "write CSV, read it back, write another CSV" steps are replaced
  by keeping everything in memory; `run_elo.py` saves one enriched results file
  at the end.
