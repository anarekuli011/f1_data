"""Configuration: file paths and tunable model constants.

Everything you might want to tweak about the model lives here so the rest of
the code doesn't have magic numbers scattered through it. A future UI can also
read/write these values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Root of the project (the folder containing this package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the raw Kaggle F1 CSVs live. Point this at your "archive2" folder.
DATA_DIR = PROJECT_ROOT / "data"

# Where computed outputs are written.
OUTPUT_DIR = PROJECT_ROOT / "output"

# Expected raw input files (Kaggle "Formula 1 World Championship" dataset).
RACES_CSV = "races.csv"
RESULTS_CSV = "results.csv"
CONSTRUCTORS_CSV = "constructors.csv"
DRIVERS_CSV = "drivers.csv"


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
@dataclass
class EloConfig:
    """Tunable parameters for the ELO model.

    Defaults reproduce the behaviour of the original notebook.
    """

    initial_elo: float = 1000.0

    # K-factors for constructors.
    k_constructor_race: float = 52.0
    k_constructor_quali: float = 18.0

    # Era-start boost for constructors. A big regulation change reshuffles
    # the order, so ratings need to move faster until they catch up. Each
    # season in `regulation_change_years` starts with an "era confidence" of
    # 0%, rising by 1/era_start_races per race (10 races -> 10% per race).
    # The constructor K slides linearly from the era-start K below (at 0%)
    # to the normal K above (at 100%), like the driver confidence.
    # None = no boost (use the normal K). Drivers are unaffected.
    regulation_change_years: tuple[int, ...] = (
        1961,  # 1.5 L engines
        1966,  # 3.0 L engines
        1983,  # flat bottoms, ground-effect ban
        1989,  # turbo ban
        1998,  # narrow track, grooved tyres
        2009,  # aero overhaul, slicks, KERS
        2014,  # turbo-hybrid power units
        2017,  # wider cars and tyres
        2022,  # ground-effect cars
        2026,  # new power units and active aero
    )
    # Races for the era confidence to reach 100%: round 1 uses the full
    # era-start K, round 11 onwards the normal K.
    era_start_races: int = 10
    k_constructor_race_era_start: float | None = 60.0
    k_constructor_quali_era_start: float | None = 24.0

    # K-factors for drivers, scaled by a confidence level.
    # Confidence runs from 0 to driver_confidence_races "races' worth"
    # (20 -> each race is 5%). The driver's K slides linearly from the rookie
    # value at 0% to the veteran value at 100%, for both race and quali.
    # After each race: a classified finish adds 1; a DNF takes 1 away; every
    # race a driver misses (once they've started at least one) takes 1 away.
    # So a full season out (~20 races) brings a driver back to 0%.
    driver_confidence_races: int = 20
    # At the first race of each season in regulation_change_years, every
    # driver's confidence drops by this share of the full scale (0.5 = 50
    # percentage points, i.e. 10 races' worth), never below 0%. New cars
    # mean even veterans' ratings should move faster for a while.
    era_start_confidence_drop: float = 0.5
    k_driver_race_rookie: float = 112.0
    k_driver_race_veteran: float = 56.0
    k_driver_quali_rookie: float = 120.0
    k_driver_quali_veteran: float = 60.0

    # Opponent-confidence weight (drivers only). A result against a driver
    # whose rating is still uncertain says less, so each side of a pair is
    # weighted by the OTHER driver's confidence:
    #     weight = floor + (1 - floor) * opponent_confidence   (0..1)
    # A veteran who beats a 0%-confidence newcomer gains floor x as much,
    # and loses floor x as much to one -- the same weight for a win and a
    # loss, so a correctly rated driver doesn't drift. Against a 100%
    # confident opponent the weight is 1. Your own confidence is already in
    # your own K. Stops part-timers, who start at 1000 but are mostly worse,
    # from feeding rating to the regulars who beat them.
    # 1.0 = off (the original behaviour). 0.5 is a first guess, to be tuned.
    driver_opponent_confidence_floor: float = 0.5

    # Weight of the car in the driver expectation: driver A vs driver B is
    # predicted from (D_a + driver_car_weight * C_a) vs
    # (D_b + driver_car_weight * C_b), C = constructor rating. 0 = drivers
    # only (the original behaviour, which lets car strength leak into driver
    # ratings: a driver in a dominant car keeps gaining for wins the car
    # earned). Teammate pairs are unaffected, since they share a car.
    # Backtested on teammate pairs, 1990-2026: best around 0.875 (0.75-1
    # effectively tied), with constructor_adjustment_exponent = 0.
    driver_car_weight: float = 0.875

    # True: the driver passes use the constructor ratings from BEFORE that
    # session's constructor update (so the car rating doesn't yet contain the
    # result being scored). False: the freshly updated ones (the engine's
    # original order). Backtests can't tell them apart (within 0.0005).
    driver_car_before_update: bool = False

    # Older, cruder car correction: each driver update is multiplied by
    # (C_b / C_a) ** exponent. Redundant once the car is in the expectation
    # (driver_car_weight), and worse on every backtest when combined with it;
    # 0 = off. Use 2.25 with driver_car_weight = 0 for the previous engine.
    constructor_adjustment_exponent: float = 0.0

    # Driver race pass only: when a classified driver is compared with a
    # driver who did not finish (position == dnf_sentinel), both sides of
    # that pair's driver delta are multiplied by this factor -- the DNF
    # driver loses less and the finisher gains correspondingly less, so no
    # rating is created or destroyed. Constructors still take the full
    # penalty (reliability is part of car strength); qualifying is
    # unaffected. 1.0 reproduces the original "DNF = last" behaviour.
    driver_dnf_penalty_factor: float = 0.25

    # Field-size correction, for every pass (drivers and constructors,
    # qualifying and race). Each entry's summed pair deltas are averaged
    # over its n - 1 opponents, then scaled by (ref - 1) / ref, so a race
    # moves ratings by the same amount whatever the grid size, and exactly
    # as much as before on a grid of this many cars (K keeps its meaning).
    # Without it, deltas are divided by the number of cars n, i.e. scaled
    # by (n - 1) / n: a 10-car race moved ratings 5% less than a 20-car one.
    # None = no correction (the original divide-by-n).
    reference_field_size: int | None = 20

    # Standard ELO logistic scale (the "400" in the classic formula).
    elo_scale: float = 400.0

    # Sentinels in the raw data.
    dnf_sentinel: str = "\\N"   # finishing "position" value for did-not-finish
    no_grid_sentinel: int = 0    # "grid" value meaning no/last qualifying spot


# A ready-to-use default instance.
DEFAULT_CONFIG = EloConfig()
