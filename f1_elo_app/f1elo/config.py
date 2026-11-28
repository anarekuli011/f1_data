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
# Optional: real qualifying positions (1994 onwards). Without it, the
# qualifying pass falls back to the starting grid.
QUALIFYING_CSV = "qualifying.csv"
# Optional, for the race features (f1elo/features.py) only; missing files
# just leave those features empty.
SPRINT_RESULTS_CSV = "sprint_results.csv"
STATUS_CSV = "status.csv"
CIRCUITS_CSV = "circuits.csv"
LAP_TIMES_CSV = "lap_times.csv"      # 1996 onwards (update_laps.py)
PIT_STOPS_CSV = "pit_stops.csv"      # 2011 onwards (update_laps.py)


# ---------------------------------------------------------------------------
# Race features (f1elo/features.py, build_features.py)
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    """Settings of the feature table for the step 6 models. Every value is
    part of the cache key, so changing one rebuilds the table."""

    # Recent form: averages over each driver's / team's last N races.
    form_races: int = 10
    # Circuit history: the circuit's results relative to the field at the
    # time, shrunk toward the average circuit by this many races' worth
    # (a circuit visited twice is mostly "average").
    circuit_shrink: float = 3.0
    # Only the circuit's last N visits count: tracks and cars change (Monaco
    # held its grid order like an average track until the 2000s, far better
    # than any since 2015).
    circuit_visits: int = 10
    # The field "at the time" for the circuit features: the last N races.
    circuit_base_races: int = 40
    # Clean laps (race pace, consistency): a lap is dropped when the
    # field's median time on that lap is more than this much slower than
    # the race's typical lap (safety car, VSC, rain shower) ...
    neutralised_lap: float = 0.05
    # ... or the driver's own lap is this much slower than the field's
    # median on that lap (pit laps before 2011, spins, damage).
    slow_lap: float = 0.10
    # Fewest clean laps for a race pace / consistency value.
    min_clean_laps: int = 5
    # Pit stops: stops longer than this many times the race's median stop
    # (repairs, red flags) are left out.
    pit_outlier: float = 2.0


DEFAULT_FEATURES = FeatureConfig()

# Street and temporary circuits (circuitRef), for the "street" flag: walls
# close to the track, hard to pass, more safety cars.
STREET_CIRCUITS = (
    "monaco", "adelaide", "albert_park", "baku", "dallas", "detroit", "jeddah",
    "las_vegas", "vegas", "long_beach", "marina_bay", "miami", "monsanto",
    "montjuic", "pedralbes", "phoenix", "valencia", "boavista", "ain-diab",
    "madring", "sochi",
)

# Driver / constructor nationality -> the country name circuits.csv uses,
# for the "home race" flags.
NATIONALITY_COUNTRY = {
    "American": "USA", "Argentine": "Argentina", "Australian": "Australia",
    "Austrian": "Austria", "Belgian": "Belgium", "Brazilian": "Brazil",
    "British": "UK", "Canadian": "Canada", "Chilean": "Chile", "Chinese": "China",
    "Colombian": "Colombia", "Czech": "Czech Republic", "Danish": "Denmark",
    "Dutch": "Netherlands", "East German": "Germany", "Finnish": "Finland",
    "French": "France", "German": "Germany", "Hong Kong": "China",
    "Hungarian": "Hungary", "Indian": "India", "Indonesian": "Indonesia",
    "Irish": "Ireland", "Italian": "Italy", "Japanese": "Japan",
    "Liechtensteiner": "Liechtenstein", "Malaysian": "Malaysia", "Mexican": "Mexico",
    "Monegasque": "Monaco", "New Zealander": "New Zealand", "Polish": "Poland",
    "Portuguese": "Portugal", "Rhodesian": "Rhodesia", "Russian": "Russia",
    "South African": "South Africa", "Spanish": "Spain", "Swedish": "Sweden",
    "Swiss": "Switzerland", "Thai": "Thailand", "Uruguayan": "Uruguay",
    "Venezuelan": "Venezuela",
}

# Why a car retired, for the retirement (survival) model's labels. Statuses
# not listed are "mechanical" (the long tail is all parts: engine, gearbox,
# hydraulics ...). Since 2023 almost every retirement is just "Retired",
# which is "unknown".
DNF_INCIDENT = ("Accident", "Collision", "Collision damage", "Spun off", "Damage",
                "Debris", "Fatal accident", "Puncture", "Tyre puncture")
DNF_OTHER = ("Disqualified", "Excluded", "Underweight", "Withdrew", "Did not start",
             "Not restarted", "Physical", "Driver unwell", "Injury", "Injured",
             "Eye injury", "Illness", "Safety concerns", "Safety")
DNF_UNKNOWN = ("Retired", "Not classified")
# Entries that never really started (not a first-lap incident).
DID_NOT_START = ("Withdrew", "Did not start", "Disqualified", "Excluded", "Not restarted")


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
@dataclass
class EloConfig:
    """Tunable parameters for the ELO model.

    Defaults are the backtested settings (backtest_constructors.py and
    backtest_drivers.py, tuned on 1990-2022 with 2023 onwards held out). The
    comments say which value restores the original notebook's behaviour.
    """

    initial_elo: float = 1000.0

    # K-factors for constructors. Qualifying adds little to the car rating
    # once real qualifying results are used (the race pass already captures
    # pace), so its K is small outside an era start.
    k_constructor_race: float = 45.0
    k_constructor_quali: float = 7.0

    # Era-start boost for constructors. A big regulation change reshuffles
    # the order, so ratings need to move faster until they catch up. Each
    # season in `regulation_change_years` starts with an "era confidence" of
    # 0%, rising by 1/era_start_races per race (7 races -> about 14% per race).
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
    # era-start K, round 8 onwards the normal K. Backtested together with the
    # era-start K-factors below, on the regulation-change seasons 1998-2022.
    # Right after a rule change qualifying is the quickest read of the new
    # order, hence the large quali boost (79 vs a normal 7).
    era_start_races: int = 7
    k_constructor_race_era_start: float | None = 94.0
    k_constructor_quali_era_start: float | None = 79.0

    # K-factors for drivers, scaled by a confidence level.
    # Confidence runs from 0 to driver_confidence_races "races' worth"
    # (40 -> each race is 2.5%, about two seasons to full confidence). The
    # driver's K slides linearly from the rookie value at 0% to the veteran
    # value at 100%, for both race and quali (rookie = 2.75 x veteran).
    # After each race: a classified finish adds 1; a DNF takes 1 away; every
    # race a driver misses (once they've started at least one) takes 1 away.
    # So a full season out (~22 races) takes away a little over half.
    driver_confidence_races: int = 40
    # At the first race of each season in regulation_change_years, every
    # driver's confidence drops by this share of the full scale (0.5 = 50
    # percentage points, i.e. 20 races' worth), never below 0%. New cars
    # mean even veterans' ratings should move faster for a while.
    era_start_confidence_drop: float = 0.5
    k_driver_race_rookie: float = 187.0
    k_driver_race_veteran: float = 68.0
    k_driver_quali_rookie: float = 214.5
    k_driver_quali_veteran: float = 78.0

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
    # 1.0 = off (the original behaviour). 0.6 backtested best, clearly so on
    # 1950-1979, where part-timers were common.
    driver_opponent_confidence_floor: float = 0.6

    # Weight of the car in the driver expectation: driver A vs driver B is
    # predicted from (D_a + driver_car_weight * C_a) vs
    # (D_b + driver_car_weight * C_b), C = constructor rating. 0 = drivers
    # only (the original behaviour, which lets car strength leak into driver
    # ratings: a driver in a dominant car keeps gaining for wins the car
    # earned). Teammate pairs are unaffected, since they share a car.
    # Backtested on teammate pairs, 1990-2022: 0.75-1 effectively tied;
    # 0.875 is best on new-teammate pairs. Use it with
    # constructor_adjustment_exponent = 0, so all of the car's effect is in
    # this weight (the race predictions use D + driver_car_weight * C).
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
    driver_dnf_penalty_factor: float = 0.4

    # Field-size correction, for every pass (drivers and constructors,
    # qualifying and race). Each entry's summed pair deltas are averaged
    # over its n - 1 opponents, then scaled by (ref - 1) / ref, so a race
    # moves ratings by the same amount whatever the grid size, and exactly
    # as much as before on a grid of this many cars (K keeps its meaning).
    # Without it, deltas are divided by the number of cars n, i.e. scaled
    # by (n - 1) / n: a 10-car race moved ratings 5% less than a 20-car one.
    # None = no correction (the original divide-by-n).
    reference_field_size: int | None = 20

    # Makes with more than two cars (common before 1980: the data identifies
    # a car by its make, so works and privateer Maseratis or Coopers share one
    # constructorId -- up to 15 in one race).
    #
    # Constructors: each make's update is the AVERAGE over its cars in that
    # session, scaled to this many cars, instead of the sum. Every car counts
    # equally, so a make's rating is its typical car, and a 10-car make no
    # longer moves 5x as far per race as a 2-car team. With 2 every two-car
    # team moves exactly as before; a one-car team moves as far as a two-car
    # team whose cars did the same. None = the original sum over cars.
    constructor_reference_cars: int | None = 2
    # Drivers: pairs within one make share a car rating, so the car term
    # cancels and the whole gap counts as driver skill -- wrong for a works
    # driver beating privateers in an older car of the same make. With this
    # on, every same-make pair is weighted 1 / (cars - 1): each driver's
    # comparisons with their own make add up to one teammate's worth,
    # however many cars it ran, all weighted equally. The driver's race is
    # then averaged over the weight of comparisons they actually had
    # (n - cars + 1 instead of n - 1), so a race moves them as far as anyone
    # else's -- the same-make pairs only count less within it. Two-car
    # teams are unchanged. False = every same-make pair at full weight
    # (original).
    driver_shared_make_weighting: bool = True

    # Order the qualifying pass by the real qualifying result (qualifying.csv,
    # 1994 onwards) instead of the starting grid. The grid includes penalties
    # (engine, gearbox, pit-lane starts): 23.5% of entries since 1994 started
    # somewhere other than where they qualified, so a penalised driver lost
    # rating for it and the drivers moved up gained it. Races without
    # qualifying data (all before 1994, a few after) still use the grid. In
    # a race with data, a driver missing from it counts as no time (last).
    # False = always the grid (the original behaviour).
    use_qualifying_results: bool = True

    # Standard ELO logistic scale (the "400" in the classic formula).
    elo_scale: float = 400.0

    # Race predictions (f1elo/strength.py), not the rating updates: each
    # car's pre-race strength is D + prediction_car_weight * C, and car A
    # beats car B with probability expected_score(S_a, S_b, prediction_scale).
    # Kept apart from driver_car_weight / elo_scale, which are tuned for the
    # engine's updates on teammate pairs; these are tuned on every pair of
    # cars in a race, a retirement losing to every classified car
    # (backtest_strength.py, 1990-2022, pre-race ratings).
    # Best 1.875 / 1175 (log loss 0.5592 vs 0.6209 with the engine's
    # 0.875 / 400, 0.5667 with the car alone); a flat ridge, w 1.5-2.25 with
    # s 1000-1400 within 0.0005. Holds on 2023-2026: 0.5246 vs 0.5403.
    # Why flatter and more car than the engine: 43% of those pairs were
    # decided by a retirement, which ratings barely predict (hence the
    # larger scale) and which is mostly the car (hence the larger weight).
    # On pairs with both cars classified the best is 1.25 / 525, with
    # 0.875 / 400 only 0.002 worse -- what a race model that handles
    # retirements separately should use (--pairs classified).
    prediction_car_weight: float = 1.875
    prediction_scale: float = 1175.0

    # Finishing order among the cars that FINISH, for the race simulation
    # (f1elo/simulate.py), which draws retirements separately (below). Same
    # strength D + w*C, Plackett-Luce weights 10^(S / finish_scale), so
    # P(A ahead of B | both classified) = expected_score(S_a, S_b,
    # finish_scale). Tuned with backtest_strength.py --pairs classified,
    # 1990-2022 (pairs with both cars classified): best 1.25 / 525, log
    # loss 0.4221 (max. likelihood 1.228 / 510; w 1-1.5 with s 450-575
    # within 0.0005); the engine's 0.875 / 400 0.4239. Holdout 2023-2026:
    # 0.4492 vs 0.4517. Use prediction_car_weight / prediction_scale for
    # a one-step view where a retirement just loses to every finisher;
    # use these when retirements are modelled on their own.
    finish_car_weight: float = 1.25
    finish_scale: float = 525.0

    # The front of the race: the first finish_top_places places are filled
    # one at a time with Plackett-Luce at finish_top_scale (sharper), the
    # rest with finish_scale. 0 places = off (one scale for the whole order).
    # Why: one scale can't fit both ends of the order. The exact
    # Plackett-Luce likelihood of the real finishing orders (1990-2022)
    # wants a scale of ~300 for the winner, ~400 for the top 3 and 650+ for
    # the whole classified order; pairs (above) settle in between at 525,
    # which made win and podium chances too flat (cars given 24% won 34% of
    # the time). Tuned on whole results (backtest_calibration.py
    # --experiments finish, 2,000 simulations per race), best podium log
    # loss on 1990-2022 at w 1.25, top 400 for 3 places; tune / holdout
    # 2023-2026, vs one scale at 525:
    #   winner log loss  1.802 -> 1.728   /  1.700 -> 1.582
    #   podium log loss  0.2524 -> 0.2481 /  0.2648 -> 0.2607
    #   points log loss  0.4720 -> 0.4720 /  0.4949 -> 0.4948
    #   RPS              0.1607 -> 0.1606 /  0.1294 -> 0.1291
    # Flat nearby: top 375-450 with 3-6 places and w 1.125-1.375 are within
    # 0.0005 on podium; winner log loss alone would prefer 350 (1.713).
    finish_top_scale: float = 400.0
    finish_top_places: int = 3

    # Extra spread for drivers the engine isn't sure of: each simulated race
    # adds normal noise with standard deviation finish_newcomer_spread *
    # (1 - driver confidence) rating points to the driver's strength.
    # 0 = off. Tried 50-200 (backtest_calibration.py --experiments
    # newcomer): every value made the 1990-2022 scores worse, rookies' own
    # included (first 20 starts: podium log loss 0.0906 at 0, 0.0915 at 50,
    # 0.0920 at 200). Rookies do beat their ratings slightly (88 podiums vs
    # 78 predicted, 438 points finishes vs 412) -- a level, not a spread.
    finish_newcomer_spread: float = 0.0

    # Shared race factor for retirements (f1elo/simulate.py,
    # draw_retirements): the correlation between cars' chances of trouble
    # in the same race (a Gaussian copula). Each car keeps its own P(DNF);
    # only how often many cars retire together changes. 0 = independent.
    # Fitted to the variance of the number of finishers per race
    # (backtest_calibration.py --experiments corr, exact distributions):
    # independent draws give 4.1 vs 5.9 actual on 1990-2022 (ratio 1.44);
    # 0.04 gives 6.0 (ratio 0.97; matched at 0.037), holdout ratio 1.32 ->
    # 1.01. Log loss of the number of finishers 2.302 -> 2.261 (tune),
    # 1.939 -> 1.914 (holdout), best at 0.04 on both. Win / podium / points
    # / RPS are unchanged within noise (podium +0.0001, intervals across 0).
    dnf_race_corr: float = 0.04

    # Retirement model (f1elo/reliability.py): each car's chance of not
    # finishing, from pre-race data only. Counts over the last N races,
    # shrunk toward the field's recent rate (base) by k entries' worth:
    #   base   = field DNF share over its last dnf_base_races races
    #   team   = (team DNFs + k_t * base) / (team entries + k_t), team's
    #            last dnf_team_races races (all its cars)
    #   driver = same over the driver's last dnf_driver_races races
    #   blend  = (1 - dnf_driver_weight) * team + dnf_driver_weight * driver
    #   logit P(DNF) = a + b * logit(blend) + c * car_rel + d * driver_rel
    # car_rel / driver_rel = the pre-race car / driver rating minus the
    # field's mean, per 100 points. a, b, c, d = 0, 1, 0, 0 is the plain
    # blend.
    # Tuned by log loss on "did this car retire" (position \N), 1990-2022,
    # 2023 on held out (backtest_reliability.py); a, b, c, d by logistic
    # regression on 1990-2022. Log loss, tune / holdout 2023-2026:
    #   base rate alone (last 40 races)   0.5665 / 0.3808
    #   blend                             0.5560 / 0.3762
    #   blend + rating step (these)       0.5534 / 0.3744
    # The grid is flat: base 20-60 races, team 20-30 races with k 0-20,
    # driver 30-60 races with k 20-100, driver weight 0.3-0.7 all within
    # 0.0005. The rating step also holds when fitted on 1990-2009 only and
    # scored on 2010-2022 (0.4491 vs 0.4494 blend, 0.4537 base). Better
    # drivers retire less (d), beyond what their own record says; the car
    # rating adds little once the team's record is in (c).
    dnf_base_races: int = 40
    dnf_team_races: int = 20
    dnf_team_shrink: float = 0.0
    dnf_driver_races: int = 50
    dnf_driver_shrink: float = 60.0
    dnf_driver_weight: float = 0.6
    dnf_logit_intercept: float = -0.116
    dnf_logit_blend: float = 0.939
    dnf_logit_car: float = -0.020
    dnf_logit_driver: float = -0.086

    # After qualifying only (backtest_calibration.py --stage post_quali):
    # the grid slot in the retirement chance,
    #   logit P' = logit P(DNF) + a + b * slot + c * slot^2 + d * pit lane
    # with slot = (grid - 1) / (n - 1), 0 = pole, 1 = last (grid 0 = last).
    # All 0 = off; the pre-race prediction never uses them. Fitted on
    # 1990-2022 by logistic regression with the model's logit as an offset:
    # the front of the grid retires less (pole -0.10 logit, last +0.09).
    # DNF log loss tune / holdout 0.5531 -> 0.5528 / 0.3737 -> 0.3726, and
    # every race score improves a little on both (points 0.4635 -> 0.4625 /
    # 0.4803 -> 0.4795). A slot^2 term adds nothing (0.5527 / 0.3727). A
    # pit-lane flag looks better on 1990-2022 (0.5520) but grid 0 also marks
    # cars that withdrew before the start, which isn't known after
    # qualifying, and it is worse on the holdout (0.3729): not used.
    dnf_grid_a: float = -0.099
    dnf_grid_b: float = 0.188
    dnf_grid_c: float = 0.0
    dnf_grid_pitlane: float = 0.0

    # Sentinels in the raw data.
    dnf_sentinel: str = "\\N"   # finishing "position" value for did-not-finish
    no_grid_sentinel: int = 0    # "grid" / qualifying value meaning no time (last)


# A ready-to-use default instance.
DEFAULT_CONFIG = EloConfig()
