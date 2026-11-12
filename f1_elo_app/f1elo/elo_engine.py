"""The ELO rating engine.

This reproduces the algorithm from the original notebook, but as a single
stateful class instead of a wall of global dicts and functions.

How it works, per race:
  1. QUALIFYING pass, ordered by grid position (0 = no qualifying time = last).
     - Update constructor ratings (low K), then driver ratings (low K).
  2. RACE pass, ordered by finishing position ("\\N" = DNF = last).
     - Update constructor ratings (high K), then driver ratings (high K).
     - Drivers only: a classified-vs-DNF pair is scaled by
       ``driver_dnf_penalty_factor`` (default 0.25), so a retirement costs
       the driver a quarter as much. Constructors take the full penalty.

Within each pass it does a full pairwise comparison: every entry is treated as
having "beaten" every entry classified below it (score 1 vs 0). The per-entry
ELO deltas are accumulated across all pairs, then averaged over the n - 1
opponents and scaled by (ref - 1) / ref, with ref = ``reference_field_size``
(20) -- so a single race moves a rating by roughly one K-step, not N of them,
and by the same amount on any grid size (exactly as before on 20 cars).
With ``reference_field_size = None`` the deltas are divided by n instead,
the original behaviour, which moved ratings 5% less on a 10-car grid than on
a 20-car one.

Driver pairs are predicted from driver AND car strength: the expected result
of driver A vs driver B is ``expected_score(D_a + lam*C_a, D_b + lam*C_b)``,
with D the driver ratings, C the constructor ratings and lam
``driver_car_weight``. A driver in a dominant car is then expected to beat
midfield drivers and gains little for doing it; their rating only rises for
results better than the car explains, which keeps car strength from leaking
into the driver rating. Teammates share C, so for them only D matters.
Updates can also be scaled by ``(C_b / C_a) ** constructor_adjustment_exponent``
(an older, cruder way to discount wins in a better car; 0 turns it off).

Driver K-factors depend on a confidence level (0-20 races' worth, 5% per
race): rookie K at 0%, veteran K at 100%, linear in between. A classified
finish adds a race of confidence; a DNF or a missed race takes one away. At
the first race of a regulation-change season, every driver loses
``era_start_confidence_drop`` (50 percentage points) of confidence.
"""
from __future__ import annotations

import pandas as pd

from .config import DEFAULT_CONFIG, EloConfig


def expected_score(ra: float, rb: float, scale: float = 400.0) -> float:
    """Classic ELO expectation: probability that A beats B."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))


class EloEngine:
    """Holds all rating state and processes races into ELO ratings."""

    def __init__(self, cfg: EloConfig = DEFAULT_CONFIG):
        self.cfg = cfg

        # Current ratings.
        self.constructor_elo: dict[int, float] = {}
        self.driver_elo: dict[int, float] = {}

        # Peak rating ever reached (the notebook tracked these too).
        self.highest_constructor: dict[int, float] = {}
        self.highest_driver: dict[int, float] = {}

        # Number of classified races (kept for reference).
        self.constructor_race_count: dict[int, int] = {}
        self.driver_race_count: dict[int, int] = {}

        # Driver confidence, in "races' worth" (0 .. driver_confidence_races).
        # Drives the rookie -> veteran K-factor slide. Only drivers who have
        # started at least one race are in here.
        self.driver_confidence: dict[int, float] = {}

        # Season of the last race processed, to spot the first race of a year.
        self._last_year: int | None = None

        # Era confidence for the race being processed: 0.0 at the first race
        # of a regulation-change season, 1.0 once the era has settled.
        self.era_conf = 1.0

    # -- K-factors ---------------------------------------------------------
    def era_confidence(self, year: int, rnd: int) -> float:
        """0.0 at round 1 of a regulation-change season, rising by
        1/era_start_races per race already run, capped at 1.0. Any other
        season is 1.0 (settled)."""
        c = self.cfg
        if year not in c.regulation_change_years or c.era_start_races <= 0:
            return 1.0
        return min(1.0, (rnd - 1) / c.era_start_races)

    def _k_constructor(self, is_race: bool) -> float:
        """Slide from the era-start K (0% era confidence) to the normal K."""
        c = self.cfg
        normal = c.k_constructor_race if is_race else c.k_constructor_quali
        boost = (c.k_constructor_race_era_start if is_race
                 else c.k_constructor_quali_era_start)
        if boost is None or self.era_conf >= 1.0:
            return normal
        return boost + (normal - boost) * self.era_conf

    def confidence_pct(self, driver_id) -> float:
        """Driver confidence as a fraction: 0.0 (none) to 1.0 (full)."""
        n = self.cfg.driver_confidence_races
        if n <= 0:
            return 1.0
        return min(self.driver_confidence.get(driver_id, 0), n) / n

    def _k_driver(self, driver_id, is_race: bool) -> float:
        """Slide linearly from the rookie K (0%) to the veteran K (100%)."""
        c = self.cfg
        conf = self.confidence_pct(driver_id)
        if is_race:
            rookie, veteran = c.k_driver_race_rookie, c.k_driver_race_veteran
        else:
            rookie, veteran = c.k_driver_quali_rookie, c.k_driver_quali_veteran
        return rookie + (veteran - rookie) * conf

    def _era_start_confidence_drop(self) -> None:
        """New regulations: knock every driver's confidence down (floor 0)."""
        drop = self.cfg.era_start_confidence_drop * self.cfg.driver_confidence_races
        for d in self.driver_confidence:
            self.driver_confidence[d] = max(0.0, self.driver_confidence[d] - drop)

    def _update_confidence(self, race: pd.DataFrame) -> None:
        """After a race: +1 for a classified finish, -1 for a DNF, and -1 for
        every driver who has raced before but wasn't in this one. Clamped to
        0 .. driver_confidence_races."""
        cap = self.cfg.driver_confidence_races
        sentinel = self.cfg.dnf_sentinel

        # A driver with several entries in one race (1950s shared drives)
        # counts as finishing if any of their entries was classified.
        finished: dict = {}
        for d, pos in zip(race["driverId"], race["position"]):
            finished[d] = finished.get(d, False) or pos != sentinel

        conf = self.driver_confidence
        for d in conf:
            if d not in finished and conf[d] > 0:
                conf[d] -= 1                      # missed this race
        for d, ok in finished.items():
            conf[d] = min(cap, max(0, conf.get(d, 0) + (1 if ok else -1)))

    # -- state helpers -----------------------------------------------------
    def _ensure_entities(self, df: pd.DataFrame) -> None:
        """Seed any not-yet-seen drivers/constructors at the initial rating."""
        init = self.cfg.initial_elo
        for cid in df["constructorId"].unique():
            self.constructor_elo.setdefault(cid, init)
            self.highest_constructor.setdefault(cid, init)
        for did in df["driverId"].unique():
            self.driver_elo.setdefault(did, init)
            self.highest_driver.setdefault(did, init)

    @staticmethod
    def _record(changes: dict, highest: dict,
                a, b, new_ra, new_rb, old_ra, old_rb) -> None:
        """Accumulate a pair's deltas and update peak ratings."""
        changes[a] = changes.get(a, 0.0) + (new_ra - old_ra)
        changes[b] = changes.get(b, 0.0) + (new_rb - old_rb)
        highest[a] = max(highest[a], new_ra)
        highest[b] = max(highest[b], new_rb)

    def _field_divisor(self, field_size: int) -> float:
        """What a pass's summed deltas are divided by.

        With ``reference_field_size`` set: (n - 1) * ref / (ref - 1), i.e. the
        average over the n - 1 opponents, scaled by (ref - 1) / ref. That is
        exactly n on a reference-size grid, so K keeps its meaning there,
        and the size of the field no longer changes how far a rating moves.
        Without it: n, the original behaviour.
        """
        ref = self.cfg.reference_field_size
        if ref is None or ref < 2 or field_size < 2:
            return float(field_size)
        return (field_size - 1) * ref / (ref - 1)

    def _apply(self, ratings: dict, changes: dict, field_size: int) -> None:
        """Apply accumulated deltas, normalised by the size of the field
        (see ``_field_divisor``)."""
        if field_size == 0:
            return
        divisor = self._field_divisor(field_size)
        for entity_id, delta in changes.items():
            ratings[entity_id] = ratings.get(entity_id, 0.0) + delta / divisor

    # -- pairwise passes ---------------------------------------------------
    def _constructor_pass(self, order: pd.DataFrame, col: str, sentinel,
                          is_race: bool) -> None:
        """One full pairwise constructor update over an ordered field."""
        cfg = self.cfg
        ids = order["constructorId"].tolist()
        vals = order[col].tolist()
        n = len(ids)

        changes: dict = {}
        for i in range(n):
            a, va = ids[i], vals[i]
            for j in range(i + 1, n):
                b, vb = ids[j], vals[j]
                if va == sentinel and vb == sentinel:
                    continue
                if a == b:  # teammates -> no comparison
                    continue
                ra, rb = self.constructor_elo[a], self.constructor_elo[b]
                k_a = self._k_constructor(is_race)
                k_b = self._k_constructor(is_race)
                ea = expected_score(ra, rb, cfg.elo_scale)
                eb = 1.0 - ea
                new_ra = ra + k_a * (1 - ea)   # i is ahead of j -> sa = 1
                new_rb = rb + k_b * (0 - eb)
                self._record(changes, self.highest_constructor,
                             a, b, new_ra, new_rb, ra, rb)

            # Race pass increments the classified-race counter.
            if is_race and va != sentinel:
                self.constructor_race_count[a] = (
                    self.constructor_race_count.get(a, 0) + 1
                )

        self._apply(self.constructor_elo, changes, n)

    def _driver_pass(self, order: pd.DataFrame, col: str, sentinel,
                     is_race: bool, car_elo: dict | None = None) -> None:
        """One full pairwise driver update, with the car in the expectation.

        ``car_elo`` is the constructor rating table to use for the cars
        (default: the current one).
        """
        cfg = self.cfg
        car_elo = self.constructor_elo if car_elo is None else car_elo
        lam = cfg.driver_car_weight
        exponent = cfg.constructor_adjustment_exponent
        driver_ids = order["driverId"].tolist()
        constructor_ids = order["constructorId"].tolist()
        vals = order[col].tolist()
        n = len(driver_ids)

        changes: dict = {}
        for i in range(n):
            a, ca, va = driver_ids[i], constructor_ids[i], vals[i]
            for j in range(i + 1, n):
                b, cb, vb = driver_ids[j], constructor_ids[j], vals[j]
                if va == sentinel and vb == sentinel:
                    continue
                if a == b:
                    continue
                ra, rb = self.driver_elo[a], self.driver_elo[b]
                con_ra = car_elo[ca]
                con_rb = car_elo[cb]

                k_a = self._k_driver(a, is_race)
                k_b = self._k_driver(b, is_race)
                # Expected result from driver + weighted car strength, so a
                # win the car explains earns little (teammates: same car).
                ea = expected_score(ra + lam * con_ra, rb + lam * con_rb,
                                    cfg.elo_scale)
                eb = 1.0 - ea

                # Optional extra discount: better car -> smaller reward for
                # the win (and vice versa). Exponent 0 -> adj = 1.
                adj = (1.0 if exponent == 0 or con_ra == 0
                       else (con_rb / con_ra) ** exponent)

                # Classified driver vs a DNF in the race: soften the pair
                # (the car may have failed). Applied to both sides so the
                # pair stays balanced. DNF-vs-DNF pairs were skipped above.
                dnf = (cfg.driver_dnf_penalty_factor
                       if is_race and vb == sentinel else 1.0)

                new_ra = ra + k_a * (1 - ea) * adj * dnf   # sa = 1
                new_rb = rb + k_b * (0 - eb) * adj * dnf    # sb = 0
                self._record(changes, self.highest_driver,
                             a, b, new_ra, new_rb, ra, rb)

            if is_race and va != sentinel:
                self.driver_race_count[a] = self.driver_race_count.get(a, 0) + 1

        self._apply(self.driver_elo, changes, n)

    # -- per-race orchestration -------------------------------------------
    @staticmethod
    def _order_by_grid(race: pd.DataFrame, no_grid_sentinel: int) -> pd.DataFrame:
        """Sort a race's entries by grid, pushing 'no grid' (0) to the back."""
        r = race.copy()
        max_grid = r["grid"].max()
        r["_grid_sort"] = r["grid"].replace(no_grid_sentinel, max_grid + 1)
        r = r.sort_values(by="_grid_sort")
        return r.drop(columns="_grid_sort")

    @staticmethod
    def _order_by_position(race: pd.DataFrame) -> pd.DataFrame:
        """Sort by finishing position, pushing DNF (non-numeric) to the back."""
        r = race.copy()
        r["_pos"] = pd.to_numeric(r["position"], errors="coerce")
        r = r.sort_values(by="_pos", na_position="last")
        return r.drop(columns="_pos")

    def _process_one_race(self, race: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
        """Run quali + race passes for a single race.

        Returns the (possibly reordered) race frame plus the post-race
        constructor / driver rating snapshots to write back.
        """
        cfg = self.cfg
        self.process_ordered(
            int(race["year"].iloc[0]), int(race["round"].iloc[0]),
            self._order_by_grid(race, cfg.no_grid_sentinel),
            self._order_by_position(race),
        )
        return race, dict(self.constructor_elo), dict(self.driver_elo)

    def process_ordered(self, year: int, rnd: int, quali_order: pd.DataFrame,
                        race_order: pd.DataFrame) -> None:
        """Process one race whose entries are already sorted.

        ``quali_order`` is sorted by grid and ``race_order`` by finishing
        position; both need driverId, constructorId and grid / position.
        Backtests call this directly so they share the exact same logic.
        """
        cfg = self.cfg
        self.era_conf = self.era_confidence(year, rnd)

        # First race of a regulation-change season: drop driver confidence
        # before this race, so the higher K applies from round 1.
        if year != self._last_year:
            if year in cfg.regulation_change_years:
                self._era_start_confidence_drop()
            self._last_year = year

        # Which car ratings the driver passes see: the ones from before this
        # session's constructor update (the session's own result not yet in
        # them), or the freshly updated ones.
        snap = dict if cfg.driver_car_before_update else (lambda d: None)

        # ----- QUALIFYING (low K), ordered by grid -----
        cars = snap(self.constructor_elo)
        self._constructor_pass(quali_order, "grid", cfg.no_grid_sentinel, is_race=False)
        self._driver_pass(quali_order, "grid", cfg.no_grid_sentinel, is_race=False,
                          car_elo=cars)

        # ----- RACE (high K), ordered by finishing position -----
        cars = snap(self.constructor_elo)
        self._constructor_pass(race_order, "position", cfg.dnf_sentinel, is_race=True)
        self._driver_pass(race_order, "position", cfg.dnf_sentinel, is_race=True,
                          car_elo=cars)

        # Confidence changes only after the whole weekend, so it affects the
        # NEXT race's K-factors.
        self._update_confidence(race_order)

    # -- public entry point ------------------------------------------------
    def run(self, df: pd.DataFrame, progress: bool = False) -> pd.DataFrame:
        """Process every race in chronological order.

        Fills the ``EloRating`` (constructor) and ``DriverEloRating`` columns
        with the rating *after* each race, and returns the same DataFrame.
        """
        self._ensure_entities(df)

        race_ids = df["raceId"].unique()
        total = len(race_ids)

        for n, race_id in enumerate(race_ids, 1):
            race = df[df["raceId"] == race_id]
            self._process_one_race(race)

            # Write the current ratings onto every row of this race.
            mask = df["raceId"] == race_id
            df.loc[mask, "EloRating"] = (
                df.loc[mask, "constructorId"].map(self.constructor_elo)
            )
            df.loc[mask, "DriverEloRating"] = (
                df.loc[mask, "driverId"].map(self.driver_elo)
            )

            if progress and (n % 100 == 0 or n == total):
                print(f"  processed {n}/{total} races", flush=True)

        return df
