"""The ELO rating engine.

This reproduces the algorithm from the original notebook, but as a single
stateful class instead of a wall of global dicts and functions.

How it works, per race:
  1. QUALIFYING pass, ordered by grid position (0 = no qualifying time = last).
     - Update constructor ratings (low K), then driver ratings (low K).
  2. RACE pass, ordered by finishing position ("\\N" = DNF = last).
     - Update constructor ratings (high K), then driver ratings (high K).
     - Drivers only: a classified-vs-DNF pair is scaled by
       ``driver_dnf_penalty_factor`` (default 0.5), so a retirement costs
       the driver half as much. Constructors take the full penalty.

Within each pass it does a full pairwise comparison: every entry is treated as
having "beaten" every entry classified below it (score 1 vs 0). The per-entry
ELO deltas are accumulated across all pairs, then divided by the size of the
field before being applied -- so a single race moves a rating by roughly one
K-step, not N of them.

Driver updates are additionally scaled by a constructor-strength adjustment:
a driver in a much stronger car gains less for a win and loses more for a loss.
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

        # Number of classified races, used to pick driver K-factors.
        self.constructor_race_count: dict[int, int] = {}
        self.driver_race_count: dict[int, int] = {}

        # True while processing a race early in a new regulation era.
        self.era_start = False

    # -- K-factors ---------------------------------------------------------
    def in_era_start(self, year: int, rnd: int) -> bool:
        """Is this race inside the boosted start of a regulation era?"""
        c = self.cfg
        return year in c.regulation_change_years and rnd <= c.era_start_races

    def _k_constructor(self, is_race: bool) -> float:
        c = self.cfg
        if self.era_start:
            boost = (c.k_constructor_race_era_start if is_race
                     else c.k_constructor_quali_era_start)
            if boost is not None:
                return boost
        return c.k_constructor_race if is_race else c.k_constructor_quali

    def _k_driver(self, race_count: int, is_race: bool) -> float:
        c = self.cfg
        rookie = race_count <= c.driver_experience_threshold
        if is_race:
            return c.k_driver_race_rookie if rookie else c.k_driver_race_veteran
        return c.k_driver_quali_rookie if rookie else c.k_driver_quali_veteran

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

    @staticmethod
    def _apply(ratings: dict, changes: dict, field_size: int) -> None:
        """Apply accumulated deltas, normalised by the size of the field."""
        if field_size == 0:
            return
        for entity_id, delta in changes.items():
            ratings[entity_id] = ratings.get(entity_id, 0.0) + delta / field_size

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
                     is_race: bool) -> None:
        """One full pairwise driver update, scaled by constructor strength."""
        cfg = self.cfg
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
                con_ra = self.constructor_elo[ca]
                con_rb = self.constructor_elo[cb]

                k_a = self._k_driver(self.driver_race_count.get(a, 0), is_race)
                k_b = self._k_driver(self.driver_race_count.get(b, 0), is_race)
                ea = expected_score(ra, rb, cfg.elo_scale)
                eb = 1.0 - ea

                # Better car -> smaller reward for the win (and vice versa).
                adj = ((con_rb / con_ra) if con_ra != 0 else 1.0) ** \
                    cfg.constructor_adjustment_exponent

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
        self.era_start = self.in_era_start(int(race["year"].iloc[0]),
                                           int(race["round"].iloc[0]))

        # ----- QUALIFYING (low K), ordered by grid -----
        quali_order = self._order_by_grid(race, cfg.no_grid_sentinel)
        self._constructor_pass(quali_order, "grid", cfg.no_grid_sentinel, is_race=False)
        self._driver_pass(quali_order, "grid", cfg.no_grid_sentinel, is_race=False)

        # ----- RACE (high K), ordered by finishing position -----
        race_order = self._order_by_position(race)
        self._constructor_pass(race_order, "position", cfg.dnf_sentinel, is_race=True)
        self._driver_pass(race_order, "position", cfg.dnf_sentinel, is_race=True)

        return race, dict(self.constructor_elo), dict(self.driver_elo)

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
