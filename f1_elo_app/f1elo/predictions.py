"""Finishing-position probability predictor.

Given a field of drivers (each with a driver ELO and, optionally, a constructor
ELO), estimate the probability that a chosen driver finishes in each position.

This reproduces the notebook's approach: for each target position, multiply the
pairwise expected scores against every other driver, then normalise across all
positions so the distribution sums to 1.
"""
from __future__ import annotations

import pandas as pd

from .elo_engine import expected_score


def position_probabilities(field: pd.DataFrame, driver_name: str,
                           driver_elo_col: str = "DriverEloRating",
                           constructor_elo_col: str | None = None,
                           scale: float = 400.0) -> dict[str, float]:
    """Probability distribution over finishing positions for one driver.

    ``field`` must contain a 'driverName' column plus the rating column(s).
    If ``constructor_elo_col`` is given, constructor strength is folded in too.
    """
    field = field.reset_index(drop=True)
    names = field["driverName"].tolist()
    driver_elo = field[driver_elo_col].tolist()
    con_elo = (field[constructor_elo_col].tolist()
               if constructor_elo_col else None)

    if driver_name not in names:
        raise ValueError(f"{driver_name!r} is not in the field.")
    target = names.index(driver_name)
    n = len(names)

    probs: list[float] = []
    for pos in range(n):
        # The `pos` strongest *other* drivers are assumed to be ahead.
        ahead = [i for i in range(n) if i != target][:pos]
        p = 1.0
        for i in range(n):
            if i == target:
                continue
            if i in ahead:
                p *= expected_score(driver_elo[i], driver_elo[target], scale)
                if con_elo is not None:
                    p *= expected_score(con_elo[i], con_elo[target], scale)
            else:
                p *= expected_score(driver_elo[target], driver_elo[i], scale)
                if con_elo is not None:
                    p *= expected_score(con_elo[target], con_elo[i], scale)
        probs.append(p)

    total = sum(probs) or 1.0
    return {f"P{pos + 1}": p / total for pos, p in enumerate(probs)}
