"""F1 ELO rating system.

Public API::

    from f1elo import build_race_frame, EloEngine, load_data

    raw = load_data()                       # dict of the 4 raw CSVs
    df = build_race_frame()                 # merged, chronological race frame
    engine = EloEngine()
    df = engine.run(df)                     # fills EloRating / DriverEloRating

    from f1elo import analysis
    analysis.top_drivers_by_peak(df, raw["drivers"])
"""
from .config import DEFAULT_CONFIG, EloConfig
from .data_loader import build_race_frame, load_data
from .elo_engine import EloEngine, expected_score

__all__ = [
    "DEFAULT_CONFIG",
    "EloConfig",
    "build_race_frame",
    "load_data",
    "EloEngine",
    "expected_score",
]
